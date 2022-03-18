# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code pertaining to training models."""

import dataclasses
import functools
import itertools
import re
import time
from typing import Optional, Text, Tuple, Union

from absl import logging
from flax import jax_utils
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from plur.model_design import checkpoint
from plur.model_design import data_generation as dg
from plur.model_design import data_types as dt
from plur.model_design import losses
from plur.model_design import measurements
from plur.model_design import metrics
from plur.model_design import model_configs
from plur.model_design import models as tocopo_models
from plur.utils import constants


KEY_SEQ_ACCURACY = 'seq_accuracy'


def is_main_process():
  """Returns whether the current process is the main one."""
  return jax.process_index() == 0


def flags_to_run_name(flag_values):
  """Converts a flags object to a string representation of the run name."""

  if flag_values.num_output_propagation_steps < 0:
    num_output_propagation_steps = flag_values.num_input_propagation_steps
  else:
    num_output_propagation_steps = flag_values.num_output_propagation_steps

  run_name_terms = [
      ('mk', flag_values.model_kind),
      ('h', flag_values.hidden_dim),
      ('nip', flag_values.num_input_propagation_steps),
      ('nop', num_output_propagation_steps),
      ('mgn', float(flag_values.max_gradient_norm)),
      ('bspd', flag_values.batch_size_per_device),
      ('lr', flag_values.learning_rate),
      ('mvb', flag_values.max_validation_batches),
  ]
  if flag_values.model_kind.startswith('transformer'):
    run_name_terms.append(('nh', flag_values.num_transformer_attention_heads))
    run_name_terms.append(('rb', flag_values.use_relational_bias))
  if flag_values.max_num_subtokens != 0:
    run_name_terms.append(('ns', flag_values.max_num_subtokens))
  if flag_values.warmup_steps_fraction:
    run_name_terms.append(
        ('wsf', float(flag_values.warmup_steps_fraction)))
  if flag_values.model_initialization_seed:
    run_name_terms.append(('seed', flag_values.model_initialization_seed))

  run_name = '_'.join([
      '{}{}'.format(short_name, value) for short_name, value in run_name_terms
  ])
  return run_name


def run_name_to_flags(checkpoint_path: Text):
  """Extracts flag settings from a saved checkpoint path name.

  TODO: Consider making a single class that converts flags to run names
  and run names to flags. Or find some existing code that does this.

  Args:
    checkpoint_path: Path where checkpoint is saved.

  Returns:
    Dictionary mapping attribute names to string representations of values.
  """
  pattern_components = [
      r'mk(?P<model_kind>[^_]+)',
      r'h(?P<hidden_dim>[^_]+)',
      r'nip(?P<num_input_propagation_steps>[^_]+)',
      r'nop(?P<num_output_propagation_steps>[^_]+)',
      r'mgn(?P<max_gradient_norm>[^_]+)',
      r'bspd(?P<batch_size_per_device>[^_]+)',
      r'lr(?P<learning_rate>[^_]+)',
      r'mvb(?P<max_validation_batches>[^_/]+)',
  ]
  if 'mktransformer' in checkpoint_path:
    pattern_components.append('nh(?P<num_transformer_attention_heads>[^_]+)')
    pattern_components.append('rb(?P<use_relational_bias>[^_/]+)')
  if '_ns' in checkpoint_path:
    pattern_components.append('ns(?P<max_num_subtokens>[^_/]+)')
  if '_dropout' in checkpoint_path:
    pattern_components.append('dropout(?P<dropout_rate>[^_/]+)')
  if '_wsf' in checkpoint_path:
    pattern_components.append('wsf(?P<warmup_steps_fraction>[^_/]+)')
  if '_seed' in checkpoint_path:
    pattern_components.append('seed(?P<model_initialization_seed>[^_/]+)')

  pattern = '_'.join(pattern_components)
  match = re.search(pattern, checkpoint_path)
  if match is not None:
    model_flags = match.groupdict()
  else:
    raise ValueError(
        f'Failed to match run name pattern {pattern}\n{checkpoint_path}')

  # Convert back from string values.

  int_flags = ('num_input_propagation_steps', 'num_output_propagation_steps',
               'hidden_dim', 'num_transformer_attention_heads',
               'model_initialization_seed', 'batch_size_per_device',
               'max_num_subtokens', 'max_validation_batches')
  for flag in int_flags:
    if flag in model_flags:
      model_flags[flag] = int(model_flags[flag])

  # A missing seed means its value should be 0.
  model_flags.setdefault('model_initialization_seed', 0)

  float_flags = ('learning_rate', 'max_gradient_norm', 'warmup_steps_fraction',
                 'dropout_rate')
  for flag in float_flags:
    if flag in model_flags:
      model_flags[flag] = float(model_flags[flag])

  if 'use_relational_bias' in model_flags:
    model_flags['use_relational_bias'] = (
        model_flags['use_relational_bias'] == 'True')

  return model_flags


def load_model_with_data_manager(data_manager, model_checkpoint_path,
                                 fail_if_not_found=True, which_model='both'):
  """Loads a model given a DataManager object."""
  model_flags = run_name_to_flags(model_checkpoint_path)

  if model_flags['model_kind'] == 'ggnn2tocopo':
    model_class = tocopo_models.GGNN2Tocopo
  elif model_flags['model_kind'] == 'transformer2tocopo':
    model_class = tocopo_models.Transformer2Tocopo
  else:
    raise ValueError('Unknown model kind passed: {}'.format(
        model_flags['model_kind']))

  logging.warning('Pointer masking is not assumed False.')
  create_model_fn = functools.partial(
      create_model,
      token_vocab_size=data_manager.token_vocab_size,
      type_vocab_size=data_manager.type_vocab_size,
      output_vocab_size=data_manager.output_vocab_size,
      node_text_pad_token_id=data_manager.node_text_pad_token_id,
      output_oov_token_id=data_manager.output_oov_token_id,
      output_pad_token_id=data_manager.output_pad_token_id,
      padding_spec=data_manager.padding_spec,
      num_input_propagation_steps=model_flags['num_input_propagation_steps'],
      num_output_propagation_steps=model_flags['num_output_propagation_steps'],
      hidden_dim=model_flags['hidden_dim'],
      num_transformer_attention_heads=model_flags[
          'num_transformer_attention_heads'],
      use_relational_bias=model_flags['use_relational_bias'],
      model_temporal_relations=False,  # Hardcoding False.
      use_pointer_candidate_masking=False,  # Hardcoding False
      model_class=model_class,
      dropout_rate=model_flags['dropout_rate'],
      seed=model_flags['model_initialization_seed'])

  def _restore_model(model, checkpoint_path):
    state = checkpoint.TrainState(step=0, model=model, optimizer_state=None)

    state = checkpoint.restore_ckpt_from_dir(
        checkpoint_path, state, fail_if_not_found=fail_if_not_found)

    return jax_utils.replicate(state.model)

  if which_model == 'train':
    train_model, _, _ = create_model_fn(is_training=True)
    train_model = _restore_model(train_model, model_checkpoint_path)
    eval_model = None
  elif which_model == 'eval':
    train_model = None
    eval_model, _, _ = create_model_fn(is_training=False)
    eval_model = _restore_model(eval_model, model_checkpoint_path)
  elif which_model == 'both':
    train_model, _, _ = create_model_fn(is_training=True)
    train_model = _restore_model(train_model, model_checkpoint_path)
    eval_model, _, _ = create_model_fn(is_training=False)
    eval_model = _restore_model(eval_model, model_checkpoint_path)
  else:
    raise ValueError(f'Unexpected value for which_model {which_model}')

  return data_manager, train_model, eval_model


def create_model(token_vocab_size: int,
                 type_vocab_size: int,
                 output_vocab_size: int,
                 node_text_pad_token_id: int,
                 output_oov_token_id: int,
                 output_pad_token_id: int,
                 padding_spec: dt.Graph2TocopoPaddingSpec,
                 num_input_propagation_steps: int,
                 num_output_propagation_steps: int,
                 hidden_dim: int,
                 num_transformer_attention_heads: int = 1,
                 use_relational_bias: bool = False,
                 model_temporal_relations: bool = False,
                 use_pointer_candidate_masking: bool = False,
                 model_class=tocopo_models.GGNN2Tocopo,
                 is_training: bool = True,
                 jax2tf_compatible: bool = False,
                 dropout_rate: float = 0.1,
                 seed: int = 0):
  """Creates the model and model helper functions.

  Args:
    token_vocab_size: Node token vocabulary size.
    type_vocab_size: Node type vocabulary size.
    output_vocab_size: Output token vocabulary size.
    node_text_pad_token_id: Id of PAD token in the input token vocabulary.
    output_oov_token_id: Id of OOV token in the output token vocabulary.
    output_pad_token_id: Id of PAD token in the output token vocabulary.
    padding_spec: A PaddingSpec storing the sizes of dimensions to pad to.
    num_input_propagation_steps: Number of input propagation step.
    num_output_propagation_steps: Number of output propagation step.
    hidden_dim: Size of the hidden dimension.
    num_transformer_attention_heads: The number of attention heads for
      Transformer models to use.
    use_relational_bias: Whether to add edge-bias to Transformer model.
    model_temporal_relations: Whether to model temporal relations, if any.
    use_pointer_candidate_masking: Whether to use or ignore pointer candidates.
      NOTE, If set, pointer candidates must be present in the input examples
      otherwise pointer prediction will be disabled since there is nothing to
      point to.
    model_class: Class to use as the model.
    is_training: Whether to create a model for training, with, e.g., drop-out.
    jax2tf_compatible: Whether to add workarounds in the computation to prevent
      using some JAX operations not supported by TF.
    dropout_rate: The dropout rate.
    seed: The model initialization seed.

  Returns:
    A tuple of (model, loss, accuracy), where the model is a instance of
      model_class, loss and accuracy are pointers to the corresponding
      function.
  """
  rng = jax.random.PRNGKey(seed)  # Same seed on each host.
  if is_training:
    config_creator_fn = model_configs.get_train_config
  else:
    config_creator_fn = model_configs.get_eval_config
  config = config_creator_fn(
      max_input_length=padding_spec.num_nodes_per_graph,
      num_input_timesteps=padding_spec.num_input_timesteps,
      model_temporal_relations=model_temporal_relations,
      node_position_dim=padding_spec.node_position_dim,
      max_output_length=padding_spec.output_length,
      token_vocab_size=token_vocab_size,
      type_vocab_size=type_vocab_size,
      output_vocab_size=output_vocab_size,
      num_transformer_attention_heads=num_transformer_attention_heads,
      num_edge_types=padding_spec.num_edge_types,
      num_time_edge_types=padding_spec.num_time_edge_types,
      num_input_propagation_steps=num_input_propagation_steps,
      num_output_propagation_steps=num_output_propagation_steps,
      hidden_dim=hidden_dim,
      use_relational_bias=use_relational_bias,
      use_pointer_candidate_masking=use_pointer_candidate_masking,
      node_text_pad_token_id=node_text_pad_token_id,
      jax2tf_compatible=jax2tf_compatible,
      dropout_rate=dropout_rate)
  model = model_class.create(rng, config)
  loss = functools.partial(losses.tocopo_loss_fn,
                           oov_token_id=output_oov_token_id,
                           pad_token_id=output_pad_token_id)
  accuracy = functools.partial(
      metrics.tocopo_accuracy_fn,
      oov_token_id=output_oov_token_id,
      pad_token_id=output_pad_token_id)
  return model, loss, accuracy


def compute_logits(model, per_device_batch: dt.BatchedTrainTocopoData,
                   rng: jnp.ndarray):
  return model(per_device_batch, rng)


def compute_loss_from_logits(loss, logits,
                             target_data: dt.BatchedTrainTocopoTargetData):
  return loss(logits, target_data)


def compute_loss(loss, model, input_data: dt.BatchedTrainTocopoData,
                 rng: jnp.ndarray):
  logits = compute_logits(model, input_data, rng)
  return compute_loss_from_logits(loss, logits, input_data.target_data)


def train_step(optimizer,
               loss_and_grad_fn,
               train_state: checkpoint.TrainState,
               per_device_batch: dt.BatchedTrainTocopoData,
               rng: jnp.ndarray,
               is_distributed=True):
  """A training step, running on each device.

  Note the use of `pmean` to combine gradients and loss across devices (and
  hosts).

  Args:
    optimizer: The optimizer.
    loss_and_grad_fn: Pointer to a loss and grad function, such as resulting
      from calling `jax.value_and_grad` on a loss from losses.py.
    train_state: The training state (model and optimizer state).
    per_device_batch: The data to train on.
    rng: A PRNG that will be combined with the `train_state.model.step`.
    is_distributed: Boolean, whether to use pmap to aggregate losses across
      devices. Use if training in a multi-device setting.

  Returns:
    A tuple of (loss_value, new_train_state).
  """
  model = train_state.model
  optimizer_state = train_state.optimizer_state
  step = train_state.step
  rng = jax.random.fold_in(rng, step)

  # `loss_value` is a scalar, `grads` matches the trainable params in `model`.
  if is_distributed:
    loss_value, grads = jax.lax.pmean(
        loss_and_grad_fn(model, per_device_batch, rng), axis_name='i')
  else:
    loss_value, grads = loss_and_grad_fn(model, per_device_batch, rng)

  update, new_optimizer_state = optimizer.update(grads, optimizer_state, model)
  new_model = optax.apply_updates(model, update)

  new_train_state = checkpoint.TrainState(
      step=step + 1, model=new_model, optimizer_state=new_optimizer_state)

  return loss_value, new_train_state


def evaluation_step(loss_from_logits_fn, model,
                    per_device_batch: dt.BatchedTrainTocopoData,
                    rng: jnp.ndarray):
  """An evaluation step, running on each device.

  Note the use of `pmean` to combine gradients and loss across devices (and
  hosts).

  Args:
    loss_from_logits_fn: Pointer to a loss function, such as those from
      losses.py.
    model: The model instance.
    per_device_batch: The data to train on.
    rng: A single-user PRNG.

  Returns:
    A tuple of (loss_value, output_logits).
  """
  logits = compute_logits(model, per_device_batch, rng)
  loss_value = jax.lax.pmean(
      loss_from_logits_fn(logits, per_device_batch.target_data), axis_name='i')

  return loss_value, logits


@dataclasses.dataclass
class BatchProcessingSizes:
  """The splitting of a batch's examples between parallel/single processing."""
  parallel_size: int
  single_device_size: int


def get_pmap_remainder(batch_size: int,
                       num_devices: int) -> BatchProcessingSizes:
  """Given a batch size, returns parallel and single device computation sizes.

  Args:
    batch_size: The size of the batch.
    num_devices: The number of local devices among which the batch must be
      split.

  Returns:
    A tuple of the number of examples to process using parallel computation and
    single device computation.
  """
  parallel_size = (batch_size // num_devices) * num_devices
  single_device_size = batch_size % num_devices
  return BatchProcessingSizes(parallel_size, single_device_size)


def _run_prediction(predict_fn,
                    replicated_model,
                    data,
                    batch_size: int,
                    get_pmap_remainder_fn=get_pmap_remainder):
  """Gets the model's predictions."""
  logits, kinds, indices = None, None, None

  # Assign work to the multi/single device phases.
  num_devices = jax.local_device_count()
  processing_sizes = get_pmap_remainder_fn(batch_size, num_devices)
  pmap_size = processing_sizes.parallel_size
  remainder_size = processing_sizes.single_device_size
  if pmap_size != batch_size:
    logging.warning('Batch size is not a multiplier of the number of devices.')

  if pmap_size:
    pmap_data = jax.tree_util.tree_map(lambda x, size=pmap_size: x[:size], data)
    # Introduce a device axis, for parallel computation.
    sharded_pmap_data = jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, -1) + x.shape[1:]), pmap_data)
    pmap_predict_fn = jax.pmap(predict_fn, axis_name='i')
    logits, kinds, indices = pmap_predict_fn(replicated_model,
                                             sharded_pmap_data)
    # Remove the device axis.
    logits = jax.tree_util.tree_map(
        lambda x: x.reshape((pmap_size,) + x.shape[2:]), logits)
    kinds = kinds.reshape((pmap_size, -1))  # (B, O)
    indices = indices.reshape((pmap_size, -1))  # (B, O)

  if remainder_size:
    remainder_data = jax.tree_util.tree_map(
        lambda x, size=pmap_size: x[size:], data)
    jit_predict_fn = jax.jit(predict_fn)
    model = jax_utils.unreplicate(replicated_model)
    remainder_logits, remainder_kinds, remainder_indices = jit_predict_fn(
        model, remainder_data)

    if pmap_size:
      logits = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)),
                                      logits, remainder_logits)
      kinds = jnp.concatenate((kinds, remainder_kinds))
      indices = jnp.concatenate((indices, remainder_indices))
    else:
      logits, kinds, indices = (remainder_logits, remainder_kinds,
                                remainder_indices)

  return logits, kinds, indices


def autoregressive_step(
    input_data: dt.BatchedTrainTocopoData,
    input_eval_data: dt.BatchedEvalTocopoData,
    eval_model: Union[tocopo_models.GGNN2Tocopo,
                      tocopo_models.Transformer2Tocopo],
    output_token_encoder,
    sample_prediction_fn
) -> Tuple[dt.NDArrayIntBO, dt.NDArrayIntBO, dt.BatchedTocopoLogits,
           dt.BatchedTrainTocopoTargetData]:
  """Generate autoregressive prediction for dt.BatchedTrainTocopoData.

  The autoreggresive prediction generates one prediction at a time, until
  we have reached the conditions in _autoregressive_continue_condition(). It is
  different from teacher forced prediction which predicts all output
  simultaneously. In teacher forced prediction, we have the ground truth output,
  therefore we can shift the output and let the model to predict all next
  outputs simultaneously. But without the ground truth output, we can only
  predict one output at a time, and feed the previous output back to predict the
  next output. This is why autoregressive prediction should only be used during
  inference, because we don't have the ground truth output. We expect that well
  trained models should have similar teacher forced and autoregressive
  predictions (output token sequences) but are not expected to be the same.
  However, teacher forced and autoregressive sequence accuracies are guaranteed
  to be identical.

  The following happens in each step during autoregressive prediction:
  1. Predict the next kind and index using the previous predicted outputs and
     the input graph. This is stored in a dt.BatchedTrainTocopoData data
     structure, where the target data of dt.BatchedTrainTocopoData is empty in
     the first iteration. The predicted kind and index are returned by
     calling sample_prediction_fn().
  2. Given the predicted next kind and index, build the token id array.
     The token id array contains the output token id predicted by the model.
     This is done in _compute_next_batched_target_data_token_ids() function.
  3. Given the predicted next kind and index, build valid copy candidate array.
     The valid copy candidate array specifies from which input node we can copy
     from the output the current output token. This is done in
     _compute_next_batched_target_data_valid_copy_candidates() function
  4. Given the predicted next kind and index, build valid pointer candidate
     array. The valid pointer candidate array specifies which input node we are
     pointing to. This is done in
     _compute_next_batched_target_data_valid_pointer_candidates() function.
  5. Update the prediction with the token id array, valid copy candidate array
     and valid pointer candidate array. This will feed into
     sample_prediction_fn() again.
  This loops until _autoregressive_continue_condition() returns False, and we
  return the predicted outputs.

  Args:
    input_data: The data to generate prediction on.
    input_eval_data: dt.BatchedEvalTocopoData that contains additional
      information about the input_data. Such as the node token strings that
      should not be feed into the model.
    eval_model: The evaluation model.
    output_token_encoder: The output token tensor2tensor encoder.
    sample_prediction_fn: Function for sampling the next prediction.
  Returns:
    A tuple of (batched_predicted_kinds, batched_predicted_indices,
    output_logit, batched_predicted_data). batched_predicted_kinds is
    the predicted kind (token/copy/pointer), batched_predicted_indices is the
    corresponding index for each kind, output_logit is the logit for the whole
    prediction, batched_predicted_data contains other prediction-related data.
  """
  batch_size = input_data.target_data.token_ids.shape[0]
  num_outputs = input_data.target_data.token_ids.shape[1]
  num_nodes = input_data.target_data.is_target_copy.shape[2]

  done_token_id = output_token_encoder.encode(constants.DONE_TOKEN)[0]
  pad_token_id = output_token_encoder.encode(constants.PAD_TOKEN)[0]

  # The comment following each numpy array is the array shape.
  # The characters should match the one defined in data_types.py.
  # B = batch size, O = number of outputs, V = number of input node,
  # S = number of subtokens.

  # batched_partial_prediction hold the prediction that we made so far,
  # we initialize the target data with zeros first.
  batched_partial_prediction = _get_partial_prediction(
      input_data,
      # The token ids are initialzed with pad_token_id. Since the autoregressive
      # prediction will update the tokens in place, and when it is finished,
      # we don't have to pad the tokens.
      dt.NDArrayIntBO(jnp.full(
          (batch_size, num_outputs), pad_token_id, dtype=jnp.int32)),  # (B, O)
      dt.NDArrayBoolBOV(jnp.zeros(
          (batch_size, num_outputs, num_nodes), dtype=jnp.int32)),  # (B, O, V)
      dt.NDArrayBoolBOV(jnp.zeros(
          (batch_size, num_outputs, num_nodes), dtype=jnp.int32)))   # (B, O, V)

  # batched_predicted_kinds holds the predicted kinds (token/copy/pointer).
  # Each cell can have value 0, 1, 2, where 0 means token, 1 means copy
  # and 2 means pointer.
  # batched_predicted_indices holds the predicted indices.
  # If batched_predicted_kinds[i, j] == 0 (token), then
  # batched_predicted_indices[i, j] is the output token id.
  # If batched_predicted_kinds[i, j] == 1 (copy), then
  # batched_predicted_indices[i, j] is the input node id that we copy from.
  # If batched_predicted_kinds[i, j] == 2 (pointer), then
  # batched_predicted_indices[i, j] is the input node id that we point to.
  # batched_predicted_kinds is initialized to dt.TocopoKind.TOKEN. The
  # Autoregressive prediction will update batched_predicted_kinds in place, and
  # when it is finished, we don't have to pad batched_predicted_kinds. Same
  # thing goes for batched_predicted_indices.
  batched_predicted_kinds = jnp.full(
      (batch_size, num_outputs), dt.TocopoKind.TOKEN, dtype=jnp.int32)  # (B, O)
  batched_predicted_indices = jnp.full(
      (batch_size, num_outputs), pad_token_id, dtype=jnp.int32)  # (B, O)

  i = 0
  while _autoregressive_continue_condition(
      batched_partial_prediction.target_data.token_ids, i, num_outputs,
      done_token_id):
    output_logits, output_kinds, output_indices = _run_prediction(
        predict_fn=sample_prediction_fn,
        replicated_model=eval_model,
        data=batched_partial_prediction,
        batch_size=batch_size)

    # Get the next predicted kind and index.
    batched_next_kind = output_kinds[:, i]  # (B)
    batched_next_index = output_indices[:, i]  # (B)

    # Update batched_predicted_kinds and batched_predicted_indices with the
    # latest prediction.
    batched_predicted_kinds = batched_predicted_kinds.at[:, i].set(
        batched_next_kind)  # (B, O)
    batched_predicted_indices = batched_predicted_indices.at[:, i].set(
        batched_next_index)  # (B, O)

    batched_token_id = _compute_next_batched_target_data_token_ids(
        input_eval_data, batched_next_kind, batched_next_index,
        output_token_encoder)  # (B)

    batched_valid_copy_candidates = (
        _compute_next_batched_target_data_valid_copy_candidates(
            input_eval_data, batched_next_kind, batched_next_index,
            output_token_encoder))  # (B, V)

    batched_valid_pointer_candidates = (
        _compute_next_batched_target_data_valid_pointer_candidates(
            input_eval_data, batched_next_kind, batched_next_index))  # (B, V)

    # Update batched_partial_prediction with the newest prediction.
    predicted_token_ids = batched_partial_prediction.target_data.token_ids.at[:, i].set(
        batched_token_id)  # (B, O)
    is_valid_copy = batched_partial_prediction.target_data.is_target_copy.at[:, i, :].set(
        batched_valid_copy_candidates)  # (B, O, V)
    is_valid_pointer = batched_partial_prediction.target_data.is_target_pointer.at[:, i, :].set(
        batched_valid_pointer_candidates)  # (B, O, V)
    batched_partial_prediction = _get_partial_prediction(
        input_data,
        predicted_token_ids,
        is_valid_copy,
        is_valid_pointer)
    i += 1

  return (batched_predicted_kinds, batched_predicted_indices, output_logits,
          batched_partial_prediction.target_data)


def _autoregressive_continue_condition(
    predicted_token_ids: dt.NDArrayIntBO, current_step: int,
    num_outputs: int, done_token_id: int) -> bool:
  """Check if we want to continue the autoregressive prediction .

  The stop criteria of autoregressive prediction depends on two conditions:
    1. If all predictions have generated the DONE token.
    2. If we have reached 'num_outputs' steps.

  Args:
    predicted_token_ids: The predicted tokens.
    current_step: The current autogressive step.
    num_outputs: Number of outputs.
    done_token_id: The id of the DONE token.

  Returns:
    A boolean indicating if we want to continue the autoregressive prediction.
  """
  # Check that we have not yet generated DONE token in all predictions.
  continue_autoregressive = not jnp.all(
      jnp.any(predicted_token_ids[:, :current_step] == done_token_id, axis=1))

  continue_autoregressive = (
      continue_autoregressive and current_step < num_outputs)
  return continue_autoregressive


def _compute_next_batched_target_data_valid_pointer_candidates(
    input_eval_data: dt.BatchedEvalTocopoData,
    batched_next_kind: dt.NDArrayIntB, batched_next_index: dt.NDArrayIntB
    ) -> dt.NDArrayIntBV:
  """Compute valid copy candidates for the predicted kind and index.

  Valid pointer candidates is a matrix of shape (batch_size, num_nodes). If the
  cell at row i and column j is 1, it means that for the current output, we
  can point to j:th node in i:th data of the batch. Valid pointer candidates
  is only computed for when the predicted kind is pointer.

  Args:
    input_eval_data: A instance of dt.BatchedEvalTocopoData, containing
      information needed at evaluation time.
    batched_next_kind: The next predicted kind for the current batch.
    batched_next_index: The next predicted index for the current batch.

  Returns:
    The valid pointer candidates for the batch, it has shape
    (batch_size, num_nodes).
  """
  (batch_size, num_timesteps,
   num_nodes) = input_eval_data.node_data.node_texts.shape

  batched_valid_pointer_candidates = jnp.zeros((batch_size, num_nodes),
                                               dtype=jnp.int32)
  for idx_in_batch, kind in enumerate(batched_next_kind):
    valid_pointer_candidates = [0] * num_timesteps * num_nodes
    # valid_pointer_candidates is only meaningful when the output kind is
    # pointer. And if the output kind is pointer, we set the corresponding
    # valid_pointer_candidates to 1.
    if kind == dt.TocopoKind.POINTER:
      pointed_node_id = batched_next_index[idx_in_batch]
      valid_pointer_candidates[pointed_node_id] = 1
    batched_valid_pointer_candidates = batched_valid_pointer_candidates.at[
        idx_in_batch, :].set(valid_pointer_candidates)
  return batched_valid_pointer_candidates


def _compute_next_batched_target_data_valid_copy_candidates(
    input_eval_data: dt.BatchedEvalTocopoData,
    batched_next_kind: dt.NDArrayIntB, batched_next_index: dt.NDArrayIntB,
    output_token_encoder) -> dt.NDArrayIntBV:
  """Compute valid copy candidates for the predicted kind and index.

  Valid copy candidates is a matrix of shape (batch_size, num_nodes). If the
  cell at row i and column j is 1, it means that for j:th node in i:th data of
  the batch, its node's token is the same as the current output token,
  meaning that the model can copy from it. Valid copy candidates is only
  computed for when the predicted kind is a token or copy. Because for a
  pointer, there are no valid copy candidate.

  Args:
    input_eval_data: A instance of dt.BatchedEvalTocopoData, containing
      information needed at evaluation time.
    batched_next_kind: The next predicted kind for the current batch.
    batched_next_index: The next predicted index for the current batch.
    output_token_encoder: The output token encoder.

  Returns:
    The valid copy candidates for the batch, it has shape
    (batch_size, num_nodes).
  """
  (batch_size, num_timesteps,
   num_nodes) = input_eval_data.node_data.node_texts.shape

  batched_valid_copy_candidates = jnp.zeros((batch_size, num_nodes),
                                            dtype=jnp.int32)
  for idx_in_batch, kind in enumerate(batched_next_kind):
    valid_copy_candidates = [0] * num_timesteps * num_nodes

    # We can only copy from input nodes if the predicted kind is token,
    # or copy. If the predicted kind is pointer, there are no
    # valid_copy_candidates.
    if kind == dt.TocopoKind.TOKEN or kind == dt.TocopoKind.COPY:
      # For token kind, we use output_token_encoder to get the output token.
      # For copy kind, we use input_eval_data.node_data.node_texts to get
      # the actual copied node token.
      if kind == dt.TocopoKind.TOKEN:
        output_token_id = batched_next_index[idx_in_batch]
        output_token = output_token_encoder.decode(
            [int(output_token_id)]).encode('utf-8')
      else:
        copied_node_id = batched_next_index[idx_in_batch]
        copied_timestep = copied_node_id // num_nodes
        copied_id_in_snapshot = copied_node_id % num_nodes
        output_token = input_eval_data.node_data.node_texts[
            idx_in_batch, copied_timestep, copied_id_in_snapshot]

      # Compare output_token with node_texts, and update valid_copy_candidates
      # if there is a match. We don't use the node token ids for two reasons:
      # 1) The node token ids can have multiple ids for one node, for example
      # when we use subtokens. It means that we have to convert multiple ids
      # back to subtokens and know how to concatenate them.
      # 2) The node token id can be the id of OOV token, which means that we
      # can not reconstruct the original token text to compute the
      # valid_copy_candidates list.
      node_texts = jnp.reshape(  # Flatten node texts across timesteps.
          input_eval_data.node_data.node_texts[idx_in_batch], [-1])
      for node_id, node_text in enumerate(node_texts):
        if output_token == node_text:
          valid_copy_candidates[node_id] = 1

    batched_valid_copy_candidates = batched_valid_copy_candidates.at[
        idx_in_batch, :].set(valid_copy_candidates)

  return jnp.array(batched_valid_copy_candidates, dtype=jnp.int32)


def _compute_next_batched_target_data_token_ids(
    input_eval_data: dt.BatchedEvalTocopoData,
    batched_next_kind: dt.NDArrayIntB, batched_next_index: dt.NDArrayIntB,
    output_token_encoder) -> dt.NDArrayIntB:
  """Compute output token ids for the predicted kind and index.

  We compute the output token ids here. If batched_next_kind[i] == 0 (token),
  then the corresponding output token id is just batched_next_index[i]. If
  batched_next_kind[i] == 1 (copy), then we will use input_eval_data to get
  the node token at batched_next_index[i], and use output_token_encoder
  to get the output token id. And if batched_next_kind[i] == 2 (pointer),
  then the corresponding output token id is
  output_token_encoder.encode(constants.POINTER_TOKEN)[0].

  Args:
    input_eval_data: A instance of dt.BatchedEvalTocopoData, containing
      information needed at evaluation time.
    batched_next_kind: The next predicted kind for the current batch.
    batched_next_index: The next predicted index for the current batch.
    output_token_encoder: The output token encoder.

  Returns:
    The output token ids of shape (batch_size). It contains the output token id
    for each data in the batch.
  """
  batch_size, _, num_nodes = input_eval_data.node_data.node_texts.shape

  pointer_output_token_id = output_token_encoder.encode(
      constants.POINTER_TOKEN)[0]

  batched_token_id = jnp.zeros(batch_size, dtype=jnp.int32)
  for idx_in_batch, kind in enumerate(batched_next_kind):
    if kind == dt.TocopoKind.TOKEN:
      # The output token id is just the value in batched_next_index.
      output_token_id = batched_next_index[idx_in_batch]
    elif kind == dt.TocopoKind.COPY:
      # The output token id is the output token id of the copied node token.
      copied_node_id = batched_next_index[idx_in_batch]
      copied_timestep = copied_node_id // num_nodes
      copied_id_in_snapshot = copied_node_id % num_nodes
      node_text = input_eval_data.node_data.node_texts[
          idx_in_batch, copied_timestep, copied_id_in_snapshot]
      # This can potentialy cause a problem when the encoder decides to split
      # on whitespaces and does not treat node_text as a single token.
      # For now we simply take the first output token id. But it is possible
      # that the encoder returns multiple output token ids for node_text.
      # TODO: Have the encode function treat node text as a
      # single token.
      # The test use text_encoder hence adding this if condition.
      if isinstance(node_text, bytes):
        node_text = node_text.decode('utf-8')
      output_token_id = output_token_encoder.encode(node_text)[0]
    else:  # Pointer
      # The output token id is the pointer_output_token_id for a pointer.
      output_token_id = pointer_output_token_id
    batched_token_id = batched_token_id.at[idx_in_batch].set(output_token_id)
  return batched_token_id


def sample_prediction(
    model,
    per_device_batch: dt.BatchedTrainTocopoData,
    rng: jnp.ndarray,
    temperature: float = 0.0,
) -> Tuple[dt.BatchedTocopoLogits, dt.NDArrayIntBO, dt.NDArrayIntBO]:
  """Sample the all next outputs given the previous outputs.

  The default behavior is to set temperature=0, which makes this function do
  greedy (non-random) decoding.

  Samples the output kinds and tocopo_samples from the output logits of the
  model. This is not autoregressive since the model predict all next output
  given all the previous outputs (teacher-forced setting). First we compute
  logsumexp for token, copy and pointer logits to sample the output kinds,
  where 0 == token, 1 == copy and 2 == pointer. Then we sample from token,
  copy and pointer logits, and use the sampled output kinds to select the
  corresponding sampled token, copy and pointer.

  Args:
    model: The model instance.
    per_device_batch: The data to sample from.
    rng: rng returned by jax.random.PRNGKey.
    temperature: Temperature to scale the output logits, must be >= 0. When
      it is equal to 0, we use argmax to sample. When it is close to 1, it is
      sampling from a categorical distribution. When temperature >> 1, it is
      purely a random sampling.

  Returns:
    A tuple of (output_logits, output_kinds, tocopo_samples), the output_logits
    is the logits returned by the model. output_kinds, tocopo_samples both are
    jnp.array with shape (batch_size, num_outputs). output_kinds contains the
    predicted output kinds (token/copy/pointer) and tocopo_samples contains the
    indices for the respective kind.
  """
  # Generate output logits for all outputs.
  output_logits = model(per_device_batch, rng)

  # Scale the logits by temperature if the temperature is > 0. And use
  # categorical as the sampling function. For temperature close to 1, it is
  # sampling from a categorical distribution. For temperature >> 1, it is
  # purely a random sampling. For the special case temperature == 0, we use
  # argmax as the sampling function.
  if temperature > 0:
    token_logits = output_logits.token_logits / temperature
    copy_logits = output_logits.copy_logits / temperature
    pointer_logits = output_logits.pointer_logits / temperature
    sampling_fn = functools.partial(jax.random.categorical, rng)
    integrating_fn = jax.scipy.special.logsumexp
  else:
    token_logits = output_logits.token_logits
    copy_logits = output_logits.copy_logits
    pointer_logits = output_logits.pointer_logits
    sampling_fn = jnp.argmax
    integrating_fn = jnp.max

  # The comment following each numpy array is the array shape.
  # The character matches the one defined in data_types.py.
  # B = batch size, O = number of outputs

  # There are two ways to the sampling:
  # 1. Jointly sample the output indices and kinds. We concatenate the logits
  # to shape(B, O, U + 2V) and sample from it. Once we have the indices, we
  # can infer the kind.
  # 2. First sample the kind, then sample the indices given the kind.
  # The following code implements the second option. Because it is more
  # explicit and easier to understand. And mathematically, option 1
  # (output, kind) ~ P(output, kind) and option 2
  # (output, kind) ~ P(output|kind)P(kind) should be the same.

  # Compute logsumexp for token, copy and pointer logits. They are used to
  # sample the output kinds. When temperature = 0, logsumexp is just max.
  token_log_zs = integrating_fn(token_logits, axis=-1)  # (B, O)
  copy_log_zs = integrating_fn(copy_logits, axis=-1)  # (B, O)
  pointer_log_zs = integrating_fn(pointer_logits, axis=-1)  # (B, O)

  # Stack all logsumexp so that we can sample from the last dimension.
  kind_logits = jnp.stack(
      [token_log_zs, copy_log_zs, pointer_log_zs], axis=-1)  # (B, O, 3)

  # Categorical sample from the kind logits. output_kinds == 0 means token,
  # output_kinds == 1 means copy and output_kinds == 2 means pointer.
  # The indices of token, copy and pointer logits should match the integer
  # enumeration of dt.TocopoKind (ie. token_log_zs at index 0 and etc.).
  output_kinds = sampling_fn(kind_logits, axis=-1)  # (B, O)

  # We sample from token, copy and pointer logits. They will be selected based
  # on the output_kinds that we sampled.
  token_samples = sampling_fn(token_logits, axis=-1)  # (B, O)
  copy_samples = sampling_fn(copy_logits, axis=-1)  # (B, O)
  pointer_samples = sampling_fn(pointer_logits, axis=-1)  # (B, O)

  # Based on output_kinds (0==token, 1==copy, 2==pointer), select corresponding
  # token, copy and pointer samples.
  tocopo_samples = jnp.zeros(output_kinds.shape, dtype=jnp.int32)  # (B, O)
  tocopo_samples = jnp.where(  # (B, O)
      output_kinds == dt.TocopoKind.TOKEN, token_samples, tocopo_samples)
  tocopo_samples = jnp.where(  # (B, O)
      output_kinds == dt.TocopoKind.COPY, copy_samples, tocopo_samples)
  tocopo_samples = jnp.where(  # (B, O)
      output_kinds == dt.TocopoKind.POINTER, pointer_samples, tocopo_samples)

  return output_logits, output_kinds, tocopo_samples


def do_evaluation(
    step: int,
    rng: jnp.ndarray,
    eval_model,
    data_generator,
    max_evaluation_batches: int,
    evaluation_step_fn,
    accuracy_fn,
    eval_flush_steps=100,
    measurement_recorder: Optional[measurements.MeasurementRecorder] = None,
    metrics_file: Optional[str] = None):
  """Evaluates the model, records measurements and updates a best checkpoint.

  Args:
    step: Current step.
    rng: Random number generator.
    eval_model: Model to be evaluated.
    data_generator: Data generator providing examples to be evaluated.
    max_evaluation_batches: Maximum number of batches to be evaluated or -1 for
      no limit.
    evaluation_step_fn: Function executing the evaluation step.
    accuracy_fn: Function for calculating accuracy.
    eval_flush_steps: Logging frequency during evaluation.
    measurement_recorder: Handle for recording metrics to TF events.
    metrics_file: Filehandle for recording metrics to disk.

  Returns:
    results_dict: A dict containing metrics from eval and other metadata.
  """
  batch_count = 0
  mean_device_loss_sum = 0
  accuracy_metrics_sum = metrics.AccuracyMetrics()
  results_dict = {}

  # Note: each device gets a different key.
  sharded_rng = jax.random.split(rng, jax.local_device_count())

  # Restrict max number of evaluation batches if configured.
  if max_evaluation_batches > -1:
    data_generator = itertools.islice(data_generator, 0, max_evaluation_batches)

  for batch_idx, (input_data, _) in enumerate(data_generator):
    sharded_input_data = jax.tree_util.tree_map(
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
        input_data)

    # TODO: consider combining the rng update into the main call.
    replicated_data = jax_utils.replicate(step + batch_idx)
    sharded_rng_for_step = jax.pmap(jax.random.fold_in)(sharded_rng,
                                                        replicated_data)

    mean_device_loss, output_logits = jax.pmap(
        evaluation_step_fn, axis_name='i')(eval_model, sharded_input_data,
                                           sharded_rng_for_step)

    batch_count += 1
    mean_device_loss_sum += jax_utils.unreplicate(mean_device_loss)

    # Compute accuracy.
    accuracy_metrics = jax.pmap(
        accuracy_fn, axis_name='i')(output_logits,
                                    sharded_input_data.target_data)
    accuracy_metrics_sum += jax_utils.unreplicate(accuracy_metrics)

    logging.log_every_n(logging.INFO, 'Evaluating batch: {}'.format(batch_idx),
                        eval_flush_steps)

  if accuracy_metrics_sum.num_element_attempts == 0:
    # The data_generator did not yield any batch.
    logging.warning('Evaluation at step %d: not computed since '
                    'total_seq_attempts is 0')
  else:
    # Record metrics.
    element_accuracy = accuracy_metrics_sum.get_element_accuracy()
    seq_accuracy = accuracy_metrics_sum.get_seq_accuracy()

    # The mean over batches of the mean loss per device.
    mean_loss = mean_device_loss_sum / batch_count
    logging.info(
        'Evaluation at step %d: avg loss=%f, element acc=%.3f (%d tokens), '
        'seq acc=%.3f (%d sequences).', step, mean_loss, element_accuracy,
        accuracy_metrics_sum.num_element_attempts, seq_accuracy,
        accuracy_metrics_sum.num_seq_attempts)
    if measurement_recorder:
      measurement_recorder.record_validation_measurements(
          step, mean_loss, accuracy_metrics_sum)

    if metrics_file:
      with open(metrics_file, 'wt') as metrics_writer:
        metrics_writer.write('seq_accuracy,{:.5f}\n'.format(seq_accuracy))
        metrics_writer.write(
            'element_accuracy,{:.5f}\n'.format(element_accuracy))
        metrics_writer.write('num_sequences,{:d}\n'.format(
            accuracy_metrics_sum.num_seq_attempts))
        metrics_writer.write('num_elements,{:d}\n'.format(
            accuracy_metrics_sum.num_element_attempts))

    results_dict = {
        KEY_SEQ_ACCURACY: seq_accuracy,
        'element_accuracy': element_accuracy,
        'num_sequences': accuracy_metrics_sum.num_seq_attempts,
        'num_elements': accuracy_metrics_sum.num_element_attempts,
        'avg_seq_loss': mean_loss
    }

  return results_dict


def train(train_data_generator: dg.TocopoDataGenerator, valid_data_generator_fn,
          train_model, eval_model, model_train_args: dt.TrainingConfiguration):
  """Trains model with args from model_train_args.

  Args:
    train_data_generator: Training data generator, it never exhausts.
    valid_data_generator_fn: Function to create validation data generator,
      need to call this function each time we want to iterate through the
      validation dataset.
    train_model: The model to train.
    eval_model: A model set for evaluation - needs to be sync'ed to `model`
      before use.
    model_train_args: Configuration for training the model, it includes the
      pointer to the loss function, pointer to the accuracy function, the
      optimizer, number of training steps and steps between evaluations of
      validation dataset.

  Returns:
    The train state after training.
  """
  # The order of the following operations is important.
  #
  # We need to make sure that the following happens in order:
  #   (1) A fresh Training state is initialized from model spec.
  #   (2) A pretrained model is loaded if present.
  #   (3) The pretrained model params are overwritten by a checkpoint in the
  #   experiment directory if present.
  #
  # Preserving the order helps us achieve the following:
  #  a) Helps initialize the model with additional param dimenstions than
  #    those in the pre-training checkpoint.
  #  b) Ensures that the params in a local checkpoint in `exp_dir` take
  #    precedence over those from the pretrained model.

  # Initialize the train state.
  optimizer_state = model_train_args.optimizer.init(train_model)
  state = checkpoint.TrainState(
      step=0, model=train_model, optimizer_state=optimizer_state)
  del train_model, optimizer_state  # Access instead through `state`.

  # Optional restoration.
  checkpoint_dir = model_train_args.checkpoint_dir
  best_saver = None
  if checkpoint_dir:
    state = checkpoint.restore_ckpt_from_dir(
        checkpoint_dir, state, fail_if_not_found=False)
    if is_main_process():
      # Only the main process saves checkpoints.
      best_saver = checkpoint.BestSaver(checkpoint_dir)
  start_step = int(state.step)

  # Replicate model/optimizer state to each local device
  state = jax_utils.replicate(state)

  # Set up functions.
  loss_fn = functools.partial(compute_loss, model_train_args.loss)
  loss_and_grad_fn = jax.value_and_grad(loss_fn)
  train_step_fn = functools.partial(train_step,
                                    model_train_args.optimizer,
                                    loss_and_grad_fn)
  loss_from_logits_fn = functools.partial(compute_loss_from_logits,
                                          model_train_args.loss)
  evaluation_step_fn = functools.partial(evaluation_step, loss_from_logits_fn)
  accuracy_fn = model_train_args.accuracy
  compute_logits_fn = compute_logits

  measurement_recorder = None
  if is_main_process():
    # Only the main process records measurements, which factor in all processes.
    measurement_recorder = measurements.MeasurementRecorder(model_train_args)

  # Training loop.

  # Note: each device gets a different key. This key will be combined with the
  # training step inside the pmap to form a different key for each step.
  rng = jax.random.PRNGKey(0)
  sharded_rng = jax.random.split(rng, jax.local_device_count())
  # Place on the host to avoid needing to pull it from the single device it's on
  # at each step.
  sharded_rng = np.array(sharded_rng)

  if start_step == 0:
    # Evaluate validation metrics at step 0.
    rng_validation = jax.random.fold_in(rng, start_step)
    valid_data_generator = valid_data_generator_fn()
    # Synchronize `eval_model` with the training model.
    # Note: `pytree_node=False` attributes, such as the dropout layers'i
    # `mode`, do not get included in the serialization.
    serialized_train_model_dict = serialization.to_state_dict(state.model)
    synchronized_eval_model = serialization.from_state_dict(
        eval_model, serialized_train_model_dict)
    eval_results_dict = do_evaluation(
        step=start_step,
        rng=rng_validation,
        eval_model=synchronized_eval_model,
        data_generator=valid_data_generator,
        max_evaluation_batches=(model_train_args.max_validation_batches //
                                jax.process_count()),
        evaluation_step_fn=evaluation_step_fn,
        accuracy_fn=accuracy_fn,
        measurement_recorder=measurement_recorder)

  ptrain_step = jax.pmap(train_step_fn, axis_name='i', donate_argnums=(0, 1))
  for step in range(start_step, model_train_args.num_training_steps):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      input_data, _ = train_data_generator()  # dt.BatchedTrainTocopoData

      # Reshape the outer dimension of each array in the
      # dt.BatchedTrainTocopoData from (batch_per_host) to (num_local_devices,
      # batch_per_device). Since`BatchedTrainTocopoData` is a flax dataclass,
      # this can be done using`jax.tree_util.tree_map`.
      sharded_input_data = jax.tree_util.tree_map(
          lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]),
          input_data)

      if (step + 1) % model_train_args.train_logging_steps == 0:
        # Compute accuracies on the training data before doing the training
        # update, so that we don't train on an example before using it to
        # compute accuracy. TODO: Consider re-using the logits
        # computed in the training step.
        # TODO: consider combining the two computations into one.
        replicated_step = jax_utils.replicate(step)
        sharded_rng_for_step = jax.pmap(jax.random.fold_in)(sharded_rng,
                                                            replicated_step)
        output_logits = (
            jax.pmap(compute_logits_fn)(state.model, sharded_input_data,
                                        sharded_rng_for_step))

        # Compute accuracy.
        accuracy_metrics = jax.pmap(
            accuracy_fn, axis_name='i')(output_logits,
                                        sharded_input_data.target_data)
        accuracy_metrics = jax_utils.unreplicate(accuracy_metrics)
        element_training_accuracy = accuracy_metrics.get_element_accuracy()
        seq_training_accuracy = accuracy_metrics.get_seq_accuracy()

        # Run parallel computation. The inputs and outputs are sharded according
        # to the number of local devices. Specifically:
        #   - `sharded_input_data` is as above
        #   - `state` is the device sharded (replicated) model and optimizer
        #     state
        #   - `loss_value` is a sharded (replicated) scalar
        loss_value, state = ptrain_step(state, sharded_input_data, sharded_rng)

        logging.info(
            'Training step %d: loss=%f, element acc=%.3f (%d elements), '
            'seq acc=%.3f (%d sequences).', step,
            jax_utils.unreplicate(loss_value), element_training_accuracy,
            accuracy_metrics.num_element_attempts, seq_training_accuracy,
            accuracy_metrics.num_seq_attempts)
        if measurement_recorder:
          measurement_recorder.record_train_measurements(
              step, jax_utils.unreplicate(loss_value), accuracy_metrics)
      else:
        # Run parallel computation. The inputs and outputs are sharded according
        # to the number of local devices. Specifically:
        #   - `sharded_input_data` is as above
        #   - `state` is the device sharded (replicated) model and optimizer
        #.    state
        #   - `loss_value` is a sharded (replicated) scalar
        start_time_sec = time.time()
        loss_value, state = ptrain_step(state, sharded_input_data, sharded_rng)
        logging.log_first_n(logging.INFO, 'Step took %f seconds.', 5,
                            time.time() - start_time_sec)

    if (step + 1) % model_train_args.valid_steps == 0:
      # Evaluate validation metrics, possibly updating the best checkpoint.
      # Updating the best model happens before updating the model for fault
      # tolerance. This ensures no best model checkpointing is skipped.
      rng_validation = jax.random.fold_in(rng, step)
      valid_data_generator = valid_data_generator_fn()
      # Synchronize `eval_model` with the training model.
      # Note: `pytree_node=False` attributes, such as the dropout layers'i
      # `mode`, do not get included in the serialization.
      serialized_train_model_dict = serialization.to_state_dict(state.model)
      synchronized_eval_model = serialization.from_state_dict(
          eval_model, serialized_train_model_dict)
      eval_results_dict = do_evaluation(
          step=step,
          rng=rng_validation,
          eval_model=synchronized_eval_model,
          data_generator=valid_data_generator,
          max_evaluation_batches=(model_train_args.max_validation_batches //
                                  jax.process_count()),
          evaluation_step_fn=evaluation_step_fn,
          accuracy_fn=accuracy_fn,
          measurement_recorder=measurement_recorder)
      if best_saver:
        try:
          seq_accuracy = eval_results_dict[KEY_SEQ_ACCURACY]
        except KeyError as e:
          raise ValueError(
              'Did not get seq_accuracy after running eval..') from e
        best_saver.update(seq_accuracy, jax_utils.unreplicate(state))

    if (step + 1) % model_train_args.checkpoint_every_n_steps == 0:
      if checkpoint_dir and is_main_process():
        # Only the main process saves the checkpoint.
        checkpoint.save_checkpoint(checkpoint_dir, jax_utils.unreplicate(state))

  return jax_utils.unreplicate(state)


def single_device_train(train_data_generator: dg.TocopoDataGenerator,
                        unused_valid_data_generator_fn, model,
                        model_train_args: dt.TrainingConfiguration):
  """Trains model with args from model_train_args without using pmap.

  This makes it easier to print intermediate results, since pmap automatically
  jits the code being pmapped.

  Args:
    train_data_generator: Training data generator, it never exhausts.
    unused_valid_data_generator_fn: Unused.
    model: A model from the models class.
    model_train_args: Configuration for training the model, it includes the
      pointer to the loss function, pointer to the accuracy function, the
      optimzer, number of training steps and steps between evluation of
      validation dataset.
  """
  # Initialize the training state.
  optimizer_state = model_train_args.optimizer.init(model)
  train_state = checkpoint.TrainState(
      step=0, model=model, optimizer_state=optimizer_state)

  # Don't jit these, since we're using this function for debugging.
  loss_fn = functools.partial(compute_loss, model_train_args.loss)
  loss_and_grad_fn = jax.value_and_grad(loss_fn)
  train_step_fn = functools.partial(train_step,
                                    model_train_args.optimizer,
                                    loss_and_grad_fn,
                                    is_distributed=False)
  accuracy_fn = model_train_args.accuracy

  # Training loop.
  rng = jax.random.PRNGKey(0)
  for step in range(model_train_args.num_training_steps):
    input_data, _ = train_data_generator()  # dt.BatchedTrainTocopoData
    loss_value, train_state = train_step_fn(train_state, input_data, rng)

    if step % 5 == 0:
      # TODO: training metrics should be computed before update.
      rng_for_step = jax.random.fold_in(rng, step)
      output_logits = compute_logits(train_state.model, input_data,
                                     rng_for_step)
      accuracy_components = accuracy_fn(
          output_logits, input_data.target_data, is_distributed=False)
      (num_token_train_correct, num_token_train_attempts, num_seq_train_correct,
       num_seq_train_attempts) = accuracy_components

      token_training_accuracy = num_token_train_correct / num_token_train_attempts
      seq_training_accuracy = num_seq_train_correct / num_seq_train_attempts
      logging.info('Training step %d: loss=%f, token acc=%.3f, seq acc=%.3f',
                   step, loss_value, token_training_accuracy,
                   seq_training_accuracy)


def _get_partial_prediction(input_data: dt.BatchedTrainTocopoData,
                            target_data_token_ids: dt.NDArrayIntBO,
                            target_data_is_target_copy: dt.NDArrayBoolBOV,
                            target_data_is_target_pointer: dt.NDArrayBoolBOV
                            ) -> dt.BatchedTrainTocopoData:
  """Create BatchedTrainTocopoData that contains the latest predictions.

  This function creates BatchedTrainTocopoData for the autoregressive
  prediction. The returned batched_partial_prediction contains the prediction
  made so far by the autoregressive prediction, notebly
  BatchedTrainTocopoTargetData.token_ids,
  BatchedTrainTocopoTargetData.is_target_copy and
  BatchedTrainTocopoTargetData.is_target_pointer. batched_partial_prediction
  should be used by the autoregressive prediction to generate the next
  prediction.

  Args:
    input_data: The input data that we generate the autoregressive prediction.
      We used it copy the BatchedTrainGraphNodeData and
      BatchedTrainGraphEdgeData. But BatchedTrainTocopoTargetData should not be
      copied from the input data since it contains the ground truth.
    target_data_token_ids: Token ids that the autoregressive prediction
      predicted so far.
    target_data_is_target_copy: is_target_copy matrix that the
      autoregressive prediction predicted so far.
    target_data_is_target_pointer: is_target_pointer that the
      autoregressive prediction predicted so far.

  Returns:
    A instance of BatchedTrainTocopoData, where the BatchedTrainGraphNodeData
    and BatchedTrainGraphEdgeData is the same as input_data. But
    BatchedTrainTocopoTargetData holds the prediction made so far.
  """
  # BatchedTrainTocopoTargetData contains the latest prediction.
  # We must not copy from input_data, but rather use the target_data_token_ids,
  # target_data_is_target_copy and target_data_is_target_pointer that are
  # predicted by the autoregressive prediction.
  batched_partial_prediction_tocopo_target_data = (
      dt.BatchedTrainTocopoTargetData(
          token_ids=target_data_token_ids,
          is_target_copy=target_data_is_target_copy,
          is_target_pointer=target_data_is_target_pointer))
  # BatchedTrainGraphNodeData and BatchedTrainGraphEdgeData is the same as the
  # input_data.
  batched_partial_prediction_graph_node_data = dt.BatchedTrainGraphNodeData(
      token_ids=input_data.node_data.token_ids,
      type_ids=input_data.node_data.type_ids,
      token_positions=input_data.node_data.token_positions,
      pointer_candidates=input_data.node_data.pointer_candidates
  )
  batched_partial_prediction_graph_edge_data = dt.BatchedTrainGraphEdgeData(
      edges=input_data.edge_data.edges,
      time_edges=input_data.edge_data.time_edges)
  batched_partial_prediction = dt.BatchedTrainTocopoData(
      node_data=batched_partial_prediction_graph_node_data,
      edge_data=batched_partial_prediction_graph_edge_data,
      target_data=batched_partial_prediction_tocopo_target_data
  )
  return batched_partial_prediction
