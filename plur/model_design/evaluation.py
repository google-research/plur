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

"""Code pertaining to evaluating models."""

import functools
import os
from typing import Any, Optional, Sequence, Text, Union

from absl import flags
from absl import logging

from flax import jax_utils
import jax
import jax.numpy as jnp

from plur.model_design import checkpoint
from plur.model_design import data_generation
from plur.model_design import data_manager as dm
from plur.model_design import data_types as dt
from plur.model_design import losses
from plur.model_design import models as tocopo_models
from plur.model_design import training
from plur.utils import constants
import tensorflow as tf


FLAGS = flags.FLAGS


def get_sharded_filename(filename: str, shard: int, shard_count: int):
  if shard >= shard_count:
    raise ValueError(f'Shard index ({shard}) must be smaller than the '
                     f'number of shards ({shard_count}).')

  root, ext = os.path.splitext(filename)
  return f'{root}-{shard}-of-{shard_count}{ext}'


class EvaluationState(object):
  """An object that maintains a fault-tolerant evaluation state."""

  def __init__(self, shard_count: int, save_dir: str,
               state_filename_prefix: str):
    if shard_count < 1:
      raise ValueError('The minimum number of shards is 1.')

    self._current_shard = 0
    self._shard_count = shard_count

    state_filename = state_filename_prefix + 'evaluation_state.txt'
    self._state_file_path = os.path.join(save_dir, state_filename)

    self._restore_state()

  def _save_state(self) -> None:
    state = '{},{}'.format(self._current_shard, self._shard_count)
    logging.info('Saving evaluation state (%s) to %s.', state,
                 self._state_file_path)
    tf.io.write_file(self._state_file_path, state)

  def _restore_state(self) -> None:
    """Restores a persisted state from disk if one is found."""
    if tf.io.gfile.exists(self._state_file_path):
      logging.info('Restoring evaluation state from %s', self._state_file_path)
      with open(self._state_file_path, mode='r') as fd:
        recovered_current_shard, recovered_shard_count = fd.read().split(',')

      recovered_shard_count = int(recovered_shard_count)
      if recovered_shard_count != self._shard_count:
        raise ValueError('Found an incompatible previous shard count (%s)' %
                         recovered_shard_count)
      self._current_shard = int(recovered_current_shard)
      logging.info('Restored state: at shard %d out of %d.',
                   self._current_shard, self._shard_count)
    else:
      logging.info('No previously persisted evaluation state found at %s.',
                   self._state_file_path)

  @property
  def current_shard(self) -> int:
    return self._current_shard

  def increment(self) -> None:
    self._current_shard += 1
    self._save_state()


def evaluate(data_generator_fn,
             model: Union[tocopo_models.GGNN2Tocopo,
                          tocopo_models.Transformer2Tocopo],
             model_evaluate_args: dt.EvaluationConfiguration,
             data_manager: Any = None,
             output_filename_prefix: str = '',
             seed: int = 0,
             num_evaluation_rounds=1,
             output_provenance: bool = False):
  """Dispatches evaluation to the appropriate function.

  Args:
    data_generator_fn: A callable that returns a data generator for evaluation.
    model: GGNN2Tocopo or Transformer2Tocopo model from the tocopo models class.
    model_evaluate_args: Configuration for evaluation. It includes the pointer
      to the loss function, pointer to the accuracy function, the optimizer, the
      directory to save the predictions and targets, and the directory to load
      the model.
    data_manager: Data manager corresponding to the generator.
    output_filename_prefix: Add prefix to output filename(s).
    seed: The seed to initialize the Jax RNG.
    num_evaluation_rounds: The number of rounds for running evaluation.
    output_provenance: Output the provenance string at the start of each
      ground truth if true.
  """
  # Restore the model.
  state = checkpoint.TrainState(step=0, model=model, optimizer_state=None)
  del model  # Access instead through `state`.
  checkpoint_dir = model_evaluate_args.checkpoint_dir
  state = checkpoint.restore_ckpt_from_dir(checkpoint_dir, state)

  if model_evaluate_args.evaluation_mode == 'autoregressive':
    generate_predictions(data_generator_fn, state.model, model_evaluate_args,
                         data_manager, output_filename_prefix,
                         seed, num_evaluation_rounds,
                         output_provenance)
  elif model_evaluate_args.evaluation_mode == 'teacher_forced':
    evaluate_teacher_forced(data_generator_fn, state, model_evaluate_args,
                            output_filename_prefix, seed)
  else:
    raise ValueError('Unsupported evaluation mode: {}'.format(
        model_evaluate_args.evaluation_mode))


def evaluate_chunk(data_generator_fn, shard: int, num_shards: int, model,
                   data_manager, sample_prediction_fn,
                   model_evaluate_args: dt.EvaluationConfiguration,
                   ground_truth_file: str, prediction_file: str,
                   score_file: str, provenance_file: str,
                   decode_utf8: bool) -> None:
  """Evaluates predictions on a chunk and saves them to disk."""
  if num_shards > 1:
    data_generator = data_generator_fn(shard=shard, shard_count=num_shards)
  else:
    data_generator = data_generator_fn()

  ground_truth_writer = open(ground_truth_file, 'wt')
  prediction_writer = open(prediction_file, 'wt')
  score_writer = open(score_file, 'wt')
  provenance_writer = open(provenance_file, 'wt') if provenance_file else None
  _evaluate_chunk(data_generator, model, data_manager, sample_prediction_fn,
                  model_evaluate_args, ground_truth_writer, prediction_writer,
                  score_writer, provenance_writer, decode_utf8)
  ground_truth_writer.close()
  prediction_writer.close()
  score_writer.close()
  if provenance_writer:
    provenance_writer.close()


def generate_predictions(data_generator_fn,
                         model: tocopo_models.GGNN2Tocopo,
                         model_evaluate_args: dt.EvaluationConfiguration,
                         data_manager: Optional[dm.DataManager] = None,
                         output_filename_prefix: str = '',
                         seed: int = 0,
                         num_evaluation_rounds: int = 1,
                         output_provenance: bool = False,
                         generate_fn=evaluate_chunk):
  """Generates predictions and saves them to disk.

  We use autoregressive_step to generate the predictions. The ground truths are
  also saved to disk for comparison.

  Args:
    data_generator_fn: A callable that returns a data generator for evaluation.
    model: GGNN2Tocopo model from the tocopo models class.
    model_evaluate_args: Configuration for evaluation. It includes the pointer
      to the loss function, pointer to the accuracy function, the optimizer, the
      directory to save the predictions and targets, and the directory to load
      the model.
    data_manager: Data manager corresponding to the generator.
    output_filename_prefix: Add prefix to output filename(s).
    seed: The seed to initialize the Jax RNG.
    num_evaluation_rounds: The number of rounds for running evaluation.
    output_provenance: Output the provenance string at the start of each
      ground truth if true.
    generate_fn: A callable that generates the predictions.
  """
  if data_manager is None:
    raise ValueError('data_manager is needed in autoregressive mode.')

  # Multi-host: revise `output_filename_prefix` as each host writes.
  output_filename_prefix = (
      f'{output_filename_prefix}host{jax.process_index()}_'
      f'of_{jax.process_count()}_')

  rng = jax.random.PRNGKey(seed)
  model = jax_utils.replicate(model)
  sample_prediction_fn = functools.partial(
      training.sample_prediction, rng=rng, temperature=0)

  num_iterative_shards = num_evaluation_rounds
  evaluation_dir = model_evaluate_args.evaluation_dir
  evaluation_state = EvaluationState(num_iterative_shards, evaluation_dir,
                                     output_filename_prefix)

  provenance_file = None
  while evaluation_state.current_shard < num_iterative_shards:
    current_shard = evaluation_state.current_shard
    logging.info('Evaluating shard %d of %d', current_shard,
                 num_iterative_shards)

    ground_truth_file = os.path.join(evaluation_dir,
                                     output_filename_prefix + 'targets.txt')
    prediction_file = os.path.join(evaluation_dir,
                                   output_filename_prefix + 'predictions.txt')
    score_file = os.path.join(evaluation_dir,
                              output_filename_prefix + 'scores.txt')
    if output_provenance:
      provenance_file = os.path.join(evaluation_dir,
                                     output_filename_prefix + 'provenances.txt')
    if num_iterative_shards > 1:
      ground_truth_file = get_sharded_filename(ground_truth_file, current_shard,
                                               num_iterative_shards)
      prediction_file = get_sharded_filename(prediction_file, current_shard,
                                             num_iterative_shards)
      score_file = get_sharded_filename(score_file, current_shard,
                                        num_iterative_shards)
      if output_provenance:
        provenance_file = get_sharded_filename(provenance_file, current_shard,
                                               num_iterative_shards)

    logging.info('Evaluating %s to %s', ground_truth_file, prediction_file)

    generate_fn(
        data_generator_fn,
        current_shard,
        num_iterative_shards,
        model,
        data_manager,
        sample_prediction_fn,
        model_evaluate_args,
        ground_truth_file,
        prediction_file,
        score_file,
        provenance_file,
        decode_utf8=True)

    evaluation_state.increment()
  logging.info('Done evaluating.')


def evaluate_teacher_forced(data_generator_fn,
                            state: checkpoint.TrainState,
                            model_evaluate_args: dt.EvaluationConfiguration,
                            output_filename_prefix: str = '',
                            seed: int = 0):
  """Evaluate a model, writing out metrics to a file.

  Args:
    data_generator_fn: A callable that returns a data generator for evaluation.
    state: A `TrainState` containing the model to evaluate.
    model_evaluate_args: Configuration for evaluation. It includes the pointer
      to the loss function, pointer to the accuracy function, the optimizer, the
      directory to save the predictions and targets, and the directory to load
      the model.
    output_filename_prefix: Add prefix to output filename(s).
    seed: The seed to initialize the Jax RNG.
  """
  rng = jax.random.PRNGKey(seed)
  data_generator = data_generator_fn()

  loss_from_logits_fn = functools.partial(training.compute_loss_from_logits,
                                          model_evaluate_args.loss)
  evaluation_step_fn = functools.partial(training.evaluation_step,
                                         loss_from_logits_fn)
  accuracy_fn = model_evaluate_args.accuracy

  metrics_file = None
  if training.is_main_process():
    # Only the main process records metrics which factor in all processes.
    metrics_file = os.path.join(
        model_evaluate_args.evaluation_dir,
        output_filename_prefix + 'teacher_forced_metrics.txt')

  training.do_evaluation(
      step=state.step,
      rng=rng,
      eval_model=jax_utils.replicate(state.model),
      data_generator=data_generator,
      max_evaluation_batches=-1,
      evaluation_step_fn=evaluation_step_fn,
      accuracy_fn=accuracy_fn,
      eval_flush_steps=model_evaluate_args.eval_flush_steps,
      metrics_file=metrics_file)


def _write_batched_tokens(batched_tokens: Sequence[Sequence[Text]], writer):
  """Writes a batch of tokens to `writer`."""
  for tokens in batched_tokens:
    # We try to find the 'DONE' token, and remove all tokens after the
    # 'DONE' token. If we cannot find the 'DONE' token, write the tokens as
    # they are.
    try:
      done_token_index = tokens.index(constants.DONE_TOKEN)
    except ValueError:
      done_token_index = len(tokens)
    line = ' '.join(tokens[:done_token_index])
    writer.write(line + '\n')


def _evaluate_chunk(data_generator: data_generation.TocopoDataGenerator,
                    eval_model, data_manager, sample_prediction_fn,
                    model_evaluate_args: dt.EvaluationConfiguration,
                    ground_truth_writer, prediction_writer, score_writer,
                    provenance_writer, decode_utf8: bool):
  """Evaluates predictions on a chunk and saves them to disk."""

  for step, (input_data, input_eval_data) in enumerate(data_generator):
    (batched_predicted_kinds, batched_predicted_indices, logits,
     predicted_target_data) = (
         training.autoregressive_step(input_data, input_eval_data, eval_model,
                                      data_manager.output_encoder,
                                      sample_prediction_fn))

    # Get the ground truth output.
    batched_ground_truth_tokens = _get_ground_truth_output(
        input_data,
        input_eval_data,
        decode_utf8=decode_utf8)
    _write_batched_tokens(batched_ground_truth_tokens, ground_truth_writer)

    # Get the predictions.
    batched_prediction_tokens = _get_predicted_output(
        batched_predicted_kinds,
        batched_predicted_indices,
        data_manager.output_encoder,
        input_eval_data,
        decode_utf8=decode_utf8)
    _write_batched_tokens(batched_prediction_tokens, prediction_writer)

    # Score the predictions
    done_token_id = data_manager.output_encoder.encode(constants.DONE_TOKEN)[0]
    batched_prediction_scores = get_prediction_scores(
        logits, predicted_target_data, data_manager.output_oov_token_id,
        data_manager.output_pad_token_id, done_token_id)
    _write_batched_scores(batched_prediction_scores, score_writer)

    if provenance_writer:
      batched_provenance_tokens = _get_provenance_output(
          input_eval_data, decode_utf8=decode_utf8)
      _write_batched_tokens(batched_provenance_tokens, provenance_writer)

    logging.log_every_n(logging.INFO, 'Evaluate at step: {}'.format(step),
                        model_evaluate_args.eval_flush_steps)
    if step % model_evaluate_args.eval_flush_steps == 0:
      prediction_writer.flush()
      ground_truth_writer.flush()
      score_writer.flush()
      if provenance_writer:
        provenance_writer.flush()


def _aggregate_log_probs(token_ids, log_probs, done_token_id):
  """Compute the sum of log-probs until (incl.) where DONE is generated."""
  log_probs_until_done = _truncate_log_probs(
      token_ids, log_probs, done_token_id)
  return jnp.sum(log_probs_until_done)


def _truncate_log_probs(token_ids, log_probs, done_token_id):
  """Zero out log-probs after where the DONE token is."""
  num_outputs = token_ids.shape[-1]
  done_token_present = jnp.any(token_ids == done_token_id)
  cutoff = jax.lax.cond(
      done_token_present,
      lambda tids: jnp.argwhere(tids == done_token_id, size=1)[0, 0] + 1,
      lambda tids: num_outputs,
      token_ids)
  log_probs_until_done = jnp.where(
      jnp.arange(num_outputs) < cutoff, log_probs, 0.)
  return log_probs_until_done


def get_prediction_scores(
    tocopo_logits: dt.BatchedTocopoLogits,
    prediction_data: dt.BatchedTrainTocopoTargetData,
    oov_token_id: int,
    pad_token_id: int,
    done_token_id: int) -> dt.NDArrayFloatB:
  """Compute prediction scores (log likelihood) for a batch of predictions."""
  log_probs = losses.compute_tocopo_log_probs(
      tocopo_logits, prediction_data, oov_token_id=oov_token_id,
      pad_token_id=pad_token_id)

  scores = jax.vmap(_aggregate_log_probs, in_axes=(0, 0, None))(
      prediction_data.token_ids, log_probs, done_token_id)
  return scores


def get_per_output_scores(
    tocopo_logits: dt.BatchedTocopoLogits,
    prediction_data: dt.BatchedTrainTocopoTargetData,
    oov_token_id: int,
    pad_token_id: int,
    done_token_id: int) -> dt.NDArrayFloatBO:
  """Compute per-output log likelihood for a batch of predictions."""
  log_probs = losses.compute_tocopo_log_probs(
      tocopo_logits, prediction_data, oov_token_id=oov_token_id,
      pad_token_id=pad_token_id)

  scores = jax.vmap(_truncate_log_probs, in_axes=(0, 0, None))(
      prediction_data.token_ids, log_probs, done_token_id)
  return scores


def _write_batched_scores(
    batched_prediction_scores: dt.NDArrayFloatB, writer):
  """Writes a batch of scores to `writer`."""
  for score in batched_prediction_scores:
    writer.write('%s\n' % score)


def _get_ground_truth_output(
    input_data: dt.BatchedTrainTocopoData,
    input_eval_data: dt.BatchedEvalTocopoData,
    decode_utf8: bool = True) -> Sequence[Sequence[str]]:
  """Get the ground truth tocopo output.

  The ground truth output is a list of strings. For token outputs, the actual
  ground truth token is stored in BatchedEvalTocopoTargetData, regardless if it
  is a token or a copy. For pointer outputs, we get the node id and label that
  the pointer is pointed to, and convert it to string as
  'POINTER({},{})'.format(node_id, node_label). For example 'POINTER(1,foo)'
  is a pointer that points to node 1 with node label 'foo'.

  Args:
    input_data: An instance of BatchedTrainTocopoData, we extract the pointer
      information from it.
    input_eval_data: An instance of BatchedEvalTocopoData, we extract the tokens
      from it.
    decode_utf8: If true, decode strings as utf-8.

  Returns:
    A list of string lists, each string list is the ground truth output.
  """
  batch_size = input_data.target_data.token_ids.shape[0]
  max_num_output = input_data.target_data.token_ids.shape[1]

  all_ground_truth = []
  for batch_index in range(batch_size):
    ground_truth = []
    for output_index in range(max_num_output):
      cur_is_target_pointer = input_data.target_data.is_target_pointer[
          batch_index, output_index, :]
      # Check if the current output is a pointer
      if jnp.sum(cur_is_target_pointer) == 1:
        pointed_node_id = jnp.nonzero(cur_is_target_pointer)[0][0]
        pointed_node_label = input_eval_data.node_data.node_texts[
            batch_index, -1, pointed_node_id]
        if decode_utf8:
          pointed_node_label = pointed_node_label.decode('utf-8')
        # Format pointer output as 'POINTER({node_id},{node_label})'
        ground_truth.append('POINTER({},{})'.format(pointed_node_id,
                                                    pointed_node_label))
      else:
        # If it is not a pointer, the ground truth token is stored in
        # BatchedEvalTocopoData.
        token_string = input_eval_data.target_data.tokens[batch_index,
                                                          output_index]
        if decode_utf8:
          token_string = token_string.decode('utf-8')
        ground_truth.append(token_string)
    all_ground_truth.append(ground_truth)
  return all_ground_truth


def _get_predicted_output(
    batch_predicted_kinds: dt.NDArrayIntBO,
    batch_predicted_indices: dt.NDArrayIntBO,
    output_token_encoder,
    input_eval_data: dt.BatchedEvalTocopoData,
    decode_utf8: bool = True) -> Sequence[Sequence[str]]:
  """Get the predicted output generated by autoregressive prediction.

  The predicted output is a list of strings. So to cast token, pointer and
  copy to a string, we do the following:
  * token: If the output is a token, use the token as it is.
  * pointer: If the output is a pointer, we get the node id and label that the
    pointer is pointed to, and formats it as
    'POINTER({},{})'.format(node_id, node_label). For example 'POINTER(1,foo)'
    is a pointer pointing to node 1 with node label 'foo'.
  * copy: If the output is a copy from the input node, we use the copied node
    value.

  Args:
    batch_predicted_kinds: The predicted kinds for the batch. 0 means that the
      predicted kind is a token. 1 means that the predicted kind is a copy. And
      2 means that the predicted kind is a pointer.
    batch_predicted_indices: The predicted indices for the batch. When
      batch_predicted_kinds is 0, the index is the output vocab id. When
      batch_predicted_kinds is 1, the index is the node id that we copy from.
      When batch_predicted_kinds is 2, the index is the node that we point to.
    output_token_encoder: Encoder for the output token.
    input_eval_data: A instance of BatchedEvalTocopoData, used to get the node
      label that we copy from.
    decode_utf8: If true, decode strings as utf-8.

  Returns:
    A list of string lists, each string list is the predicted output.
  """
  batch_size = batch_predicted_kinds.shape[0]
  max_num_output = batch_predicted_kinds.shape[1]

  predictions = []
  for batch_index in range(batch_size):
    prediction = []
    for output_index in range(max_num_output):
      if batch_predicted_kinds[
          batch_index, output_index] == dt.TocopoKind.TOKEN:
        # Use output_token_encoder to get the token.
        vocab_id = batch_predicted_indices[batch_index, output_index]
        prediction.append(output_token_encoder.decode((int(vocab_id),)))
      elif batch_predicted_kinds[
          batch_index, output_index] == dt.TocopoKind.COPY:
        copied_node_id = batch_predicted_indices[batch_index, output_index]
        copied_node_token = input_eval_data.node_data.node_texts[
            batch_index, -1, copied_node_id]
        if decode_utf8:
          copied_node_token = copied_node_token.decode('utf-8')
        prediction.append(copied_node_token)
      # Else the prediction is a pointer
      else:
        pointed_node_id = batch_predicted_indices[batch_index, output_index]
        pointed_node_label = input_eval_data.node_data.node_texts[
            batch_index, -1, pointed_node_id]
        if decode_utf8:
          pointed_node_label = pointed_node_label.decode('utf-8')
        # Format pointer output as 'POINTER({node_id},{node_label})'
        prediction.append('POINTER({},{})'.format(pointed_node_id,
                                                  pointed_node_label))
    predictions.append(prediction)
  return predictions


def _get_provenance_output(input_eval_data: dt.BatchedEvalTocopoData,
                           decode_utf8: bool = True) -> Sequence[Sequence[str]]:
  """Get the provenances of the evaluation data.

  Args:
    input_eval_data: A instance of BatchedEvalTocopoData, used to get the
      provenance.
    decode_utf8: If true, decode strings as utf-8.

  Returns:
    A list of string lists, each string list is the provenance.
  """
  batch_size = input_eval_data.provenance.shape[0]
  provenances = []
  for batch_index in range(batch_size):
    provenance = input_eval_data.provenance[batch_index]
    if decode_utf8:
      provenance = provenance.decode('utf-8')
    provenances.append([provenance])
  return provenances
