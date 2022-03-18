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

"""Main script for training in parallel using JAX.
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import optax

from plur.model_design import data_manager as dm
from plur.model_design import data_types as dt
from plur.model_design import evaluation
from plur.model_design import models
from plur.model_design import training
import tensorflow as tf

FLAGS = flags.FLAGS


# On TPU, this line raises the default matmul precision. See b/193404861.
jax.config.update('jax_default_matmul_precision', 'tensorfloat32')


# Problem.
flags.DEFINE_string('data_dir', '/tmp/code2seq_dataset/stage_2',
                    'Directory in which to store problem data.')
# Model.
flags.DEFINE_enum('model_kind', 'ggnn2tocopo',
                  ['ggnn2tocopo', 'transformer2tocopo'],
                  'Kind of model to use.')
flags.DEFINE_integer('max_num_subtokens', 0,
                     'The number of subtokens to consider per node. If 0, no'
                     'subtokenization is used at all; in this case, every node'
                     'must have precisely one token.')
flags.DEFINE_integer('node_position_dim', 1,
                     'The number of positional indicators to use per node.')
flags.DEFINE_integer('hidden_dim', 128, 'Size of hidden states.')
flags.DEFINE_integer('num_input_propagation_steps', 8,
                     'Number of propagation steps to run on input graph.')
flags.DEFINE_integer(
    'num_transformer_attention_heads', 1, 'Number of attention'
    'heads used by Transformer encoders.')
flags.DEFINE_integer(
    'num_output_propagation_steps', -1,
    'Number of propagation steps to run on output graph. If '
    'this is -1, use `num_input_propagation_steps`.')
flags.DEFINE_bool(
    'use_relational_bias', False, 'Whether to add relational'
    'attention bias based on edge data. Only relevant when'
    'using the transformer2tocopo model_kind.')
flags.DEFINE_bool(
    'use_pointer_candidate_masking', False,
    'Boolean indicating whether to use or ignore '
    'pointer candidates. NOTE: If no pointer candidates are provided in '
    'training examples and if `_use_pointer_candidates` is True, then all '
    'pointers will be disabled since there are no candidates to point to. '
    'Therefore, only use pointer candidates if the training examples '
    'contain such candidates.')
# Training.
flags.DEFINE_integer('batch_size_per_device', 8, 'Batch size per device.')
flags.DEFINE_integer('num_training_steps', 50000,
                     'Number of training steps to run.')
flags.DEFINE_float('learning_rate', 1e-5, 'Base learning rate.')
# Use warmup of 0.1 for the BERT default.
flags.DEFINE_float(
    'warmup_steps_fraction', 0.0, 'Fraction of the training steps during which '
    'to warm up the learning rate linearly starting at 0.')
# Use EPS of 1e-6 for the BERT default.
flags.DEFINE_float('adam_eps', 1e-8, 'The EPS parameter of the Adam optimizer.')
flags.DEFINE_float('dropout_rate', 0.1, 'The dropout rate.')
flags.DEFINE_integer('model_initialization_seed', 0,
                     'Seed used to initialize the model.')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Gradient clipping parameter.')
# Training loop details.
flags.DEFINE_integer('valid_steps', 5000, 'Run validation every X steps.')
flags.DEFINE_integer('train_logging_steps', 1000,
                     'Num steps to log training metrics.')
flags.DEFINE_integer(
    'max_validation_batches', 50,
    'Maximum number of batches to run validation for. If -1, then run till exhaustion.'
)
flags.DEFINE_integer(
    'checkpoint_every_n_steps', 10000,
    'How often to save checkpoints in terms of number of steps.')
# Experiment directory.
flags.DEFINE_string('exp_dir', '/tmp/experiments/',
                    'Root directory in which to store experiment data.')
flags.DEFINE_boolean(
    'evaluate', False, 'If `True`, it will output two files containing the '
    'ground truths and predictions for comparison.')
flags.DEFINE_enum(
    'data_split_to_evaluate', 'test', ['train', 'validation', 'test'],
    'Dataset selection to generate ground truths and predictions.')
flags.DEFINE_enum(
    'checkpoint_type_to_evaluate', 'best', ['best', 'latest'],
    'Type of checkpoint to use for generating predictions for evaluation. '
    'If `best`, use the best checkpoint per validation metric. '
    'If `latest` use the last saved checkpoint after training.'
)
flags.DEFINE_enum(
    'evaluation_mode', 'autoregressive', ['autoregressive', 'teacher_forced'],
    'Mode of generating predictions. '
    'If `autoregressive`, feed back generated predictions to predict '
    'next token. '
    'If `teacher-forced` use ground truth to generate next predicted token.')
flags.DEFINE_integer('num_evaluation_rounds', 5,
                     'Number of rounds of iterative evaluation for datasets '
                     'that support it.')
flags.DEFINE_integer('eval_flush_steps', 100,
                     'Num steps to flush eval results.')
flags.DEFINE_boolean(
    'drop_remainder', True, 'If `True`, it will drop a final batch that is '
    'incomplete.')
flags.DEFINE_boolean(
    'output_provenance_when_evaluate', False, 'If true, output the example '
    'provenance when evaluating.')
flags.DEFINE_string(
    'evaluation_dir', '', 'Directory to store the evaluation '
    'result. Default to {exp_dir}/evaluation.')


def _log_compute_info():
  """Logs information about the compute setup (#processes and devices)."""
  # Jax can do single-program-multiple-data parallel programming using `pmap`.
  # This setup is based on a number of processes, each connected to local
  # devices. Typically, we consider the following setups (others are possible):
  #   CPU:
  #     - a single process with a single local device
  #   GPU:
  #     - single process with one or more local devices (GPUs)
  #   TPU:
  #     - There are a few varieties. See https://cloud.google.com/tpu/docs/tpus.
  logging.info('This is host %d of %d with %d local devices (%d in total)',
               jax.process_index(), jax.process_count(),
               jax.local_device_count(), jax.device_count())
  logging.info('JAX devices: %r', jax.devices())


def _join_processes_for_exit():
  if jax.host_count() > 1:
    # The TPU runtime typically crashes if you interact with it while the
    # device mesh is only partially available. This can happen if one host does
    # special work (e.g. checkpointing). The following ensures all processes
    # exit together.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()


def _get_num_output_propagation_steps(num_input_propagation_steps,
                                      num_output_propagation_steps):
  if num_output_propagation_steps < 0:
    return num_input_propagation_steps
  return num_output_propagation_steps


def main(unused_argv):

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  _log_compute_info()

  # Compute the experiment and checkpoint directory names.
  run_name = training.flags_to_run_name(FLAGS)
  exp_dir = os.path.join(FLAGS.exp_dir, run_name)
  checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

  # Ensure the experiment and checkpoint directories exist.
  tf.io.gfile.makedirs(exp_dir)
  tf.io.gfile.makedirs(checkpoint_dir)

  logging.info('Experiment directory: %s', exp_dir)

  # Create a data generator that will yield enough data for all the local
  # devices.
  batch_size_per_host = FLAGS.batch_size_per_device * jax.local_device_count()

  logging.info('Data directory: %s', FLAGS.data_dir)
  data_manager = dm.PlurDataManager(
      FLAGS.data_dir, batch_size_per_host, drop_remainder=FLAGS.drop_remainder)

  # Create the model.
  num_output_propagation_steps = _get_num_output_propagation_steps(
      FLAGS.num_input_propagation_steps, FLAGS.num_output_propagation_steps)

  if FLAGS.model_kind == 'ggnn2tocopo':
    model_class = models.GGNN2Tocopo
  elif FLAGS.model_kind == 'transformer2tocopo':
    model_class = models.Transformer2Tocopo
  else:
    raise ValueError('Unknown model kind passed: {}'.format(FLAGS.model_kind))

  create_model_fn = functools.partial(
      training.create_model,
      token_vocab_size=data_manager.token_vocab_size,
      type_vocab_size=data_manager.type_vocab_size,
      output_vocab_size=data_manager.output_vocab_size,
      node_text_pad_token_id=data_manager.node_text_pad_token_id,
      output_oov_token_id=data_manager.output_oov_token_id,
      output_pad_token_id=data_manager.output_pad_token_id,
      padding_spec=data_manager.padding_spec,
      num_input_propagation_steps=FLAGS.num_input_propagation_steps,
      num_output_propagation_steps=num_output_propagation_steps,
      hidden_dim=FLAGS.hidden_dim,
      num_transformer_attention_heads=FLAGS.num_transformer_attention_heads,
      use_relational_bias=FLAGS.use_relational_bias,
      model_temporal_relations=False,
      use_pointer_candidate_masking=FLAGS.use_pointer_candidate_masking,
      model_class=model_class,
      dropout_rate=FLAGS.dropout_rate,
      seed=FLAGS.model_initialization_seed)

  # Always disable dropout in evaluate mode.
  model, loss, accuracy = create_model_fn(is_training=not FLAGS.evaluate)

  if FLAGS.evaluate:
    if FLAGS.evaluation_dir:
      evaluation_dir = FLAGS.evaluation_dir
    else:
      evaluation_dir = os.path.join(exp_dir, 'evaluation')
    tf.io.gfile.makedirs(evaluation_dir)

    if FLAGS.checkpoint_type_to_evaluate == 'best':
      checkpoint_dir = os.path.join(checkpoint_dir, 'best')

    if FLAGS.evaluation_mode == 'teacher_forced' and not FLAGS.drop_remainder:
      raise app.UsageError('`--evaluation_mode=teacher_forced` and '
                           '`--nodrop_remainder` should to be combined '
                           '(b/188216682).')

    model_evaluate_args = dt.EvaluationConfiguration(
        eval_flush_steps=FLAGS.eval_flush_steps,
        checkpoint_dir=checkpoint_dir,
        evaluation_dir=evaluation_dir,
        evaluation_mode=FLAGS.evaluation_mode,
        loss=loss,
        accuracy=accuracy)

    logging.info('Using split: %s for evaluation.',
                 FLAGS.data_split_to_evaluate)
    if FLAGS.data_split_to_evaluate == 'train':
      eval_data_generator_fn = data_manager.train_data_generator_fn
    elif FLAGS.data_split_to_evaluate == 'validation':
      eval_data_generator_fn = data_manager.valid_data_generator_fn
    elif FLAGS.data_split_to_evaluate == 'test':
      eval_data_generator_fn = data_manager.test_data_generator_fn
    else:
      raise ValueError('Unsupported data split for evaluation: {}'.format(
          FLAGS.data_split_to_evaluate))

    output_filename_prefix = f'{FLAGS.data_split_to_evaluate}_'
    num_evaluation_rounds = 1
    is_iterative_evaluation_supported = (
        FLAGS.evaluation_mode == 'autoregressive' and
        FLAGS.num_evaluation_rounds > 1)
    if is_iterative_evaluation_supported:
      num_evaluation_rounds = FLAGS.num_evaluation_rounds
      logging.info('Evaluation will proceed iteratively over %d rounds.',
                   num_evaluation_rounds)
    else:
      logging.info('Evaluation will proceed non-iteratively in a single round.')
    evaluation.evaluate(
        eval_data_generator_fn,
        model,
        model_evaluate_args,
        data_manager,
        output_filename_prefix=output_filename_prefix,
        num_evaluation_rounds=num_evaluation_rounds,
        output_provenance=FLAGS.output_provenance_when_evaluate)
  else:
    # A model with `is_training=False`, e.g. `Dropout` in `eval` mode.
    eval_model, _, _ = create_model_fn(is_training=False)

    # Optimizer: use Adam and gradient clipping by max norm.
    # Note that this is meant to construct standard Adam updates but with
    # clipped gradients. The negative learning rate is so that we minimize
    # loss. First we clip gradients, then we rescale according to Adam
    # statistics, then finally we multiply by the negative learning rate to get
    # our update for minimizing loss.
    adam_b1 = .9
    adam_b2 = .999
    adam_eps = FLAGS.adam_eps
    learning_rate_transition_steps = int(FLAGS.num_training_steps *
                                         FLAGS.warmup_steps_fraction)
    if learning_rate_transition_steps == 0:
      # If transition steps is set to 0, the learning rate is set to a constant
      # equal to the initial value.
      initial_learning_rate = FLAGS.learning_rate
    else:
      initial_learning_rate = 0.0
    learning_rate_schedule = optax.linear_schedule(
        init_value=initial_learning_rate,
        end_value=FLAGS.learning_rate,
        transition_steps=learning_rate_transition_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_gradient_norm),
        optax.scale_by_adam(b1=adam_b1, b2=adam_b2, eps=adam_eps),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))

    model_train_args = dt.TrainingConfiguration(
        loss=loss,
        accuracy=accuracy,
        optimizer=optimizer,
        num_training_steps=FLAGS.num_training_steps,
        valid_steps=FLAGS.valid_steps,
        train_logging_steps=FLAGS.train_logging_steps,
        max_validation_batches=FLAGS.max_validation_batches,
        exp_dir=exp_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_steps=FLAGS.checkpoint_every_n_steps)

    training.train(data_manager.train_data_generator,
                   data_manager.valid_data_generator_fn, model, eval_model,
                   model_train_args)

  _join_processes_for_exit()


if __name__ == '__main__':
  app.run(main)
