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

"""Checkpoint related functionality."""

import dataclasses
import os
from typing import Any, Optional, TYPE_CHECKING, Text

from absl import logging
import flax
from flax import serialization
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from tensorflow.io import gfile

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


@flax_dataclass
class TrainState:
  step: int = 0
  model: Any = None
  optimizer_state: Optional[optax.OptState] = None


def restore_ckpt_from_path(ckpt_path: Text, state: Optional[TrainState] = None):
  """Load a checkpoint from a path."""
  if not gfile.exists(ckpt_path):
    raise ValueError('Could not find checkpoint: {}'.format(ckpt_path))

  logging.info('Restoring checkpoint from %s', ckpt_path)
  with gfile.GFile(ckpt_path, 'rb') as fp:
    if state is None:
      # Returns a dict in MsgPack format. This is useful when the loaded
      # checkpoint needs to be sliced and diced to extract only relevant
      # parameters.
      # E.g. The optimizer state may be ignored when loading from a pretrained
      # model.
      return serialization.msgpack_restore(fp.read())
    else:
      return serialization.from_bytes(state, fp.read())


def restore_ckpt_from_dir(checkpoint_dir: Text, state: TrainState,
                          fail_if_not_found: bool = True):
  """Load checkpoint from a directory.

  There maybe multiple checkpoints in the directory. The one with the largest
  step number is loaded by default.

  Args:
    checkpoint_dir: Fully qualified path to directory containing ckpt.
    state: training state to be populated.
    fail_if_not_found: Raise an exception if no checkpoint is found in
      checkpoint_dir.

  Returns:
    state: training state restored with checkpoint from dir.
  """
  restored_model = checkpoints.restore_checkpoint(checkpoint_dir, state)
  # If no checkpoint is found, `restore_checkpoint` returns `state` back.
  if restored_model is state and fail_if_not_found:
    raise ValueError(f'Did not find any checkpoints in {checkpoint_dir}')

  # Send (the array parts of) the model to JAX device. This avoids the following
  # bug, occurring when indexing a NumPy array with a JAX one.
  # https://github.com/google/jax/issues/620#issuecomment-484344945
  # Other types, such as strings, should not be transferred (b/185819710).

  def to_jax_array(x):
    if isinstance(x, str):
      return x
    else:
      # Setting `copy=False` to avoid useless copies, such as when the object is
      # already a jax array.
      return jnp.array(x, copy=False)

  jax_model = jax.tree_map(to_jax_array, restored_model)

  return jax_model


def save_checkpoint(checkpoint_dir: Text,
                    state: TrainState,
                    keep: int = 3,
                    overwrite: bool = False):
  step = int(state.step)
  checkpoints.save_checkpoint(
      checkpoint_dir, state, step, keep=keep, overwrite=overwrite)


class BestSaver(object):
  """A class for maintaining a checkpoint corresponding to the best accuracy.

  In a multi-process setting, this class should only be instantiated on the main
  process.

  The implementation assumes determinism, of the accuracy evaluation and of
  training, which implies small corner cases. For instance, if a fault occurs
  after best checkpointing but prior to fault tolerance checkpointing, it would
  be possible to resume training without updating the best model again. In this
  case, the best model might not correspond to a point on the learning curve.
  """

  def __init__(self, save_dir: Text):
    self._save_dir = os.path.join(save_dir, 'best')
    self._best_step_and_accuracy_path = os.path.join(
        self._save_dir, 'best_step_and_accuracy.txt')

    self._best_accuracy = float('-inf')
    self._best_step = -1

    # Attempt to restore a previous best state.
    if gfile.exists(self._best_step_and_accuracy_path):
      with gfile.GFile(self._best_step_and_accuracy_path, mode='r') as fd:
        best_step_and_accuracy = fd.read().split(',')
        self._best_step = int(best_step_and_accuracy[0])
        self._best_accuracy = float(best_step_and_accuracy[1])
      logging.info(
          'Restored best checkpoint\'s step and accuracy: step %d, accuracy %.3f.',
          self._best_step, self._best_accuracy)

  def update(self, accuracy: float, state: TrainState):
    """Updates the best model state based on a new observation.

    Args:
      accuracy: The accuracy corresponding to `state`.
      state: The training state matching `accuracy`.
    """
    if accuracy > self._best_accuracy:  # Opting for a strict improvement.
      self._best_accuracy = accuracy
      self._best_step = state.step
      # Save checkpoint before the state - in case of fault, the state will not
      # have been updated, leading to both being updated once training resumes.
      save_checkpoint(self._save_dir, state, keep=1, overwrite=True)

      best_step_and_accuracy = '{},{}'.format(self._best_step,
                                              self._best_accuracy)
      tf.io.write_file(self._best_step_and_accuracy_path,
                       best_step_and_accuracy)
      logging.info(
          'Updated best checkpoint step and accuracy: step %d, accuracy %.3f.',
          self._best_step, self._best_accuracy)
