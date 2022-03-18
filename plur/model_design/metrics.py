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

"""Metrics functions for models.

Metrics functions typically should *not* be differentiable.
"""
import dataclasses
from typing import TYPE_CHECKING

import flax
import jax
import jax.numpy as jnp

from plur.model_design import data_types as dt


flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


def pointer_accuracy_fn(pointer_logits: dt.BatchedTocopoLogits,
                        target_data: dt.BatchedTrainTocopoTargetData):
  """Computes average 0-1 accuracy of pointer predictions at timestep 1."""
  one_hot_argmax_predictions = jax.nn.one_hot(
      jnp.argmax(pointer_logits, axis=1), pointer_logits.shape[1])

  # The pointer should always be at timestep 1 in copy-paste data. Slice out
  # the BV array of those pointer targets.
  is_pointer_targets = target_data.is_target_pointer[:, 1, :]

  num_correct = jnp.sum(one_hot_argmax_predictions * is_pointer_targets)
  num_attempts = is_pointer_targets.shape[0]
  return num_correct, num_attempts


@flax_dataclass
class AccuracyMetrics:
  """Accuracy metrics.

  Note Flax dataclasses are immutable.

  Attributes:
    num_element_correct: The number of elements (Pointers or ToCo) correctly
        predicted.
    num_element_attempts: The number of elements (Pointers or ToCo) in the
        target.
    num_seq_correct: The number of sequences correctly predicted.
    num_seq_attempts: The number of sequences in the target.
    num_pointer_correct: The number of pointer elements correctly predicted.
    num_pointer_attempts: The number of pointer elements in the target.
    num_pointer_seq_correct: The number of sequences for which all pointer
        elements are correctly predicted (or without pointer elements).
    num_toco_correct: The number of ToCo elements correctly predicted.
    num_token_correct: The number of ToCo elements correctly predicted using the
        Token output head.
    num_toco_attempts: The number of ToCo elements in the target
    num_toco_seq_correct: The number of sequences for which all ToCo
        elements are correctly predicted (or without ToCo elements).
  """
  num_element_correct: int = 0
  num_element_attempts: int = 0
  num_seq_correct: int = 0
  num_seq_attempts: int = 0
  num_pointer_correct: int = 0
  num_pointer_attempts: int = 0
  num_pointer_seq_correct: int = 0
  num_toco_correct: int = 0
  num_token_correct: int = 0
  num_toco_attempts: int = 0
  num_toco_seq_correct: int = 0

  def __add__(self, other: 'AccuracyMetrics'):
    return dataclasses.replace(
        self,
        num_element_correct=(self.num_element_correct +
                             other.num_element_correct),
        num_element_attempts=(self.num_element_attempts +
                              other.num_element_attempts),
        num_seq_correct=self.num_seq_correct + other.num_seq_correct,
        num_seq_attempts=self.num_seq_attempts + other.num_seq_attempts,
        num_pointer_correct=(self.num_pointer_correct +
                             other.num_pointer_correct),
        num_pointer_attempts=(self.num_pointer_attempts +
                              other.num_pointer_attempts),
        num_pointer_seq_correct=(self.num_pointer_seq_correct +
                                 other.num_pointer_seq_correct),
        num_toco_correct=self.num_toco_correct + other.num_toco_correct,
        num_token_correct=self.num_token_correct + other.num_token_correct,
        num_toco_attempts=self.num_toco_attempts + other.num_toco_attempts,
        num_toco_seq_correct=(self.num_toco_seq_correct +
                              other.num_toco_seq_correct))

  def get_element_accuracy(self) -> float:
    return self.num_element_correct / self.num_element_attempts

  def get_seq_accuracy(self) -> float:
    return self.num_seq_correct / self.num_seq_attempts

  def get_pointer_accuracy(self) -> float:
    return self.num_pointer_correct / self.num_pointer_attempts

  def get_pointer_seq_accuracy(self) -> float:
    return self.num_pointer_seq_correct / self.num_seq_attempts

  def get_toco_accuracy(self) -> float:
    return self.num_toco_correct / self.num_toco_attempts

  def get_toco_seq_accuracy(self) -> float:
    return self.num_toco_seq_correct / self.num_seq_attempts

  def get_token_prediction_ratio(self) -> float:
    """Gets the fraction of correct ToCo predictions made using Tokens."""
    return self.num_token_correct / self.num_toco_correct


def tocopo_accuracy_fn(tocopo_logits: dt.BatchedTocopoLogits,
                       target_data: dt.BatchedTrainTocopoTargetData,
                       oov_token_id: int,
                       pad_token_id: int,
                       is_distributed: bool = True) -> AccuracyMetrics:
  """Computes accuracy metrics.

  Args:
    tocopo_logits: Predictions from model (unnormalized log scores).
    target_data: target data to compare prediction against.
    oov_token_id: Id of out of vocabulary token.
    pad_token_id: Id of pad token.
    is_distributed: Whether to perform cross-device aggregation.

  Returns:
    A `AccuracyMetrics` instance.
  """
  vocab_size = tocopo_logits.token_logits.shape[2]

  one_hot_target_tokens = jax.nn.one_hot(target_data.token_ids,
                                         vocab_size)  # (B, O, U)

  # Don't give credit for OOV tokens.
  one_hot_target_tokens = one_hot_target_tokens.at[:, :, oov_token_id].set(
      jnp.zeros_like(target_data.token_ids))

  # Disable predictions for all tokens when there is a pointer.
  # Mask indicating absence of a pointer at target.
  not_pointer_mask = target_data.is_target_pointer.sum(axis=2) == 0  # (B, O)
  one_hot_target_tokens = one_hot_target_tokens * jnp.expand_dims(
      not_pointer_mask, axis=2)

  few_hot_targets = jnp.concatenate([
      one_hot_target_tokens, target_data.is_target_copy,
      target_data.is_target_pointer
  ],
                                    axis=2)  # (B, O, U+2V)

  # Get the one hot predictions.
  tocopo_logits_stacked = jnp.concatenate([
      tocopo_logits.token_logits, tocopo_logits.copy_logits,
      tocopo_logits.pointer_logits
  ],
                                          axis=2)  # (B, O, U+2V)

  prediction_indices = jnp.argmax(tocopo_logits_stacked, axis=2)  # (B, O)
  one_hot_predictions = jax.nn.one_hot(
      prediction_indices, tocopo_logits_stacked.shape[2])  # (B, O, U+2V)

  # (B, O)
  is_pad = (target_data.token_ids == pad_token_id)
  # (B, O, U+2V) -> (B, O)
  # If the target is a pad token, then we remove it from consideration when
  # calculating accuracies. `element_correct_or_pad` array always assign a 1 to
  # padded prediction (this property is used in the sequence accuracy
  # computation).
  element_correct = jnp.sum(one_hot_predictions * few_hot_targets, axis=-1)
  element_correct_or_pad = jnp.where(is_pad, 1, element_correct)
  per_element_correct = jnp.sum(element_correct_or_pad * (1 - is_pad))
  per_element_attempts = jnp.sum(1 - is_pad)

  per_sequence_correct = jnp.sum(jnp.prod(element_correct_or_pad, axis=-1))
  per_sequence_attempts = element_correct_or_pad.shape[0]

  pointer_mask = jnp.logical_and(
      jnp.logical_not(not_pointer_mask), jnp.logical_not(is_pad))

  pointer_correct = jnp.sum(element_correct * pointer_mask)
  pointer_attempts = jnp.sum(pointer_mask)

  # Pointer sequence accuracy: construct an array of 1s everywhere except where
  # a pointer is incorrectly predicted. Note: this counts a sequence without
  # pointers as accurately predicted.
  pointer_correct_or_toco_or_pad = jnp.where(not_pointer_mask, 1,
                                             element_correct_or_pad)
  per_sequence_po_correct = jnp.sum(
      jnp.prod(pointer_correct_or_toco_or_pad, axis=-1))

  toco_mask = jnp.logical_and(not_pointer_mask, jnp.logical_not(is_pad))

  toco_correct = jnp.sum(element_correct * toco_mask)
  toco_attempts = jnp.sum(toco_mask)

  # ToCo sequence accuracy: construct an array of 1s everywhere except where
  # a To/Co is incorrectly predicted. Note: this counts a sequence without
  # ToCo as accurately predicted.
  toco_correct_or_po_or_pad = jnp.where(pointer_mask, 1, element_correct_or_pad)
  per_sequence_toco_correct = jnp.sum(
      jnp.prod(toco_correct_or_po_or_pad, axis=-1))

  # Correct predictions using the To head.
  is_prediction_token_mask = prediction_indices < vocab_size
  token_correct = jnp.sum(
      element_correct *
      jnp.logical_and(is_prediction_token_mask, jnp.logical_not(is_pad)))

  # Aggregate across devices.
  if is_distributed:
    per_element_correct = jax.lax.psum(per_element_correct, axis_name='i')
    per_element_attempts = jax.lax.psum(per_element_attempts, axis_name='i')
    per_sequence_correct = jax.lax.psum(per_sequence_correct, axis_name='i')
    per_sequence_attempts = jax.lax.psum(per_sequence_attempts, axis_name='i')
    pointer_correct = jax.lax.psum(pointer_correct, axis_name='i')
    pointer_attempts = jax.lax.psum(pointer_attempts, axis_name='i')
    toco_correct = jax.lax.psum(toco_correct, axis_name='i')
    token_correct = jax.lax.psum(token_correct, axis_name='i')
    toco_attempts = jax.lax.psum(toco_attempts, axis_name='i')
    per_sequence_po_correct = jax.lax.psum(
        per_sequence_po_correct, axis_name='i')
    per_sequence_toco_correct = jax.lax.psum(
        per_sequence_toco_correct, axis_name='i')

  return AccuracyMetrics(
      num_element_correct=per_element_correct,
      num_element_attempts=per_element_attempts,
      num_seq_correct=per_sequence_correct,
      num_seq_attempts=per_sequence_attempts,
      num_pointer_correct=pointer_correct,
      num_pointer_attempts=pointer_attempts,
      num_pointer_seq_correct=per_sequence_po_correct,
      num_toco_correct=toco_correct,
      num_token_correct=token_correct,
      num_toco_attempts=toco_attempts,
      num_toco_seq_correct=per_sequence_toco_correct)
