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

"""Loss functions for models.

Note: Loss functions are differentiable.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from plur.model_design import data_types as dt

# A large negative value that leads to a probability close to zero when
# exponentiated.
_LOG_OF_SMALL_VALUE = -1e9


def tocopo_loss_fn(tocopo_logits: dt.BatchedTocopoLogits,
                   target_data: dt.BatchedTrainTocopoTargetData,
                   oov_token_id: int,
                   pad_token_id: int) -> float:
  """Computes cross entropy loss for predictions."""
  # Get tocopo logprobs. Padded out entries are set to 0.
  # (B, O)
  tocopo_log_probs = compute_tocopo_log_probs(
      tocopo_logits, target_data, oov_token_id, pad_token_id)

  # See b/165283293 for more context on the choice to normalize by batch size.
  return -jnp.sum(tocopo_log_probs) / tocopo_log_probs.shape[0]


def compute_tocopo_log_probs(
    tocopo_logits: dt.BatchedTocopoLogits,
    target_data: dt.BatchedTrainTocopoTargetData,
    oov_token_id: int,
    pad_token_id: int) -> dt.NDArrayFloatBO:
  """Tocopo log probabilities on target data.

  Calculates log probabilities assigned to target data by a model that produces
  Tocopo logit predictions. The log probs are calculated by taking a big softmax
  over all token, copy and pointer logits jointly.

  Args:
    tocopo_logits: Logits from token, copy and pointer predictions.
    target_data: Unshifted ground target data.
    oov_token_id: Integer id corresponding to output token <OOV>.
    pad_token_id: Integer id corresponding to output token <PAD>.

  Returns:
    log_probs: The tocopo log probs for each output slot. i.e.
    log p(s_t|s_1:t-1). For padded entries, the log prob is set to 0.
  """
  token_target_scores, token_log_zs = _extract_token_target_scores(
      tocopo_logits.token_logits, target_data.token_ids)  # (B, O)

  # mask[_,j]==1 iff output target is a pointer.
  # (B, O)
  target_pointer_mask = jnp.sum(target_data.is_target_pointer, axis=-1)
  # We mask out `token_target_scores` for all candidates where the target is a
  # pointer since we don't want the model to reward the model getting
  # the token "POINTER" right but instead want to encourage it to get the
  # pointer logits right.
  # The same shouldn't be done for `token_log_zs` so that we can propagate
  # negative gradients back and discourage the model from predicting a token as
  # the model doesn't know in advance that the candidate is not a token.
  token_target_scores = jnp.where(target_pointer_mask == 0, token_target_scores,
                                  _LOG_OF_SMALL_VALUE)  # (B, O)
  # We mask out `token_target_scores` for all candidates where the target is a
  # OOV token since we don't want the model to reward the model getting
  # the token "<OOV>" right but instead want to encourage it to try to get
  # the COPY right if it is possible to indeed copy.
  # The same shouldn't be done for `token_log_zs` so that we can propagate
  # negative gradients back and discourage the model from predicting a token as
  # the model doesn't know in advance that the candidate is <OOV>.
  # TODO: Experiment with alternate ways to give credit to <OOV>.
  oov_token_mask = target_data.token_ids == oov_token_id
  token_target_scores = jnp.where(oov_token_mask == 0, token_target_scores,
                                  _LOG_OF_SMALL_VALUE)  # (B, O)

  copy_target_log_zs, copy_log_zs = _masked_logprob_terms(
      tocopo_logits.copy_logits, target_data.is_target_copy)  # (B, O)

  pointer_target_log_zs, pointer_log_zs = _masked_logprob_terms(
      tocopo_logits.pointer_logits, target_data.is_target_pointer)  # (B, O)

  # Note: if we were predicting tokens and copies independently, then the
  # cross entropy losses would be as follows:
  #  token_losses = -(token_target_scores - token_log_zs)
  #  copy_losses = -(copy_target_log_zs - copy_log_zs)
  #  pointer_losses = -(pointer_target_log_zs - pointer_log_zs)

  # Combine terms. Concat and take logsum exp over type of output head.
  # 3X(B, O, 1) -> (B, O, 3) -> (B, O)
  tocopo_target_log_zs = jnp.stack(
      [token_target_scores, copy_target_log_zs, pointer_target_log_zs],
      axis=-1)  # (B, O, 3)
  tocopo_log_zs = jnp.stack([token_log_zs, copy_log_zs, pointer_log_zs],
                            axis=-1)  # (B, O, 3)
  tocopo_logprobs = (
      jax.scipy.special.logsumexp(tocopo_target_log_zs, axis=-1) -
      jax.scipy.special.logsumexp(tocopo_log_zs, axis=-1)).squeeze()  # (B, O)

  # (B, O)
  is_pad = (target_data.token_ids == pad_token_id)
  tocopo_logprobs = jnp.where(is_pad, 0.0, tocopo_logprobs)
  return tocopo_logprobs


def _extract_token_target_scores(
    token_logits: dt.NDArrayFloatBOU,
    token_ids: dt.NDArrayIntBO) -> Tuple[dt.NDArrayFloatBO, dt.NDArrayFloatBO]:
  """Select score for target token_id."""
  # Break these out into helper functions.
  # (B, O) -> (B, O, 1)
  expanded_token_ids = jnp.expand_dims(token_ids, axis=-1)
  # (B, O, U) -> (B, O)
  token_target_scores = jnp.take_along_axis(
      token_logits, expanded_token_ids, axis=-1).squeeze()
  # (B, O, U) -> (B, O)
  token_log_zs = jax.scipy.special.logsumexp(token_logits, axis=-1)

  return token_target_scores, token_log_zs


def _masked_logprob_terms(
    scores: dt.NDArrayFloatBOV,
    mask: dt.NDArrayBoolBOV) -> Tuple[dt.NDArrayFloatBO, dt.NDArrayFloatBO]:
  """Gets log-numerator and log-denominator for logprob terms.

  We refrain from subtracting the terms (which would give logprobs) so that we
  can use these terms as a subset of outputs in a large softmax involving other
  terms as well. That is, to form a joint softmax over scores1 and scores2, we
  can compute target_score1, target_log_z1, target_score2, target_log_z2, and
  then the probability of correct output if either target1 or target2 is
  generated from the union of candidates given scores in scores1 and scores2
  is
      logsumexp(target_score1, target_score2)
      - logsumexp(target_log_z1, target_log_z2).

  Additionally note that the logsumexp is associative. i.e.
      logsumexp(logsumexp(a, b), c) = logsumexp(a, logsumexp(b, c))

  This allows us to calculate the logsumexp for one component and then later
  merge it with the logsumexp of other components.

  Args:
    scores: Dense scores that will be masked out and reduced over the last
      dimension. Usually these correspond to logits from predictions.
    mask: A boolean mask specifying which scores to retain. This is the same
      shape as `scores`. If mask[i, j] = 1 then scores[i, j] will be retained
      for calculations.

  Returns:
    A pair of two elements,
      The per-output total log masses assigned to target candidates, and
      the per-output total log mass assigned to all candidates.
  """
  # Use appropriate value for elements that should not be attended to.
  masked_scores = jnp.where(mask, scores, _LOG_OF_SMALL_VALUE)  # BTV

  # BTV -> BT
  target_log_zs = jax.scipy.special.logsumexp(masked_scores, axis=-1)
  # BTV -> BT
  log_zs = jax.scipy.special.logsumexp(scores, axis=-1)

  return target_log_zs, log_zs
