# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute metrics that are computed in the retrieve and edit paper."""
import collections
import re
from typing import Sequence, Tuple, Union

from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import util


class RetrieveAndEditEval(PlurEval):
  """The eval class for the retrieve and edit dataset."""

  def compute_metric_for_one_target(
      self, beam_prediction_lines: Sequence[str],
      target_line: str) -> Tuple[float, int, float]:
    """Return stats computed in the retrieve and edit dataset.

    The target line is a whitespace separated string representing the
    ground truth function tokens. The predicted lines are in the same format.
    We compute the average and maximum number of successive tokens correctly
    predicted, along with the BLEU score for the first prediction.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        whitespace separated tokens representing the predicted function tokens.
      target_line: The ground truth line.

    Returns:
      Average and maximum number of successive tokens correctly predicted, along
      with the BLEU score for the first prediction.
    """
    assert beam_prediction_lines

    target_tokens = target_line.split(' ')

    beam_expected_tokens_predicted = []
    beam_max_tokens_predicted = []
    for prediction_line in beam_prediction_lines:
      prediction_tokens = prediction_line.split(' ')
      matches = _get_matches(prediction_tokens, target_tokens)
      beam_expected_tokens_predicted.append(
          _expected_correctly_predicted_tokens(matches))
      beam_max_tokens_predicted.append(
          _max_correctly_predicted_tokens(matches))

    # Use the best one from the average tokens predicted.
    best_index = np.argmax(beam_expected_tokens_predicted)

    # BLEU score is only calculated for the first prediction in the beam.
    target_bleu_eval_tokens = _tokenize_for_bleu_eval(target_line)
    prediction_bleu_eval_tokens = _tokenize_for_bleu_eval(
        beam_prediction_lines[0])
    bleu_score = _bleu(target_bleu_eval_tokens, prediction_bleu_eval_tokens)

    return (beam_expected_tokens_predicted[best_index],
            beam_max_tokens_predicted[best_index], bleu_score * 100)

  def evaluate_once(self, grouped_prediction_lines: Sequence[Sequence[str]],
                    target_lines: Sequence[str]) -> eval_util.Results:
    all_expected_successive_tokens = []
    all_max_successive_tokens = []
    all_bleu_scores = []
    for beam_prediction_lines, target_line in zip(grouped_prediction_lines,
                                                  target_lines):
      expected_successive_tokens, max_successive_tokens, bleu_score = (
          self.compute_metric_for_one_target(beam_prediction_lines,
                                             target_line))
      all_expected_successive_tokens.append(expected_successive_tokens)
      all_max_successive_tokens.append(max_successive_tokens)
      all_bleu_scores.append(bleu_score)

    average_expected_successive_tokens = util.safe_division(
        sum(all_expected_successive_tokens),
        len(all_expected_successive_tokens))
    average_max_successive_tokens = util.safe_division(
        sum(all_max_successive_tokens),
        len(all_max_successive_tokens))
    average_bleu_score = util.safe_division(
        sum(all_bleu_scores), len(all_bleu_scores))

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['average_expected_successive_tokens'] = (
        average_expected_successive_tokens)
    metrics_format['average_expected_successive_tokens'] = '{:.5f}'
    metrics['average_max_successive_tokens'] = average_max_successive_tokens
    metrics_format['average_max_successive_tokens'] = '{:.5f}'
    metrics['bleu'] = average_bleu_score
    metrics_format['bleu'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self) -> str:
    return str(self.evaluate())


def _get_matches(prediction: Sequence[str], reference: Sequence[str]
                 ) -> Sequence[bool]:
  """Get the token matches as a list of boolean.

  Args:
    prediction: The predicted token list.
    reference: The ground truth token list.

  Returns:
    A list of boolean. l[i] == True means that i:th token matched between the
    prediction and reference
  """
  matches = []
  for i in range(max(len(prediction), len(reference))):
    if i < len(prediction) and i < len(reference):
      matches.append(prediction[i] == reference[i])
    else:
      matches.append(False)
  return matches


def _run_length_encoding(
    inarray: Sequence[bool]) -> Tuple[Union[Sequence[int], None],
                                      Union[Sequence[int], None],
                                      Union[Sequence[bool], None]]:
  """The run length encoding algorithm.

  This function is to compute information about sequences in which the same data
  values occurs. It is originally a data compression algorithm. For example
  [True, True, False, False, True] can be compressed to 2-True 2-False 1-True.
  This algorithm keeps the track of:
    1. The length of each sequences with the same value.
    2. The start position of each sequence.
    3. The value in each sequence.

  It is the 'def rle(inarray)' function from 'github_eval.py' in the published
  retrieve and edit source code.

  Args:
    inarray: The input array.

  Returns:
    A tuple of (run lengths, start positions, values). Run lengths is a list
    of integers representing the length  of each sequences with the same value.
    Start positions is a list of integers representing the start position of
    each sequence. Values is the values in each sequences.
  """
  ia = np.asarray(inarray)               # force numpy
  n = len(ia)
  if n == 0:
    return (None, None, None)
  else:
    y = ia[1:] != ia[:-1]                # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)    # must include last element posi
    z = np.diff(np.append(-1, i))        # run lengths
    p = np.cumsum(np.append(0, z))[:-1]  # positions
    return (z, p, ia[i])


def _max_correctly_predicted_tokens(matches: Sequence[bool]) -> int:
  """Get the length of the longest consecutive correctly predicted sequence.

  This is the slightly modifed version of 'def correct_runlen(rankin, cut)'
  function from 'github_eval.py' in the published retrieve and edit source code.

  Args:
    matches: A list of boolean, True means that the token matched between the
      prediction and the target, False otherwise.

  Returns:
    The length of the longest consecutive correctly predicted sequence.
  """
  if any(matches):
    # lengths is np.ndarray of the sequence lengths.
    # values is the respective values of each sequence.
    lengths, _, values = _run_length_encoding(matches)
    return np.max(lengths[values])
  else:
    return 0


def _expected_correctly_predicted_tokens(matches: Sequence[bool]) -> float:
  """Get the expected length of the consecutive correctly predicted sequences.

  This is the slightly modifed version of 'def avg_runlen(rankin, cut)'
  function from 'github_eval.py' in the published retrieve and edit source code.

  Args:
    matches: A list of boolean, True means that the token matched between the
      prediction and the target, False otherwise.

  Returns:
    The expected length of consecutive correctly predicted sequences.
  """
  if any(matches):
    # lengths is np.ndarray of the sequence lengths.
    # values is the respective values of each sequence.
    lengths, _, values = _run_length_encoding(matches)
    # The probability of choosing the sequences that matches the ground truth.
    seq_prob = lengths[values] / float(np.sum(lengths))
    # The expected matching sequence length if we sample uniformly.
    # lengths[values] is the length of all matching sequences. For each matching
    # sequence, the expected consecutive correctly predicted tokens is
    # 1 * (1/n) + 2 * (1/n) + 3 * (1/n) + ... = n(n+1)/2n = (n+1)/n
    expected_dist = (lengths[values]+1.0)/2.0
    return np.sum(np.array(expected_dist)*np.array(seq_prob))
  else:
    return 0.0


def _tokenize_for_bleu_eval(code: str) -> Sequence[str]:
  """Tokenize the input for computing the bleu score.

  We perform the same tokenizing step as the retrieve and edit dataset. It is
  the 'def tokenize_for_bleu_eval(code)' function from 'github_eval.py' in the
  published retrieve and edit source code.

  Args:
    code: A string representing the code.

  Returns:
    The tokenized code as a list.
  """
  code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
  code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
  code = re.sub(r'\s+', ' ', code)
  code = code.replace('"', '`')
  code = code.replace('\'', '`')
  tokens = [t for t in code.split(' ') if t]
  return tokens


def _bleu(reference: Sequence[str], predict: Sequence[str]) -> float:
  """Compute the bleu score.

  We compute the bleu score in the same way as the retrieve and edit dataset.
  It is the 'def bleu(reference, predict)' function from 'gtd/utils.py' in the
  published retrieve and edit source code.

  Args:
    reference: The reference token list.
    predict: The predicted token list.

  Returns:
    The bleu score.
  """
  if not predict:
    if not reference:
      return 1.0
    else:
      return 0.0

  # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
  n = max(min(4, len(reference), len(predict)), 1)
  weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
  return sentence_bleu([reference], predict, weights)
