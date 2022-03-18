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

"""Compute metrics used in code2seq."""
import collections

from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import constants
from plur.utils import util


class Code2seqEval(PlurEval):
  """The eval class for Code2seq dataset."""

  def __init__(self, prediction_file, target_file, top_n=1):
    """Overwrite the superclass init.

    For Code2seq, it is unclear how to compute the number of true positives,
    false positives and false negatives for multiple predictions per target.
    Therefore we limit top_n to 1 here.

    Args:
      prediction_file: The file containing the predictions, the order of
        predictions must match the order in the target file.
      target_file: The file containing the ground truths.
      top_n: Only evaluate top_n of beam_size predictions per target. Must be
        1 here.
    """
    assert top_n == 1
    super().__init__(prediction_file, target_file, top_n=top_n)

  def _filter_impossible_name(self, name):
    if name in constants.RESERVED_TOKENS:
      return False
    else:
      return True

  def compute_metric_for_one_target(self, beam_prediction_lines, target_line):
    """Compute TP, FP and FN for tokens between the prediction and target.

    The target line is a whitespace separated string representing the
    ground truth method name, ie. 'get result'. The predicted lines should
    have the same format.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        whitespace separated tokens representing the predicted function name.
        For code2seq predictions, the length of beam_prediction_lines must be 1.
      target_line: The ground truth line.
    Returns:
      A tuple of true positive, false positive and false negative counts for
      tokens between beam_prediction_lines and target_line.
    """
    assert len(beam_prediction_lines) == 1
    prediction_line_tokens = beam_prediction_lines[0].split(' ')
    prediction_line_tokens = list(filter(self._filter_impossible_name,
                                         prediction_line_tokens))
    # No need to filter the target tokens, since it can never contain oov
    # tokens.
    target_line_tokens = target_line.split(' ')

    # Similar to how code2seq computes TP, FP and FN.

    # https://github.com/tech-srl/code2seq/blob/b16a5ba0abe0d259dc8b1c4481d0867b341d3d7b/model.py#L268

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for token in prediction_line_tokens:
      # TP = number of predicted tokens in target tokens.
      # FP = number of predicted tokens not in target tokens.
      if token in target_line_tokens:
        true_positives += 1
      else:
        false_positives += 1

    # FN = number of target tokens not in predicted tokens
    for token in target_line_tokens:
      if token not in prediction_line_tokens:
        false_negatives += 1
    return true_positives, false_positives, false_negatives

  def evaluate_once(self, grouped_prediction_lines, target_lines):
    # Sum all TP, FP and FN, and calculate precision, recall and F1 score.
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    # Sum TP, FP and FN for all predictions.
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      true_positives, false_positives, false_negatives = (
          self.compute_metric_for_one_target(
              beam_prediction_lines, target_line))
      total_true_positives += true_positives
      total_false_positives += false_positives
      total_false_negatives += false_negatives

    precision = util.safe_division(
        total_true_positives, total_true_positives + total_false_positives)
    recall = util.safe_division(
        total_true_positives, total_true_positives + total_false_negatives)
    f1_score = 2 * util.safe_division(precision * recall, precision + recall)

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['precision'] = precision
    metrics_format['precision'] = '{:.5f}'
    metrics['recall'] = recall
    metrics_format['recall'] = '{:.5f}'
    metrics['f1_score'] = f1_score
    metrics_format['f1_score'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
