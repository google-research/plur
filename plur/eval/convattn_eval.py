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

"""Compute metrics used in Convolutional Attention Network.
"""
import collections
import operator

from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import util


class ConvattnEval(PlurEval):
  """Eval class for Convattn dataset."""

  def compute_metric_for_one_target(self, beam_prediction_lines, target_line):
    """Compute metric between predictions and target.

    In Convattn, the authors only select the prediction with the highest
    F1 score for each target. Therefore we compute F1 score for all predictions
    and return precision, recall and F1 score for the prediction with the
    highest F1 score, along with exact match (1 or 0). The target line is a
    whitespace separated string representing the ground truth method name, ie.
    'get result'. The predicted lines should have the same format.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        whitespace separated tokens representing the predicted function name.
      target_line: The ground truth line.
    Returns:
      A tuple of precision, recall, F1 score and exact match counts for tokens
      in beam_prediction_lines that has the highest F1 score.
    """
    target_line_tokens = target_line.split(' ')
    target_line_tokens = [token.lower() for token in target_line_tokens]

    # List to store all the numbers, later we will use the f1_score_list
    # to select the numbers.
    precision_list = []
    recall_list = []
    f1_score_list = []
    exact_match_list = []
    for prediction_line in beam_prediction_lines:
      if prediction_line == target_line:
        exact_match = 1
      else:
        exact_match = 0

      true_positives = 0
      false_positives = 0
      false_negatives = 0
      prediction_line_tokens = prediction_line.split(' ')
      prediction_line_tokens = [
          token.lower() for token in prediction_line_tokens]
      for token in set(prediction_line_tokens):
        # TP = number of predicted tokens in target tokens.
        # FP = number of predicted tokens not in target tokens.
        if token in target_line_tokens:
          true_positives += 1
        else:
          false_positives += 1
      # FN = number of target tokens not in predicted tokens.
      for token in target_line_tokens:
        if token not in prediction_line_tokens:
          false_negatives += 1

      precision = util.safe_division(
          true_positives, true_positives + false_positives)
      recall = util.safe_division(
          true_positives, true_positives + false_negatives)
      f1_score = 2 * util.safe_division(
          precision * recall, precision + recall)

      precision_list.append(precision)
      recall_list.append(recall)
      f1_score_list.append(f1_score)
      exact_match_list.append(exact_match)

    # Find the highest F1 score, and then return the corresponding numbers.
    max_f1_score_index = max(enumerate(f1_score_list),
                             key=operator.itemgetter(1))[0]
    return (precision_list[max_f1_score_index],
            recall_list[max_f1_score_index],
            f1_score_list[max_f1_score_index],
            exact_match_list[max_f1_score_index])

  def evaluate_once(self, grouped_prediction_lines, target_lines):
    """Compute the average precision, recall, f1_score and exact match."""

    precision_list = []
    recall_list = []
    f1_score_list = []
    total_exact_matches = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      precision, recall, f1_score, exact_match = (
          self.compute_metric_for_one_target(
              beam_prediction_lines, target_line))
      precision_list.append(precision)
      recall_list.append(recall)
      f1_score_list.append(f1_score)
      total_exact_matches += exact_match

    average_precision = util.safe_division(
        sum(precision_list), len(target_lines))
    average_recall = util.safe_division(
        sum(recall_list), len(target_lines))
    average_f1_score = util.safe_division(
        sum(f1_score_list), len(target_lines))
    exact_match_percentage = util.safe_division(total_exact_matches,
                                                len(target_lines))

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['precision'] = average_precision
    metrics_format['precision'] = '{:.5f}'
    metrics['recall'] = average_recall
    metrics_format['recall'] = '{:.5f}'
    metrics['f1_score'] = average_f1_score
    metrics_format['f1_score'] = '{:.5f}'
    metrics['exact_match_percentage'] = exact_match_percentage
    metrics_format['exact_match_percentage'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
