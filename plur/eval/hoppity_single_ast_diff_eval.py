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

"""Compute metrics that are computed in Hoppity.
"""
import collections

from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import util


class HoppitySingleAstDiffEval(PlurEval):
  """Eval class for Hoppity single AST diff dataset."""

  def compute_metric_for_one_target(self, beam_prediction_lines, target_line):
    """Compute the overall and transformation accuracy.

    We keep count of the overall accuracy and number of attempts, but also
    the accuracy and number of attempts of each transformation. The target line
    is a string of whitespace separated tokens representing the ground truth
    transformation. The first token is a transformation keyword, it can be one
    of ['add_node', 'del_node', 'replace_val', 'replace_val']. The subsequent
    outputs are transformations specific outputs. But for computing the metrics,
    we only care that they all match the ground truth. 'add_node POINTER(1)
    POINTER(2) TYPE VALUE' is an example of a target line. It means that it is
    adding a new node under node 1 with node 2 as sibling, the inserted node
    should have type TYPE and value VALUE. The predicted lines should have the
    same format.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        whitespace separated tokens representing the predicted transformation.
      target_line: The ground truth line.
    Returns:
      The number of correct and tried attempts for add, replace value, replace
      type, del transformation, and the overall correct and tried attempts.
    """
    correct = 0
    add_correct = 0
    add_attempts = 0
    rep_val_correct = 0
    rep_val_attempts = 0
    rep_type_correct = 0
    rep_type_attempts = 0
    del_correct = 0
    del_attempts = 0

    target_tokens = target_line.split(' ')
    # Check the transformation type, then check for each transformation if
    # the prediction matches the transformation. Then, we update each counter
    # accordingly.
    if target_tokens[0] == 'add_node':
      add_attempts = 1
      for prediction_line in beam_prediction_lines:
        if prediction_line == target_line:
          add_correct = 1
          correct = 1
    elif target_tokens[0] == 'del_node':
      del_attempts = 1
      for prediction_line in beam_prediction_lines:
        if prediction_line == target_line:
          del_correct = 1
          correct = 1
    elif target_tokens[0] == 'replace_val':
      rep_val_attempts = 1
      for prediction_line in beam_prediction_lines:
        if prediction_line == target_line:
          rep_val_correct = 1
          correct = 1
    elif target_tokens[0] == 'replace_type':
      rep_type_attempts = 1
      for prediction_line in beam_prediction_lines:
        if prediction_line == target_line:
          rep_type_correct = 1
          correct = 1

    return (correct, add_correct, add_attempts, rep_val_correct,
            rep_val_attempts, rep_type_correct, rep_type_attempts,
            del_correct, del_attempts)

  def evaluate_once(self, grouped_prediction_lines, target_lines):

    # Sum numbers for all predictions and compute the final metrics.
    total_correct = 0
    total_add_correct = 0
    total_add_attempts = 0
    total_rep_val_correct = 0
    total_rep_val_attempts = 0
    total_rep_type_correct = 0
    total_rep_type_attempts = 0
    total_del_correct = 0
    total_del_attempts = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      (correct, add_correct, add_attempts, rep_val_correct,
       rep_val_attempts, rep_type_correct, rep_type_attempts, del_correct,
       del_attempts) = self.compute_metric_for_one_target(
           beam_prediction_lines, target_line)

      total_correct += correct
      total_add_correct += add_correct
      total_add_attempts += add_attempts
      total_rep_val_correct += rep_val_correct
      total_rep_val_attempts += rep_val_attempts
      total_rep_type_correct += rep_type_correct
      total_rep_type_attempts += rep_type_attempts
      total_del_correct += del_correct
      total_del_attempts += del_attempts

    total_acc = util.safe_division(total_correct, len(target_lines))
    add_acc = util.safe_division(total_add_correct, total_add_attempts)
    rep_val_acc = util.safe_division(
        total_rep_val_correct, total_rep_val_attempts)
    rep_type_acc = util.safe_division(
        total_rep_type_correct, total_rep_type_attempts)
    del_acc = util.safe_division(total_del_correct, total_del_attempts)

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['total_acc'] = total_acc
    metrics_format['total_acc'] = '{:.5f}'
    metrics['add_acc'] = add_acc
    metrics_format['add_acc'] = '{:.5f}'
    metrics['rep_val_acc'] = rep_val_acc
    metrics_format['rep_val_acc'] = '{:.5f}'
    metrics['rep_type_acc'] = rep_type_acc
    metrics_format['rep_type_acc'] = '{:.5f}'
    metrics['del_acc'] = del_acc
    metrics_format['del_acc'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
