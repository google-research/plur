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

"""Compute class accuracy for manysstubs4j.
"""
import collections

from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import util


class Manysstubs4jEval(PlurEval):
  """Eval class for manysstubs4j dataset."""

  def compute_metric_for_one_target(self, beam_prediction_lines, target_line):
    """Compute class accuracy between the prediction and target.

    We define manysstubs4j as a classification task. Currently there is
    no tool that has evaluations on this task, so for now we define the
    evaluation metric as the class accuracy. The target line is the ground truth
    bug type, ie. 'WRONG_FUNCTION_NAME'. The predictions should have the same
    format.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        predicted class name.
      target_line: The ground truth line.
    Returns:
      A integer, 1 means that one of the predicted class is correct, and 0
      otherwise.
    """
    for line in beam_prediction_lines:
      if line == target_line:
        return 1
    return 0

  def evaluate_once(self, grouped_prediction_lines, target_lines):

    # Sum all numbers for predictions and compute the final metric.
    total_correct = 0
    total_attempts = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      total_correct += self.compute_metric_for_one_target(
          beam_prediction_lines, target_line)
      total_attempts += 1

    class_acc = util.safe_division(total_correct, total_attempts)

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['class_acc'] = class_acc
    metrics_format['class_acc'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
