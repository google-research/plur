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

"""Compute metrics used in Great VarMisuse.
"""
import collections

from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval
from plur.utils import util


class GreatVarMisuseEval(PlurEval):
  """Eval class for Great VarMisuse dataset."""

  def __init__(self, prediction_file, target_file, top_n=1):
    """Overwrite the superclass init.

    For Great dataset, it is unclear how to compute all accuracies when we have
    multiple predictions per target. Therefore we limit top_n to 1.

    Args:
      prediction_file: The file containing the predictions, the order of
        predictions must match the order in the target file.
      target_file: The file containing the ground truths.
      top_n: Only evaluate top_n of beam_size predictions per target. Must be
        1 here.
    """
    assert top_n == 1
    super().__init__(prediction_file, target_file, top_n=top_n)

  def assert_target_line(self, target_line):
    assert target_line.startswith('POINTER(')

  def compute_metric_for_one_target(self, beam_prediction_lines, target_line):
    """Compute class accuracy, location and repair accuracy.

    The target line is a whitespace separated string representing the
    ground truth classification and/or repair. The target line can be a single
    'NO_BUG' token, meaning that there are no bugs. The target line can also
    have two outputs, where the first output is a pointer pointing to the
    bug location, and the second output is a token that should replace the
    variable at the bug location. For example 'POINTER(1) foo' is a valid
    target line, it means to replace variable at node 1 with 'foo'. The
    predicted lines should have the same format.

    Args:
      beam_prediction_lines: A list of prediction strings. Each string is a
        whitespace separated tokens representing the predicted variable
        misuse location and repair.
      target_line: The ground truth line.
    Returns:
      The number of correct and tried attempts of class prediction, and the
      number of correct and tried attempts of location and/or repair.
    """
    assert len(beam_prediction_lines) == 1
    prediction_line = beam_prediction_lines[0]

    seq_correct = 0
    bug_free_class_correct = 0
    loc_correct = 0
    rep_correct = 0
    loc_rep_correct = 0
    loc_rep_attempt = 0

    if prediction_line == target_line:
      seq_correct = 1

    # Check bug-free examples first.
    if target_line == 'NO_BUG':
      if prediction_line == 'NO_BUG':
        bug_free_class_correct = 1
    else:
      self.assert_target_line(target_line)
      # The target is not 'NO_BUG', meaning that we should predict a location
      # and repair here.
      loc_rep_attempt = 1

      # Check if we got both location and repair correct.
      if prediction_line == target_line:
        loc_rep_correct = 1

      # A bug prediction has 2 components:
      # The first component is the pointer (location) and the second
      # component is the repair. The repair is usually one token but can
      # consist of multiple tokens as well. e.g. `def _get(`
      target_tokens = target_line.split(' ')
      prediction_tokens = prediction_line.split(' ')
      # Check if we got the location correct, ie. the first token matches.
      if prediction_tokens[0] == target_tokens[0]:
        loc_correct = 1
      # Check if we got the repair correct, i.e. the list of tokens after the
      # first matches.
      if prediction_tokens[1:] == target_tokens[1:]:
        rep_correct = 1

    return (seq_correct, bug_free_class_correct, loc_correct, rep_correct,
            loc_rep_correct, loc_rep_attempt)

  def evaluate_once(self, grouped_prediction_lines, target_lines):

    # Sum all numbers and compute the final metric.
    total_seq_correct = 0
    total_bug_free_class_correct = 0
    total_loc_correct = 0
    total_rep_correct = 0
    total_loc_rep_correct = 0
    total_loc_rep_attempt = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      (seq_correct, bug_free_class_correct, loc_correct, rep_correct,
       loc_rep_correct, loc_rep_attempt) = self.compute_metric_for_one_target(
           beam_prediction_lines, target_line)
      total_seq_correct += seq_correct
      total_bug_free_class_correct += bug_free_class_correct
      total_loc_correct += loc_correct
      total_rep_correct += rep_correct
      total_loc_rep_correct += loc_rep_correct
      total_loc_rep_attempt += loc_rep_attempt

    loc_and_rep_acc = util.safe_division(
        total_loc_rep_correct, total_loc_rep_attempt)
    seq_acc = util.safe_division(total_seq_correct, len(target_lines))
    bug_free_class_acc = util.safe_division(
        total_bug_free_class_correct,
        len(target_lines) - total_loc_rep_attempt)
    loc_acc = util.safe_division(
        total_loc_correct, total_loc_rep_attempt)
    rep_acc = util.safe_division(
        total_rep_correct, total_loc_rep_attempt)

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['seq_acc'] = seq_acc
    metrics_format['seq_acc'] = '{:.5f}'
    metrics['loc_and_rep_acc'] = loc_and_rep_acc
    metrics_format['loc_and_rep_acc'] = '{:.5f}'
    metrics['bug_free_class_acc'] = bug_free_class_acc
    metrics_format['bug_free_class_acc'] = '{:.5f}'
    metrics['loc_acc'] = loc_acc
    metrics_format['loc_acc'] = '{:.5f}'
    metrics['rep_acc'] = rep_acc
    metrics_format['rep_acc'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
