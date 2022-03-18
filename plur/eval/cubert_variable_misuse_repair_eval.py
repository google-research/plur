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

"""Compute evaluation metrics for CuBERT VMR."""
import dataclasses
import re
from typing import Optional, Sequence

from absl import flags
from absl import logging
from plur.eval.plur_eval import PlurEval
from plur.utils import util


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'cubert_vmr_raw_test_total_examples', 0, 'The number of raw test '
    'examples, before any pruning of graphination losses. 0 means that the raw '
    'number is unavailable, and only the observed numbers will be used for '
    'reporting metrics.')
flags.DEFINE_integer(
    'cubert_vmr_raw_test_total_buggy_examples', 0,
    'The number of raw test buggy '
    'examples, before any pruning of graphination losses.'
    '0 means that the raw '
    'number is unavailable, and only the observed numbers will be used for '
    'reporting metrics. This flag must be 0 if and only if '
    '`cubert_vmr_raw_test_total_examples` is 0.')


@dataclasses.dataclass
class _PredictionResult():
  """Collects details of a single prediction evaluation."""
  target: str
  prediction: str
  is_buggy: bool
  true_positive: bool
  true: bool
  localized: bool
  localized_and_repaired: bool
  syntax_error: bool
  sequence_match: bool


@dataclasses.dataclass
class _AllResults():
  """Collects statistics about entire evaluation."""
  total: int
  total_buggy: int

  true_positive: float
  true: float
  localized: float
  localized_and_repaired: float

  raw_total: Optional[int]
  raw_total_buggy: Optional[int]

  raw_true_positive: Optional[float]
  raw_true: Optional[float]
  raw_localized: Optional[float]
  raw_localized_and_repaired: Optional[float]

  total_syntax_errors: int
  total_localized_but_not_repaired: int

  sequence_accuracy: float


class CuBertVariableMisuseRepairEval(PlurEval):
  """Eval class for CuBERT VMR dataset.

  It computes true positive rate (classification accuracy for the BUGGY class),
  classification accuracy (for both classes together), localization accuracy,
  and localization and repair accuracy.
  """
  _RE = re.compile(r'^POINTER\((\d*),.*$')

  def __init__(self,
               prediction_file: str,
               target_file: str,
               top_n: int = 1) -> None:
    """As per superclass."""
    assert top_n == 1
    super().__init__(prediction_file, target_file, top_n=top_n)

  def parse_pointer(self, line: str) -> Optional[int]:
    """Parses a target/prediction line produced during evaluation."""
    match = self._RE.match(line)
    if not match:
      logging.warning('Failed to match %r', line)
      return None
    matching_groups = list(match.groups())
    if len(matching_groups) != 1:
      logging.warning('Did not find exactly one POINTER match in %r.', line)
      return None
    error_pointer = int(matching_groups[0])
    return error_pointer

  def compute_metric_for_one_target(self, beam_prediction_lines: Sequence[str],
                                    target_line: str) -> _PredictionResult:
    """As per superclass. Returs class and correctness."""
    assert len(beam_prediction_lines) == 1
    line = beam_prediction_lines[0]

    actual = line.strip()
    expected = target_line.strip()

    expected_error_pointer = self.parse_pointer(expected)
    assert expected_error_pointer is not None, (
        f'Ground truth should be well-formed, but was not in {expected}')
    expected_buggy = expected_error_pointer != 1

    actual_error_pointer = self.parse_pointer(actual)
    if actual_error_pointer is None:
      # The model didn't produce a valid input pointer first. Fail everything.
      return _PredictionResult(
          target=expected,
          prediction=actual,
          is_buggy=expected_buggy,
          syntax_error=True,
          sequence_match=False,
          true_positive=False,
          true=False,
          localized=False,
          localized_and_repaired=False)
    actual_buggy = actual_error_pointer != 1

    true_positive = actual_buggy and expected_buggy
    true = actual_buggy == expected_buggy
    localized = (
        actual_buggy and (actual_error_pointer == expected_error_pointer))
    localized_and_repaired = actual_buggy and (actual == expected)
    sequence_match = actual == expected

    result = _PredictionResult(
        target=expected,
        prediction=actual,
        is_buggy=expected_buggy,
        syntax_error=False,
        true_positive=true_positive,
        true=true,
        sequence_match=sequence_match,
        localized=localized,
        localized_and_repaired=localized_and_repaired)
    if localized and not localized_and_repaired:
      logging.log_every_n(logging.INFO, f'Repair failed: {result}', 100)

    return result

  def evaluate_once(self, grouped_prediction_lines, target_lines):

    # Sum all numbers for predictions and compute the final metric.
    true_positive = 0
    true = 0
    localized = 0
    localized_and_repaired = 0
    total = 0
    total_buggy = 0
    total_syntax_errors = 0
    matches = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      result = self.compute_metric_for_one_target(
          beam_prediction_lines=beam_prediction_lines, target_line=target_line)
      total_buggy += result.is_buggy
      total += 1

      matches += result.sequence_match
      if result.syntax_error:
        total_syntax_errors += 1
      else:
        true_positive += result.true_positive
        true += result.true
        localized += result.localized
        localized_and_repaired += result.localized_and_repaired

    raw_total = FLAGS.cubert_vmr_raw_test_total_examples
    raw_total_buggy = FLAGS.cubert_vmr_raw_test_total_buggy_examples

    if bool(raw_total) != bool(raw_total_buggy):
      raise AssertionError('The flags `raw_test_total_examples` and '
                           '`raw_test_total_buggy_examples` must either both '
                           'be set or both unset (or 0), but instead, they are '
                           f'({raw_total} and {raw_total_buggy}, respectively.')
    if raw_total:
      # TODO: Split _AllResults from _AllMetrics to avoid inconsistent
      # computation.
      return _AllResults(
          total=total,
          total_buggy=total_buggy,
          true=util.safe_division(true, total),
          sequence_accuracy=util.safe_division(matches, total),
          true_positive=util.safe_division(true_positive, total_buggy),
          localized=util.safe_division(localized, total_buggy),
          localized_and_repaired=util.safe_division(localized_and_repaired,
                                                    total_buggy),

          raw_total=raw_total,
          raw_total_buggy=raw_total_buggy,
          raw_true=util.safe_division(true, raw_total),
          raw_true_positive=util.safe_division(true_positive, raw_total_buggy),
          raw_localized=util.safe_division(localized, raw_total_buggy),
          raw_localized_and_repaired=util.safe_division(localized_and_repaired,
                                                        raw_total_buggy),

          total_localized_but_not_repaired=localized - localized_and_repaired,
          total_syntax_errors=total_syntax_errors)
    else:
      return _AllResults(
          total=total,
          total_buggy=total_buggy,
          true=util.safe_division(true, total),
          sequence_accuracy=util.safe_division(matches, total),
          true_positive=util.safe_division(true_positive, total_buggy),
          localized=util.safe_division(localized, total_buggy),
          localized_and_repaired=util.safe_division(localized_and_repaired,
                                                    total_buggy),

          raw_total=None,
          raw_total_buggy=None,
          raw_true=None,
          raw_true_positive=None,
          raw_localized=None,
          raw_localized_and_repaired=None,

          total_localized_but_not_repaired=localized - localized_and_repaired,
          total_syntax_errors=total_syntax_errors)

  def get_metric_as_string(self) -> str:
    all_results = self.evaluate()
    return str(all_results)
