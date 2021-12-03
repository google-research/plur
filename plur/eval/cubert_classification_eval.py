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

"""Compute class and mean-per-class accuracy for CuBERT classification tasks."""
import collections
import dataclasses
from typing import AbstractSet, Mapping, MutableMapping, Sequence

from plur.eval.plur_eval import PlurEval
from plur.utils import util


@dataclasses.dataclass
class _PredictionResult():
  target: str
  prediction: str
  correct: bool


@dataclasses.dataclass
class _AllResults():
  total: int
  accuracy: float
  per_class_accuracies: Mapping[str, float]
  mean_per_class_accuracy: float


class CuBertClassificationEval(PlurEval):
  """Eval class for CuBERT classification datasets.

  It computes total classification accuracy, as well as mean per-class accuracy.
  """

  def __init__(self,
               prediction_file: str,
               target_file: str,
               all_classes: AbstractSet[str],
               top_n: int = 1) -> None:
    """As per superclass."""
    assert top_n == 1
    self.all_classes = all_classes
    super().__init__(prediction_file, target_file, top_n=top_n)

  def compute_metric_for_one_target(self, beam_prediction_lines: Sequence[str],
                                    target_line: str) -> _PredictionResult:
    """As per superclass. Returs class and correctness."""
    assert len(beam_prediction_lines) == 1
    line = beam_prediction_lines[0]

    actual = line.strip()
    expected = target_line.strip()

    return _PredictionResult(
        prediction=actual,
        target=expected,
        correct=(actual == expected))

  def evaluate_once(self, grouped_prediction_lines, target_lines):

    # Sum all numbers for predictions and compute the final metric.
    per_class_correct = collections.defaultdict(int)
    per_class_total = collections.defaultdict(int)
    correct = 0
    total = 0
    for beam_prediction_lines, target_line in zip(
        grouped_prediction_lines, target_lines):
      result = self.compute_metric_for_one_target(
          beam_prediction_lines=beam_prediction_lines, target_line=target_line)
      if result.correct:
        per_class_correct[result.target] += 1
        correct += 1
      per_class_total[result.target] += 1
      total += 1

    per_class_accuracy: MutableMapping[str, float] = {}
    for class_name in per_class_total:
      assert 'CLASS_' in class_name
      original_class_name = class_name.replace('CLASS_', '', 1)
      assert (
          original_class_name
          in self.all_classes), (
              f'Label {original_class_name} should be in the label set '
              f'{self.all_classes} but is not.')
      per_class_accuracy[class_name] = util.safe_division(
          per_class_correct[class_name], per_class_total[class_name])
    accuracy: float = util.safe_division(correct, total)
    # We ignore in the computation of mean accuracy those classes that did not
    # appear at all in the test dataset.
    mean_per_class_accuracy: float = util.safe_division(
        sum(per_class_accuracy.values()), len(per_class_accuracy))

    return _AllResults(
        total=total,
        accuracy=accuracy,
        per_class_accuracies=per_class_accuracy,
        mean_per_class_accuracy=mean_per_class_accuracy)

  def get_metric_as_string(self) -> str:
    all_results = self.evaluate()
    return str(all_results)
