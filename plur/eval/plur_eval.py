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

"""The abstract class for implementing evaluation functions.
"""
import abc
import glob
import operator
import random

from plur.eval import util as eval_util


class PlurEval(abc.ABC):
  r"""The abstract class for implementing evaluation functions.

  Each dataset uses different evaluation function, such as exact match,
  F1 score, BLEU score, etc. We use this abstract class to implement all
  evaluation functions used in PLUR datasets. We assume that the predictions
  and ground truth are stored in the following format:
  * predictions.txt:
    - {prediction1_1}\t{prediction1_2}
    - {prediction2_1}\t{prediction2_2}
    - ...
  * targets.txt:
    - target1
    - target2
    - ...

  In this case, the first line in predictions.txt are the predictions of
  target1 in targets.txt. The predictions are separated by a tab (\t) if there
  are multiple predictions per target. The predictions must be in the same order
  as the targets.

  The inherited class must implement the following three functions:
  * compute_metric_for_one_target: This function compares a list of predicted
      lines, to a single ground truth target line. It should return relevant
      numbers for computing the final metric for the dataset.
  * evaluate: This function is used to evaluate prediction_file_pattern on
      target_file_pattern.
      It should first call read_prediction_and_target_file to get the
      predicted lines and targets lines, and then call
      compute_metric_for_one_target on each paris of predicted lines and target
      line. Then, it should aggregate all numbers from
      compute_metric_for_one_target and compute the final metric.
  * get_metric_as_string: This function should call evaluate and return the
      computed metrics as a single string.
  See code2seq_eval.py as an example on how it is implemented.
  """

  def __init__(self, prediction_file_pattern, target_file_pattern, top_n=1):
    """Constructor function.

    Args:
      prediction_file_pattern: The glob file pattern containing the predictions,
        the order of predictions must match the order in the target file.
      target_file_pattern: The glob file pattern containing the ground truths.
      top_n: Only evaluate top_n of beam_size predictions per target, the
        top_n must be smaller or equal than the beam_size.
    """
    self.prediction_file_pattern = prediction_file_pattern
    self.target_file_pattern = target_file_pattern
    self.top_n = top_n

  def read_prediction_and_target_file(self):
    """Read files matching prediction and target file patterns.

    We read the prediction and target files matching the file patterns.
    For the predictions, we group them according to the beam_size, wherein
    returned predictions are a list of lists. The nested list contains
    predictions for a single target.

    Returns:
      It returns two lists. grouped_prediction_lines is a list of lists,
      where the nested list contains the predictions for one target.
      target_lines is a list of target lines.
    """
    prediction_lines = []
    target_lines = []
    for filename in sorted(glob.glob(self.prediction_file_pattern)):
      with open(filename) as f:
        prediction_lines.extend(f.read().splitlines())
    for filename in sorted(glob.glob(self.target_file_pattern)):
      with open(filename) as f:
        target_lines.extend(f.read().splitlines())

    assert len(prediction_lines) == len(target_lines)
    # Multiple predictions are separated by a tab (\t). It is designed in this
    # way since we know for sure that tabs are not part of the prediction
    # vocabulary.
    grouped_prediction_lines = [
        line.split('\t')
        for line in prediction_lines
    ]

    return grouped_prediction_lines, target_lines

  @abc.abstractmethod
  def compute_metric_for_one_target(self):
    """This function computes numbers relevant for one target."""
    pass

  @abc.abstractmethod
  def evaluate_once(self, grouped_prediction_lines, target_lines, **kwargs):
    """Runs a single bootstrap iteration to get final metrics."""
    pass

  def evaluate(self, num_bootstraps=1, seed=42, ci_intervals=(90, 95, 99)):
    """Run evaluation to get final metrics.

    Args:
      num_bootstraps: Number of times to run bootstrap resampling.
      seed: Random seed
      ci_intervals: list of confidence intervals desired in percentages.

    Returns:
      results: A results object containing metrics from task specific
          evaluators.
    """
    grouped_prediction_lines, target_lines = (
        self.read_prediction_and_target_file())

    results = self.evaluate_once(grouped_prediction_lines, target_lines)

    if num_bootstraps > 1:
      bootstraps = []
      rng = random.Random(seed)
      results.lower_confidence_interval = {}
      results.upper_confidence_interval = {}
      for unused_bootstrap_id in range(num_bootstraps):
        resamples = rng.choices(
            list(zip(grouped_prediction_lines, target_lines)),
            k=len(target_lines))
        resampled_grouped_prediction_lines = list(map(
            operator.itemgetter(0), resamples))
        resampled_target_lines = list(map(operator.itemgetter(1), resamples))
        bootstraps.append(
            self.evaluate_once(resampled_grouped_prediction_lines,
                               resampled_target_lines))
      # Add confidence intervals to results.
      cints = eval_util.get_confidence_intervals(
          bootstraps, intervals=ci_intervals)
      for key in results.metrics.keys():
        results.lower_confidence_interval[key] = cints[key]['lower']
        results.upper_confidence_interval[key] = cints[key]['upper']

    return results

  @abc.abstractmethod
  def get_metric_as_string(self) -> str:
    """This function returns the metrics as a string."""
    pass
