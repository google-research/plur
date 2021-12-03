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
"""Util files for eval."""

import dataclasses
import math
from typing import Mapping, Optional, Sequence


def _get_lower_index(size, interval):
  return math.floor(size * (100 - interval) / 100)


def _get_upper_index(size, interval):
  return math.ceil(size * interval / 100) - 1


def get_confidence_intervals(bootstraps, intervals):
  """Get confidence intervals for metrics.

  Args:
    bootstraps: A list containing results from different bootstrap resamplings.
    intervals: A liist/tuple of confidence intervals to generate.

  Returns:
    A dict containing lower and upper confidence intervals for metrics in
    bootstrapped results.
  """
  assert bootstraps
  size = len(bootstraps)
  keys = bootstraps[0].metrics.keys()
  cints = {}
  for key in keys:
    metrics = sorted([b.metrics[key] for b in bootstraps])
    cints[key] = {}
    cints[key]['lower'] = []
    cints[key]['upper'] = []
    for interval in intervals:
      assert 50 <= interval < 100
      lower_idx = _get_lower_index(size, interval)
      upper_idx = _get_upper_index(size, interval)
      cints[key]['lower'].append(metrics[lower_idx])
      cints[key]['upper'].append(metrics[upper_idx])
  return cints


@dataclasses.dataclass
class Results():
  """Container for PLUR eval results."""
  total: int
  metrics: Mapping[str, float]
  # Print string format.
  metrics_format: Mapping[str, str]

  lower_confidence_interval: Optional[Mapping[str, Sequence[float]]] = None
  upper_confidence_interval: Optional[Mapping[str, Sequence[float]]] = None

  def __str__(self, sort_keys=False):
    keys = sorted(self.metrics.keys()) if sort_keys else self.metrics.keys()
    results_str_list = ['total: {:d}'.format(self.total)]
    for key in keys:
      metric = self.metrics[key]
      print_format = self.metrics_format[key]

      results_str_list.append('{}: {}'.format(key, print_format).format(metric))

      if self.lower_confidence_interval:
        lower_ci = self.lower_confidence_interval[key]
        num_intervals = len(lower_ci)
        ci_list_str = ', '.join([print_format] *
                                num_intervals).format(*lower_ci)
        results_str_list.append('{}: [{}]'.format(key + '_lower_ci',
                                                  ci_list_str))

      if self.upper_confidence_interval:
        upper_ci = self.upper_confidence_interval[key]
        num_intervals = len(upper_ci)
        ci_list_str = ', '.join([print_format] *
                                num_intervals).format(*upper_ci)
        results_str_list.append('{}: [{}]'.format(key + '_upper_ci',
                                                  ci_list_str))

    return '\n'.join(results_str_list)
