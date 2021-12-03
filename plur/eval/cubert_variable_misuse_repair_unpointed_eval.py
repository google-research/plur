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

"""Compute evaluation metrics for CuBERT VMR-U."""
import re
from typing import Optional

from absl import logging

from plur.eval.cubert_variable_misuse_repair_eval import CuBertVariableMisuseRepairEval


class CuBertVariableMisuseRepairUnpointedEval(CuBertVariableMisuseRepairEval):
  """Eval class for CuBERT VMR-U dataset."""
  # The regular expression means "ERROR_LOCATION_" followed by at least one
  # digit, follows by either the end of the line, or some non-digit followed by
  # other stuff. The second group (end of line or non-digit and stuff) is non-
  # capturing, so it is not returned in match groups.
  #
  # This will successfully reject a token "ERROR_LOCATION_13B" because it
  # does not end with a whole integer number and end of sequence or a non-digit
  _RE_TOKENIZED_POINTER = re.compile(r'^ERROR_LOCATION_(\d+)(?:\s.*$|$)')

  def parse_pointer(self, line: str) -> Optional[int]:
    """It parses the tokenized pointer/no-bug token to match the pointed VMR."""
    if line.startswith('BUG_FREE'):
      return 1  # This means bug free in the canonical VMR graphination.

    match = self._RE_TOKENIZED_POINTER.match(line)
    if not match:
      logging.warning('Failed to match %r', line)
      return None
    matching_groups = list(match.groups())
    if len(matching_groups) != 1:
      logging.warning('Did not find exactly one ERROR_LOCATION match in %r.',
                      line)
      return None
    error_pointer = int(matching_groups[0])
    if error_pointer == 1:
      raise AssertionError('Got an error location, but it points to location '
                           '1, which means there is no error. This should not '
                           f'happen: {line}')

    return error_pointer
