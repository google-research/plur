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
"""Converts the CuBERT VMR dataset (w/o input pointers in output) to PLUR."""
# TODO: Add a test for this dataset.

from plur.stage_1 import cubert_variable_misuse_repair_dataset
from plur.stage_1 import plur_dataset


class CuBertVariableMisuseRepairUnpointedDataset(
    cubert_variable_misuse_repair_dataset.CuBertVariableMisuseRepairDataset):
  """As per superclass, but does not use pointer into the input for errors."""

  def __init__(self,
               stage_1_dir,
               configuration: plur_dataset.Configuration,
               *args,
               **kwargs) -> None:
    """As per superclass."""
    super().__init__(
        stage_1_dir, configuration, use_pointer_output=False, *args, **kwargs)

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_variable_misuse_repair_unpointed_dataset'

  def dataset_description(self) -> str:
    """As per superclass."""
    return (super().dataset_description() +
            'Instead of pointers, it uses token to localize errors.')
