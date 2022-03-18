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
"""Converts the CuBERT VMR NC dataset (w/o output token copies) to PLUR."""
# TODO: Add a test for this dataset.

from plur.stage_1 import cubert_variable_misuse_repair_dataset
from plur.stage_1 import plur_dataset


class CuBertVariableMisuseRepairNoCopyDataset(
    cubert_variable_misuse_repair_dataset.CuBertVariableMisuseRepairDataset):
  """As per superclass, but does not allow output copying."""

  def __init__(self,
               stage_1_dir,
               configuration: plur_dataset.Configuration,
               *args,
               **kwargs) -> None:
    """As per superclass."""
    super().__init__(
        stage_1_dir, configuration, allow_output_copy=False, *args, **kwargs)

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_variable_misuse_repair_nocopy_dataset'

  def dataset_description(self) -> str:
    """As per superclass."""
    return (super().dataset_description() +
            'It confounds all output tokens, to prevent copying from input.')
