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

"""Compute class and mean-per-class accuracy for CuBERT VM."""
from plur.eval.cubert_classification_eval import CuBertClassificationEval
from plur.stage_1.cubert_variable_misuse_classification_dataset import CuBertVariableMisuseClassificationDataset


class CuBertVariableMisuseClassificationEval(CuBertClassificationEval):
  """Eval class for CuBERT VM dataset."""

  def __init__(self,
               prediction_file: str,
               target_file: str,
               top_n: int = 1) -> None:
    """As per superclass."""
    assert top_n == 1
    super().__init__(
        prediction_file=prediction_file,
        target_file=target_file,
        all_classes=CuBertVariableMisuseClassificationDataset.ALL_CLASSES,
        top_n=top_n)
