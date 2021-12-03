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
"""Converts the CuBERT Wrong Operator Classification dataset to PLUR."""
from typing import Any, Mapping, Optional

from plur.stage_1 import cubert_dataset
from plur.utils.graph_to_output_example import GraphToOutputExample


class CuBertWrongOperatorClassificationDataset(cubert_dataset.CuBertDataset):
  """Converts CuBERT Wrong Operator Classification data to a PLUR dataset.

  The dataset is created by: Aditya Kanade, Petros Maniatis, Gogul Balakrishnan,
  Kensen Shi Proceedings of the 37th International Conference on Machine
  Learning, PMLR 119:5110-5121, 2020.

  The task is to predict whether a function contains an incorrect operator.

  The context consists of the body of a Python function. This context is
  tokenized using the CuBERT Python tokenizer, and encoded as WordPiece
  vocabulary IDs from the CuBERT-released Python vocabulary. The graph
  representation is as a chain of nodes, each holding a WordPiece subtoken. The
  output is one of the two classification labels.
  """

  ALL_CLASSES = frozenset((
      'Correct',
      'Wrong binary operator',
  ))

  def folder_path(self) -> str:
    """As per superclass."""
    return '20200621_Python/wrong_binary_operator_datasets/'

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_wrong_operator_classification_dataset'

  def dataset_description(self) -> str:
    """As per superclass."""
    return """From CuBERT website:

  Here we describe the 6 Python benchmarks we created. All 6 benchmarks were
  derived from ETH Py150 Open. All examples are stored as sharded text files.
  Each text line corresponds to a separate example encoded as a JSON object. For
  each dataset, we release separate training/validation/testing splits along the
  same boundaries that ETH Py150 Open splits its files to the corresponding
  splits. The fine-tuned models are the checkpoints of each model with the
  highest validation accuracy.

  Combinations of functions where one binary operator has been swapped with
  another, to create a buggy example, or left undisturbed, along with a label
  indicating if this bug-injection has occurred. The JSON fields are:

    function: string, the source code of a function as text.

    label: string, one of (“Correct”, “Wrong binary operator”) indicating if
      this is a buggy or bug-free example.

    info: string, an unformatted description of how the example was constructed,
      including the source dataset (always “ETHPy150Open”), the repository and
      filepath, the function, and whether the example is bugfree (marked
      “original”) or the operator replacement has occurred (e.g., “== -> !=”).
  """

  def data_to_graph_to_output_example(
      self, data: Mapping[str, Any],
      max_graph_size: int,
      split: str) -> Optional[GraphToOutputExample]:
    """Convert data example to the unified GraphToOutputExample data structure.

    The input is a function string, and the output is a class.

    There are no edges in this task, since it was targeting BERT.

    Args:
      data: A dictionary with 'function', 'label', and 'info' as keys.
      max_graph_size: The maximum number of input nodes allowed for the example.
      split: The split of the example.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A GraphToOutputExample.
    """
    return (
        self.single_classification_data_dictionary_to_graph_to_output_example(
            data, classes=self.ALL_CLASSES, max_graph_size=max_graph_size,
            split=split))
