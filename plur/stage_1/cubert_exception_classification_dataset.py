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
"""Converts the CuBERT Exception Classification dataset to PLUR."""
from typing import Any, Mapping, Optional

from plur.stage_1 import cubert_dataset
from plur.utils.graph_to_output_example import GraphToOutputExample


class CuBertExceptionClassificationDataset(cubert_dataset.CuBertDataset):
  """Converts CuBERT Exception Classification data to a PLUR dataset.

  The dataset is created by: Aditya Kanade, Petros Maniatis, Gogul Balakrishnan,
  Kensen Shi Proceedings of the 37th International Conference on Machine
  Learning, PMLR 119:5110-5121, 2020.

  The task is to predict the exception type (one of 20) that should be caught
  by an exception block in Python.

  The classes (i.e., available exception types) are:
  ValueError, KeyError, AttributeError, TypeError, OSError, IOError,
  ImportError, IndexError, DoesNotExist, KeyboardInterrupt, StopIteration,
  AssertionError, SystemExit, RuntimeError, HTTPError, UnicodeDecodeError,
  NotImplementedError, ValidationError, ObjectDoesNotExist, NameError.

  The context consists of the body of a Python function, where the exception
  type to predict has been replaced by a special __HOLE__ token. This context is
  tokenized using the CuBERT Python tokenizer, and encoded as WordPiece
  vocabulary IDs from the CuBERT-released Python vocabulary. The graph
  representation is as a chain of nodes, each holding a WordPiece subtoken. The
  output is one of the classification labels above.
  """

  ALL_CLASSES = frozenset((
      'AssertionError',
      'AttributeError',
      'DoesNotExist',
      'HTTPError',
      'IOError',
      'ImportError',
      'IndexError',
      'KeyError',
      'KeyboardInterrupt',
      'NameError',
      'NotImplementedError',
      'OSError',
      'ObjectDoesNotExist',
      'RuntimeError',
      'StopIteration',
      'SystemExit',
      'TypeError',
      'UnicodeDecodeError',
      'ValidationError',
      'ValueError',
  ))

  def folder_path(self) -> str:
    """As per superclass."""
    return '20200621_Python/exception_datasets/'

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_exception_classification_dataset'

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

  Combinations of functions where one exception type has been masked, along with
  a label indicating the masked exception type. The JSON fields are:
    function: string, the source code of a function as text, in which one
      exception type has been replaced with the special token “HOLE”
    label: string, one of (ValueError, KeyError, AttributeError, TypeError,
      OSError, IOError, ImportError, IndexError, DoesNotExist,
      KeyboardInterrupt,
      StopIteration, AssertionError, SystemExit, RuntimeError, HTTPError,
      UnicodeDecodeError, NotImplementedError, ValidationError,
      ObjectDoesNotExist, NameError, None), the masked exception type. Note that
      None never occurs in the data and will be removed in a future release.
    info: string, an unformatted description of how the example was constructed,
      including the source dataset (always “ETHPy150Open”), the repository and
      filepath, and the fully-qualified function name.
  """

  def data_to_graph_to_output_example(
      self, data: Mapping[str, Any],
      max_graph_size: int,
      split: str) -> Optional[GraphToOutputExample]:
    """Convert data example to the unified GraphToOutputExample data structure.

    The input is a function string, and the output is the full name of the
    exception-type class.

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
            data, self.ALL_CLASSES, max_graph_size, split))
