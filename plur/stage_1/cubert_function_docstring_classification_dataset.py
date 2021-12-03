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
"""Converts the CuBERT Function Docstring Classification dataset to PLUR."""
import itertools
from typing import Any, List, Mapping, MutableSequence, Optional

from plur.stage_1 import cubert_dataset
from plur.utils.graph_to_output_example import GraphToOutputExample
from cubert import code_to_subtokenized_sentences


def _truncate_seq_pair(tokens_a: MutableSequence[str],
                       tokens_b: MutableSequence[str], max_length: int) -> None:
  """BERT's truncation of two token sequences."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


class CuBertFunctionDocstringClassificationDataset(cubert_dataset.CuBertDataset
                                                  ):
  """Converts CuBERT Function Docstring Classification data to a PLUR dataset.

  The dataset is created by: Aditya Kanade, Petros Maniatis, Gogul Balakrishnan,
  Kensen Shi Proceedings of the 37th International Conference on Machine
  Learning, PMLR 119:5110-5121, 2020.

  The task is to predict whether a function and a docstring match, or if they
  come from distinct contexts.

  The context consists of the body of a Python function and a docstring. This
  context is tokenized using the CuBERT Python tokenizer, and encoded as
  WordPiece vocabulary IDs from the CuBERT-released Python vocabulary. The graph
  representation is as a chain of nodes, each holding a WordPiece subtoken. The
  output is one of the two classification labels. We use separate node types
  for docstrings and for function bodies.
  """

  ALL_CLASSES = frozenset((
      'Correct',
      'Incorrect',
  ))

  def folder_path(self) -> str:
    """As per superclass."""
    return '20200621_Python/function_docstring_datasets/'

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_function_docstring_classification_dataset'

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

  Combinations of functions with their correct or incorrect documentation
  string, used to train a classifier that can tell which pairs go together. The
  JSON fields are:

    function: string, the source code of a function as text

    docstring: string, the documentation string for that function

    label: string, one of (“Incorrect”, “Correct”), the label of the example.

    info: string, an unformatted description of how the example was constructed,
    including the source dataset (always “ETHPy150Open”), the repository and
    filepath, the function name and, for “Incorrect” examples, the function
    whose docstring was substituted.
  """

  def data_to_graph_to_output_example(
      self, data: Mapping[str, Any],
      max_graph_size: int,
      split: str) -> Optional[GraphToOutputExample]:
    """Convert data example to the unified GraphToOutputExample data structure.

    The input is a function string, and the output is a class.

    There are no edges in this task, since it was targeting BERT.

    Args:
      data: A dictionary with 'function', 'docstring', 'label', and 'info' as
        keys.
      max_graph_size: The maximum number of input nodes allowed for the example.
      split: The split of the example.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A GraphToOutputExample.
    """
    del split  # Unused.
    function = data['function']
    docstring = data['docstring']
    label = data['label']
    assert label in self.ALL_CLASSES
    provenance = data['info']

    graph_to_output_example = GraphToOutputExample()

    # The input graph nodes are the source code tokens and docstring. We don't
    # filter any examples based on size. Instead, we trim the suffix of the
    # source-code sequence or docstring sequence, whichever is longest. This is
    # the same logic used by BERT for two-context examples. Note that we trim so
    # that the number of tokens plus the three delimiters (one extra between
    # function and docstring) is at most `max_graph_size`.
    sentences: List[List[str]] = (
        code_to_subtokenized_sentences.code_to_cubert_sentences(
            function, self.tokenizer, self.subword_text_encoder))
    docstring: List[List[str]] = (
        code_to_subtokenized_sentences.code_to_cubert_sentences(
            f'"""{docstring}"""', self.tokenizer, self.subword_text_encoder))
    docstring_tokens = sum(docstring, [])
    function_tokens = sum(sentences, [])
    # This updates `docstring_tokens` and `function_tokens` in place.
    _truncate_seq_pair(docstring_tokens, function_tokens, max_graph_size - 3)

    number_of_docstring_tokens = len(docstring_tokens)
    number_of_function_tokens = len(function_tokens)
    delimited_tokens = tuple(itertools.chain(('[CLS]_',), docstring_tokens,
                                             ('[SEP]_',), function_tokens,
                                             ('[SEP]_',)))
    types = tuple(['DOCSTRING'] * (number_of_docstring_tokens + 2) +
                  ['TOKEN'] * (number_of_function_tokens + 1))
    assert len(types) == len(delimited_tokens)
    assert len(delimited_tokens) <= max_graph_size
    for index, (token, token_type) in enumerate(zip(delimited_tokens, types)):
      graph_to_output_example.add_node(
          node_id=index, node_type=token_type, node_label=token)

    graph_to_output_example.add_class_output(label)
    graph_to_output_example.set_provenance(provenance)

    return graph_to_output_example
