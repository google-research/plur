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
"""Converts the CuBERT VMR dataset to PLUR."""
# TODO: Add a test for this dataset.
import itertools
from typing import Any, Mapping, Optional

import apache_beam as beam
from plur.stage_1 import cubert_dataset
from plur.stage_1 import plur_dataset
from plur.utils.graph_to_output_example import GraphToOutputExample

from cubert import code_to_subtokenized_sentences


_BEAM_METRIC_NAMESPACE = 'cubert_variable_misuse_repair_dataset'


class CuBertVariableMisuseRepairDataset(cubert_dataset.CuBertDataset):
  """Converts CuBERT Variable Misuse Repair data to a PLUR dataset.

  The dataset is created by: Aditya Kanade, Petros Maniatis, Gogul Balakrishnan,
  Kensen Shi Proceedings of the 37th International Conference on Machine
  Learning, PMLR 119:5110-5121, 2020.

  The task is to predict where a variable misuse may be, and how to repair it,
  by pointing to the correct variable to use instead.

  The context consists of the body of a Python function. This context is
  tokenized using the CuBERT Python tokenizer, and encoded as WordPiece
  vocabulary IDs from the CuBERT-released Python vocabulary. The graph
  representation is as a chain of nodes, each holding a WordPiece subtoken. The
  output is one of the two classification labels.
  """

  ALL_CLASSES = frozenset((
      'Correct',
      'Variable misuse',
  ))

  _CONFOUNDING_STRING = '###DO_NOT_COPY_ME###'

  def __init__(self,
               stage_1_dir,
               configuration: plur_dataset.Configuration,
               *args,
               use_pointer_output: bool = True,
               allow_output_copy: bool = True,
               **kwargs) -> None:
    """As per superclass. It initializes variations (unpointed, no-copy)."""
    super().__init__(stage_1_dir, configuration, *args, **kwargs)
    self.use_pointer_output = use_pointer_output
    self.allow_output_copy = allow_output_copy

  def folder_path(self) -> str:
    """As per superclass."""
    return '20200621_Python/variable_misuse_repair_datasets/'

  def dataset_name(self) -> str:
    """As per superclass."""
    return 'cubert_variable_misuse_repair_dataset'

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

  Variable-misuse localization and repair. Combinations of functions where one
  use of a variable may have been replaced with another variable defined in the
  same context, along with information that can be used to localize and repair
  the bug, as well as the location of the bug if such a bug exists. The JSON
  fields are:

  function: a list of strings, the source code of a function, tokenized with the
  vocabulary common vocabulary. Note that, unlike other task datasets, this
  dataset gives a tokenized function, rather than the code as a single string.

  target_mask: a list of integers (0 or 1). If the integer at some position is
  1, then the token at the corresponding position of the function token list is
  a correct repair for the introduced bug. If a variable has been split into
  multiple tokens, only the first subtoken is marked in this mask. If the
  example is bug-free, all integers are 0.

  error_location_mask: a list of integers (0 or 1). If the integer at some
  position is 1, then there is a variable-misuse bug at the corresponding
  location of the tokenized function. In a bug-free example, the first integer
  is 1. There is exactly one integer set to 1 for all examples. If a variable
  has been split into multiple tokens, only the first subtoken is marked in this
  mask.

  candidate_mask: a list of integers (0 or 1). If the integer at some position
  is 1, then the variable starting at that position in the tokenized function is
  a candidate to consider when repairing a bug. Candidates are all variables
  defined in the function parameters or via variable declarations in the
  function. If a variable has been split into multiple tokens, only the first
  subtoken is marked in this mask, for each candidate.

  provenance: string, an unformatted description of how the example was
  constructed, including the source dataset (always “ETHPy150Open”), the
  repository and filepath, the function, and whether the example is bugfree
  (marked “original”) or the buggy/repair token positions and variables (e.g.,
  “16/18 kwargs → self”). 16 is the position of the introduced error, 18 is the
  location of the repair.
  """

  def data_to_graph_to_output_example(
      self, data: Mapping[str, Any],
      max_graph_size: int,
      split: str) -> Optional[GraphToOutputExample]:
    """Convert data example to the unified GraphToOutputExample data structure.

    The input is a function string, and the output is localization pointer and
    if the example is buggy, a sequence of subtokens for the repair.

    There are no edges in this task, since it was targeting BERT.

    Args:
      data: A dictionary with 'function', 'target_mask', 'candidate_mask',
        'error_location_mask', and 'provenance' as keys.
      max_graph_size: The maximum number of input nodes allowed for the example.
      split: The split of the example.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A GraphToOutputExample. None if this example cannot be created.
    """

    # The target mask points to the first subtoken of the repair. However, there
    # might be multiple subtokens in the repair identifier. Since we want to put
    # in the output the full repair token (i.e., all of its subtokens), we must
    # figure it out from the input. We start with the first target mask
    # occurrence and we decode the first whole token.
    target_mask = data['target_mask']
    function_tokens = data['function']
    target_indices = tuple(i for i, x in enumerate(target_mask) if x == 1)
    # We interpret all targets and require them to be identical, since we
    # have not done any trimming yet, and we assume the dataset to be correct.
    target_interpretations = []
    whole_tokens = []
    for target_index in target_indices:
      whole_token, end_target_index = (
          code_to_subtokenized_sentences.next_whole_token(
              function_tokens[target_index:], self.tokenizer,
              self.subword_text_encoder))
      target_subtokens = tuple(function_tokens[target_index:target_index +
                                               end_target_index])
      target_interpretations.append(target_subtokens)
      whole_tokens.append(whole_token)

    # Is this a valid example?
    original_error_location_mask = data['error_location_mask']
    original_is_bug_free = original_error_location_mask[0] == 1
    assert sum(original_error_location_mask) == 1, (
        'Exactly one error location mask must be set to 1, but instead got '
        f'{original_error_location_mask} in {data}')
    assert not target_indices or not original_is_bug_free, (
        # bug_free => not target_indices
        # not bug_free or not target_indices
        # Reminder: it's not possible for the first subtoken to be the site of
        # an error, so using the first subtoken as an indication of the absence
        # of bug is sound for this task, and in the absence of any prefix
        # trimming.
        'Bug-free (error mask on at first location) implies there must be '
        f'no targets in target mask, but instead got {data}')
    assert original_is_bug_free or target_indices, (
        # buggy => target_indices
        # not bug_free => target_indices
        # bug_free or target_indices
        'Buggy (error mask not on at first location) implies there must be '
        f'some targets in error mask, but instead got {data}')
    assert (
        len(target_indices) == len(target_interpretations) == sum(target_mask)
    ), (f'Expected target mask size {sum(target_mask)} to be the same as '
        f'target indices count {len(target_indices)}, and target '
        f'target interpretations count {len(target_interpretations)} '
        'but not quite: '
        f'{target_mask, target_indices, target_interpretations}')
    assert len(set(target_interpretations)) <= 1, (
        'Expected all target interpretations (if any) to be the same, but '
        f'instead got {target_interpretations}.')
    assert len(set(whole_tokens)) <= 1, (
        'Expected all whole tokens (if any) to be the same, but '
        f'instead got {whole_tokens}.')
    beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                 f'input.example_{split}.count').inc()
    if original_is_bug_free:
      beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                   f'input.is_bug_free_{split}.count').inc()
    else:
      beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                   f'input.is_buggy_{split}.count').inc()

    # Now that we have recovered a target string, we go back to graphinating the
    # input. First prune to size.
    pruned_tokens = function_tokens[:max_graph_size - 2]
    error_location_mask = original_error_location_mask[:max_graph_size - 2]
    is_bug_free = error_location_mask[0] == 1
    assert is_bug_free == original_is_bug_free
    candidate_mask = data['candidate_mask'][:max_graph_size - 2]
    pruned_target_mask = target_mask[:max_graph_size - 2]
    provenance = data['provenance']

    # Is this example impossible?
    if sum(error_location_mask) == 0:
      # There's an error location, but we pruned it.
      beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                   f'pruned_error_location_{split}.count').inc()
      return None
    if sum(pruned_target_mask) == 0 and not is_bug_free:
      # There's an error but the targets have been pruned.
      beam.metrics.Metrics.counter(
          _BEAM_METRIC_NAMESPACE,
          f'pruned_all_repair_targets_{split}.count').inc()
      return None

    graph_to_output_example = GraphToOutputExample()

    # The input graph nodes are the source code tokens. We don't filter any
    # examples based on size. Instead, we trim the suffix of the token sequence.
    # Note that we trim so that the number of tokens plus the two delimiters
    # is at most `max_graph_size`.
    delimited_tokens = tuple(
        itertools.chain(('[CLS]',), pruned_tokens, ('[SEP]',)))

    number_of_delimited_tokens = len(delimited_tokens)
    for index, token in enumerate(delimited_tokens):
      # index is in [0, number_of_delimited_tokens). We want it to index into
      # the candidate mask, which maps to [1, number_of_delimited_tokens - 1),
      # since the delimiters are at the two ends, after any trimming. We
      # definitely don't want to mark 0 and `number_of_delimited_tokens` - 1 as
      # repair candidates. For the indices in between, we want to look them up
      # into `candidate_mask` but offset by 1 to the left.
      if index == 0 or index == number_of_delimited_tokens - 1:
        is_repair_candidate = False
      else:
        original_index = index - 1
        is_repair_candidate = bool(candidate_mask[original_index])
      graph_to_output_example.add_node(
          node_id=index,
          node_type='TOKEN',
          node_label=token,
          is_repair_candidate=is_repair_candidate)

    # The first output is a pointer pointing to node 1 (for bug-free examples)
    # or some other element otherwise.
    error_location_index = error_location_mask.index(1) + 1
    if self.use_pointer_output:
      # We encode pointers as a Po entry (from ToCoPo).
      graph_to_output_example.add_pointer_output(error_location_index)
    else:
      # As an ablation, instead of using a Po entry for error localization, we
      # use a special token with the location.
      if is_bug_free:
        graph_to_output_example.add_token_output('BUG_FREE')
      else:
        graph_to_output_example.add_token_output(
            f'ERROR_LOCATION_{error_location_index}')

    # Now we output the repair, as a sequence of subtokens.
    if target_interpretations:
      confounder = '' if self.allow_output_copy else self._CONFOUNDING_STRING
      for repair_subtoken in target_interpretations[0]:
        graph_to_output_example.add_token_output(confounder + repair_subtoken)
      graph_to_output_example.add_additional_field('repair_text',
                                                   whole_tokens[0])

    graph_to_output_example.set_provenance(provenance)

    if original_is_bug_free:
      beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                   f'output.is_bug_free_{split}.count').inc()
    else:
      beam.metrics.Metrics.counter(_BEAM_METRIC_NAMESPACE,
                                   f'output.is_buggy_{split}.count').inc()
    return graph_to_output_example
