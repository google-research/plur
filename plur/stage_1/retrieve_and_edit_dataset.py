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

"""Classes for converting the retrieve and edit dataset to a PLUR dataset.
"""
import os
from typing import Callable, Iterator, Mapping, Sequence, Tuple, Union

import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import typing_extensions


class DataDict(typing_extensions.TypedDict):
  split: str
  function_name: str
  block_comment: str
  arguments: str
  function_tokens: str


class RetrieveAndEditDataset(PlurDataset):
  """Converting data from retrieve and edit dataset to a PLUR dataset.

  The dataset is used in: Hashimoto, Tatsunori B., et al. "A retrieve-and-edit
  framework for predicting structured outputs." arXiv preprint arXiv:1812.01194
  (2018).

  The task is to predict the next token given the block comment, function name
  and arguments, predict the functions tokens. The performance of a model is
  defined in terms of the average or maximum number of successive tokens
  correctly predicted.

  The provided dataset by the authors are the tokenized block comment, function
  name, tokenized arguments and tokenized function tokens. We build three
  different trees for the block comment, function name and the arguments. They
  are connected with a dataset root node. The output will be the entire
  function tokens.
  """


  _URLS = {
      'train.tsv': {
          'url': 'https://worksheets.codalab.org/rest/bundles/0xfa69890526c04899a1eb286afb17d37a/contents/blob/github/train.tsv',
          'sha1sum': 'c9e2ddd492e048f136c2d8e14d663593e1dd182c',
      },
      'valid.tsv': {
          'url': 'https://worksheets.codalab.org/rest/bundles/0xfa69890526c04899a1eb286afb17d37a/contents/blob/github/valid.tsv',
          'sha1sum': '6947e2b0c960844a76b0ce670c4e17a88c4efba4',
      },
      'test.tsv': {
          'url': 'https://worksheets.codalab.org/rest/bundles/0xfa69890526c04899a1eb286afb17d37a/contents/blob/github/test.tsv',
          'sha1sum': '1f44bae4a03bf7f1b3cf882943af404368939f46',
      }
  }

  _GIT_URL = {}
  _DATASET_NAME = 'retrieve_and_edit_dataset'
  _DATASET_DESCRIPTION = """\
  The dataset is from 'A Retrieve-and-Edit Framework for Predicting Structured
  Outputs'. Here is the description of this dataset from the paper:

  Autocomplete on Python GitHub code:
  Given a natural language description of a Python function and a partially
  written code fragment, the task is to return a candidate list of k = 1, 5, 10
  next tokens (Figure 2). A model predicts correctly if the ground truth token
  is in the candidate list. The performance of a model is defined in terms of
  the average or maximum number of successive tokens correctly predicted.

  Our Python autocomplete dataset is a representative sample of Python code from
  GitHub, obtained from Google Bigquery by retrieving Python code containing at
  least one block comment with restructured text (reST) formatting (See Appendix
  C for details). We use this data to form a code prediction task where each
  example consists of four inputs: the block comment, function name, arguments,
  and a partially written function body. The output is the next token in the
  function body.
  """

  def __init__(
      self,
      stage_1_dir: str,
      configuration: Configuration = Configuration(),
      transformation_funcs: Sequence[Callable[..., GraphToOutputExample]] = (),
      filter_funcs: Sequence[Callable[..., bool]] = (),
      user_defined_split_range: Union[Tuple[()], Tuple[int, int, int]]=(),
      num_shards: int = 1000,
      seed: int = 0,
      deduplicate: bool = False):
    super().__init__(
        self._DATASET_NAME,
        self._URLS,
        self._GIT_URL,
        self._DATASET_DESCRIPTION,
        stage_1_dir,
        transformation_funcs=transformation_funcs,
        filter_funcs=filter_funcs,
        user_defined_split_range=user_defined_split_range,
        num_shards=num_shards,
        seed=seed,
        configuration=configuration,
        deduplicate=deduplicate)

  def download_dataset(self) -> None:
    """Download the dataset using requests."""
    super().download_dataset_using_requests()

  def get_all_raw_data_paths(self) -> Sequence[str]:
    """Get paths to all raw data."""
    train_file = os.path.join(self.raw_data_dir, 'train.tsv')
    validation_file = os.path.join(self.raw_data_dir, 'valid.tsv')
    test_file = os.path.join(self.raw_data_dir, 'test.tsv')
    return [train_file, validation_file, test_file]

  def raw_data_paths_to_raw_data_do_fn(self) -> 'TSVExtractor':
    """Returns a beam.DoFn subclass that reads the raw data."""
    return TSVExtractor(super().get_random_split,
                        bool(self.user_defined_split_range))

  def raw_data_to_graph_to_output_example(
      self, raw_data: DataDict
  ) -> Mapping[str, Union[str, GraphToOutputExample, None]]:
    """Convert raw data to the unified GraphToOutputExample data structure.

    We create a graph using the function name, block comment and arguments.
    For all of them we will simply split them by whitespace and create a chain
    of nodes, and connecting each consective nodes with next token edge. Each
    chain will be connect to each function_name/block_comment/argument root node
    and these root nodes will be connected to a single root node representing
    this etrieve and edit example.

    Args:
      raw_data: A DataDict with 'split', 'function_name', 'block_comment',
        'arguments' and 'function_tokens' as keys. The 'split' field is the
        split it belongs (train/valid/test). The 'function_name' field is a
        single token repsenting the function name. The 'block_comment' field is
        a whitespace separated comment tokens string. The 'arguments' field is a
        whitespace separated argument tokens string. The 'function_tokens' field
        is a whitespace separated function tokens string.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    function_name = raw_data['function_name']
    block_comment = raw_data['block_comment']
    arguments = raw_data['arguments']
    function_tokens = raw_data['function_tokens']

    cur_node_id = 0

    graph_to_output_example = GraphToOutputExample()
    graph_to_output_example.add_node(cur_node_id, 'RETRIEVE_AND_EDIT_ROOT',
                                     'RETRIEVE_AND_EDIT_ROOT')
    retrieve_and_edit_root_node = cur_node_id
    cur_node_id += 1

    cur_node_id, function_name_root_node_id = create_tree_from_string(
        graph_to_output_example, cur_node_id, function_name,
        'NEXT_FUNCTION_NAME_TOKEN', 'FUNCTION_NAME_TOKEN', 'FUNCTION_NAME')
    cur_node_id, block_comment_root_node_id = create_tree_from_string(
        graph_to_output_example, cur_node_id, block_comment,
        'NEXT_BLOCK_COMMENT_TOKEN', 'BLOCK_COMMENT_TOKEN', 'BLOCK_COMMENTS')
    cur_node_id, arguments_root_node_id = create_tree_from_string(
        graph_to_output_example, cur_node_id, arguments,
        'NEXT_ARGUMENT_TOKEN', 'ARGUMENT_TOKEN', 'ARGUMENTS')

    graph_to_output_example.add_edge(
        retrieve_and_edit_root_node, function_name_root_node_id,
        'FUNCTION_NAME')
    graph_to_output_example.add_edge(
        retrieve_and_edit_root_node, block_comment_root_node_id,
        'BLOCK_COMMENTS')
    graph_to_output_example.add_edge(
        retrieve_and_edit_root_node, arguments_root_node_id,
        'ARGUMENTS')

    for function_token in function_tokens.split(' '):
      graph_to_output_example.add_token_output(function_token)

    for transformation_fn in self.transformation_funcs:
      graph_to_output_example = transformation_fn(graph_to_output_example)

    if not graph_to_output_example.check_if_valid():
      raise GraphToOutputExampleNotValidError(
          'Invalid GraphToOutputExample found {}'.format(
              graph_to_output_example))

    for filter_fn in self.filter_funcs:
      if not filter_fn(graph_to_output_example):
        graph_to_output_example = None
        break

    return {'split': split, 'GraphToOutputExample': graph_to_output_example}


class TSVExtractor(beam.DoFn):
  """Class to read the retrieve and edit dataset."""

  def __init__(self, random_split_fn: Callable[[], str],
               use_random_split: bool):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def _read_data(self, file_path: str) -> Iterator[Sequence[str]]:
    r"""Read and parse the retrieve and edit raw data file.

    Each line in the retrieve and edit raw data file has the following format:
    '<function name>\t<block coment>\t<arguments>\t<function tokens>'

    Args:
      file_path: Path to a retrieve and edit data file.

    Yields:
      A tuple of function_name, block_comment, arguments and function_tokens.
      Each string is still whitespace separated.
    """
    with open(file_path) as f:
      for line in f:
        fields = line.strip().split('\t')
        # Some data are malformed, therefore we check that we have four fields.
        if len(fields) == 4:
          yield fields

  def _get_split(self, file_path: str) -> str:
    """Get the data split based on the filename suffix."""
    if file_path.endswith('train.tsv'):
      return constants.TRAIN_SPLIT_NAME
    elif file_path.endswith('valid.tsv'):
      return constants.VALIDATION_SPLIT_NAME
    else:
      return constants.TEST_SPLIT_NAME

  def process(self, file_path: str) -> Iterator[DataDict]:
    split = self._get_split(file_path)
    for fields in self._read_data(file_path):
      yield DataDict(
          split=self.random_split_fn() if self.use_random_split else split,
          function_name=fields[0],
          block_comment=fields[1],
          arguments=fields[2],
          function_tokens=fields[3],
      )


def create_tree_from_string(graph_to_output_example: GraphToOutputExample,
                            cur_node_id: int, string: str,
                            next_token_edge_type: str, token_node_type: str,
                            root_node_name: str) -> Tuple[int, int]:
  """Create a tree from a whitespace separated string.

  The input 'string' will be split by whitespace to create a token list. Each
  token has the node type 'token_node_type' and the node label is the token
  it self. Each token is connected to the next token via the
  'next_token_edge_type' edge. A root node with node type 'root_node_name'
  and node label 'root_node_name' will be connected to all tokens with
  'root_token_edge_type' edges.

  Args:
    graph_to_output_example: A GraphToOutputExample instance.
    cur_node_id: Next available node id.
    string: A whitespace separated string.
    next_token_edge_type: The edge type connecting two consective tokens.
    token_node_type: The node type for the token, will also be used as the
      edge type connecting the root node to the token.
    root_node_name: The node type and label for the root node.

  Returns:
    A tuple of cur_node_id and root_node_id. cur_node_id is the next available
    node id and root_node_id is the root node id.
  """
  string_tokens = string.split(' ')
  token_node_ids = []
  prev_token_node_id = -1
  for token in string_tokens:
    graph_to_output_example.add_node(
        cur_node_id, token_node_type, token)
    token_node_ids.append(cur_node_id)
    if prev_token_node_id != -1:
      graph_to_output_example.add_edge(prev_token_node_id, cur_node_id,
                                       next_token_edge_type)

    prev_token_node_id = cur_node_id
    cur_node_id += 1

  graph_to_output_example.add_node(
      cur_node_id, root_node_name, root_node_name)
  root_node_id = cur_node_id
  cur_node_id += 1

  for token_node_id in token_node_ids:
    graph_to_output_example.add_edge(root_node_id, token_node_id,
                                     token_node_type)

  return cur_node_id, root_node_id



