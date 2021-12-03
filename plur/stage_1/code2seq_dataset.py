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

"""Classes for converting the Code2Seq dataset to a PLUR dataset.
"""
import os
import tarfile

import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils import util
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import tqdm


class Code2SeqDataset(PlurDataset):
  # pylint: disable=line-too-long
  """Converting data from code2seq dataset to a PLUR dataset.

  The dataset is used in: Alon, Uri, et al. 'code2seq: Generating sequences from
  structured representations of code.' arXiv preprint arXiv:1808.01400 (2018).

  The task is to predict the function name given the function body.

  The provided dataset by code2seq are the tokenized function name, and the AST
  paths. Therefore we have to create our own graph representation of code2seq.
  We try to mimic the code2seq model by constructing a graph similar to figure
  3 in the code2seq paper. An example of such graph is shown in
  https://drive.google.com/file/d/1-cH0FzYIMikgTkUpzVkEZDGjoiqBB9C1/view?usp=sharing.
  In short, we build the AST path subtree and connects all AST paths with a
  code2seq root node to make it a graph.
  """

  _URLS_SMALL = {
      'java-small-preprocessed.tar.gz': {
          'url': 'https://s3.amazonaws.com/code2seq/datasets/java-small-preprocessed.tar.gz',
          'sha1sum': '857c2495785f606ab99676c7bbae601ea2160f66',
      }
  }
  _URLS_MED = {
      'java-med-preprocessed.tar.gz': {
          'url': 'https://s3.amazonaws.com/code2seq/datasets/java-med-preprocessed.tar.gz',
          'sha1sum': '219e558ddf46678ef322ff75bf1982faa1b6204d',
      }
  }
  _URLS_LARGE = {
      'java-large-preprocessed.tar.gz': {
          'url': 'https://s3.amazonaws.com/code2seq/datasets/java-large-preprocessed.tar.gz',
          'sha1sum': 'ebc229ba1838a3c8f3a69ab507eb26fa5460152a',
      }
  }
  # pylint: enable=line-too-long
  _GIT_URL = {}
  _DATASET_NAME = 'code2seq_dataset'
  _DATASET_DESCRIPTION = """\
  This dataset is used to train the code2seq model. The task is to predict the
  function name, given the ast paths sampled the function AST. An AST path is
  a path between two leaf nodes in the AST.
  """

  def __init__(self,
               stage_1_dir,
               configuration: Configuration = Configuration(),
               transformation_funcs=(),
               filter_funcs=(),
               user_defined_split_range=(),
               num_shards=1000,
               seed=0,
               dataset_size='small',
               deduplicate=False):
    # dataset_size can only be 'small', 'med' or 'large'.
    valid_dataset_size = {'small', 'med', 'large'}
    if dataset_size not in valid_dataset_size:
      raise ValueError('{} not in {}'.format(dataset_size,
                                             str(valid_dataset_size)))
    if dataset_size == 'small':
      urls = self._URLS_SMALL
    elif dataset_size == 'med':
      urls = self._URLS_MED
    else:
      urls = self._URLS_LARGE
    self.dataset_size = dataset_size
    super().__init__(self._DATASET_NAME, urls, self._GIT_URL,
                     self._DATASET_DESCRIPTION, stage_1_dir,
                     transformation_funcs=transformation_funcs,
                     filter_funcs=filter_funcs,
                     user_defined_split_range=user_defined_split_range,
                     num_shards=num_shards, seed=seed,
                     configuration=configuration, deduplicate=deduplicate)

  def download_dataset(self):
    """Download the dataset using requests and extract the tarfile."""
    super().download_dataset_using_requests()
    # Extract the tarfile depending on the dataset size.
    if self.dataset_size == 'small':
      self.code2seq_extracted_dir = os.path.join(
          self.raw_data_dir, 'java-small')
      tarfile_name = 'java-small-preprocessed.tar.gz'
    elif self.dataset_size == 'med':
      self.code2seq_extracted_dir = os.path.join(
          self.raw_data_dir, 'java-med')
      tarfile_name = 'java-med-preprocessed.tar.gz'
    else:
      self.code2seq_extracted_dir = os.path.join(
          self.raw_data_dir, 'java-large')
      tarfile_name = 'java-large-preprocessed.tar.gz'

    tarfiles_to_extract = []
    tarfiles_to_extract = util.check_need_to_extract(
        tarfiles_to_extract, self.code2seq_extracted_dir,
        tarfile_name)
    for filename in tarfiles_to_extract:
      dest = os.path.join(self.raw_data_dir, filename)
      with tarfile.open(dest, 'r:gz') as tf:
        for member in tqdm.tqdm(
            tf.getmembers(),
            unit='file',
            desc='Extracting {}'.format(filename)):
          tf.extract(member, self.raw_data_dir)

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    # Get the filenames depending on the dataset size.
    if self.dataset_size == 'small':
      train_file = os.path.join(
          self.code2seq_extracted_dir, 'java-small.train.c2s')
      validation_file = os.path.join(
          self.code2seq_extracted_dir, 'java-small.val.c2s')
      test_file = os.path.join(
          self.code2seq_extracted_dir, 'java-small.test.c2s')
    elif self.dataset_size == 'med':
      train_file = os.path.join(
          self.code2seq_extracted_dir, 'java-med.train.c2s')
      validation_file = os.path.join(
          self.code2seq_extracted_dir, 'java-med.val.c2s')
      test_file = os.path.join(
          self.code2seq_extracted_dir, 'java-med.test.c2s')
    else:
      train_file = os.path.join(
          self.code2seq_extracted_dir, 'java-large.train.c2s')
      validation_file = os.path.join(
          self.code2seq_extracted_dir, 'java-large.val.c2s')
      test_file = os.path.join(
          self.code2seq_extracted_dir, 'java-large.test.c2s')
    return [train_file, validation_file, test_file]

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return C2SExtractor(super().get_random_split,
                        bool(self.user_defined_split_range))

  def _construct_token_subtree(self, graph_to_output_example, token,
                               cur_node_id, token_root_name):
    # pylint: disable=line-too-long
    """Construct the token subtree in a AST path.

    We create a node for each subtoken in the token, all subtokens are connected
    to the next subtoken via the 'NEXT_SUBTOKEN' edge. All subtokens are
    connected to the token root node via the 'SUBTOKEN' edge. See the draw.io
    figure mentioned in the class doc for the visualization.

    Args:
      graph_to_output_example: A GraphToOutputExample instance.
      token: Starting or ending token in the AST path.
      cur_node_id: Next available node id.
      token_root_name: Node type and label for the token root node.

    Returns:
      A tuple of graph_to_output_example, cur_node_id, token_node_id.
      graph_to_output_example is updated with the token subtree, cur_node_id is
      the next available node id after all the token subtree nodes are added,
      and token_node_id is the node id of the root token node.
    """
    subtokens = token.split('|')
    subtoken_node_ids = []
    prev_subtoken_id = -1
    # Create a node each subtoken.
    for subtoken in subtokens:
      graph_to_output_example.add_node(cur_node_id, 'SUBTOKEN', subtoken)
      subtoken_node_ids.append(cur_node_id)
      # Connects to the previous subtoken node
      if prev_subtoken_id != -1:
        graph_to_output_example.add_edge(prev_subtoken_id, cur_node_id,
                                         'NEXT_SUBTOKEN')
      prev_subtoken_id = cur_node_id
      cur_node_id += 1

    # Add a root node for the token subtree.
    graph_to_output_example.add_node(cur_node_id, token_root_name,
                                     token_root_name)
    token_node_id = cur_node_id
    cur_node_id += 1
    # Connect all subtoken nodes to the token subtree root node.
    for node_id in subtoken_node_ids:
      graph_to_output_example.add_edge(token_node_id, node_id, 'SUBTOKEN')

    return graph_to_output_example, cur_node_id, token_node_id

  def _construct_ast_nodes_subtree(self, graph_to_output_example, ast_nodes,
                                   cur_node_id):
    """Construct the AST nodes subtree in a AST path.

    We create a node for each AST node in the AST path. Each AST node are
    connected to the next AST node via the 'NEXT_AST_NODE' edge. See the draw.io
    figure mentioned in the class doc for the visualization.

    Args:
      graph_to_output_example: A GraphToOutputExample instance.
      ast_nodes: AST nodes in the AST path.
      cur_node_id: Current available node id.
    Returns:
      A tuple of graph_to_output_example, cur_node_id, ast_node_ids.
      graph_to_output_example is updated with the ast nodes subtree,
      cur_node_id is the next available node id after all the ast nodes are
      added, and ast_node_ids the node ids of all AST nodes.
    """
    ast_nodes = ast_nodes.split('|')
    ast_node_ids = []
    prev_ast_node_id = -1
    # Create a node each AST node.
    for ast_node in ast_nodes:
      graph_to_output_example.add_node(cur_node_id, 'AST_NODE', ast_node)
      ast_node_ids.append(cur_node_id)
      # Connects to the previous AST node.
      if prev_ast_node_id != -1:
        graph_to_output_example.add_edge(prev_ast_node_id, cur_node_id,
                                         'NEXT_AST_NODE')
      prev_ast_node_id = cur_node_id
      cur_node_id += 1
    return graph_to_output_example, cur_node_id, ast_node_ids

  def raw_data_to_graph_to_output_example(self, raw_data):
    # pylint: disable=line-too-long
    """Convert raw data to the unified GraphToOutputExample data structure.

    The Code2Seq raw data contains the target function name, and the sampled
    AST paths. Each AST path starts and ends with a token, and a series of
    AST nodes that connects the two tokens. We use _construct_token_subtree
    to build the token subtree and _construct_ast_nodes_subtree to build the
    AST nodes subtree. Then, all AST paths' nodes are connected to a AST root
    node.
    All AST root nodes are connected to a single code2seq root node.
    https://drive.google.com/file/d/1-cH0FzYIMikgTkUpzVkEZDGjoiqBB9C1/view?usp=sharing
    shows an example of such a graph and the original AST path.

    Args:
      raw_data: A dictionary with 'split', 'target_label' and 'ast_paths' as keys.
        The value of the 'split' field is the split (train/valid/test) that the
        data belongs to. The value of the 'target_label' field is the function
        name. The value of the 'ast_paths' field is a list of AST paths.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs to, and the
      GraphToOutputExample instance.
    """
    # pylint: enable=line-too-long
    split = raw_data['split']
    target_label = raw_data['target_label']
    ast_paths = raw_data['ast_paths']
    graph_to_output_example = GraphToOutputExample()

    cur_node_id = 0
    ast_path_root_node_ids = []

    # This is the root node of all AST path nodes.
    graph_to_output_example.add_node(cur_node_id, 'C2C_ROOT', 'C2C_ROOT')
    c2c_root_node_id = cur_node_id
    cur_node_id += 1

    for ast_path in ast_paths:
      # The start_token subtree
      start_token = ast_path[0]
      graph_to_output_example, cur_node_id, start_token_node_id = (
          self._construct_token_subtree(
              graph_to_output_example, start_token, cur_node_id, 'START_TOKEN'))

      # The ast_nodes subtree
      ast_nodes = ast_path[1]
      graph_to_output_example, cur_node_id, ast_node_ids = (
          self._construct_ast_nodes_subtree(
              graph_to_output_example, ast_nodes, cur_node_id))

      # The end_token subtree
      end_token = ast_path[2]
      graph_to_output_example, cur_node_id, end_token_node_id = (
          self._construct_token_subtree(
              graph_to_output_example, end_token, cur_node_id, 'END_TOKEN'))

      # Connects the start_token root node with the first node in the
      # ast_nodes subtree.
      graph_to_output_example.add_edge(
          start_token_node_id, ast_node_ids[0], 'START_AST_PATH')
      # Connects the end_token root node with the last node in the
      # ast_nodes subtree.
      graph_to_output_example.add_edge(
          end_token_node_id, ast_node_ids[-1], 'END_AST_PATH')

      # Add a root AST path node representing the AST path.
      graph_to_output_example.add_node(
          cur_node_id, 'ROOT_AST_PATH', 'ROOT_AST_PATH')
      ast_path_root_node_id = cur_node_id
      ast_path_root_node_ids.append(ast_path_root_node_id)
      cur_node_id += 1

      # Connects the root AST path node with the start_token and end_token
      # subtree.
      graph_to_output_example.add_edge(
          ast_path_root_node_id, start_token_node_id, 'START_TOKEN')
      graph_to_output_example.add_edge(
          ast_path_root_node_id, end_token_node_id, 'END_TOKEN')
      # Connects the root AST path node with all nodes in the ast_nodes subtree.
      for node_id in ast_node_ids:
        graph_to_output_example.add_edge(ast_path_root_node_id, node_id,
                                         'AST_NODE')

    # Connects the code2seq root node with all AST path root node.
    for ast_path_root_node_id in ast_path_root_node_ids:
      graph_to_output_example.add_edge(c2c_root_node_id, ast_path_root_node_id,
                                       'AST_PATH')

    for subtoken in target_label.split('|'):
      graph_to_output_example.add_token_output(subtoken)

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


class C2SExtractor(beam.DoFn):
  """Class to read the code2seq dataset."""

  def __init__(self, random_split_fn, use_random_split):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def _read_data(self, file_path):
    """Read and parse the code2seq raw data file.

    Each line in the code2seq raw data file has the following format:
    '<token> <token>,<node1>,<node2>,<token> <token>,<node3>,<token>'
    The first token is the function name. The rest are the AST paths, separated
    with a whitespace.

    Args:
      file_path: Path to a code2seq data file.

    Yields:
      A tuple of the function name, and a list of AST paths.
    """
    with open(file_path) as f:
      for line in f:
        fields = line.rstrip().split(' ')
        # The subtokens are still separated by '|', we handle them
        # together in self.raw_data_to_graph_to_output_example()
        target_label = fields[0]
        ast_paths = []
        for field in fields[1:]:
          if field:
            # The subtokens are still separated by '|', we handle them
            # together in self.raw_data_to_graph_to_output_example()
            ast_paths.append(field.split(','))
        yield target_label, ast_paths

  def _get_split(self, file_path):
    """Get the data split based on the filename suffix."""
    if file_path.endswith('train.c2s'):
      return constants.TRAIN_SPLIT_NAME
    elif file_path.endswith('val.c2s'):
      return constants.VALIDATION_SPLIT_NAME
    else:
      return constants.TEST_SPLIT_NAME

  def process(self, file_path):
    split = self._get_split(file_path)
    for target_label, ast_paths in self._read_data(file_path):
      yield {
          'split': self.random_split_fn() if self.use_random_split else split,
          'target_label': target_label,
          'ast_paths': ast_paths
      }
