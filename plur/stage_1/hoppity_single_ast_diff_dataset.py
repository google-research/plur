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
"""Classes for converting the Hoppity single AST diff dataset to a PLUR dataset."""
import glob
import json
import os
import tarfile

from absl import logging
import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils import util
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import tqdm


class HoppitySingleAstDiffDataset(PlurDataset):
  """Converting data from Hoppity single AST diff dataset to a PlurDataset.

  The dataset is used in: Dinella, Elizabeth, et al. 'Hoppity: Learning Graph
  Transformations to Detect and Fix Bugs in Programs.' International Conference
  on Learning Representations. 2019.

  The task is to predict a transformation to fix the bug, given the buggy AST.
  The transformation can be adding a new node, replacing a node's type,
  replacing a node's label, deleting a node or no transformation.

  The data is already represented as a graph, therefore we use the graph as
  it is. For the output, we output the transformation as a sequence of tokens
  and pointers. For:
  * add_node: The output is 'add_node {parent node pointer}
    {sibling node pointer} {node type token} {node label token}'
  * replace_type: The output is 'replace_type {node pointer} {node type token}'
  * replace_val: The output is 'replace_val {node pointer} {label token}'
  * del_node: The output is 'del_node {node pointer}'
  * 'NoOp': The output is 'NoOp'.
  """


  _URLS = {
      'cooked-one-diff.gz': {
          'url':
              'https://drive.google.com/uc?id=1kEJBCH1weMioTcnmG6fmqz6VP-9KjH7x&export=download',
          'sha1sum':
              '25e4f10db00dcf3a8cd63fcd41130baf66fb5c7d',
      },
      'hoppity_cg.tar.gz': {
          'url': 'https://drive.google.com/u/0/uc?id=1JdXaehWO4UocjXqIXzWtUmVpRWWBtqmE&export=download',
          'sha1sum': '9f4a635408f86974a8e9739769d3ed2a52c2b907',
      }
  }

  _GIT_URL = {}
  _DATASET_NAME = 'Hoppity_Single_AST_diff_dataset'
  _DATASET_DESCRIPTION = """\
  This dataset is used to train Hoppity, which a learning based approach to
  detect and fix bugs in Javascript programs. It contains bug fixes where the
  bug are fixed with a single operation, the operation can be add_node,
  del_node, replace_val, replace_type and NoOp.
  """

  def __init__(self,
               stage_1_dir,
               configuration: Configuration = Configuration(),
               transformation_funcs=(),
               filter_funcs=(),
               user_defined_split_range=(),
               num_shards=1000,
               seed=0,
               deduplicate=False):
    self.cooked_one_diff_extracted_dir = None
    self.hoppity_cg_extracted_dir = None
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

  def download_dataset(self):
    """Download and extract the tar.gz files."""
    super().download_dataset_using_requests()
    self.cooked_one_diff_extracted_dir = os.path.join(
        self.raw_data_dir, 'cooked-full-fmt-shift_node')
    self.hoppity_cg_extracted_dir = os.path.join(self.raw_data_dir,
                                                 'hoppity_cg')
    tarfiles_to_extract = []
    tarfiles_to_extract = util.check_need_to_extract(
        tarfiles_to_extract, self.cooked_one_diff_extracted_dir,
        'cooked-one-diff.gz')

    tarfiles_to_extract = util.check_need_to_extract(
        tarfiles_to_extract, self.hoppity_cg_extracted_dir, 'hoppity_cg.tar.gz')

    for filename in tarfiles_to_extract:
      dest = os.path.join(self.raw_data_dir, filename)
      with tarfile.open(dest, 'r:gz') as tf:
        for member in tqdm.tqdm(
            tf.getmembers(), unit='file',
            desc='Extracting {}'.format(filename)):
          tf.extract(member, self.raw_data_dir)

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    return glob.glob(
        os.path.join(self.hoppity_cg_extracted_dir, 'part-*', '*_buggy.json'))

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    with open(os.path.join(self.cooked_one_diff_extracted_dir,
                           'train.txt')) as f:
      train_data_filenames = set(f.read().splitlines())
    with open(os.path.join(self.cooked_one_diff_extracted_dir, 'val.txt')) as f:
      validation_data_filenames = set(f.read().splitlines())
    with open(os.path.join(self.cooked_one_diff_extracted_dir,
                           'test.txt')) as f:
      testing_data_filenames = set(f.read().splitlines())
    return JsonExtractor(super().get_random_split,
                         bool(self.user_defined_split_range),
                         train_data_filenames, validation_data_filenames,
                         testing_data_filenames,
                         self.cooked_one_diff_extracted_dir)

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Convert raw data to the unified GraphToOutputExample data structure.

    The hoppity single AST diff task is already represented as a graph.
    We used their input graph structure. For the output, we have 1 to 1 mapping
    to our output format. We try to mirror the implementation details from
    the hoppity data processing stage.

    Args:
      raw_data: A hoppity single AST diff task from the dataset. It is a
        dictionary with keys 'split', 'buggy_graph' and 'edit_operation'. Values
        are the split(train/valid/test) the data belongs, the buggy_graph read
        from the json file and 'edit_operation' read from the text file.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    buggy_graph = raw_data['buggy_graph']
    edit_operation = raw_data['edit_operation']
    graph_to_output_example = GraphToOutputExample()

    # Each node is [id, type, value]
    for node in buggy_graph['nodes']:
      graph_to_output_example.add_node(node[0], node[1], node[2])

    # Each edge is [src node id, dst node id, edge type]
    for edge in buggy_graph['edges']:
      # There are duplicate edges in the raw dataset, therefore we ignore them
      # here.
      try:
        graph_to_output_example.add_edge(edge[0], edge[1], str(edge[2]))
      except ValueError:
        logging.warning('Duplicate edge in %s, ignoring it for now.',
                        self.dataset_name)

    # The Hoppity specific separator, used to concat node types that have more
    # than 1 token.
    separator = '!#@$'
    edit_operation_tokens = edit_operation.split(separator)
    # Add the transformation keyword as the first output as a token.
    graph_to_output_example.add_token_output(edit_operation_tokens[0])

    if edit_operation_tokens[0] == constants.HOPPITY_ADD_NODE_OP_NAME:
      # The parent node id, use a pointer to point to it.
      graph_to_output_example.add_pointer_output(int(edit_operation_tokens[1]))
      # The sibling node id, use a pointer to point to it.
      graph_to_output_example.add_pointer_output(int(edit_operation_tokens[2]))
      # If the edit operation is 6 tokens long, then there are two tokens
      # representing the node type. This information is taken from

      # https://github.com/AI-nstein/hoppity/blob/master/gtrans/common/dataset.py#L34

      if len(edit_operation_tokens) == 6:
        # The type token.
        graph_to_output_example.add_token_output(
            separator.join([edit_operation_tokens[3],
                            edit_operation_tokens[4]]))
        # The value token.
        graph_to_output_example.add_token_output(edit_operation_tokens[5])
      else:
        # The type token.
        graph_to_output_example.add_token_output(edit_operation_tokens[3])
        # The value token
        graph_to_output_example.add_token_output(edit_operation_tokens[4])
    elif edit_operation_tokens[0] == constants.HOPPITY_DEL_NODE_OP_NAME:
      # The deleted node id, use a pointer to point to it.
      graph_to_output_example.add_pointer_output(int(edit_operation_tokens[1]))
    elif edit_operation_tokens[0] == constants.HOPPITY_REPLACE_VAL_OP_NAME:
      # The id of the node to replace its node value, use a pointer to point to
      # it.
      graph_to_output_example.add_pointer_output(int(edit_operation_tokens[1]))
      # The value token
      graph_to_output_example.add_token_output(edit_operation_tokens[2])
    elif edit_operation_tokens[0] == constants.HOPPITY_REPLACE_TYPE_OP_NAME:
      # The id of the token to replace its node type, use a pointer to point to
      # it.
      graph_to_output_example.add_pointer_output(int(edit_operation_tokens[1]))
      # Again, the node type can be more than 1 token long, we concatenate the
      # tokens it that is the case.
      node_type = edit_operation_tokens[2:]
      if isinstance(node_type, list):
        node_type = separator.join(node_type)
      graph_to_output_example.add_token_output(node_type)
    elif edit_operation_tokens[0] == constants.HOPPITY_REPLACE_NOOP_OP_NAME:
      # For NoOp, only the 'NoOp' token is enough
      pass
    else:
      # Unknown operation, should never happen, this indicates a error in the
      # dataset.
      raise ValueError('Unexpected edit operation: {}'.format(
          edit_operation_tokens[0]))

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


class JsonExtractor(beam.DoFn):
  """Class to read the Hoppity dataset files."""

  def __init__(self, random_split_fn, use_random_split, train_data_filenames,
               validation_data_filenames, testing_data_filenames,
               cooked_one_diff_extracted_dir):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split
    self.train_data_filenames = train_data_filenames
    self.validation_data_filenames = validation_data_filenames
    self.testing_data_filenames = testing_data_filenames
    self.cooked_one_diff_extracted_dir = cooked_one_diff_extracted_dir

  def _read_data(self, file_path):
    """Read the buggy graph and edit operation JSON files."""
    filename_prefix = os.path.basename(file_path)[:-11]
    edit_operation_file_path = os.path.join(self.cooked_one_diff_extracted_dir,
                                            (filename_prefix + '_gedit.txt'))
    with open(file_path) as f:
      buggy_graph = json.load(f)
    with open(edit_operation_file_path) as f:
      edit_operation = json.load(f)[0]['edit']
    return buggy_graph, edit_operation

  def _get_split(self, file_path):
    """Get the Hoppity predefined split with the filename prefix."""
    filename_prefix = os.path.basename(file_path)[:-11]
    if filename_prefix in self.train_data_filenames:
      return constants.TRAIN_SPLIT_NAME
    elif filename_prefix in self.validation_data_filenames:
      return constants.VALIDATION_SPLIT_NAME
    elif filename_prefix in self.testing_data_filenames:
      return constants.TEST_SPLIT_NAME
    else:
      return None

  def process(self, file_path):
    """Function to read each json file.

    Args:
      file_path: Path to a raw data file.

    Yields:
      A dictionary with 'split', 'buggy_graph' and 'edit_operation' as keys.
      The value of the 'split' field is the split (train/valid/test) that the
      data belongs to. The value of the 'buggy_graph' is the parsed buggy graph.
      The value of the 'edit_operation' is the edit operation string.
    """
    split = self._get_split(file_path)
    if split is None:
      return
    buggy_graph, edit_operation = self._read_data(file_path)
    yield {
        'split': self.random_split_fn() if self.use_random_split else split,
        'buggy_graph': buggy_graph,
        'edit_operation': edit_operation
    }
