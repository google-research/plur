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

"""Classes for converting the GREAT VarMisuse dataset to a PLUR dataset."""
import glob
import json
import os

from absl import logging
import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError


class GreatVarMisuseDataset(PlurDataset):
  """Converting data from GREAT VarMisuse dataset to a PLUR dataset.

  The dataset is used in: Hellendoorn, Vincent J., et al. 'Global relational
  models of source code.' International Conference on Learning Representations.
  2019.

  The task is to predict the location of a variable misuse, and a location
  of a variable to replace the mis-used variable, given the method body. The
  prediction can also predict that there is no variable misuse by pointing
  to a specific node.

  The data is already represented as a graph, therefore we use the graph as
  it is. For the output, we generate a special 'NO_BUG' token if there is no
  variable misuse. And we generate one pointer and one token if there is a
  variable misuse. The pointer points to a location of a variable being
  mis-used. And the token is one of the input node labels, which should be a
  variable that replaces the mis-used variable.
  """

  _URLS = {}
  _GIT_URL = {
      'url': 'https://github.com/google-research-datasets/great.git',
      'commit_id': 'd53603435aa62e598feef5ac0723ec975852cfcc',
  }
  _DATASET_NAME = 'great_varmisuse_dataset'
  _DATASET_DESCRIPTION = """\
  From GitHub README:

  The dataset for the variable-misuse task, used in the ICLR 2020 paper
  'Global Relational Models of Source Code'
  [https://openreview.net/forum?id=B1lnbRNtwr]
  This dataset was generated synthetically from the corpus of Python code in
  the ETH Py150 Open dataset
  [https://github.com/google-research-datasets/eth_py150_open].
  The dataset is presented in 3 splits: the training dataset train, the
  validation dataset dev, and the evaluation (test) dataset eval. Each of
  these was derived from the corresponding split of ETH Py150 Open.
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
    super().__init__(self._DATASET_NAME, self._URLS, self._GIT_URL,
                     self._DATASET_DESCRIPTION, stage_1_dir,
                     transformation_funcs=transformation_funcs,
                     filter_funcs=filter_funcs,
                     user_defined_split_range=user_defined_split_range,
                     num_shards=num_shards, seed=seed,
                     configuration=configuration, deduplicate=deduplicate)

  def download_dataset(self):
    """Download the dataset using git."""
    super().download_dataset_using_git()

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    # pattern match the raw data files, the pattern is
    # git_repo/*/*VARIABLE_MISUSE__SStuB.txt*
    return glob.glob(
        os.path.join(self.git_repo_dir, '*', '*VARIABLE_MISUSE__SStuB.txt*'))

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return JsonExtractor(super().get_random_split,
                         bool(self.user_defined_split_range))

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Convert raw data to the unified GraphToOutputExample data structure.

    The GREAT VarMisuse task is naturally represented as a graph. We simply
    copy the graph structure. For the output:
    1) If there is a variable misuse: The first output is a pointer pointing to
    the error location. The second output is a token that should replace the
    node label pointed by the first pointer.
    2) If there is no variable misuse: The output is a 'NO_BUG' token.

    Args:
      raw_data: A dictionary with 'split' and 'data' as keys. The value of the
        'split' field is the split (train/valid/test) that the data belongs to.
        The value of the 'data' field is the parsed raw json data.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs to, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    data = raw_data['data']
    graph_to_output_example = GraphToOutputExample()

    repair_candidates_ids_list = data['repair_candidates']
    # The repair candidates should be a list of integers. But it can contain
    # some strings which are just noise, we simply remove them from the list.
    repair_candidates_ids_set = set([
        node_id
        for node_id in repair_candidates_ids_list
        if isinstance(node_id, int)
    ])

    # The input graph nodes are the source code tokens.
    for index, node_token in enumerate(data['source_tokens']):
      graph_to_output_example.add_node(
          index, 'TOKEN', node_token,
          is_repair_candidate=index in repair_candidates_ids_set)

    # We use the edges as it is, since the data is already in graph format.
    for [before_index, after_index, _, edge_type_name] in data['edges']:
      # There are duplicate edges in the raw dataset, therefore we ignore them
      # here.
      try:
        graph_to_output_example.add_edge(
            before_index, after_index, edge_type_name)
      except ValueError:
        logging.warning('Duplicate edge in %s, ignoring it for now.',
                        self.dataset_name)

    # has_bug field tells us if it has a variable misuse.
    if data['has_bug']:
      # This is a flaw in the dataset, where there is a variable misuse but no
      # fixes. We ignore it for now.
      if not data['repair_targets']:
        return {'split': split, 'GraphToOutputExample': None}

      # The error_location points to the mis-used variable.
      graph_to_output_example.add_pointer_output(data['error_location'])
      # repair_targets stores all possible nodes that we can point to to replace
      # the mis-used variable.
      # Since all data['repair_targets'] should point to node with same label.
      # We simply choose the first one. However, at commit id
      # d53603435aa62e598feef5ac0723ec975852cfcc, there is a bug in the dataset
      # that not all repair targets have the same label. This roughly impacts
      # 0.5% of all data.
      index = data['repair_targets'][0]
      graph_to_output_example.add_token_output(data['source_tokens'][index])
    else:
      graph_to_output_example.add_token_output('NO_BUG')

    for transformation_fn in self.transformation_funcs:
      graph_to_output_example = transformation_fn(graph_to_output_example)

    if not graph_to_output_example.check_if_valid():
      raise GraphToOutputExampleNotValidError(
          'Invalid GraphToOutputExample found {}'.format(
              graph_to_output_example))

    graph_to_output_example.add_additional_field(
        'provenances', data['provenances'])

    for filter_fn in self.filter_funcs:
      if not filter_fn(graph_to_output_example):
        graph_to_output_example = None
        break

    return {'split': split, 'GraphToOutputExample': graph_to_output_example}


class JsonExtractor(beam.DoFn):
  """Class to read the GREAT VarMisuse json files."""

  def __init__(self, random_split_fn, use_random_split):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def _get_split(self, file_path):
    """Use parent filename to determine the split.

    Args:
      file_path: Path to a raw data file.

    Returns:
      The split that the raw data belongs to. It is training data if it is
      stored under 'train', it is validation data if it is stored under 'dev',
      and it is test data otherwise.
    """
    file_parent_dirname = os.path.basename(os.path.dirname(file_path))
    if file_parent_dirname == 'train':
      return constants.TRAIN_SPLIT_NAME
    elif file_parent_dirname == 'dev':
      return constants.VALIDATION_SPLIT_NAME
    else:
      return constants.TEST_SPLIT_NAME

  def process(self, file_path):
    """Function to read each json file.

    Args:
      file_path: Path to a raw data file.

    Yields:
      A dictionary with 'split' and 'data' as keys. The value of the 'split'
      field is the split (train/valid/test) that the data belongs to. The value
      of the 'data' is the parsed raw json data.
    """
    split = self._get_split(file_path)
    with open(file_path) as f:
      for line in f:
        json_data = json.loads(line)
        yield {
            'split': self.random_split_fn() if self.use_random_split else split,
            'data': json_data
        }
