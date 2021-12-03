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

"""Classes for converting the OGB code dataset to a PLUR dataset."""
import csv
import gzip
import os
import shutil
import zipfile

from absl import logging
import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils import util
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import tqdm


class OgbCodeDataset(PlurDataset):
  """Converting the data from OGB code dataset to a PLUR dataset.

  The dataset is from: Hu, Weihua, et al. 'Open graph benchmark: Datasets for
  machine learning on graphs.' arXiv preprint arXiv:2005.00687 (2020).

  The task is to predict the method name given the method body.

  The data is already represented as a graph, and we use the graph as it is.
  The output is the method name tokens.
  """

  _URLS = {
      'code.zip': {
          'url': 'https://snap.stanford.edu/ogb/data/graphproppred/code2.zip',
          'sha1sum': 'b27404677ff72c55561b380224acc4147d113c5f',
      }
  }
  _GIT_URL = {}
  _DATASET_NAME = 'OGB_code_dataset'
  _DATASET_DESCRIPTION = """\
  From the paper:

  The ogbg-code dataset is a collection of Abstract Syntax Trees (ASTs) obtained
  from approximately 450 thousands Python method definitions. Methods are
  extracted from a total of 13,587 different repositories across the most
  popular projects on GITHUB (where “popularity” is defined as number of stars
  and forks). Our collection of Python methods originates from GITHUB
  CodeSearchNet (Husain et al., 2019), a collection of datasets and benchmarks
  for machine-learning-based code retrieval. The authors paid particular
  attention to avoid common shortcomings of previous source code datasets
  (Allamanis, 2019), such as duplication of code and labels, low number of
  projects, random splitting, etc. In ogbg-code, we contribute an additional
  feature extraction step, which includes:
  AST edges, AST nodes (associated with features such as their types and
  attributes), tokenized method name. Altogether, ogbg-code allows us to capture
  source code with its underlying graph structure, beyond its token sequence
  representation.
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
    """Download the dataset using requests and extract the zip file."""
    super().download_dataset_using_requests()
    self.dataset_extracted_dir = os.path.join(self.raw_data_dir, 'code2')
    zipfiles_to_extract = []
    zipfiles_to_extract = util.check_need_to_extract(
        zipfiles_to_extract, self.dataset_extracted_dir, 'code.zip')

    for filename in zipfiles_to_extract:
      zipfile_path = os.path.join(self.raw_data_dir, filename)
      with zipfile.ZipFile(zipfile_path) as zf:
        for member in tqdm.tqdm(zf.infolist(), desc='Extracting'):
          try:
            zf.extract(member, self.raw_data_dir)
          except zipfile.error:
            pass

    for root, _, filenames in os.walk(self.dataset_extracted_dir):
      for filename in filenames:
        if filename.endswith('.gz'):
          gz_filename = os.path.join(root, filename)
          extracted_filename = gz_filename[:-3]
          logging.info('Extracting %s ...', gz_filename)
          with gzip.open(gz_filename, 'rb') as f_gz:
            with open(extracted_filename, 'wb') as f_ex:
              shutil.copyfileobj(f_gz, f_ex)

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    mapping_dir = os.path.join(self.dataset_extracted_dir, 'mapping')
    raw_dir = os.path.join(self.dataset_extracted_dir, 'raw')
    split_dir = os.path.join(self.dataset_extracted_dir, 'split', 'project')

    path_dict = {}
    path_dict.update({'train_split': os.path.join(split_dir, 'train.csv')})
    path_dict.update({'valid_split': os.path.join(split_dir, 'valid.csv')})
    path_dict.update({'test_split': os.path.join(split_dir, 'test.csv')})
    path_dict.update({'edges': os.path.join(raw_dir, 'edge.csv')})
    path_dict.update({'node-feat': os.path.join(raw_dir, 'node-feat.csv')})
    path_dict.update({'labels': os.path.join(raw_dir, 'graph-label.csv')})
    path_dict.update(
        {'num-edge-list': os.path.join(raw_dir, 'num-edge-list.csv')})
    path_dict.update(
        {'num-node-list': os.path.join(raw_dir, 'num-node-list.csv')})
    path_dict.update(
        {'attridx2attr': os.path.join(mapping_dir, 'attridx2attr.csv')})
    path_dict.update(
        {'typeidx2type': os.path.join(mapping_dir, 'typeidx2type.csv')})
    return [path_dict]

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return CsvExtractor(super().get_random_split,
                        bool(self.user_defined_split_range))

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Convert raw data to the unified GraphToOutputExample data structure.

    The ogb code task is already represented as a graph, and we simply use it as
    it is. The output is the function name tokens.

    Args:
      raw_data: A dictionary with 'split', 'label', 'edges', 'node_feats',
        'attr_mapping' and 'type_mapping' as keys. The value of the 'split'
        field is the split (train/valid/test) that the data belongs to.
        The value of the 'label' field is the function name tokens. The value
        of the 'edges' field is the edges. The value of the 'node_feats' is
        the node types and labels. The value of the 'attr_mapping' is the
        integer to node label string mapping.  The value of the 'type_mapping'
        is the integer to node type string mapping.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs to, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    label = raw_data['label']
    edges = raw_data['edges']
    node_feats = raw_data['node_feats']
    attr_mapping = raw_data['attr_mapping']
    type_mapping = raw_data['type_mapping']

    graph_to_output_example = GraphToOutputExample()
    for node_id, node_feat in enumerate(node_feats):
      graph_to_output_example.add_node(node_id, type_mapping[node_feat[0]],
                                       attr_mapping[node_feat[1]])
    for edge in edges:
      graph_to_output_example.add_edge(edge[0], edge[1], 'EDGE')

    for output_token in label.split(' '):
      graph_to_output_example.add_token_output(output_token)

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


class CsvExtractor(beam.DoFn):
  """Class to read the OGB code dataset files."""

  def __init__(self, random_split_fn, use_random_split):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def _get_num_edge_list(self, num_edge_list_csv):
    """Read the csv file containing the number of edges for each graph."""
    with open(num_edge_list_csv) as f:
      num_edge_list = f.read().splitlines()
    return [int(num_edge) for num_edge in num_edge_list]

  def _get_num_node_list(self, num_node_list_csv):
    """Read the csv file containing the number of nodes for each graph."""
    with open(num_node_list_csv) as f:
      num_node_list = f.read().splitlines()
    return [int(num_node) for num_node in num_node_list]

  def _get_labels(self, label_csv):
    """Read the csv file containing the function names."""
    with open(label_csv) as f:
      labels = f.read().splitlines()
    return labels

  def _get_split_index_range(self, train_split_csv, valid_split_csv,
                             test_split_csv):
    """Read the csv files containing the data splits."""
    with open(train_split_csv) as f:
      last_train_id = int(f.read().splitlines()[-1])

    with open(valid_split_csv) as f:
      last_valid_id = int(f.read().splitlines()[-1])

    with open(test_split_csv) as f:
      last_test_id = int(f.read().splitlines()[-1])

    return last_train_id, last_valid_id, last_test_id

  def _get_attr_mapping(self, attr_csv):
    """Read the csv file containing node label mapping."""
    with open(attr_csv) as f:
      lines = f.read().splitlines()

    # The attr_csv file is malformed, because special characters such as the
    # newline character are not properly escaped. Therefore we cannot read the
    # file line by line. But we know what is the correct attr_csv file format,
    # each line should be <ID>,<NODE LABEL>. We read the file line by line, and
    # check if the line starts with '<ID>,'. If that is the case, we know that
    # it is a new mapping, otherwise it is a continuation of the last mapping.
    next_expected_id = 0
    attr_mapping = {}
    # The first line is the column names.
    for line in lines[1:]:
      # A new mapping.
      if line.startswith(str(next_expected_id)+','):
        next_expected_id += 1
        attr_mapping[next_expected_id-1] = line.split(',')[1]
      # Otherwise it is a continuation of the last mapping.
      else:
        attr_mapping[next_expected_id-1] += line
    return attr_mapping

  def _get_type_mapping(self, type_csv):
    """Read the csv file containing node type mapping."""
    type_mapping = {}
    with open(type_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      next(csv_reader)
      for row in csv_reader:
        type_mapping[int(row[0])] = row[1]
    return type_mapping

  def _get_edges(self, edges_csv):
    """Read the csv file containing edges for all graphs."""
    with open(edges_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      edges = [[int(row[0]), int(row[1])] for row in csv_reader]
    return edges

  def _get_node_feat(self, node_feat_csv):
    """Read the csv file containing node type id and label id for all graphs."""
    with open(node_feat_csv) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      node_feats = [[int(row[0]), int(row[1])] for row in csv_reader]
    return node_feats

  def process(self, path_dict):
    """Function to read all files and yield all raw data.

    The raw data of OGB code dataset are stored in different files. All
    filenames are stored in the input argument path_dict as a dictionary.
    We use different functions to read these files and yield the parsed raw
    data.

    Args:
      path_dict: A dictionary containing paths to all files needed for parse the
        raw data.

    Yields:
      A dictionary with 'split', 'label', 'edges', 'node_feats', 'attr_mapping'
      and 'type_mapping' as keys. The value of the 'split' field is the split
      (train/valid/test) that the data belongs to. The value of the 'label'
      field is the function name tokens. The value of the 'edges' field is the
      edges. The value of the 'node_feats' is the node types and labels. The
      value of the 'attr_mapping' is the integer to node label string mapping.
      The value of the 'type_mapping' is the integer to node type string
      mapping.
    """
    # Read all the files.
    num_edge_list = self._get_num_edge_list(path_dict['num-edge-list'])
    num_node_list = self._get_num_node_list(path_dict['num-node-list'])
    labels = self._get_labels(path_dict['labels'])
    last_train_id, last_valid_id, _ = self._get_split_index_range(
        path_dict['train_split'], path_dict['valid_split'],
        path_dict['test_split'])
    attr_mapping = self._get_attr_mapping(path_dict['attridx2attr'])
    type_mapping = self._get_type_mapping(path_dict['typeidx2type'])
    edges = self._get_edges(path_dict['edges'])
    node_feats = self._get_node_feat(path_dict['node-feat'])

    num_edge_start_index = 0
    num_node_start_index = 0
    for index, (num_edge, num_node) in enumerate(
        zip(num_edge_list, num_node_list)):
      # labels stores all the function names, edges stores edges for all graphs,
      # node_feats stores node type/label ids for all graphs.
      cur_label = labels[index]
      cur_edges = edges[num_edge_start_index:num_edge_start_index + num_edge]
      cur_node_feats = node_feats[
          num_node_start_index:num_node_start_index + num_node]
      num_edge_start_index += num_edge
      num_node_start_index += num_node

      # If we use random split, or use predefined split.
      if self.use_random_split:
        split = self.random_split_fn()
      else:
        if index <= last_train_id:
          split = constants.TRAIN_SPLIT_NAME
        elif index <= last_valid_id:
          split = constants.VALIDATION_SPLIT_NAME
        else:
          split = constants.TEST_SPLIT_NAME
      yield {'split': split, 'label': cur_label, 'edges': cur_edges,
             'node_feats': cur_node_feats, 'attr_mapping': attr_mapping,
             'type_mapping': type_mapping}
