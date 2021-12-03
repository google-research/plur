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

"""Classes for converting the Convattn dataset to a PLUR dataset.
"""
import glob
import json
import os
import random
import zipfile

import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils import constants
from plur.utils import util
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import tqdm


class ConvAttnDataset(PlurDataset):
  """Converting the data from ConvAttn dataset to a PLUR dataset.

  The dataset is used in: Allamanis, Miltiadis, Hao Peng, and Charles Sutton.
  'A convolutional attention network for extreme summarization of source code.'
  International conference on machine learning. 2016.

  The task is to predict the method name given the method body.

  In the dataset, we have the method name and method body already tokenized.
  Since the data is not represented as a graph, but represented as tokens, we
  transform it into a graph. We will create a node for each source code tokens
  and connect all nodes as a chain with 'NEXT_TOKEN' edges. The output will
  simply be the method name tokens.
  """

  _URLS = {
      'dataset.zip': {
          'url': 'http://groups.inf.ed.ac.uk/cup/codeattention/dataset.zip',
          'sha1sum': 'a4d587a97617b79c990f1d4aa2f9159ed8c0cc9b',
      }
  }
  _GIT_URL = {}
  _DATASET_NAME = 'convolutional_attention_dataset'
  _DATASET_DESCRIPTION = """\
  From http://groups.inf.ed.ac.uk/cup/codeattention/dataset.txt:

  The dataset used for the paper experiments contains the following folders:

  train: Contains all the training (.java) files

  test: Contains all the test (.java) files

  json: Contains a parsed form of the data, that can be easily input into
    machine learning models. The format of the json files is explained below.

  ========================== JSON file format ==========================
  Each .json file is a list of methods.

  Each method is described by a dictionary that contains the following key-value
  pairs:
    filename: the origin of the method
    name: a list of the normalized subtokens of the method name
    tokens: a list of the tokens of the code within the body of the method. The
      code tokens are padded with a special <SENTENCE_START> and <SENTENCE_END>
      symbol. Source code identifiers (ie. variable, method and type names) are
      annotated by surrounding them with `<id>` and `</id>` tags. These tags
      were removed as a preprocessing step in this paper.
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
    """Download the dataset using requests."""
    super().download_dataset_using_requests()
    # We extract the zip file here. First we check if it exists, and then
    # ask the user about rextracting the zip file.
    self.dataset_extracted_dir = os.path.join(self.raw_data_dir, 'json')
    zipfiles_to_extract = []
    zipfiles_to_extract = util.check_need_to_extract(
        zipfiles_to_extract, self.dataset_extracted_dir,
        'dataset.zip')

    for filename in zipfiles_to_extract:
      zipfile_path = os.path.join(self.raw_data_dir, filename)
      with zipfile.ZipFile(zipfile_path) as zf:
        for member in tqdm.tqdm(zf.infolist(), desc='Extracting'):
          try:
            zf.extract(member, self.raw_data_dir)
          except zipfile.error:
            pass

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    # All data are stored in all_train_methodnaming.json and
    # *test_methodnaming.json.
    all_raw_data_file_paths = []
    all_raw_data_file_paths.append(
        os.path.join(self.dataset_extracted_dir, 'all_train_methodnaming.json')
    )
    all_raw_data_file_paths.extend(glob.glob(
        os.path.join(self.dataset_extracted_dir, '*test_methodnaming.json')))
    return all_raw_data_file_paths

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return JsonExtractor(super().get_random_split,
                         bool(self.user_defined_split_range))

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Convert raw data to the unified GraphToOutputExample data structure.

    We create a node for each source code token and connect all nodes as a chain
    with 'NEXT_TOKEN' edges. The output is the function name tokens.

    Args:
      raw_data: A dictionary with 'split' and 'data' as keys. The value of the
        'split' field is the split (train/valid/test) that the data belongs to.
        The value of the 'data' is the parsed raw json data.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    data = raw_data['data']

    graph_to_output_example = GraphToOutputExample()
    # The nodes are the source code tokens
    for token_id, token in enumerate(data['tokens']):
      graph_to_output_example.add_node(token_id, 'TOKEN', token)

    # We connect the nodes as a chain with the 'NEXT_TOKEN' edge between
    # each consecutive source code token node.
    for token_id in range(len(graph_to_output_example.get_nodes())-1):
      graph_to_output_example.add_edge(token_id, token_id + 1, 'NEXT_TOKEN')

    # The output is just the method name tokens.
    for output_name in data['name']:
      graph_to_output_example.add_token_output(output_name)

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
  """Class to extract data from the JSON files in the ConvAttn dataset."""

  def __init__(self, random_split_fn, use_random_split):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def process(self, file_path):
    """Function to read each json file.

    Args:
      file_path: Path to a raw data file.

    Yields:
      A dictionary with 'split' and 'data' as keys. The value of the 'split'
      field is the split (train/valid/test) that the data belongs. The value of
      the 'data' is the parsed raw json data.
    """
    with open(file_path) as f:
      json_data = json.load(f)
      for data in json_data:
        # If use_random_split is True, we use random_split_fn to get the split.
        if self.use_random_split:
          yield {'split': self.random_split_fn(), 'data': data}
        else:
          # ConvAttn only defines train and test data. Therefore we leave
          # test data as it is. But take 20% of train data as valid data.
          if file_path.endswith('test_methodnaming.json'):
            yield {'split': constants.TEST_SPLIT_NAME, 'data': data}
          else:
            if random.random() <= 0.8:
              split = constants.TRAIN_SPLIT_NAME
            else:
              split = constants.VALIDATION_SPLIT_NAME
            yield {'split': split, 'data': data}
