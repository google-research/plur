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

"""Classes for converting the funcom dataset to a PLUR dataset."""
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


class FuncomDataset(PlurDataset):
  """Converting data from funcom dataset to a PLUR dataset.

  The dataset is created by: LeClair, A., McMillan, C., 'Recommendations for
  Datasets for Source Code Summarization', in Proc. of the 2019 Annual
  Conference of the North American Chapter of the Association for Computational
  Linguistics (NAACL'19), Short Research Paper Track, Minneapolis, USA,
  June 2-7, 2019.

  The task is to predict the method docstring given the method.

  The docstring and the method is already tokenized. Since the data is not
  represented as a graph, but represented as tokens, we transform it into a
  graph. We will create a node for each source code tokens and connect all nodes
  as a chain with 'NEXT_TOKEN' edges. The output will simply be the docstring
  tokens.
  """


  _URLS = {
      'funcom_tokenized.tar.gz': {
          'url': 'https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_tokenized.tar.gz',
          'sha1sum': '5b25d3dfcf156ef48e1d6ab5c61029a67a6cf2a1',
      }
  }

  _GIT_URL = {}
  _DATASET_NAME = 'funcom_dataset'
  _DATASET_DESCRIPTION = """\
  From http://leclair.tech/data/funcom/:

  Funcom is a collection of ~2.1 million Java methods and their associated
  Javadoc comments. This data set was derived from a set of 51 million Java
  methods and only includes methods that have an associated comment, comments
  that are in the English language, and has had auto-generated files removed.
  Each method/comment pair also has an associated method_uid and project_uid so
  that it is easy to group methods by their parent project.
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
    self.funcom_extracted_dir = os.path.join(self.raw_data_dir,
                                             'funcom_tokenized')
    # We extract the tar file here. First we check if it exists, and then
    # ask the user about rextracting the tar file.
    tarfiles_to_extract = []
    tarfiles_to_extract = util.check_need_to_extract(
        tarfiles_to_extract, self.funcom_extracted_dir,
        'funcom_tokenized.tar.gz')

    for filename in tarfiles_to_extract:
      tarfile_path = os.path.join(self.raw_data_dir, filename)
      with tarfile.open(tarfile_path, 'r:gz') as tf:
        for member in tqdm.tqdm(
            tf.getmembers(),
            unit='file',
            desc='Extracting {}'.format(tarfile_path)):
          tf.extract(member, self.raw_data_dir)

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    train_functions_file = os.path.join(self.funcom_extracted_dir, 'train',
                                        'functions.train')
    train_comments_file = os.path.join(self.funcom_extracted_dir, 'train',
                                       'comments.train')
    validation_functions_file = os.path.join(self.funcom_extracted_dir, 'valid',
                                             'functions.valid')
    validation_comments_file = os.path.join(self.funcom_extracted_dir, 'valid',
                                            'comments.valid')
    test_functions_file = os.path.join(self.funcom_extracted_dir, 'test',
                                       'functions.test')
    test_comments_file = os.path.join(self.funcom_extracted_dir, 'test',
                                      'comments.test')
    return [(train_functions_file, train_comments_file),
            (validation_functions_file, validation_comments_file),
            (test_functions_file, test_comments_file)]

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return FuncomExtractor(super().get_random_split,
                           bool(self.user_defined_split_range))

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Convert raw data to the unified GraphToOutputExample data structure.

    We create a node for each source code token and connect all nodes as a chain
    with 'NEXT_TOKEN' edges. The output is the function comment tokens.

    Args:
      raw_data: A dictionary with 'split', 'function' and 'comment' as keys. The
        value of the 'split' field is the split (train/valid/test) that the data
        belongs to. The value of the 'function' field is the function tokens as
        a string. The value of the 'comment' field is the comment tokens as a
        string.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs, and the
      GraphToOutputExample instance.
    """
    split = raw_data['split']
    function = raw_data['function']
    comment = raw_data['comment']
    graph_to_output_example = GraphToOutputExample()

    # The function is already tokenized, but it is separated by whitespaces.
    # We use split(' ') to get the tokens.
    for index, token in enumerate(function.split(' ')):
      graph_to_output_example.add_node(index, 'TOKEN', token)

    # We connect the nodes as a chain with the 'NEXT_TOKEN' edge between
    # each consecutive source code token node.
    for i in range(len(graph_to_output_example.get_nodes())-1):
      graph_to_output_example.add_edge(i, i+1, 'NEXT_TOKEN')

    # The output is the comment tokens.
    for token in comment.split(' '):
      graph_to_output_example.add_token_output(token)

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


class FuncomExtractor(beam.DoFn):
  """Class to read the funcom dataset."""

  def __init__(self, random_split_fn, use_random_split):
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split

  def _read_data(self, functions_file, comments_file):
    r"""Read the function and comment files.

    The data are stored as rows of '<ID>\t<TOKENS>', where the <ID> is a
    unique identifier for its origin, and <TOKENS> is the corresponding
    function/comment tokens.

    Args:
      functions_file: Path to funcom file storing the function tokens.
      comments_file: Path to funcom files storing the comment tokens.

    Yields:
      A tuple of two strings. The first string is the function tokens, and the
      second string is the comment tokens.
    """
    with open(functions_file) as ff, open(comments_file) as cf:
      for function_line, comment_line in zip(ff, cf):
        yield (function_line.rstrip().split('\t')[1],
               comment_line.rstrip().split('\t')[1])

  def _get_split(self, file_path):
    """Get the split depending on the path ending."""
    if file_path.endswith('train'):
      return constants.TRAIN_SPLIT_NAME
    elif file_path.endswith('valid'):
      return constants.VALIDATION_SPLIT_NAME
    else:
      return constants.TEST_SPLIT_NAME

  def process(self, file_paths):
    """Function to read each text file.

    Args:
      file_paths: Path to a raw data file.

    Yields:
      A dictionary with 'split', 'function' and 'comment' as keys. The value of
      the 'split' field is the split (train/valid/test) that the data belongs
      to. The value of the 'function' field is the function tokens as a string.
      The value of the 'comment' field is the comment tokens as a string.
    """
    functions_file = file_paths[0]
    comments_file = file_paths[1]
    split = self._get_split(functions_file)
    for function, comment in self._read_data(functions_file, comments_file):
      # Call random_split_fn() if use_random_split, otherwise use predefined
      # funcom split.
      yield {
          'split': self.random_split_fn() if self.use_random_split else split,
          'function': function,
          'comment': comment
      }
