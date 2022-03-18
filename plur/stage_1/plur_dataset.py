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
"""The abstract class that unifies public datasets to PLUR datasets."""

import abc
import dataclasses
import hashlib
import os
import random
from typing import Dict, Mapping


# This env variable disables error from GitPython if git is not found in the
# path. This can happen when running beam in distributed settings, and git
# is not installed (it is also not needed in that case).
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


from absl import logging
import apache_beam as beam
import git
from plur.utils import constants
from plur.utils import util
import requests
import tqdm



@dataclasses.dataclass
class Configuration():
  """Collects configuration parameters, for datasets that need them."""
  max_graph_sizes: Mapping[str, int] = dataclasses.field(default_factory=dict)


class PlurDataset(abc.ABC):
  """This is the abstract class of all datasets.

  PlurDataset is the superclass of all datasets. It requires the class that
  inherits from it to implement:
  * download_dataset(): Code to download the dataset, we provided
      download_dataset_using_git() to download from git and
      download_dataset_using_requests() to download with URL, which also works
      with a Google Drive URL. In download_dataset_using_git() we are
      downloading the dataset from a specific commit id. In
      download_dataset_using_requests() we check the sha1sum for the downloaded
      files. This is to ensure that the same version of PLUR downloads the same
      raw data.
  * get_all_raw_data_paths(): It should return a list of paths, where each path
    is a file containing the raw data in the datasets.
  * raw_data_paths_to_raw_data_do_fn(): It should return a beam.DoFn class that
      overrides process(). The process() should tell beam how to open the files
      returned by get_all_raw_data_paths. It is also where we define if the data
      belongs to any split (train/validation/test).
  * raw_data_to_graph_to_output_example(): This function transforms raw data
      from raw_data_paths_to_raw_data_do_fn to GraphToOutputExample.
  See comments above get_all_raw_data_paths that explains how
  get_all_raw_data_paths, raw_data_paths_to_raw_data_do_fn and
  raw_data_to_graph_to_output_example are used. You can also check any
  *_dataset.py under plur/stage_1 on how these functions are overwritten.

  Once these functions are implemented, the rest are handled by the PlurDataset
  class. It stores GraphToOutputExample as json files and we also output another
  json file containing the metadata of the dataset. See utils/constants.py
  on the output directory structure of PLUR.
  """

  def __init__(self, dataset_name: str, urls: Dict[str, Dict[str, str]],
               git_url: Dict[str, str], dataset_description: str,
               stage_1_dir: str,
               configuration: Configuration = Configuration(),
               transformation_funcs=(), filter_funcs=(),
               user_defined_split_range=(), num_shards=1000, seed=0,
               deduplicate=False):
    """PlurDataset constructor arguments.

    Args:
      dataset_name: The dataset name.
      urls: A dictionary containing the URLs and sha1sum for downloading, it is
        a mapping between the filename and a nested dictionary. The nested
        dictionary contains the url and sha1sum for the file.
      git_url: A dictionary containing git URL and commit id if the dataset is
        stored on git.
      dataset_description: A string describing the dataset.
      stage_1_dir: Directory to store stage 1 files.
      configuration: Runtime parameters.
      transformation_funcs: A tuple of functions that are applied on
        GraphToOutputExample, each function should take GraphToOutputExample as
        argument and returns GraphToOutputExample. It can be used to transform
        GraphToOutputExample. This should be applied on all GraphToOutputExample
        (train/val/test), so it should be something that you want to
        fundamentally changed in the dataset. If not, use transformation_funcs
        in stage 2.
      filter_funcs: A tuple of functions used to filter GraphToOutputExample.
        Each function take GraphToOutputExample as input, and returns True if
        we want to keep it, and False otherwise. Note that all datasets should
        apply transformation_funcs before filter_funcs. This should be applied
        on all GraphToOutputExample (train/val/test), so it should be something
        that you want to fundamentally changed in the dataset. If not, use
        filter_funcs in stage 2.
      user_defined_split_range: A tuple of user defined split range. If it is
        specified, it should contain three integers (eg. (80, 10, 10)), where
        each integer represents the train/validation/testing split range. The
        sum of the three integers should also be 100.
      num_shards: Number of shards used to store the PlurDataset.
      seed: The random seed.
      deduplicate: If true, remove duplicate GraphToOutputExample.
    """
    self.dataset_name = dataset_name
    self.urls = urls
    self.git_url = git_url
    self.dataset_description = dataset_description
    self.stage_1_dir = stage_1_dir
    self.configuration = configuration
    self.raw_data_dir = os.path.join(self.stage_1_dir,
                                     constants.RAW_DATA_DIRNAME)
    self.git_repo_dir = os.path.join(self.raw_data_dir,
                                     constants.GIT_REPO_DIRNAME)
    self.graph_to_output_example_dir = os.path.join(
        self.stage_1_dir, constants.GRAPH_TO_OUTPUT_EXAMPLE_DIRNAME)
    self.train_graph_to_output_example_dir = os.path.join(
        self.graph_to_output_example_dir, constants.TRAIN_SPLIT_NAME)
    self.valid_graph_to_output_example_dir = os.path.join(
        self.graph_to_output_example_dir, constants.VALIDATION_SPLIT_NAME)
    self.test_graph_to_output_example_dir = os.path.join(
        self.graph_to_output_example_dir, constants.TEST_SPLIT_NAME)
    self.metadata_file = os.path.join(self.stage_1_dir,
                                      constants.METADATA_FILENAME)
    self.transformation_funcs = transformation_funcs
    self.filter_funcs = filter_funcs
    self.user_defined_split_range = user_defined_split_range
    if self.user_defined_split_range:
      assert sum(self.user_defined_split_range) == 100
      assert len(self.user_defined_split_range) == 3
      self._train_split_range = self.user_defined_split_range[0]
      self._validation_split_range = (
          self.user_defined_split_range[1] + self._train_split_range)
    self.num_shards = num_shards
    self.seed = seed
    random.seed(self.seed)
    self.random = random
    self.deduplicate = deduplicate

  def stage_1_mkdirs(self):
    logging.info('Creating all neccsary stage 1 directiories for %s.',
                 self.dataset_name)
    os.makedirs(self.git_repo_dir, exist_ok=True)
    os.makedirs(self.train_graph_to_output_example_dir, exist_ok=True)
    os.makedirs(self.valid_graph_to_output_example_dir, exist_ok=True)
    os.makedirs(self.test_graph_to_output_example_dir, exist_ok=True)

  @abc.abstractmethod
  def download_dataset(self):
    """Abstract method for downloading the raw dataset.

    If the raw dataset should be download via URLs, for example
    'https://s3.amazonaws.com/foo/bar/hello_world.tar.gz'. Then the class that
    inherits PlurDataset can set it as the 'urls' attribute and call
    super().download_dataset_using_requests().

    And if the raw dataset should be download via git, for example
    'https://github.com/foo/bar.git'. Then the class that inherits PlurDataset
    can set it as the 'git_url' attribute and call
    super().download_dataset_using_git().

    If the dataset should be downloaded in another way, the class that inherits
    PlurDataset should implement the code that handles the downloading in this
    function.
    """
    pass

  def download_dataset_using_git(self):
    """Download the dataset using GitPython."""
    logging.info('Downloading the dataset using GitPython.')
    try:
      self.repo = git.Repo(self.git_repo_dir)
      assert self.repo.remotes.origin.url == self.git_url['url']
      checkout_from_commit = input(
          '{} already exists, checkout from commit {}?(y/n): '.format(
              self.git_repo_dir, self.git_url['commit_id']))
      if checkout_from_commit.lower() == 'y':
        logging.info('Checkout from %s.', self.git_url['commit_id'])
        self.repo.git.checkout(self.git_url['commit_id'])
    except git.InvalidGitRepositoryError:
      logging.info('Cloning from commit %s of %s to %s.', self.git_url['url'],
                   self.git_url['commit_id'], self.git_repo_dir)
      self.repo = git.Repo.clone_from(
          self.git_url['url'], self.git_repo_dir, progress=util.GitProgress(),
          no_checkout=True)
      self.repo.git.checkout(self.git_url['commit_id'])

  def download_dataset_using_requests(self):
    """Download the dataset using requests."""
    def _get_confirm_token(response):
      """Get the cookie value for download confirmation."""
      for key, value in response.cookies.items():
        if key.startswith('download_warning'):
          return value

      return None

    logging.info('Downloading the dataset using requests.')
    for filename in self.urls:
      dest = os.path.join(self.raw_data_dir, filename)
      if os.path.exists(dest):
        re_download = input('{} already exists, re-download?(y/n)'.format(dest))
        if re_download.lower() == 'y':
          os.remove(dest)
        else:
          continue

      session = requests.Session()
      response = session.get(self.urls[filename]['url'], stream=True)
      response.raise_for_status()
      token = _get_confirm_token(response)
      if token:
        params = {'confirm': token}
        response = session.get(self.urls[filename]['url'], params=params,
                               stream=True)

      sha1 = hashlib.sha1()
      chunk_size = 1024
      content_length = int(response.headers.get('content-length', 0))
      with open(dest, 'wb') as f, tqdm.tqdm(
          desc='Downloading {}'.format(filename),
          total=content_length,
          unit='iB',
          unit_scale=True,
          unit_divisor=chunk_size) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
          if chunk:
            size = f.write(chunk)
            sha1.update(chunk)
            bar.update(size)
      # Check the sha1sum hash, warns the user if it is not the same as the
      # hash stored by PLUR. It is most likely because the dataset owner
      # has updated the dataset.
      if sha1.hexdigest() != self.urls[filename]['sha1sum']:
        logging.error('sha1sum of %s does not match, continue anyway. '
                      'But be aware that the generated data can be different '
                      'from data generated from PLUR of the same version.',
                      filename)

  def get_random_split(self):
    """Return a random split depending on the user_defined_split_range."""
    random_int = self.random.randint(1, 100)
    if random_int <= self._train_split_range:
      return constants.TRAIN_SPLIT_NAME
    elif random_int <= self._validation_split_range:
      return constants.VALIDATION_SPLIT_NAME
    else:
      return constants.TEST_SPLIT_NAME

  @abc.abstractmethod
  def get_all_raw_data_paths(self):
    """Abstract method for getting all the raw data paths.

    This function should return a list of paths, for example
    ['/foo/bar/train.txt', '/foo/bar/val.txt', '/foo/bar/test.txt'].
    """
    pass

  @abc.abstractmethod
  def raw_data_paths_to_raw_data_do_fn(self):
    """Abstract method for reading all the raw data files.

    It should return a beam.DoFn class that overwrites the process function,
    which tells the beam.DoFn class how to open each path returned by
    get_all_raw_data_paths.
    """
    pass

  @abc.abstractmethod
  def raw_data_to_graph_to_output_example(self, raw_data):
    """Abstract method for converting raw data to GraphToOutputExample.

    This function will receive a dictionary as input, which is the output
    yielded by raw_data_paths_to_raw_data_do_fn. Then, it should create
    GraphToOutputExample and add nodes, edges and output. The output MUST
    be a list of dictionaries, where each dictionary has the 'split' and
    'GraphToOutputExample' fields. The 'split' field contains the split
    that GraphToOutputExample belongs, and the 'GraphToOutputExample' field
    contains the GraphToOutputExample instance.

    Args:
      raw_data: The dictionary return by raw_data_paths_to_raw_data_do_fn
        function. It must have 'split' fields, and other fields that are
        neccessary to construct the GraphToOutputExample instance.
    """
    pass

  def exists_graph_to_output_example(self):
    """Check existing GraphToOutputExample stored on disk.

    We check the existing GraphToOutputExample stored on disk, and ask the user
    if they want to regenerate the GraphToOutputExample.

    Returns:
      True if we found GraphToOutputExample and user want to keep it. False if
      the user want to (re)generate GraphToOutputExample.
    """
    existing_training_graph_to_output_example = [
        os.path.join(self.train_graph_to_output_example_dir, filename)
        for filename in os.listdir(self.train_graph_to_output_example_dir)
        if os.path.splitext(filename)[1] == '.json'
    ]
    existing_validation_graph_to_output_example = [
        os.path.join(self.valid_graph_to_output_example_dir, filename)
        for filename in os.listdir(self.valid_graph_to_output_example_dir)
        if os.path.splitext(filename)[1] == '.json'
    ]
    existing_test_graph_to_output_example = [
        os.path.join(self.test_graph_to_output_example_dir, filename)
        for filename in os.listdir(self.test_graph_to_output_example_dir)
        if os.path.splitext(filename)[1] == '.json'
    ]
    existing_graph_to_output_example = (
        existing_training_graph_to_output_example +
        existing_validation_graph_to_output_example +
        existing_test_graph_to_output_example)
    if existing_graph_to_output_example:
      regenerate_graph_to_output_example = input(
          '{} already exists and contains json files, delete it and regenerate '
          'GraphToOutputExample?(y/n): '.format(
              self.graph_to_output_example_dir))
      if regenerate_graph_to_output_example.lower() == 'y':
        for file_path in tqdm.tqdm(
            existing_graph_to_output_example, desc='Deleting'):
          os.remove(file_path)
      else:
        return True
    return False

  def convert_and_write_pipeline(self, root):
    """Beam pipeline that reads, converts, and writes GraphToOutputExample."""
    # raw_data_dicts is a Pcollection of dictionaries, which is returned by
    # the raw_data_paths_to_raw_data_do_fn function.
    raw_data_dicts = (
        root
        | 'Get all raw data paths' >>
        beam.Create(self.get_all_raw_data_paths())
        | 'Read all raw data' >>
        beam.ParDo(self.raw_data_paths_to_raw_data_do_fn())
        | 'Reshuffle after reading all raw data' >>
        beam.Reshuffle()
    )

    # graph_to_output_example_dicts is a Pcollection of dictionaries. Each
    # dictionary has 'field' and 'GraphToOutputExample' fields. The 'split'
    # field specifies the split the GraphToOutputExample belongs, and the
    # 'GraphToOutputExample' contains the GraphToOutputExample instance.
    graph_to_output_example_dicts = (
        raw_data_dicts
        | 'Convert raw data to GraphToOutputExample' >> beam.Map(
            self.raw_data_to_graph_to_output_example)
        | 'Reshuffle after converting raw data to GraphToOutputExample' >>
        beam.Reshuffle()
        | 'Filter None GraphToOutputExample' >> beam.Filter(
            util.graph_to_output_example_is_not_none)
    )

    if self.deduplicate:
      graph_to_output_example_dicts = (
          graph_to_output_example_dicts
          | 'Compute hash and use it as key for each GraphToOutputExample' >>
          beam.Map(lambda x: (x['GraphToOutputExample'].compute_hash(), x))
          | 'Remove duplicate GraphToOutputExample using hash' >>
          beam.CombinePerKey(lambda x: next(iter(x)))
          | 'Remove hash as key' >> beam.Map(lambda x: x[1]))

    metadata = (root | 'Create metadata' >> beam.Create([{}]))
    fieldname = 'dataset_name'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, self.dataset_name)
    fieldname = 'urls'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, self.urls)
    fieldname = 'git_url'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, self.git_url)
    fieldname = 'dataset_description'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, self.dataset_description)

    for split in [
        constants.TRAIN_SPLIT_NAME, constants.VALIDATION_SPLIT_NAME,
        constants.TEST_SPLIT_NAME
    ]:
      file_path_prefix = os.path.join(self.graph_to_output_example_dir, split,
                                      self.dataset_name)
      file_name_suffix = '.json'

      split_raw_data_dicts = (
          raw_data_dicts
          | 'Filter {} raw data'.format(split) >>
          beam.Filter(util.filter_split, split)
      )

      split_graph_to_output_example_dicts = (
          graph_to_output_example_dicts
          | 'Filter {} GraphToOutputExample'.format(split) >>
          beam.Filter(util.filter_split, split)
      )

      metadata = self.update_dataset_metadata(
          metadata, split, split_raw_data_dicts,
          split_graph_to_output_example_dicts)

      _ = (
          split_graph_to_output_example_dicts
          | 'Extract {} GraphToOutputExample for write'.format(split) >>
          beam.Map(lambda x: x['GraphToOutputExample'].get_data())
          | 'Write {} GraphToOutputExample to disk'.format(split) >>
          beam.io.WriteToText(
              file_path_prefix,
              file_name_suffix=file_name_suffix,
              num_shards=self.num_shards,
              coder=util.JsonCoder()))

    _ = (
        metadata
        | 'Write metadata to file' >>
        beam.io.WriteToText(
            self.metadata_file,
            num_shards=1,
            shard_name_template='',
            coder=util.JsonCoder())
    )

  def update_dataset_metadata(self, metadata, split, raw_data_dicts,
                              graph_to_output_example_dicts):
    """Update the dataset metadata.

    We update the metadata dictionary (Pcollection), using the raw data
    dictionaries, GraphToOutputExample dictionaries and the split
    (train/validation/test). We keep count of metadata like maximum number of
    nodes, maximum number of edges etc.

    Args:
      metadata: A Pcollection containining the metadata dict.
      split: The split that the raw data/GraphToOutputExample belongs.
      raw_data_dicts: A Pcollection containining the raw data dictionaries.
      graph_to_output_example_dicts: A Pcollection containining the
        GraphToOutputExample dictionaries.
    Returns:
      The updated metadata Pcollection.
    """
    raw_data_split_count = beam.pvalue.AsSingleton(
        raw_data_dicts
        | 'Count {} raw_data'.format(split) >>
        beam.combiners.Count.Globally()
    )
    fieldname = split + '_raw_data' + '_count'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, raw_data_split_count)

    graph_to_output_example_dicts_split_count = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Count {} GraphToOutputExample'.format(split) >>
        beam.combiners.Count.Globally()
    )
    fieldname = split + '_GraphToOutputExample' + '_count'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, graph_to_output_example_dicts_split_count)

    graph_to_output_example_split_max_num_nodes = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Get nodes from {} GraphToOutputExample for max_num_nodes'.format(
            split) >>
        beam.Map(lambda x: len(x['GraphToOutputExample'].get_nodes()))
        | 'Compute max_num_nodes for {} GraphToOutputExample'.format(split) >>
        beam.CombineGlobally(lambda values: max(values, default=0))
    )
    fieldname = split + '_GraphToOutputExample' + '_max_num_nodes'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, graph_to_output_example_split_max_num_nodes)

    graph_to_output_example_split_max_num_edges = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Get edges from {} GraphToOutputExample for max_num_edges'.format(
            split) >>
        beam.Map(lambda x: len(x['GraphToOutputExample'].get_edges()))
        | 'Compute max_num_edges for {} GraphToOutputExample'.format(split) >>
        beam.CombineGlobally(lambda values: max(values, default=0))
    )
    fieldname = split + '_GraphToOutputExample' + '_max_num_edges'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, graph_to_output_example_split_max_num_edges)

    graph_to_output_example_split_max_num_output = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Get output from {} GraphToOutputExample for max_num_output'.format(
            split) >>
        beam.Map(lambda x: len(x['GraphToOutputExample'].get_output()))
        | 'Compute max_num_output for {} GraphToOutputExample'.format(split) >>
        beam.CombineGlobally(lambda values: max(values, default=0))
    )
    fieldname = split + '_GraphToOutputExample' + '_max_num_output'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, graph_to_output_example_split_max_num_output)

    graph_to_output_example_split_max_num_edge_types = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Get num_edge_types from {} GraphToOutputExample for '
          'max_num_edge_types'.format(split) >>
        beam.Map(lambda x: x['GraphToOutputExample'].get_num_edge_types())
        | 'Compute max_num_edge_types for {} GraphToOutputExample'.format(
            split) >>
        beam.CombineGlobally(lambda values: max(values, default=0)))
    fieldname = split + '_GraphToOutputExample' + '_max_num_edge_types'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, graph_to_output_example_split_max_num_edge_types)

    for max_num_nodes in [128, 256, 512, 1024, 2048, 4096, 8192]:
      graph_to_output_example_count = (
          self._count_graph_to_output_example_with_max_num_nodes(
              graph_to_output_example_dicts, split, max_num_nodes))
      fieldname = split + '_GraphToOutputExample' + 'max_num_nodes_{}'.format(
          max_num_nodes)
      metadata = util.add_field_to_metadata(
          metadata, fieldname, graph_to_output_example_count)

    return metadata

  def _count_graph_to_output_example_with_max_num_nodes(
      self, graph_to_output_example_dicts, split, max_num_nodes):
    """Count number of GraphToOutputExample <= max_num_nodes nodes.

    Args:
     graph_to_output_example_dicts: A Pcollection containining the
       GraphToOutputExample dictionaries.
     split: The split that graph_to_output_example_dicts belongs.
     max_num_nodes: The maximum number of nodes, we count number of
       GraphToOutputExample with less or equal than max_num_nodes nodes.
    Returns:
      Number of GraphToOutputExample with less or equal than max_num_nodes node,
      returns as Pcollection which can be used as a side input.
    """
    graph_to_output_example_count = beam.pvalue.AsSingleton(
        graph_to_output_example_dicts
        | 'Get nodes from {} GraphToOutputExample for counting '
          '<= {} nodes'.format(split, max_num_nodes) >>
        beam.Map(lambda x: len(x['GraphToOutputExample'].get_nodes()))
        | 'Filter {} GraphToOutputExample with <= {} nodes'.format(
            split, max_num_nodes) >>
        beam.Filter(lambda x: x <= max_num_nodes)
        | 'Count {} GraphToOutputExample with <= {} nodes'.format(
            split, max_num_nodes) >>
        beam.combiners.Count.Globally()
    )
    return graph_to_output_example_count

  def run_pipeline(self):
    logging.info('Running stage 1 pipeline.')
    if not self.exists_graph_to_output_example():
      with beam.Pipeline() as p:
        self.convert_and_write_pipeline(p)
