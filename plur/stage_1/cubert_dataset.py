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
"""Converts CuBERT dataset to PLUR."""
import abc
import glob
import itertools
import json
import os
import pathlib
from typing import AbstractSet, Any, Iterable, List, Mapping, Optional

import apache_beam as beam
from google.cloud import storage
from plur.stage_1 import plur_dataset
from plur.utils import constants
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
from tensor2tensor.data_generators import text_encoder
import tqdm

from cubert import code_to_subtokenized_sentences
from cubert import python_tokenizer
from cubert import unified_tokenizer


class CuBertDataset(plur_dataset.PlurDataset):
  """Converts CuBERT data to a PLUR dataset.

  The datasets are created by: Aditya Kanade, Petros Maniatis, Gogul
  Balakrishnan, Kensen Shi Proceedings of the 37th International Conference on
  Machine Learning, PMLR 119:5110-5121, 2020.

  Subclasses handle specific benchmarks in the dataset. This class focuses on
  common functionality, e.g., retrieving data from Google Storage, etc.
  """

  _CUBERT_BUCKET = 'cubert'
  _VOCABULARY = '20200621_Python/github_python_minus_ethpy150open_deduplicated_vocabulary.txt'

  def __init__(
      self,
      stage_1_dir,
      configuration: plur_dataset.Configuration = plur_dataset.Configuration(),
      transformation_funcs=(),
      filter_funcs=(),
      user_defined_split_range=(),
      num_shards=1000,
      seed=0,
      deduplicate=False):
    super().__init__(
        dataset_name=self.dataset_name(),
        urls={},
        git_url={},
        dataset_description=self.dataset_description(),
        stage_1_dir=stage_1_dir,
        transformation_funcs=transformation_funcs,
        filter_funcs=filter_funcs,
        user_defined_split_range=user_defined_split_range,
        num_shards=num_shards,
        seed=seed,
        configuration=configuration,
        deduplicate=deduplicate)
    # Will be filled after download.
    self.dataset_extracted_dir = os.path.join(self.raw_data_dir, 'json')
    self.tokenizer = None
    self.subword_text_encoder = None

  @abc.abstractmethod
  def dataset_name(self) -> str:
    """Returns the CuBERT dataset name."""
    raise NotImplementedError()

  @abc.abstractmethod
  def dataset_description(self) -> str:
    """Returns the CuBERT dataset description."""
    raise NotImplementedError()

  @abc.abstractmethod
  def folder_path(self) -> str:
    """Returns the dataset folder path within the CuBERT bucket."""
    raise NotImplementedError()

  def make_dirs(self, directory: str) -> None:
    """Creates directory and all parents."""
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

  def fetch_file(self, blob: storage.Blob, local_path: str) -> None:
    """Places a Blob onto a local path."""
    blob.download_to_filename(local_path)

  def download_dataset(self):
    """Download the dataset using Google Cloud requests."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(self._CUBERT_BUCKET)
    vocab_blob = bucket.blob(blob_name=self._VOCABULARY)
    file_blobs = client.list_blobs(
        bucket_or_name=self._CUBERT_BUCKET,
        prefix=self.folder_path(),
        delimiter='/')
    all_blobs = list(itertools.chain(file_blobs, (vocab_blob,)))
    self.make_dirs(self.dataset_extracted_dir)
    with tqdm.tqdm(desc='Downloading', total=len(all_blobs)) as bar:
      for blob in all_blobs:
        bar.update(1)
        pathname = blob.name
        basename = os.path.basename(pathname)
        local_path = os.path.join(self.dataset_extracted_dir, basename)
        self.fetch_file(blob, local_path)

    self._initialize_tokenizers()

  def _initialize_tokenizers(self) -> None:
    """Initializes the CuBERT tokenizer and the WordPiece encoder."""
    self.local_vocabulary_path = os.path.join(
        self.dataset_extracted_dir, os.path.basename(self._VOCABULARY))
    self.tokenizer = python_tokenizer.PythonTokenizer()
    self.subword_text_encoder = text_encoder.SubwordTextEncoder(
        self.local_vocabulary_path)
    self.tokenizer.update_types_to_skip(
        (unified_tokenizer.TokenKind.COMMENT,
         unified_tokenizer.TokenKind.WHITESPACE))

  def get_all_raw_data_paths(self):
    """Get paths to all raw data."""
    glob_expression = os.path.join(self.dataset_extracted_dir, '*jsontxt-*')
    files = [f for f in glob.glob(glob_expression) if 'githubcommits' not in f]
    return files

  def raw_data_paths_to_raw_data_do_fn(self):
    """Returns a beam.DoFn subclass that reads the raw data."""
    return JsonExtractor(self.get_random_split,
                         bool(self.user_defined_split_range))

  @abc.abstractmethod
  def data_to_graph_to_output_example(
      self, data: Mapping[str, Any],
      max_graph_size: int,
      split: str) -> Optional[GraphToOutputExample]:
    """Performs the task-specific parsing/pruning of an example."""
    raise NotImplementedError()

  def raw_data_to_graph_to_output_example(
      self, raw_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert raw data to the unified GraphToOutputExample data structure.

    Code components are turned into a "graph" by tokenizing (using CuBERT's
    Python tokenizer), splitting each token into WordPiece subtokens (using
    CuBERT's released Python WordPiece vocabulary), trimming to the intended
    graph size, and appending and prepending the BERT delimiter tokens [CLS] and
    [SEP].

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
    max_graph_size = self.configuration.max_graph_sizes[split]

    data = raw_data['data']
    graph_to_output_example = self.data_to_graph_to_output_example(
        data=data, max_graph_size=max_graph_size, split=split)

    if graph_to_output_example:
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

  def single_classification_data_dictionary_to_graph_to_output_example(
      self, data: Mapping[str, Any], classes: AbstractSet[str],
      max_graph_size: int,
      split: str) -> GraphToOutputExample:
    """Convenience method mapping classification dictionaries to examples."""
    function = data['function']
    label = data['label']
    del split  # Unused.
    assert label in classes
    provenance = data['info']

    graph_to_output_example = GraphToOutputExample()

    # The input graph nodes are the source code tokens. We don't filter any
    # examples based on size. Instead, we trim the suffix of the token sequence.
    # Note that we trim so that the number of tokens plus the two delimiters
    # is at most `max_graph_size`.
    sentences: List[List[str]] = (
        code_to_subtokenized_sentences.code_to_cubert_sentences(
            function, self.tokenizer, self.subword_text_encoder))
    pruned_tokens = sum(sentences, [])[:max_graph_size-2]
    delimited_tokens = itertools.chain(('[CLS]',), pruned_tokens, ('[SEP]',))
    for index, token in enumerate(delimited_tokens):
      graph_to_output_example.add_node(
          node_id=index, node_type='TOKEN', node_label=token)

    graph_to_output_example.add_class_output(label)
    graph_to_output_example.set_provenance(provenance)

    return graph_to_output_example


class JsonExtractor(beam.DoFn):
  """Class to read the CuBERT Dataset json files."""

  def __init__(self, random_split_fn, use_random_split: bool) -> None:
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
    if 'dev.jsontxt' in file_path:
      return constants.VALIDATION_SPLIT_NAME
    elif 'train.jsontxt' in file_path:
      return constants.TRAIN_SPLIT_NAME
    elif 'eval.jsontxt' in file_path:
      return constants.TEST_SPLIT_NAME
    else:
      raise ValueError(f'Cannot detect the split of filename {file_path}.')

  def process(self, file_path: str) -> Iterable[Mapping[str, Any]]:
    """Function to read each json file.

    Args:
      file_path: Path to a raw data file.

    Yields:
      A dictionary with 'split' and 'data' as keys. The value of the 'split'
      field is the split (train/valid/test) that the data belongs to. The value
      of the 'data' is the parsed raw json data.
    """
    split = (
        self.random_split_fn()
        if self.use_random_split else self._get_split(file_path))
    with open(file_path) as f:
      for line in f:
        json_data = json.loads(line)
        yield {
            'split': split,
            'data': json_data
        }
