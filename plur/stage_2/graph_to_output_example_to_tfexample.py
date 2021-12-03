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

"""Class for converting PlurDataset to TfExample, and stored in TfRecords.
"""
import collections
import functools
import json
import os
import shutil

from absl import logging
import apache_beam as beam
from plur.utils import constants
from plur.utils import tfexample_utils
from plur.utils import util
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError
import tqdm


class GraphToOutputExampleToTfexample():
  """Main class for converting GraphToOutputExample to TfExample."""

  def __init__(self,
               stage_1_dir,
               stage_2_dir,
               dataset_name,
               train_transformation_funcs=(),
               train_filter_funcs=(),
               validation_transformation_funcs=(),
               validation_filter_funcs=(),
               test_transformation_funcs=(),
               test_filter_funcs=(),
               max_node_type_vocab_size=10000,
               max_node_label_vocab_size=10000,
               max_edge_type_vocab_size=10000,
               max_output_token_vocab_size=10000,
               num_shards=1000,
               copy_vocab_dir='',
               copy_metadata_file=''):
    """GraphToOutputExampleToTfexample constructor arguments.

    Compared to plur_dataset.py in stage_1, here the transformation and filter
    functions can be applied to different split of data when loading the
    GraphToOutputExample. For example it is often the case that training data
    will be filtered, but the testing data should remain the same. For all
    data splits, transformation_funcs will be applied before filter_funcs. And
    all transformation and filter function are guaranteed to be applied
    in order.

    Args:
      stage_1_dir: Path to stage 1.
      stage_2_dir: Path to stage 2.
      dataset_name: The name of the dataset.
      train_transformation_funcs: Transformation functions that are applied on
        the train split of GraphToOutputExample.
      train_filter_funcs: Filter functions that are applied on the train split
        of GraphToOutputExample.
      validation_transformation_funcs: Transformation functions that are applied
        on the validation split of GraphToOutputExample.
      validation_filter_funcs: Filter functions that are applied on the
        validation split of GraphToOutputExample.
      test_transformation_funcs: Transformation functions that are applied on
        the test split of GraphToOutputExample.
      test_filter_funcs: Filter functions that are applied on the
        test split of GraphToOutputExample.
      max_node_type_vocab_size: The maximum vocabulary size of node type.
      max_node_label_vocab_size: The maximum vocabulary size of node label.
      max_edge_type_vocab_size: The maximum vocabulary size of edge type.
      max_output_token_vocab_size: The maximum vocabulary size of output token.
      num_shards: Number of shards used to store the TFRecords.
      copy_vocab_dir: Copy the vocabulary files in this directory. When set,
        we ignore the vocabulary sizes and don't build the vocabulary, instead
        we just copy the files in it. This can be set to ensure that two PLUR
        datasets have the same vocabulary.
      copy_metadata_file: Copy the stage 2 metadata file. When set, we simply
        copy the metadata file. We add a warning in the metadata file that it is
        copied. This can be set to ensure two PLUR datasets have the same
        padding values (max_num_nodes/max_num_edges/max_num_output/
        max_edge_types).
        Only set this when two PLUR dataset are generated in the same way, ie.
        all the filter and transformation functions are the same.
    """
    self.stage_1_dir = stage_1_dir
    self.graph_to_output_example_dir = os.path.join(
        self.stage_1_dir, constants.GRAPH_TO_OUTPUT_EXAMPLE_DIRNAME)
    self.stage_2_dir = stage_2_dir
    self.vocab_dir = os.path.join(self.stage_2_dir,
                                  constants.VOCAB_FILES_DIRNAME)
    self.node_type_vocab_file = os.path.join(self.vocab_dir,
                                             constants.NODE_TYPE_VOCAB_FILENAME)
    self.node_label_vocab_file = os.path.join(
        self.vocab_dir, constants.NODE_LABEL_VOCAB_FILENAME)
    self.edge_type_vocab_file = os.path.join(self.vocab_dir,
                                             constants.EDGE_TYPE_VOCAB_FILENAME)
    self.output_token_vocab_file = os.path.join(
        self.vocab_dir, constants.OUTPUT_TOKEN_VOCAB_FILENAME)
    self.tfrecord_dir = os.path.join(self.stage_2_dir,
                                     constants.TFRECORD_DIRNAME)
    self.train_tfrecord_dir = os.path.join(self.tfrecord_dir,
                                           constants.TRAIN_SPLIT_NAME)
    self.validation_tfrecord_dir = os.path.join(self.tfrecord_dir,
                                                constants.VALIDATION_SPLIT_NAME)
    self.test_tfrecord_dir = os.path.join(self.tfrecord_dir,
                                          constants.TEST_SPLIT_NAME)
    self.metadata_file = os.path.join(self.stage_2_dir,
                                      constants.METADATA_FILENAME)
    self.dataset_name = dataset_name
    self.transformation_funcs = {
        constants.TRAIN_SPLIT_NAME: train_transformation_funcs,
        constants.VALIDATION_SPLIT_NAME: validation_transformation_funcs,
        constants.TEST_SPLIT_NAME: test_transformation_funcs
    }
    self.filter_funcs = {
        constants.TRAIN_SPLIT_NAME: train_filter_funcs,
        constants.VALIDATION_SPLIT_NAME: validation_filter_funcs,
        constants.TEST_SPLIT_NAME: test_filter_funcs
    }
    self.max_node_type_vocab_size = max_node_type_vocab_size
    self.max_node_label_vocab_size = max_node_label_vocab_size
    self.max_edge_type_vocab_size = max_edge_type_vocab_size
    self.max_output_token_vocab_size = max_output_token_vocab_size
    self.num_shards = num_shards
    self.copy_vocab_dir = copy_vocab_dir
    self.copy_metadata_file = copy_metadata_file

  def stage_2_mkdirs(self):
    logging.info('Creating all necessary stage 2 directories for %s.',
                 self.dataset_name)
    os.makedirs(self.vocab_dir, exist_ok=True)
    os.makedirs(self.train_tfrecord_dir, exist_ok=True)
    os.makedirs(self.validation_tfrecord_dir, exist_ok=True)
    os.makedirs(self.test_tfrecord_dir, exist_ok=True)

  def exists_tfrecords(self):
    """Check existing TFRecords.

    We check the existing TFRecords stored on disk, and ask the user if they
    want to regenerate the TFRecords.

    Returns:
      True if we found TFRecords and the user wants to keep it. False if the
      user wants to (re)generate TFRecords.
    """
    existing_training_tfrecords = [
        os.path.join(self.train_tfrecord_dir, filename)
        for filename in os.listdir(self.train_tfrecord_dir)
        if os.path.splitext(filename)[1] == '.tfrecord'
    ]
    existing_validation_tfrecords = [
        os.path.join(self.validation_tfrecord_dir, filename)
        for filename in os.listdir(self.validation_tfrecord_dir)
        if os.path.splitext(filename)[1] == '.tfrecord'
    ]
    existing_test_tfrecords = [
        os.path.join(self.test_tfrecord_dir, filename)
        for filename in os.listdir(self.test_tfrecord_dir)
        if os.path.splitext(filename)[1] == '.tfrecord'
    ]
    existing_tfrecords = (
        existing_training_tfrecords + existing_validation_tfrecords +
        existing_test_tfrecords
    )
    if existing_tfrecords:
      regenerate_tfexample = input(
          '{} already exists and contains TFRecords, delete it and regenerate '
          'TfRecords?(y/n): '.format(self.tfrecord_dir))
      if regenerate_tfexample.lower() == 'y':
        for file_path in tqdm.tqdm(existing_tfrecords, desc='Deleting'):
          os.remove(file_path)
      else:
        return True
    return False

  def read_vocab_files(self):
    """Reading the vocabulary files."""
    logging.info('Reading all vocabulary files.')
    node_type_vocab_dict = {}
    with open(self.node_type_vocab_file) as f:
      lines = f.read().splitlines()
    for line in lines:
      node_type_vocab_dict[line] = len(node_type_vocab_dict)

    node_label_vocab_dict = {}
    with open(self.node_label_vocab_file) as f:
      lines = f.read().splitlines()
    for line in lines:
      node_label_vocab_dict[line] = len(node_label_vocab_dict)

    edge_type_vocab_dict = {}
    with open(self.edge_type_vocab_file) as f:
      lines = f.read().splitlines()
    for line in lines:
      edge_type_vocab_dict[line] = len(edge_type_vocab_dict)

    output_token_vocab_dict = {}
    with open(self.output_token_vocab_file) as f:
      lines = f.read().splitlines()
    for line in lines:
      output_token_vocab_dict[line] = len(output_token_vocab_dict)

    return (node_type_vocab_dict, node_label_vocab_dict, edge_type_vocab_dict,
            output_token_vocab_dict)

  def convert_and_write_tfexample(
      self, p, additional_train_transformation_funcs=(),
      additional_train_filter_funcs=(),
      additional_valid_transformation_funcs=(),
      additional_valid_filter_funcs=(),
      additional_test_transformation_funcs=(),
      additional_test_filter_funcs=()):
    """Pipeline that converts GraphToOutputExample to TFExample.

    We add the option to add additional transformation and filter functions.
    It is for transformation or filter functions that need the vocabulary as
    input.

    Args:
      p: The beam pipeline root.
      additional_train_transformation_funcs: A tuple of additional
        transformation functions to be applied on the training data.
      additional_train_filter_funcs: A tuple of additional filter functions to
        be applied on the training data.
      additional_valid_transformation_funcs: A tuple of additional
        transformation functions to be applied on the validation data.
      additional_valid_filter_funcs: A tuple of additional filter functions to
        be applied on the validation data.
      additional_test_transformation_funcs: A tuple of additional
        transformation functions to be applied on the testing data.
      additional_test_filter_funcs: A tuple of additional filter functions to be
        applied on the testing data.
    """
    self.transformation_funcs[constants.TRAIN_SPLIT_NAME] += (
        additional_train_transformation_funcs)
    self.transformation_funcs[constants.VALIDATION_SPLIT_NAME] += (
        additional_valid_transformation_funcs)
    self.transformation_funcs[constants.TEST_SPLIT_NAME] += (
        additional_test_transformation_funcs)

    self.filter_funcs[constants.TRAIN_SPLIT_NAME] += (
        additional_train_filter_funcs)
    self.filter_funcs[constants.VALIDATION_SPLIT_NAME] += (
        additional_valid_filter_funcs)
    self.filter_funcs[constants.TEST_SPLIT_NAME] += (
        additional_test_filter_funcs)

    (node_type_vocab_dict, node_label_vocab_dict, edge_type_vocab_dict,
     output_token_vocab_dict) = self.read_vocab_files()
    get_tfexample_feature_fn = functools.partial(
        tfexample_utils.get_tfexample_feature,
        node_type_vocab_dict=node_type_vocab_dict,
        node_label_vocab_dict=node_label_vocab_dict,
        edge_type_vocab_dict=edge_type_vocab_dict,
        output_token_vocab_dict=output_token_vocab_dict)

    metadata = (
        p
        | f'Create metadata {self.dataset_name}' >> beam.Create([{}])
    )

    for split in [
        constants.TRAIN_SPLIT_NAME, constants.VALIDATION_SPLIT_NAME,
        constants.TEST_SPLIT_NAME
    ]:
      split_graph_to_output_example_dir = os.path.join(
          self.graph_to_output_example_dir, split)
      file_pattern = os.path.join(split_graph_to_output_example_dir, '*.json')
      graph_to_output_examples = (
          p
          | 'Read {} GraphToOutputExample for TFExample {}'.format(
              split, self.dataset_name) >> beam.io.ReadFromText(file_pattern)
          | 'Parse {} GraphToOutputExample as json for TFExample {}'.format(
              split, self.dataset_name) >> beam.Map(json.loads)
          | 'Reconstruct {} GraphToOutputExample for TFExample {}'.format(
              split, self.dataset_name) >> beam.ParDo(
                  ParseAndProcessGraphToOutputExample(
                      self.transformation_funcs[split],
                      self.filter_funcs[split]))
          | 'Reshuffle {} GraphToOutputExample for TFExample {}'.format(
              split, self.dataset_name) >> beam.Reshuffle())

      tfexample_feature_dicts = (
          graph_to_output_examples
          | '{} GraphToOutputExample to tfexample feature {}'.format(
              split, self.dataset_name) >> beam.Map(get_tfexample_feature_fn))

      if split == constants.TRAIN_SPLIT_NAME:
        # Filter impossible examples in the training data to avoid infinite
        # loss. This should not be done on the test split.
        filter_impossible_tfexample_fn = functools.partial(
            util.filter_impossible_tfexample,
            output_token_vocab_dict=output_token_vocab_dict)
        tfexample_feature_dicts = (
            tfexample_feature_dicts
            |
            'Filter impossible {} tfexample {}'.format(split, self.dataset_name)
            >> beam.Filter(filter_impossible_tfexample_fn))

      metadata = self.update_dataset_metadata(
          metadata, split, tfexample_feature_dicts)

      file_path_prefix = os.path.join(self.tfrecord_dir, split,
                                      self.dataset_name)
      _ = (
          tfexample_feature_dicts
          | 'Serialize {} tfexample feature {}'.format(split, self.dataset_name)
          >> beam.Map(tfexample_utils.serialize_tfexample_feature)
          | 'Writing {} tfexample to disk {}'.format(
              split, self.dataset_name) >> beam.io.WriteToTFRecord(
                  file_path_prefix,
                  file_name_suffix='.tfrecord',
                  num_shards=self.num_shards))

    # Don't save the metadata when we copy from an existing metadata file.
    if not self.copy_metadata_file:
      _ = (
          metadata
          |
          f'Write metadata to file {self.dataset_name}' >> beam.io.WriteToText(
              self.metadata_file,
              num_shards=1,
              shard_name_template='',
              coder=util.JsonCoder()))

  def update_dataset_metadata(self, metadata, split, tfexample_feature_dicts):
    """Update the dataset metadata.

    We update the metadata dictionary (Pcollection), using the tfexample feature
    dictionaries and the split(train/validation/test). We keep count of metadata
    like maximum number of nodes, maximum number of edges etc.

    Args:
      metadata: A Pcollection containing the metadata dict.
      split: The split that tf_example_feature belongs.
      tfexample_feature_dicts: A Pcollection containing the tfexample feature
        dictionaries.
    Returns:
      The updated metadata Pcollection.
    """
    tfexample_count = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        | 'Count {} tfexample {}'.format(split, self.dataset_name) >>
        beam.combiners.Count.Globally()
    )
    fieldname = split + '_tfexample' + '_count'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, tfexample_count, self.dataset_name)

    tfexample_max_num_nodes = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        | 'Get node GUIDs in {} tfexample {}'.format(split, self.dataset_name)
        >> beam.Map(lambda x: x[constants.KEY_NODE_GUIDS])
        | 'Return len ofnode GUIDs in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(len)
        | 'Get maximum number of nodes in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.CombineGlobally(
                lambda values: max(values, default=0)))
    fieldname = split + '_tfexample' + '_max_num_nodes'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, tfexample_max_num_nodes, self.dataset_name)

    tfexample_max_num_edges = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        | 'Get edge srcs in {} tfexample {}'.format(split, self.dataset_name) >>
        beam.Map(lambda x: x[constants.KEY_EDGE_SOURCE_INDICES])
        | 'Return len of edge srcs in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(len)
        | 'Get maximum number of edges in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.CombineGlobally(
                lambda values: max(values, default=0)))
    fieldname = split + '_tfexample' + '_max_num_edges'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, tfexample_max_num_edges, self.dataset_name)

    tfexample_max_num_output = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        | 'Get output token ids in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(
                lambda x: x[constants.KEY_OUTPUT_TOKEN_IDS])
        | 'Return len of output token ids in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(len)
        | 'Get maximum number of output token in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.CombineGlobally(
                lambda values: max(values, default=0)))
    fieldname = split + '_tfexample' + '_max_num_output'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, tfexample_max_num_output, self.dataset_name)

    tfexample_max_num_edge_types = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        |
        'Get edge type ids in {} tfexample {}'.format(split, self.dataset_name)
        >> beam.Map(lambda x: x[constants.KEY_EDGE_TYPE_IDS])
        | 'Get all unique edge type ids in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(lambda x: list(set(x)))
        | 'Return len of unique edge type ids in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.Map(len)
        | 'Get maximum number of edge types in {} tfexample {}'.format(
            split, self.dataset_name) >> beam.CombineGlobally(
                lambda values: max(values, default=0)))
    fieldname = split + '_tfexample' + '_max_num_edge_types'
    metadata = util.add_field_to_metadata(
        metadata, fieldname, tfexample_max_num_edge_types, self.dataset_name)

    for max_num_nodes in [128, 256, 512, 1024, 2048, 4096, 8192]:
      tfexample_count = self._count_tfexample_with_max_num_nodes(
          tfexample_feature_dicts, split, max_num_nodes)
      fieldname = split + '_tfexample' + 'max_num_nodes_{}'.format(
          max_num_nodes)
      metadata = util.add_field_to_metadata(
          metadata, fieldname, tfexample_count, self.dataset_name)

    return metadata

  def exists_vocab_files(self):
    """Check existing vocab files.

    We check the existing vocab files stored on disk, and ask the user if they
    want to regenerate the vocab files.

    Returns:
      True if we found vocab files and user want to keep it. False if the user
      want to (re)generate vocab files.
    """
    existing_vocab_files = []
    for vocab_file in [self.node_type_vocab_file, self.node_label_vocab_file,
                       self.edge_type_vocab_file, self.output_token_vocab_file]:
      if os.path.exists(vocab_file):
        existing_vocab_files.append(vocab_file)
    if existing_vocab_files:
      regenerate_vocab_files = input(
          '{} already exists and contains vocab files, delete it and regenerate '
          'vocab files?(y/n): '.format(self.vocab_dir))
      if regenerate_vocab_files.lower() == 'y':
        for file_path in tqdm.tqdm(existing_vocab_files, desc='Deleting'):
          os.remove(file_path)
      else:
        return True
    return False

  def build_and_save_vocab(
      self, p, node_type_reserved_tokens=constants.RESERVED_TOKENS,
      node_label_reserved_tokens=constants.RESERVED_TOKENS,
      edge_type_reserved_tokens=constants.RESERVED_TOKENS,
      output_token_reserved_tokens=constants.RESERVED_TOKENS):
    """Build and save vocab files.

    We compute vocabulary for the node types, node labels, edge types and
    output tokens.

    Args:
      p: The root of beam pipeline.
      node_type_reserved_tokens: A list of reserved tokens for node types.
      node_label_reserved_tokens: A list of reserved tokens for node labels.
      edge_type_reserved_tokens: A list of reserved tokens for edge types.
      output_token_reserved_tokens: A list of reserved tokens for output tokens.
    """
    train_split_graph_to_output_example_dir = os.path.join(
        self.graph_to_output_example_dir, constants.TRAIN_SPLIT_NAME)
    file_pattern = os.path.join(
        train_split_graph_to_output_example_dir, '*.json')
    graph_to_output_examples = (
        p
        | f'Read train GraphToOutputExample for vocab {self.dataset_name}' >>
        beam.io.ReadFromText(file_pattern)
        | f'Parse it as json for vocab {self.dataset_name}' >> beam.Map(
            json.loads)
        | f'Reconstruct GraphToOutputExample for vocab {self.dataset_name}' >>
        beam.ParDo(
            ParseAndProcessGraphToOutputExample(
                self.transformation_funcs[constants.TRAIN_SPLIT_NAME],
                self.filter_funcs[constants.TRAIN_SPLIT_NAME]))
        | f'Reshuffle GraphToOutputExample for vocab {self.dataset_name}' >>
        beam.Reshuffle())

    self._build_and_save_individual_vocab(graph_to_output_examples,
                                          'get_node_types',
                                          self.max_node_type_vocab_size,
                                          node_type_reserved_tokens,
                                          self.node_type_vocab_file)
    self._build_and_save_individual_vocab(graph_to_output_examples,
                                          'get_node_labels',
                                          self.max_node_label_vocab_size,
                                          node_label_reserved_tokens,
                                          self.node_label_vocab_file)
    self._build_and_save_individual_vocab(graph_to_output_examples,
                                          'get_edge_types',
                                          self.max_edge_type_vocab_size,
                                          edge_type_reserved_tokens,
                                          self.edge_type_vocab_file)
    self._build_and_save_individual_vocab(graph_to_output_examples,
                                          'get_output_as_tokens',
                                          self.max_output_token_vocab_size,
                                          output_token_reserved_tokens,
                                          self.output_token_vocab_file)

  def _build_and_save_individual_vocab(self, graph_to_output_examples,
                                       method_name, vocab_size,
                                       reserved_tokens, vocab_filename):
    """Build and save a single vocabulary.

    Args:
      graph_to_output_examples: A Pcollection of GraphToOutputExample.
      method_name: The method name of GraphToOutputExample that we want
        to generate the vocabulary. For example it can be 'get_node_types'.
      vocab_size: The maximum vocab size.
      reserved_tokens: The reserved tokens that will always be included in the
        vocabulary.
      vocab_filename: The filename to save the vocabulary.
    """
    _ = (
        graph_to_output_examples
        |
        '{} from GraphToOutputExample {}'.format(method_name, self.dataset_name)
        >> beam.FlatMap(lambda x: getattr(x, method_name)())
        | 'Reshuffle after {} from GraphToOutputExample {}'.format(
            method_name, self.dataset_name) >> beam.Reshuffle()
        | 'Building {} vocabulary {}'.format(method_name, self.dataset_name) >>
        beam.CombineGlobally(VocabCombinerFn(vocab_size, reserved_tokens))
        | 'Flatten the {} vocabulary {}'.format(
            method_name, self.dataset_name) >> beam.FlatMap(lambda x: x)
        | 'Write {} vocabulary to disk {}'.format(
            method_name, self.dataset_name) >> beam.io.WriteToText(
                vocab_filename, num_shards=1, shard_name_template=''))

  def _count_tfexample_with_max_num_nodes(self, tfexample_feature_dicts, split,
                                          max_num_nodes):
    """Count number of tfexample with less than a certain number of nodes.

    Args:
     tfexample_feature_dicts: A Pcollection containing the tfexample
       dictionaries.
     split: The split (train/valid/test) that tfexample_feature_dicts belongs
       to.
     max_num_nodes: The maximum number of nodes, we count number of tfexample
       with less or equal than max_num_nodes nodes.
    Returns:
      Number of tfexamples with less or equal than max_num_nodes node, returns
      as Pcollection which can be used as a side input.
    """
    tfexample_count = beam.pvalue.AsSingleton(
        tfexample_feature_dicts
        | 'Get nodes from {} tfexample for counting <= {} nodes {}'.format(
            split, max_num_nodes, self.dataset_name) >> beam.Map(
                lambda x: x[constants.KEY_NODE_TYPE_IDS])
        | 'Filter {} tfexample with <= {} nodes {}'.format(
            split, max_num_nodes, self.dataset_name) >> beam.Filter(
                lambda x: len(x) <= max_num_nodes)
        | 'Count {} tfexample with <= {} nodes {}'.format(
            split, max_num_nodes,
            self.dataset_name) >> beam.combiners.Count.Globally())
    return tfexample_count

  def copy_vocab_files(self, copy_fn):
    """Copy vocabulary files from self.copy_vocab_dir.

    Args:
      copy_fn: A function that takes two files 'src' and 'dst', and calling
        copy_fn(src, dst) copies the 'src' file to 'dst'.
    """
    source_node_type_vocab_file = os.path.join(
        self.copy_vocab_dir, constants.NODE_TYPE_VOCAB_FILENAME)
    source_node_label_vocab_file = os.path.join(
        self.copy_vocab_dir, constants.NODE_LABEL_VOCAB_FILENAME)
    source_edge_type_vocab_file = os.path.join(
        self.copy_vocab_dir, constants.EDGE_TYPE_VOCAB_FILENAME)
    source_output_token_vocab_file = os.path.join(
        self.copy_vocab_dir, constants.OUTPUT_TOKEN_VOCAB_FILENAME)

    copy_fn(source_node_type_vocab_file, self.node_type_vocab_file)
    copy_fn(source_node_label_vocab_file, self.node_label_vocab_file)
    copy_fn(source_edge_type_vocab_file, self.edge_type_vocab_file)
    copy_fn(source_output_token_vocab_file, self.output_token_vocab_file)

  def copy_metadata(self, open_fn):
    """Copy the metadata from self.copy_metadata_file.

    Args:
      open_fn: A function that opens the file and returns a file object. The
        input arguments should be the same as the built-in 'open' function.
    """
    with open_fn(self.copy_metadata_file) as f:
      metadata = json.load(f)
    metadata['WARNING'] = (
        f'This metadata is copied from {self.copy_metadata_file}. Therefore '
        'all numbers might be inaccurate.')
    with open_fn(self.metadata_file, 'w') as f:
      json.dump(metadata, f)

  def run_pipeline(self):
    """Run the stage 2 pipeline."""
    logging.info('Running stage 2 pipeline.')
    if self.copy_vocab_dir:
      self.copy_vocab_files(shutil.copyfile)
    else:
      if not self.exists_vocab_files():
        with beam.Pipeline() as p:
          self.build_and_save_vocab(p)

    if not self.exists_tfrecords():
      if self.copy_metadata_file:
        self.copy_metadata(open)
      with beam.Pipeline() as p:
        self.convert_and_write_tfexample(p)


class ParseAndProcessGraphToOutputExample(beam.DoFn):
  """Class to parse GraphToOutputExample in a beam pipeline."""

  def __init__(self, transformation_funcs=(), filter_funcs=()):
    self.transformation_funcs = transformation_funcs
    self.filter_funcs = filter_funcs

  def process(self, element):
    graph_to_output_example = GraphToOutputExample()
    graph_to_output_example.set_data(element)
    for transformation_fn in self.transformation_funcs:
      graph_to_output_example = transformation_fn(graph_to_output_example)

    for filter_fn in self.filter_funcs:
      if not filter_fn(graph_to_output_example):
        return

    if not graph_to_output_example.check_if_valid():
      raise GraphToOutputExampleNotValidError(
          'Invalid GraphToOutputExample found {}'.format(
              graph_to_output_example))

    yield graph_to_output_example


class VocabCombinerFn(beam.CombineFn):
  """Class to generate the vocabulary in a beam pipeline."""

  def __init__(self, max_vocab_size, reserved_tokens):
    self.max_vocab_size = max_vocab_size
    self.reserved_tokens = reserved_tokens

  def create_accumulator(self):
    """Use collections.Counter for counting token frequency."""
    return collections.Counter()

  def add_input(self, accumulator, token):
    """Update the collections.Counter."""
    accumulator.update([token])
    return accumulator

  def merge_accumulators(self, accumulators):
    """Merge collections.Counter by adding them."""
    merged = collections.Counter()
    for acc in accumulators:
      merged += acc
    return merged

  def extract_output(self, accumulator):
    """Convert collections.Counter to a list of vocabulary.

    We use collections.Counter.most_common to find the top k most common token.
    However, we always include reserved_tokens, and then adding top k most
    common token in order until we reached max_vocab_size.

    Args:
      accumulator: A collections.Counter.
    Returns:
      A list of tokens which represents the vocabulary.
    """
    most_common_vocab = accumulator.most_common(self.max_vocab_size)
    vocabulary = self.reserved_tokens
    for token, _ in most_common_vocab:
      if len(vocabulary) >= self.max_vocab_size:
        break
      elif token in self.reserved_tokens:
        continue
      vocabulary.append(token)
    return vocabulary
