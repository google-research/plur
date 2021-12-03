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

"""PLUR data loader using tf.data.TFRecordDataset.
"""
import glob
import json
import os

from absl import logging
import jax

from plur.utils import constants
from plur.utils import tfexample_utils
import tensorflow as tf


class PlurDataLoader:
  """Main class that converts tfexamples to numpy arrays.

  This class loads TFRecords created by stage 2 of PLUR data generation, and
  transforms them so that they can be used as input to a machine learning model.
  First, we read the vocabulary files to get the OOV and PAD token id. Then,
  we read dataset_metadata.json file to get the padding lengths such as maximum
  number of outputs. Last, we load the train/valid/test TFRecords. The tensors
  stored in TFRecords are combined and padded, then returned as numpy arrays.
  PlurDataLoader uses several optimizations implemented in tf.data.Dataset,
  and returns the batched data ready to be trained/evaluated on.

  Assumes use of JAX to determine sharding.
  """

  def __init__(self,
               stage_2_dir,
               split,
               batch_size,
               repeat_count,
               drop_remainder,
               max_data_count=-1,
               create_dataset=True,
               shuffle_batch_count: int = 1,
               shard: int = 0,
               shard_count: int = 1):
    """The PlurDataLoader constructor.

    Args:
      stage_2_dir: Path to stage 2 directory created by PLUR data generation.
      split: The data split that we load, must be one of
        [constants.TRAIN_SPLIT_NAME, constants.VALIDATION_SPLIT_NAME,
        constants.TEST_SPLIT_NAME].
      batch_size: The batch size.
      repeat_count: The number of times to repeat the dataset. If set to -1,
        repeat indefinitely.
      drop_remainder: Boolean indicating if it will drop the last batch if it
       is smaller than batch_size.
      max_data_count: Maximum number of examples we take from the dataset (or
        the current shard, if applicable). No assumptions should be made about
        this being a deterministic subset.
      create_dataset: Boolean indicating if we will call create_dataset()
        function. It can be set to false to only read the padding values.
      shuffle_batch_count: An integer that specifies the size of the shuffle
        buffer in number of batches. A value of 1 or smaller implies no
        shuffling.
      shard: Along with `shard_count`, specifies a subset of the host's files
        to restrict to.
      shard_count: A value greater than one indicates loading should be
        restricted to a subset of the host's files.
    """
    self.stage_2_dir = stage_2_dir
    self.split = split
    self.batch_size = batch_size
    self.repeat_count = repeat_count
    self.max_data_count = max_data_count
    self.drop_remainder = drop_remainder
    self._shuffle_batch_count = shuffle_batch_count
    self._shard = shard
    self._shard_count = shard_count

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
    self.split_tfrecord_dir = os.path.join(self.tfrecord_dir, self.split)
    self.metadata_file = os.path.join(self.stage_2_dir,
                                      constants.METADATA_FILENAME)

    self.read_padding_ids()
    self.read_padding_values()
    if create_dataset:
      self.dataset, self.dataset_iter = self.create_dataset()

  def read_padding_ids(self):
    """Read padding ids from vocabulary files."""
    # All vocabulary id start with 0, starting from the top to the bottom of the
    # vocab files.
    node_type_vocab_dict = {}
    with open(self.node_type_vocab_file) as f:
      node_type_vocab_lines = f.read().splitlines()
    for idx, line in enumerate(node_type_vocab_lines):
      node_type_vocab_dict[line] = idx

    node_label_vocab_dict = {}
    with open(self.node_label_vocab_file) as f:
      node_label_vocab_lines = f.read().splitlines()
    for idx, line in enumerate(node_label_vocab_lines):
      node_label_vocab_dict[line] = idx

    edge_type_vocab_dict = {}
    with open(self.edge_type_vocab_file) as f:
      edge_type_vocab_lines = f.read().splitlines()
    for idx, line in enumerate(edge_type_vocab_lines):
      edge_type_vocab_dict[line] = idx

    output_token_vocab_dict = {}
    with open(self.output_token_vocab_file) as f:
      output_token_vocab_lines = f.read().splitlines()
    for idx, line in enumerate(output_token_vocab_lines):
      output_token_vocab_dict[line] = idx

    self.node_type_vocab_size = len(node_type_vocab_dict)
    self.node_label_vocab_size = len(node_label_vocab_dict)
    self.edge_type_vocab_size = len(edge_type_vocab_dict)
    self.output_token_vocab_size = len(output_token_vocab_dict)

    self.node_type_oov_id = node_type_vocab_dict[constants.OOV_TOKEN]
    self.node_label_oov_id = node_label_vocab_dict[constants.OOV_TOKEN]
    self.edge_type_oov_id = edge_type_vocab_dict[constants.OOV_TOKEN]
    self.output_token_oov_id = output_token_vocab_dict[constants.OOV_TOKEN]

    self.node_type_pad_id = node_type_vocab_dict[constants.PAD_TOKEN]
    self.node_label_pad_id = node_label_vocab_dict[constants.PAD_TOKEN]
    self.edge_type_pad_id = edge_type_vocab_dict[constants.PAD_TOKEN]
    self.output_token_pad_id = output_token_vocab_dict[constants.PAD_TOKEN]

    self.node_type_vocab_dict = node_type_vocab_dict
    self.node_label_vocab_dict = node_label_vocab_dict
    self.edge_type_vocab_dict = edge_type_vocab_dict
    self.output_token_vocab_dict = output_token_vocab_dict

  def read_padding_values(self):
    """Read padding values from metadata and vocabulary."""
    with open(self.metadata_file) as f:
      dataset_metadata = json.load(f)

    # Maximum number of nodes and maximum number of outputs are the maximum
    # in all splits (train/valid/test).
    max_num_nodes = 0
    max_num_output = 0
    max_num_subtokens = 0
    for split in [
        constants.TRAIN_SPLIT_NAME, constants.VALIDATION_SPLIT_NAME,
        constants.TEST_SPLIT_NAME
    ]:
      max_num_nodes = max(max_num_nodes,
                          dataset_metadata[split+'_tfexample_max_num_nodes'])
      max_num_output = max(max_num_output,
                           dataset_metadata[split+'_tfexample_max_num_output'])
      # In case there's no subtoken metadata use 1 by default.
      max_num_subtokens = max(
          max_num_subtokens,
          dataset_metadata.get(split + '_tfexample_max_num_subtokens', 1))

    # max_num_edge_types is set to the edge type vocab size, instead of
    # the value of max_num_edge_types in the metadata_file. Because in the edge
    # type vocabulary, OOV and PAD tokens are always added. The problem is that
    # it is very rare that we have OOV or PAD edge types, therefore most of the
    # time, max_num_edge_types in the metadata_file does not count OOV or PAD
    # edge type because they don't exist, but they are valid edge types.
    self.max_num_edge_types = self.edge_type_vocab_size
    self.max_num_nodes = max_num_nodes
    self.max_num_output = max_num_output
    self.max_num_subtokens = max_num_subtokens

  def _parse_tfexample_and_add_padding(self, tfexample):
    """Parse TFExample and add padding, returns them as numpy arrays.

    We use native tf transformation here so that we can call tf.data.Dataset.map
    with this function. This is because we rely on AutoGraph to convert the
    code into an equivalent graph computation. But AutoGraph can only transform
    some Python code, using pure TF transformations avoids this problem. We also
    add paddings so that we can just call tf.data.Dataset.Batch, since the
    batch function requires that all tensors have the same shape.

    Args:
      tfexample: A TFExample generated by stage 2 of PLUR data generation. See
      utils/tfexamples_utils.py on what it contains.
    Returns:
      A dictionary containing the following values:
        target_token_ids: Tensor of target token ids, shape = (max_num_output).
        copy_indices: Tensor of valid copies, shape =
          (max_num_output, max_num_nodes). copy_indices[i, j] == 1 means that we
          can copy from j'th input node to i'th output token.
        pointer_indices: Tensor of valid pointers, shape =
          (max_num_output, max_num_nodes). pointer_indices[i, j] == 1, means
          that we can point from i'th output to j'th input node.
        node_id_sequences: Tensor of input node subtoken ids. shape =
          (max_num_nodes, max_num_subtokens_per_node).
        node_types: Tensor of input node types, shape = (max_num_nodes).
        edge_indicators: Tensor of all edges, shape =
          (max_num_edge_types, max_num_nodes, max_num_nodes).
          edge_indicators[k, i, j] == 1 means that we have a edge of edge type
          k from i'th input node to j'th input node.
        node_text_sequences: Tensor of input node texts, shape =
          (max_num_nodes).
        target_texts: Tensor of output tokens, shape = (max_num_output).

      All tensors are padded.
    """
    tfexample_dict = tf.io.parse_single_example(
        tfexample, tfexample_utils.FEATURE_DESCRIPTION)

    # Pad KEY_OUTPUT_TOKEN_IDS to self.max_num_output long, with
    # output_token_pad_id as the pad value.
    num_output_token = tf.shape(
        tfexample_dict[constants.KEY_OUTPUT_TOKEN_IDS].values)[0]
    target_token_ids = tf.pad(
        tfexample_dict[constants.KEY_OUTPUT_TOKEN_IDS].values,
        [[0, self.max_num_output - num_output_token]],
        mode='CONSTANT',
        constant_values=self.output_token_pad_id)

    # copy_indices_sparse_tensor has dense_shape [self.max_num_output,
    # self.max_num_nodes]. Each index [i, j] with value 1 in
    # copy_indices_sparse_tensor means that i:th target token can be copied
    # from j:th input node.
    copy_indices_sparse_tensor = tf.SparseTensor(
        # KEY_COPY_OUTPUT_INDICES stores the output that can be copied to,
        # KEY_COPY_INPUT_INDICES stores the node that can be copied from.
        # We stack them to get the SparseTensor indices.
        tf.stack([tfexample_dict[constants.KEY_COPY_OUTPUT_INDICES].values,
                  tfexample_dict[constants.KEY_COPY_INPUT_INDICES].values],
                 axis=-1),
        # Values are 1, meaning that we can copy from the nodes and target pairs
        # in the indices.
        tf.fill(tf.shape(
            tfexample_dict[constants.KEY_COPY_INPUT_INDICES].values), 1),
        [self.max_num_output, self.max_num_nodes])
    copy_indices_sparse_tensor = tf.sparse.reorder(copy_indices_sparse_tensor)
    # Set default value to 0, meaning that we can not copy rest of the nodes
    # and target token pairs.
    # And by specifying the dense_shape in SparseTensor, it nicely adds padding.
    copy_indices_dense_tensor = tf.sparse.to_dense(
        copy_indices_sparse_tensor, default_value=0)

    # pointer_indices_sparse_tensor has dense_shape [self.max_num_output,
    # self.max_num_nodes]. Each index [i, j] with value 1 in
    # pointer_indices_sparse_tensor means that i:th target output can point to
    # j:th input node.
    pointer_indices_sparse_tensor = tf.SparseTensor(
        # KEY_POINTER_OUTPUT_INDICES stores the output that can be pointed to,
        # KEY_POINTER_INPUT_INDICES stores the node that can be pointed from.
        # We stack them to get the SparseTensor indices.
        tf.stack([tfexample_dict[constants.KEY_POINTER_OUTPUT_INDICES].values,
                  tfexample_dict[constants.KEY_POINTER_INPUT_INDICES].values],
                 axis=-1),
        tf.fill(tf.shape(
            tfexample_dict[constants.KEY_POINTER_INPUT_INDICES].values), 1),
        [self.max_num_output, self.max_num_nodes])
    pointer_indices_sparse_tensor = tf.sparse.reorder(
        pointer_indices_sparse_tensor)
    # Same as copy_indices_dense_tensor, default value is 0 and dense_shape
    # handles the padding.
    pointer_indices_dense_tensor = tf.sparse.to_dense(
        pointer_indices_sparse_tensor, default_value=0)

    # Use RaggedTensor to create the node_id_sequences, to_tensor
    # nicely handles the padding.
    # node_id_sequences has shape [max_num_nodes, max_num_subtokens].
    node_id_sequences = tf.RaggedTensor.from_row_lengths(
        tfexample_dict[constants.KEY_NODE_TOKEN_IDS].values,
        row_lengths=tfexample_dict[
            constants.KEY_NODE_TOKEN_LENGTHS].values).to_tensor(
                shape=[self.max_num_nodes, self.max_num_subtokens],
                default_value=self.node_label_pad_id)

    num_node_type_ids = tf.shape(
        tfexample_dict[constants.KEY_NODE_TYPE_IDS].values)[0]
    node_types = tf.pad(
        tfexample_dict[constants.KEY_NODE_TYPE_IDS].values,
        [[0, self.max_num_nodes-num_node_type_ids]],
        mode='CONSTANT',
        constant_values=self.node_type_pad_id)

    # edge_indicators_sparse_tensor has dense_shape [self.max_num_edge_types,
    # self.max_num_nodes, self.max_num_nodes]. Each index [k, i, j] with value 1
    # in edge_indicators_sparse_tensor there is a edge with edge type k from
    # i:th input node to j:th input node.
    edge_indicators_sparse_tensor = tf.SparseTensor(
        # KEY_EDGE_TYPE_IDS stores the encoded edge types.
        # KEY_EDGE_SOURCE_INDICES stores the edge sources.
        # KEY_EDGE_SOURCE_INDICES stores the edge targets.
        # We stack them to get the SparseTensor indices.
        tf.stack([tfexample_dict[constants.KEY_EDGE_TYPE_IDS].values,
                  tfexample_dict[constants.KEY_EDGE_SOURCE_INDICES].values,
                  tfexample_dict[
                      constants.KEY_EDGE_DESTINATION_INDICES].values],
                 axis=-1),
        tf.fill(tf.shape(
            tfexample_dict[constants.KEY_EDGE_TYPE_IDS].values), 1),
        [self.max_num_edge_types, self.max_num_nodes, self.max_num_nodes])
    edge_indicators_sparse_tensor = tf.sparse.reorder(
        edge_indicators_sparse_tensor)
    # Use default_value as 0 to indicate there are no edges between the rest
    # of nodes. Specifying dense_shape nicely handles the padding.
    edge_indicators_dense_tensor = tf.sparse.to_dense(
        edge_indicators_sparse_tensor, default_value=0)

    num_node_texts = tf.shape(
        tfexample_dict[constants.KEY_NODE_TEXTS].values)[0]
    node_text_sequences = tf.pad(
        tfexample_dict[constants.KEY_NODE_TEXTS].values,
        [[0, self.max_num_nodes - num_node_texts]],
        mode='CONSTANT',
        constant_values=constants.PAD_TOKEN.encode('utf-8')
    )

    num_target_texts = tf.shape(
        tfexample_dict[constants.KEY_OUTPUT_TOKEN_TEXTS].values)[0]
    target_texts = tf.pad(
        tfexample_dict[constants.KEY_OUTPUT_TOKEN_TEXTS].values,
        [[0, self.max_num_output - num_target_texts]],
        mode='CONSTANT',
        constant_values=constants.PAD_TOKEN.encode('utf-8'))

    pointer_candidate_indices = tfexample_dict[
        constants.KEY_MASKING_CANDIDATE_INDICES].values  # N
    pointer_candidates = self._get_pointer_candidates(pointer_candidate_indices)

    return {
        constants.PROVENANCE_TENSOR_NAME:
            tfexample_dict[constants.KEY_PROVENANCE],
        constants.TARGET_TOKEN_IDS_TENSOR_NAME:
            target_token_ids,
        constants.COPY_INDICES_TENSOR_NAME:
            copy_indices_dense_tensor,
        constants.POINTER_INDICES_TENSOR_NAME:
            pointer_indices_dense_tensor,
        constants.NODE_ID_SEQUENCES_TENSOR_NAME:
            node_id_sequences,
        constants.NODE_TYPES_TENSOR_NAME:
            node_types,
        constants.EDGE_INDICATORS_TENSOR_NAME:
            edge_indicators_dense_tensor,
        constants.NODE_TEXT_SEQUENCES_TENSOR_NAME:
            node_text_sequences,
        constants.TARGET_TEXTS_TENSOR_NAME:
            target_texts,
        constants.MASKING_CANDIDATE_TENSOR_NAME:
            pointer_candidates
    }

  def _get_pointer_candidates(self, pointer_candidate_indices):
    # Creat a boolean mask over all nodes, such that it is is set to True at
    # candidate indices. e.g.
    # max_num_nodes = 5, candidates = [2, 3] => mask = [0, 0, 1, 1, 0]
    pointer_candidates = tf.zeros((self.max_num_nodes,), dtype=tf.bool)
    updates = tf.ones_like(pointer_candidate_indices, dtype=tf.bool)
    pointer_candidates = tf.tensor_scatter_nd_update(
        pointer_candidates, pointer_candidate_indices[:, None], updates)
    return pointer_candidates

  def _get_tfrecord_filenames(self):
    """Get all tfrecord filenames."""
    return sorted(
        glob.glob(os.path.join(self.split_tfrecord_dir, '*.tfrecord')))

  def create_dataset(self):
    """Create tf.data.TFRecordDataset.

    Create the dataset using tf.data.TFRecordDataset. The most important
    function here is _parse_tfexample_and_add_padding(). This function
    consists of pure tf transformation so that we use it in
    tf.data.Dataset.map(). We also add paddings in
    _parse_tfexample_and_add_padding() so that we can call
    tf.data.Dataset.batch(). We try to optimize the performance by having
    parallel data extraction and transformation, along with prefetch.

    Returns:
      A tuple of iterable of tf.data.Dataset and iterator of tf.data.Dataset,
      with their tensors converted to numpy arrays.
    """
    # Determinism isn't specified as part of `_get_tfrecord_filenames`'s
    # contract. Ensure a deterministic order.
    sorted_filenames = sorted(self._get_tfrecord_filenames())

    dataset = tf.data.Dataset.from_tensor_slices(sorted_filenames)

    # All operations leading to sharding must be deterministic.

    # First level of file sharding.
    num_host_shards = jax.process_count()
    logging.info('Cross host sharding: shard %d of %d.', jax.process_index(),
                 num_host_shards)
    if len(sorted_filenames) < 10 * num_host_shards:
      logging.warning('Each host handles fewer than 10 files. Ensure files are '
                      'well balanced.')
    dataset = dataset.shard(num_host_shards, jax.process_index())

    # Second level of file sharding.
    if self._shard_count > 1:
      logging.info('Intra-host sharding: shard %d of %d.', self._shard,
                   self._shard_count)
      dataset = dataset.shard(self._shard_count, self._shard)

    # File level shuffling.
    if self._shuffle_batch_count > 1:
      dataset = dataset.shuffle(len(sorted_filenames))

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply `take` early, to avoid useless processing. Note this operation
    # isn't deterministic due to parallelism and possible shuffling of the files
    # to read from.
    dataset = dataset.take(self.max_data_count)

    # Example level shuffling happens before parsing for efficiency.
    if self._shuffle_batch_count > 1:
      shuffle_example_count = self._shuffle_batch_count * self.batch_size
      logging.info('%s shuffle buffer size: %d.', self.split,
                   shuffle_example_count)
      dataset = dataset.shuffle(shuffle_example_count)

    dataset = dataset.map(
        self._parse_tfexample_and_add_padding,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.drop_remainder)
    dataset = dataset.repeat(count=self.repeat_count)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.as_numpy_iterator()

    return dataset, iter(dataset)

  def __iter__(self):
    return self

  def __next__(self):
    return self.__call__()

  def __call__(self):
    """Get next data from the dataset and return it.

    The data from dataset_iter is a batch of numpy arrays returned by
    _parse_tfexample_and_add_padding(). For example target_token_ids returned by
    _parse_tfexample_and_add_padding() will have shape (batch_size,
    max_num_output) instead of (max_num_output). This is because we call
    tf.data.Dataset.batch() in create_dataset(). But if drop_remainder is set
    to False, the batch_size of the batched numpy arrays can be smaller than
    the user defined batch_size.

    Returns:
      A batch of numpy arrays returned by _parse_tfexample_and_add_padding().
      Meaning that all numpy arrays returned by
      _parse_tfexample_and_add_padding() will have shape (batch_size, ...). But
      if drop_remainder is set to False, the batch_size of the batched numpy
      arrays can be smaller than the user defined batch_size.
    """
    return next(self.dataset_iter)
