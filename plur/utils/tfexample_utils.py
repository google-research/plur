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

"""Utility function for TfExample used in PLUR.
"""

from plur.utils import constants
import tensorflow as tf

FEATURE_DESCRIPTION = {
    constants.KEY_PROVENANCE: tf.io.FixedLenFeature(
        [], tf.string, default_value=''),
    constants.KEY_OUTPUT_TOKEN_IDS: tf.io.VarLenFeature(tf.int64),
    constants.KEY_OUTPUT_TOKEN_TEXTS: tf.io.VarLenFeature(tf.string),
    constants.KEY_COPY_INPUT_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_COPY_OUTPUT_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_POINTER_INPUT_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_POINTER_OUTPUT_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_NODE_TOKEN_IDS: tf.io.VarLenFeature(tf.int64),
    constants.KEY_NODE_TEXTS: tf.io.VarLenFeature(tf.string),
    constants.KEY_NODE_TOKEN_LENGTHS: tf.io.VarLenFeature(tf.int64),
    constants.KEY_NODE_TYPE_IDS: tf.io.VarLenFeature(tf.int64),
    constants.KEY_EDGE_TYPE_IDS: tf.io.VarLenFeature(tf.int64),
    constants.KEY_EDGE_SOURCE_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_EDGE_DESTINATION_INDICES: tf.io.VarLenFeature(tf.int64),
    constants.KEY_MASKING_CANDIDATE_INDICES: tf.io.VarLenFeature(tf.int64),
}


def _bytes_feature(values):
  values = [v.encode('utf-8') for v in values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def serialize_tfexample_feature(tfexample_feature_dict):
  """Serialize TfExampleFeature namedtuple to TfExample."""
  feature = {
      constants.KEY_PROVENANCE:
          _bytes_feature([tfexample_feature_dict[constants.KEY_PROVENANCE]]),
      constants.KEY_OUTPUT_TOKEN_IDS:
          _int64_feature(tfexample_feature_dict[constants.KEY_OUTPUT_TOKEN_IDS]
                        ),
      constants.KEY_OUTPUT_TOKEN_TEXTS:
          _bytes_feature(
              tfexample_feature_dict[constants.KEY_OUTPUT_TOKEN_TEXTS]),
      constants.KEY_COPY_INPUT_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_COPY_INPUT_INDICES]),
      constants.KEY_COPY_OUTPUT_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_COPY_OUTPUT_INDICES]),
      constants.KEY_POINTER_INPUT_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_POINTER_INPUT_INDICES]),
      constants.KEY_POINTER_OUTPUT_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_POINTER_OUTPUT_INDICES]),
      constants.KEY_NODE_TOKEN_IDS:
          _int64_feature(tfexample_feature_dict[constants.KEY_NODE_TOKEN_IDS]),
      constants.KEY_NODE_TEXTS:
          _bytes_feature(tfexample_feature_dict[constants.KEY_NODE_TEXTS]),
      constants.KEY_NODE_TOKEN_LENGTHS:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_NODE_TOKEN_LENGTHS]),
      constants.KEY_NODE_TYPE_IDS:
          _int64_feature(tfexample_feature_dict[constants.KEY_NODE_TYPE_IDS]),
      constants.KEY_EDGE_TYPE_IDS:
          _int64_feature(tfexample_feature_dict[constants.KEY_EDGE_TYPE_IDS]),
      constants.KEY_EDGE_SOURCE_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_EDGE_SOURCE_INDICES]),
      constants.KEY_EDGE_DESTINATION_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_EDGE_DESTINATION_INDICES]),
      constants.KEY_MASKING_CANDIDATE_INDICES:
          _int64_feature(
              tfexample_feature_dict[constants.KEY_MASKING_CANDIDATE_INDICES]),
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def get_tfexample_feature(graph_to_output_example,
                          node_type_vocab_dict=None,
                          node_label_vocab_dict=None,
                          edge_type_vocab_dict=None,
                          output_token_vocab_dict=None):
  """Get all features needed in the to create the tfexample.

  Args:
    graph_to_output_example: A GraphToOutputExample instance.
    node_type_vocab_dict: Dictionary containing the node type vocabulary.
    node_label_vocab_dict: Dictionary containing the node label vocabulary.
    edge_type_vocab_dict: Dictionary containing the edge type vocabulary.
    output_token_vocab_dict: Dictionary containing the output token vocabulary.
  Returns:
    A tfexmaple feature dictionary.
  """
  node_type_oov_id = node_type_vocab_dict[constants.OOV_TOKEN]
  node_label_oov_id = node_label_vocab_dict[constants.OOV_TOKEN]
  edge_type_oov_id = edge_type_vocab_dict[constants.OOV_TOKEN]
  output_token_oov_id = output_token_vocab_dict[constants.OOV_TOKEN]

  # We mirror the key names defined in constants.py
  # Whole-example keys.
  graph_global_id = -1  # Unused for now
  provenance = graph_to_output_example.get_provenance()

  # ToCoPo TF Example keys.
  targets_token_ids = []
  target_tokens = []
  nodes_copy_indices = []
  targets_copy_indices = []
  nodes_pointer_indices = []
  targets_pointer_indices = []

  # GNN TF Example keys.
  nodes_node_id_sequences = []
  nodes_node_texts = []
  nodes_node_sequence_lengths = []
  nodes_global_node_id = []  # Unused for now

  nodes_node_types = []

  edges_types = []
  edges_source_indices = []
  edges_dest_indices = []

  # Masking keys
  masking_candidate_indices = []

  for node in graph_to_output_example.get_nodes():
    nodes_node_types.append(
        node_type_vocab_dict.get(node['type'], node_type_oov_id))
    nodes_node_id_sequences.append(
        node_label_vocab_dict.get(node['label'], node_label_oov_id))
    nodes_node_texts.append(node['label'])
    if 'is_repair_candidate' in node and node['is_repair_candidate']:
      masking_candidate_indices.append(node['id'])
    # The tokens/subtokens are created by the dataset, we don't create any
    # tokenizers. Our vocabulary also assumes that the label and type of each
    # node is a single token, even if it is separated by a whitespace. Therefore
    # we always add 1 here. If we are using our own subtokenization, update
    # this and how we insert values into nodes_node_types and
    # nodes_node_id_sequences.
    nodes_node_sequence_lengths.append(1)
    nodes_global_node_id.append(-1)  # Update this when we use global node id.

  for edge in graph_to_output_example.get_edges():
    edges_types.append(
        edge_type_vocab_dict.get(edge['type'], edge_type_oov_id))
    edges_source_indices.append(edge['src'])
    edges_dest_indices.append(edge['dst'])

  for output_token in graph_to_output_example.get_output_as_tokens():
    targets_token_ids.append(
        output_token_vocab_dict.get(output_token, output_token_oov_id))
    target_tokens.append(output_token)

  output_tokens_and_indices = (
      graph_to_output_example.get_output_tokens_and_index())
  for index, output_token in output_tokens_and_indices:
    for node in graph_to_output_example.get_nodes():
      if output_token == node['label']:
        targets_copy_indices.append(index)
        nodes_copy_indices.append(node['id'])

  for index, node_id in graph_to_output_example.get_output_pointers_and_index():
    targets_pointer_indices.append(index)
    nodes_pointer_indices.append(node_id)

  tfexample_feature_dict = {
      constants.EXAMPLE_GUID: graph_global_id,
      constants.KEY_PROVENANCE: provenance,
      constants.KEY_OUTPUT_TOKEN_IDS: targets_token_ids,
      constants.KEY_OUTPUT_TOKEN_TEXTS: target_tokens,
      constants.KEY_COPY_INPUT_INDICES: nodes_copy_indices,
      constants.KEY_COPY_OUTPUT_INDICES: targets_copy_indices,
      constants.KEY_POINTER_INPUT_INDICES: nodes_pointer_indices,
      constants.KEY_POINTER_OUTPUT_INDICES: targets_pointer_indices,
      constants.KEY_NODE_TOKEN_IDS: nodes_node_id_sequences,
      constants.KEY_NODE_TEXTS: nodes_node_texts,
      constants.KEY_NODE_TOKEN_LENGTHS: nodes_node_sequence_lengths,
      constants.KEY_NODE_GUIDS: nodes_global_node_id,
      constants.KEY_NODE_TYPE_IDS: nodes_node_types,
      constants.KEY_EDGE_TYPE_IDS: edges_types,
      constants.KEY_EDGE_SOURCE_INDICES: edges_source_indices,
      constants.KEY_EDGE_DESTINATION_INDICES: edges_dest_indices,
      constants.KEY_MASKING_CANDIDATE_INDICES: masking_candidate_indices
  }

  return tfexample_feature_dict
