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

"""Constants used in PLUR."""

from tensor2tensor.data_generators import text_encoder


###############################################################################
#
#  Constants for directory and file naming
#
###############################################################################
# The overall structure for the files in one dataset:
#├── STAGE_1_DIRNAME
#│   ├── METADATA_FILENAME
#│   ├── GRAPH_TO_OUTPUT_EXAMPLE_DIRNAME
#│   │   ├── TEST_SPLIT_NAME
#│   │   ├── TRAIN_SPLIT_NAME
#│   │   └── VALIDATION_SPLIT_NAME
#│   └── RAW_DATA_DIRNAME
#│       ├── GIT_REPO_DIRNAME
#└── STAGE_2_DIRNAME
#    ├── METADATA_FILENAME
#    ├── TFRECORD_DIRNAME
#    │   ├── TEST_SPLIT_NAME
#    │   ├── TRAIN_SPLIT_NAME
#    │   └── VALIDATION_SPLIT_NAME
#    └── VOCAB_FILES_DIRNAME
#        ├── NODE_TYPE_VOCAB_FILENAME
#        ├── NODE_LABEL_VOCAB_FILENAME
#        ├── EDGE_TYPE_VOCAB_FILENAME
#        └── OUTPUT_TOKEN_VOCAB_FILENAME

STAGE_1_DIRNAME = 'stage_1'
RAW_DATA_DIRNAME = 'raw_data'
GIT_REPO_DIRNAME = 'git_repo'
GRAPH_TO_OUTPUT_EXAMPLE_DIRNAME = 'graph_to_output_example'
TRAIN_SPLIT_NAME = 'train'
VALIDATION_SPLIT_NAME = 'valid'
TEST_SPLIT_NAME = 'test'
METADATA_FILENAME = 'dataset_metadata.json'
STAGE_2_DIRNAME = 'stage_2'
VOCAB_FILES_DIRNAME = 'vocab_files'
NODE_TYPE_VOCAB_FILENAME = 'node_type_vocab.txt'
NODE_LABEL_VOCAB_FILENAME = 'node_label_vocab.txt'
EDGE_TYPE_VOCAB_FILENAME = 'edge_type_vocab.txt'
OUTPUT_TOKEN_VOCAB_FILENAME = 'output_token_vocab.txt'
TFRECORD_DIRNAME = 'tfrecords'

###############################################################################
#
#  Constants for vocabulary
#
###############################################################################

POINTER_TOKEN = 'POINTER'
CLASS_TOKEN_PREFIX = 'CLASS_'
DONE_TOKEN = 'DONE'
PAD_TOKEN = text_encoder.PAD
OOV_TOKEN = '<OOV>'
RESERVED_TOKENS = [OOV_TOKEN, PAD_TOKEN]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)

###############################################################################
#
#  Constants for TfExample message
#
###############################################################################

# Whole-example keys.
EXAMPLE_GUID = 'graph/global_id'  # For future use.
KEY_PROVENANCE = 'metadata/provenance'

# ToCoPo TF Example keys.
KEY_OUTPUT_TOKEN_IDS = 'targets/token_ids'
KEY_OUTPUT_TOKEN_TEXTS = 'targets/texts'
KEY_COPY_INPUT_INDICES = 'nodes/copy_indices'
KEY_COPY_OUTPUT_INDICES = 'targets/copy_indices'
KEY_POINTER_INPUT_INDICES = 'nodes/pointer_indices'
KEY_POINTER_OUTPUT_INDICES = 'targets/pointer_indices'

# GNN TF Example keys.
KEY_NODE_TOKEN_IDS = 'nodes/node_id_sequences'
KEY_NODE_TEXTS = 'nodes/node_text_sequences'
KEY_NODE_TOKEN_LENGTHS = 'nodes/node_sequence_lengths'
KEY_NODE_GUIDS = 'nodes/global_node_id'  # For future use.

KEY_NODE_TYPE_IDS = 'nodes/node_types'

KEY_EDGE_TYPE_IDS = 'edges/types'
KEY_EDGE_SOURCE_INDICES = 'edges/source_indices'
KEY_EDGE_DESTINATION_INDICES = 'edges/dest_indices'

# Masking keys.
KEY_MASKING_CANDIDATE_INDICES = 'masking/candidate_indices'

###############################################################################
#
#  Constants for PLUR data loader
#
###############################################################################

PROVENANCE_TENSOR_NAME = 'provenance'
TARGET_TOKEN_IDS_TENSOR_NAME = 'target_token_ids'
COPY_INDICES_TENSOR_NAME = 'copy_indices'
POINTER_INDICES_TENSOR_NAME = 'pointer_indices'
NODE_ID_SEQUENCES_TENSOR_NAME = 'node_id_sequences'
NODE_TYPES_TENSOR_NAME = 'node_types'
EDGE_INDICATORS_TENSOR_NAME = 'edge_indicators'
NODE_TEXT_SEQUENCES_TENSOR_NAME = 'node_text_sequences'
TARGET_TEXTS_TENSOR_NAME = 'target_texts'
MASKING_CANDIDATE_TENSOR_NAME = 'candidate_indices'

###############################################################################
#
#  Dataset specific constants
#
###############################################################################

HOPPITY_ADD_NODE_OP_NAME = 'add_node'
HOPPITY_DEL_NODE_OP_NAME = 'del_node'
HOPPITY_REPLACE_VAL_OP_NAME = 'replace_val'
HOPPITY_REPLACE_TYPE_OP_NAME = 'replace_type'
HOPPITY_REPLACE_NOOP_OP_NAME = 'NoOp'
HOPPITY_OUTPUT_UNKNOWN_TOKEN = 'UNKNOWN'

NATURALIZE_CONTEXT_TOKEN_NODE_NAME = 'CONTEXT_TOKEN'
NATURALIZE_NEXT_TOKEN_EDGE_NAME = 'NEXT_TOKEN'
NATURALIZE_CONTEXT_ROOT_NODE_NAME = 'CONTEXT_ROOT'
NATURALIZE_CONTEXT_TOKEN_EDGE_NAME = 'CONTEXT_TOKEN'
NATURALIZE_FEATURE_NODE_NODE_NAME = 'FEATURE_NODE'
NATURALIZE_NEXT_FEATURE_EDGE_NAME = 'NEXT_FEATURE'
NATURALIZE_FEATURE_ROOT_NODE_NAME = 'FEATURE_ROOT'
NATURALIZE_FEATURE_EDGE_NAME = 'FEATURE'
NATURALIZE_NATURALIZE_ROOT_NODE_NAME = 'NATURALIZE_ROOT'
NATURALIZE_LOCAL_CONTEXT_EDGE_NAME = 'LOCAL_CONTEXT'
NATURALIZE_FEATURES_EDGE_NAME = 'FEATURES'
