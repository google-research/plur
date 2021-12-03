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

"""Functions for data generation, combines stage 1 and stage 2."""
import functools

from absl import app
from absl import flags
import immutabledict
from plur.stage_1.code2seq_dataset import Code2SeqDataset
from plur.stage_1.convattn_dataset import ConvAttnDataset
from plur.stage_1.cubert_exception_classification_dataset import CuBertExceptionClassificationDataset
from plur.stage_1.cubert_function_docstring_classification_dataset import CuBertFunctionDocstringClassificationDataset
from plur.stage_1.cubert_multitask_dataset import CuBertMultitaskDataset
from plur.stage_1.cubert_multitask_rebalanced_dataset import CuBertMultitaskRebalancedDataset
from plur.stage_1.cubert_swapped_operand_classification_dataset import CuBertSwappedOperandClassificationDataset
from plur.stage_1.cubert_variable_misuse_classification_dataset import CuBertVariableMisuseClassificationDataset
from plur.stage_1.cubert_variable_misuse_repair_dataset import CuBertVariableMisuseRepairDataset
from plur.stage_1.cubert_variable_misuse_repair_nocopy_dataset import CuBertVariableMisuseRepairNoCopyDataset
from plur.stage_1.cubert_variable_misuse_repair_unpointed_dataset import CuBertVariableMisuseRepairUnpointedDataset
from plur.stage_1.cubert_variable_misuse_repair_unpointed_nocopy_dataset import CuBertVariableMisuseRepairUnpointedNoCopyDataset
from plur.stage_1.cubert_wrong_operator_classification_dataset import CuBertWrongOperatorClassificationDataset
from plur.stage_1.dummy_dataset import DummyDataset
from plur.stage_1.funcom_dataset import FuncomDataset
from plur.stage_1.great_var_misuse_dataset import GreatVarMisuseDataset
from plur.stage_1.hoppity_single_ast_diff_dataset import HoppitySingleAstDiffDataset
from plur.stage_1.manysstubs4j_dataset import ManySStuBs4JDataset
from plur.stage_1.ogb_code_dataset import OgbCodeDataset
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.retrieve_and_edit_dataset import RetrieveAndEditDataset
from plur.stage_2.cubert_multitask_graph_to_output_example_to_tfexample import CuBertMultitaskGraphToOutputExampleToTfexample
from plur.stage_2.graph_to_output_example_to_tfexample import GraphToOutputExampleToTfexample
from plur.stage_2.hoppity_graph_to_output_example_to_tfexample import HoppityGraphToOutputExampleToTfexample
from plur.utils import constants

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'dataset_name',
    'dummy_dataset',
    (
        'code2seq_dataset',
        'convattn_dataset',
        'dummy_dataset',
        'funcom_dataset',
        'great_var_misuse_dataset',
        'hoppity_single_ast_diff_dataset',
        'manysstubs4j_dataset',
        'ogb_code_dataset',
        'cubert_multitask_dataset',
        'cubert_multitask_rebalanced_dataset',
        'cubert_exception_classification_dataset',
        'cubert_variable_misuse_classification_dataset',
        'cubert_variable_misuse_repair_dataset',
        'cubert_variable_misuse_repair_unpointed_dataset',
        'cubert_variable_misuse_repair_nocopy_dataset',
        'cubert_variable_misuse_repair_unpointed_nocopy_dataset',
        'cubert_swapped_operand_classification_dataset',
        'cubert_function_docstring_classification_dataset',
        'cubert_wrong_operator_classification_dataset',
        'retrieve_and_edit_dataset',
    ),
    'Name of the dataset to generate data.')
flags.DEFINE_string('stage_1_dir', '/tmp/dummy_dataset/stage_1',
                    'Where to store stage_1 files.')
flags.DEFINE_string('stage_2_dir', '/tmp/dummy_dataset/stage_2',
                    'Where to store stage_2 files.')
flags.DEFINE_enum('stages', '12', ['1', '2', '12'],
                  'Stages to be run.')
flags.DEFINE_integer('train_data_percentage', 0,
                     'Percentage of data used as training data.')
flags.DEFINE_integer('validation_data_percentage', 0,
                     'Percentage of data used as validation data.')
flags.DEFINE_integer('test_data_percentage', 0,
                     'Percentage of data used as test data.')
flags.DEFINE_integer('num_shards', 1000, 'Number of shards.')
flags.DEFINE_integer('seed', 0, 'The random seed.')
flags.DEFINE_integer('max_node_type_vocab_size', 10000,
                     'Maximum node type vocabulary size.')
flags.DEFINE_integer('max_node_label_vocab_size', 10000,
                     'Maximum node label vocabulary size.')
flags.DEFINE_integer('max_edge_type_vocab_size', 10000,
                     'Maximum edge type vocabulary size.')
flags.DEFINE_integer('max_output_token_vocab_size', 10000,
                     'Maximum output token vocabulary size.')
flags.DEFINE_integer('train_max_graph_size', 1024,
                     'Maximum number of nodes in a graph from training data, '
                     'graphs with more nodes will be filtered. They are '
                     'filtered in stage 2.')
flags.DEFINE_integer('valid_max_graph_size', 1024,
                     'Maximum number of nodes in a graph from validation data, '
                     'graphs with more nodes will be filtered. They are '
                     'filtered in stage 2.')
flags.DEFINE_integer('test_max_graph_size', 1024,
                     'Maximum number of nodes in a graph from testing data, '
                     'graphs with more nodes will be filtered. They are '
                     'filtered in stage 2.')
flags.DEFINE_bool('deduplicate', False, 'Deduplicate GraphToOutputExample.')


def get_dataset_class(dataset_name):
  """Get the dataset class based on dataset_name."""
  if dataset_name == 'code2seq_dataset':
    return Code2SeqDataset
  elif dataset_name == 'convattn_dataset':
    return ConvAttnDataset
  elif dataset_name == 'dummy_dataset':
    return DummyDataset
  elif dataset_name == 'funcom_dataset':
    return FuncomDataset
  elif dataset_name == 'great_var_misuse_dataset':
    return GreatVarMisuseDataset
  elif dataset_name == 'hoppity_single_ast_diff_dataset':
    return HoppitySingleAstDiffDataset
  elif dataset_name == 'manysstubs4j_dataset':
    return ManySStuBs4JDataset
  elif dataset_name == 'ogb_code_dataset':
    return OgbCodeDataset
  elif dataset_name == 'cubert_multitask_dataset':
    return CuBertMultitaskDataset
  elif dataset_name == 'cubert_multitask_rebalanced_dataset':
    return CuBertMultitaskRebalancedDataset
  elif dataset_name == 'cubert_exception_classification_dataset':
    return CuBertExceptionClassificationDataset
  elif dataset_name == 'cubert_variable_misuse_classification_dataset':
    return CuBertVariableMisuseClassificationDataset
  elif dataset_name == 'cubert_variable_misuse_repair_dataset':
    return CuBertVariableMisuseRepairDataset
  elif dataset_name == 'cubert_variable_misuse_repair_unpointed_dataset':
    return CuBertVariableMisuseRepairUnpointedDataset
  elif dataset_name == 'cubert_variable_misuse_repair_nocopy_dataset':
    return CuBertVariableMisuseRepairNoCopyDataset
  elif dataset_name == 'cubert_variable_misuse_repair_unpointed_nocopy_dataset':
    return CuBertVariableMisuseRepairUnpointedNoCopyDataset
  elif dataset_name == 'cubert_swapped_operand_classification_dataset':
    return CuBertSwappedOperandClassificationDataset
  elif dataset_name == 'cubert_function_docstring_classification_dataset':
    return CuBertFunctionDocstringClassificationDataset
  elif dataset_name == 'cubert_wrong_operator_classification_dataset':
    return CuBertWrongOperatorClassificationDataset
  elif dataset_name == 'retrieve_and_edit_dataset':
    return RetrieveAndEditDataset
  else:
    raise ValueError('{} is not supported.'.format(dataset_name))


def get_stage_2_class(dataset_name):
  if dataset_name == 'hoppity_single_ast_diff_dataset':
    return HoppityGraphToOutputExampleToTfexample
  elif dataset_name == 'cubert_multitask_dataset':
    return CuBertMultitaskGraphToOutputExampleToTfexample
  elif dataset_name == 'cubert_multitask_rebalanced_dataset':
    return CuBertMultitaskGraphToOutputExampleToTfexample
  else:
    return GraphToOutputExampleToTfexample


def create_dataset(dataset_name, stage_1_dir, stage_2_dir, stage_1_kwargs,
                   stage_2_kwargs, run_stage_1=True, run_stage_2=True):
  """Run stage 1 and/or stage 2 to create the PLUR dataset.

  Args:
    dataset_name: Name of the dataset, used to get the dataset class.
    stage_1_dir: Directory to store stage 1 files.
    stage_2_dir: Directory to store stage 2 files.
    stage_1_kwargs: Dictionary for named parameters in stage 1.
    stage_2_kwargs: Dictionary for named parameters in stage 2.
    run_stage_1: Boolean indicating running stage 1 or not.
    run_stage_2: Boolean indicating running stage 2 or not.
  """
  if run_stage_1:
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(stage_1_dir, **stage_1_kwargs)
    dataset.stage_1_mkdirs()
    dataset.download_dataset()
    dataset.run_pipeline()

  if run_stage_2:
    stage_2_class = get_stage_2_class(dataset_name)
    dataset = stage_2_class(
        stage_1_dir, stage_2_dir, dataset_name, **stage_2_kwargs)
    dataset.stage_2_mkdirs()
    dataset.run_pipeline()


def main(_):
  user_defined_split_range = ()
  split_percentage_sum = sum([FLAGS.train_data_percentage,
                              FLAGS.validation_data_percentage,
                              FLAGS.test_data_percentage])
  if split_percentage_sum == 100:
    user_defined_split_range = (FLAGS.train_data_percentage,
                                FLAGS.validation_data_percentage,
                                FLAGS.test_data_percentage)
  elif split_percentage_sum != 0:
    raise ValueError('Sum of train, validation and test split percentage '
                     'is {}, but should be 100.'.format(split_percentage_sum))
  stage_1_kwargs = dict(
      user_defined_split_range=user_defined_split_range,
      num_shards=FLAGS.num_shards,
      seed=FLAGS.seed,
      configuration=Configuration(max_graph_sizes=immutabledict.immutabledict({
          constants.TRAIN_SPLIT_NAME: FLAGS.train_max_graph_size,
          constants.VALIDATION_SPLIT_NAME: FLAGS.valid_max_graph_size,
          constants.TEST_SPLIT_NAME: FLAGS.test_max_graph_size})),
      deduplicate=FLAGS.deduplicate)

  def _filter_graph_size(graph_to_output_example, graph_size):
    return len(graph_to_output_example.get_nodes()) <= graph_size
  # Functions that filter different data split based on graph size.
  train_filter_graph_size = functools.partial(
      _filter_graph_size, graph_size=FLAGS.train_max_graph_size)
  valid_filter_graph_size = functools.partial(
      _filter_graph_size, graph_size=FLAGS.valid_max_graph_size)
  test_filter_graph_size = functools.partial(
      _filter_graph_size, graph_size=FLAGS.test_max_graph_size)

  stage_2_kwargs = dict(
      max_node_type_vocab_size=FLAGS.max_node_type_vocab_size,
      max_node_label_vocab_size=FLAGS.max_node_label_vocab_size,
      max_edge_type_vocab_size=FLAGS.max_edge_type_vocab_size,
      max_output_token_vocab_size=FLAGS.max_output_token_vocab_size,
      num_shards=FLAGS.num_shards,
      train_filter_funcs=(train_filter_graph_size,),
      validation_filter_funcs=(valid_filter_graph_size,),
      test_filter_funcs=(test_filter_graph_size,)
  )

  run_stage_1 = False
  run_stage_2 = False
  if '1' in FLAGS.stages:
    run_stage_1 = True
  if '2' in FLAGS.stages:
    run_stage_2 = True

  create_dataset(FLAGS.dataset_name, FLAGS.stage_1_dir, FLAGS.stage_2_dir,
                 stage_1_kwargs, stage_2_kwargs, run_stage_1=run_stage_1,
                 run_stage_2=run_stage_2)


if __name__ == '__main__':
  app.run(main)
