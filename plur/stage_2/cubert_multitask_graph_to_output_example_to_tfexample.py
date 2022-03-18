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

"""Class for converting the CuBERT multitask PLUR dataset to TFExamples."""
import functools
import os
import shutil
from typing import Type, TypeVar

from absl import logging
import apache_beam as beam
from plur.stage_1 import cubert_multitask_dataset
from plur.stage_2.graph_to_output_example_to_tfexample import GraphToOutputExampleToTfexample
from plur.utils.graph_to_output_example import GraphToOutputExample


_SubclassType = TypeVar('_SubclassType', bound=GraphToOutputExampleToTfexample)


class CuBertMultitaskGraphToOutputExampleToTfexample(
    GraphToOutputExampleToTfexample):
  """Main class for converting GraphToOutputExample to TfExample.

  For the CuBERT Multitask, we want separate stage_2 directories for the test
  splits of each task. So we perform all operations as normal, but also create
  separate GraphToOutputExampleToTfexample subobjects separately for each task.

  Training/validation will happen on the main stage_2. Testing can happen on
  the main stage_2 (but there can be no per-task evaluator, since we can't
  tell which example is from which task), but only full-sequence evaluator.
  Testing can also happen on the per-task stage_2 directories, where the
  per-task evaluators can be applied to the examples of that task.
  """

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
               max_node_label_vocab_size=10000, max_edge_type_vocab_size=10000,
               max_output_token_vocab_size=10000, num_shards=1000):
    """As per superclass."""
    # This will generate the `stage_2` for training/validation. It will also
    # generate a common test dataset for all tasks combined together, but
    # it won't be possible to run per-task evaluators on it, since we don't
    # know which example came from which task. To evaluate per task, we
    # we generate separate `stage_2` subdirectories with test-only datasets
    # below, via per-task converters in `self.per_task_converters`.
    super().__init__(stage_1_dir, stage_2_dir, dataset_name,
                     train_transformation_funcs, train_filter_funcs,
                     validation_transformation_funcs, validation_filter_funcs,
                     test_transformation_funcs, test_filter_funcs,
                     max_node_type_vocab_size, max_node_label_vocab_size,
                     max_edge_type_vocab_size, max_output_token_vocab_size,
                     num_shards)
    self.per_task_converters = {}
    subtask_class = self._subtask_class()
    for task_acronym in (
        cubert_multitask_dataset.CuBertMultitaskDataset._UNITASK_DATSETS):
      # The unitask stage_2's will be subdirectories of the overall stage_2
      # directory.
      stage_2_subdir = os.path.join(stage_2_dir, task_acronym)
      keep_only_task_examples = functools.partial(
          keep_only_this_task, task_acronym=task_acronym)
      # We won't produce any train/validation examples, just test.
      self.per_task_converters[task_acronym] = subtask_class(
          stage_1_dir=stage_1_dir,  # Shared for all stage_2's.
          stage_2_dir=stage_2_subdir,
          dataset_name=f'{dataset_name}.{task_acronym}',
          train_transformation_funcs=train_transformation_funcs,
          # We don't need any training examples in the unitask stage_2's.
          train_filter_funcs=(drop_all_examples,),
          validation_transformation_funcs=validation_transformation_funcs,
          # We only want the validation examples from the particular unitask.
          validation_filter_funcs=[keep_only_task_examples] +
          list(validation_filter_funcs),
          test_transformation_funcs=test_transformation_funcs,
          # We only want the test examples from the particular unitask.
          test_filter_funcs=[keep_only_task_examples] + list(test_filter_funcs),
          max_node_type_vocab_size=max_node_type_vocab_size,
          max_node_label_vocab_size=max_node_label_vocab_size,
          max_edge_type_vocab_size=max_edge_type_vocab_size,
          max_output_token_vocab_size=max_output_token_vocab_size,
          num_shards=num_shards)

  def _subtask_class(self) -> Type[_SubclassType]:
    return GraphToOutputExampleToTfexample

  def _copy(self, from_path: str, to_path: str) -> None:
    shutil.copyfile(from_path, to_path)

  def _make_pipeline(self) -> beam.Pipeline:
    return beam.Pipeline()

  def copy_vocabularies(self) -> None:
    print('Copying vocabularies into unitask directories.')
    for converter in self.per_task_converters.values():
      self._copy(self.node_type_vocab_file,
                 converter.node_type_vocab_file)
      self._copy(self.node_label_vocab_file,
                 converter.node_label_vocab_file)
      self._copy(self.output_token_vocab_file,
                 converter.output_token_vocab_file)
      self._copy(self.edge_type_vocab_file,
                 converter.edge_type_vocab_file)

  def exists_tfrecords(self) -> bool:
    return False

  def exists_vocab_files(self) -> bool:
    return False

  def run_pipeline(self):
    """As per superclass, but iterates on the subtasks too."""
    logging.info('Running stage 2 pipeline.')
    print('Running stage 2 pipeline.')
    print('Vocabularies.')
    with self._make_pipeline() as p:
      self.build_and_save_vocab(p)

    # Now all vocabularies are built. We don't want to build the vocabularies
    # again for the unitask stage_2s, so we just copy them over.
    self.copy_vocabularies()

    # And now we generate tf Examples for the per-task converters. These will
    # only be test examples.
    print('Create tf examples.')
    with self._make_pipeline() as p:
      self.convert_and_write_tfexample(p)
      for converter in self.per_task_converters.values():
        print(f'Create tf examples for {converter.dataset_name}.')
        converter.convert_and_write_tfexample(p)

  def stage_2_mkdirs(self):
    """As per superclass, but iterates on the subtasks too."""
    super().stage_2_mkdirs()
    for converter in self.per_task_converters.values():
      converter.stage_2_mkdirs()


def drop_all_examples(graph_to_output_example: GraphToOutputExample) -> bool:
  del graph_to_output_example
  return False


def keep_only_this_task(graph_to_output_example: GraphToOutputExample,
                        task_acronym: str) -> bool:
  node_data = graph_to_output_example.get_data()

  assert cubert_multitask_dataset.TASK_TYPE_FIELD_NAME in node_data, (
      f'Expected to find {cubert_multitask_dataset.TASK_TYPE_FIELD_NAME} in '
      f'the keys, but only found these: {node_data.keys()}')
  example_task = node_data[cubert_multitask_dataset.TASK_TYPE_FIELD_NAME]

  return example_task == task_acronym
