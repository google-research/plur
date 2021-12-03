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
"""Converts multiple CuBERT datasets into a single PLUR task."""
import json
import os
import random
from typing import Any, Iterable, Mapping

import apache_beam as beam
from plur.stage_1 import cubert_dataset
from plur.stage_1 import plur_dataset
from plur.stage_1.cubert_exception_classification_dataset import CuBertExceptionClassificationDataset
from plur.stage_1.cubert_function_docstring_classification_dataset import CuBertFunctionDocstringClassificationDataset
from plur.stage_1.cubert_swapped_operand_classification_dataset import CuBertSwappedOperandClassificationDataset
from plur.stage_1.cubert_variable_misuse_classification_dataset import CuBertVariableMisuseClassificationDataset
from plur.stage_1.cubert_variable_misuse_repair_dataset import CuBertVariableMisuseRepairDataset
from plur.stage_1.cubert_wrong_operator_classification_dataset import CuBertWrongOperatorClassificationDataset
from plur.utils import constants
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError


# We add a new field name in GraphToOutputExample, to remember the unitask
# from which the example came. This will help downstream to generate
# per-unitask test datasets.
TASK_TYPE_FIELD_NAME = 'task'
PATH_FIELD_NAME = 'actual_path'


class CuBertMultitaskDataset(plur_dataset.PlurDataset):
  """Converts CuBERT data from different tasks into a PLUR dataset.

  The datasets are created by: Aditya Kanade, Petros Maniatis, Gogul
  Balakrishnan, Kensen Shi Proceedings of the 37th International Conference on
  Machine Learning, PMLR 119:5110-5121, 2020.

  This class combines multiple CuBERT task classes into a single multi-task
  dataset/task.
  """

  _UNITASK_DATSETS = {
      'EC': CuBertExceptionClassificationDataset,
      'FD': CuBertFunctionDocstringClassificationDataset,
      'SO': CuBertSwappedOperandClassificationDataset,
      'VM': CuBertVariableMisuseClassificationDataset,
      'VMR': CuBertVariableMisuseRepairDataset,
      'WB': CuBertWrongOperatorClassificationDataset,
  }

  # This is drawn from Table 1 in the paper.
  _UNITASK_TRAIN_SIZES = {
      'EC': 18480,
      'FD': 340846,
      'SO': 236246,
      'VM': 700708,
      'VMR': 700708,
      'WB': 459400,
  }

  def __init__(
      self,
      stage_1_dir,
      *args,
      configuration: plur_dataset.Configuration = plur_dataset.Configuration(),
      rebalance_skew: bool = False,
      **kwargs):
    """Initializes with and without rebalance skew.

    Args:
      stage_1_dir: As per superclass.
      *args: args.
      configuration: Any configuration options.
      rebalance_skew: If True, duplicate examples from each task to make the
        sizes of training datasets similar. If false, examples are left as
        input.
      **kwargs: kwargs.
    """

    # We initialize `unitask_datasets` first, because doing in later confuses
    # the type checker into thinking it may not be defined.
    if rebalance_skew:
      max_train_size = max(self._UNITASK_TRAIN_SIZES.values())
      self.unitask_duplication = {k: max_train_size/s
                                  for k, s in self._UNITASK_TRAIN_SIZES.items()}
    else:
      self.unitask_duplication = {}
    assert all((d >= 1.0 for d in self.unitask_duplication.values()))

    self.unitask_datasets: Mapping[str, cubert_dataset.CuBertDataset] = {}
    # Now we create a dataset object for each included unitask dataset.
    def initialize_dataset(acronym, dataset_class):
      return dataset_class(
          # We will store the per-task stage_1's in subdirectories named by
          # the acronym.
          *args,
          stage_1_dir=os.path.join(stage_1_dir, acronym),
          configuration=configuration,
          **kwargs)
    self.unitask_datasets = {
        acronym: initialize_dataset(acronym, dataset_class)
        for acronym, dataset_class in self.unitask_dataset_classes().items()}

    # The description for the dataset will be a concatenation of all unitask
    # descriptions.
    unitask_description_content = (
        (dataset.dataset_name,
         dataset.dataset_description)
        for dataset in self.unitask_datasets.values())
    unitask_descriptions = (
        f'{name}\n{description}'
        for name, description in unitask_description_content)
    single_description = '\n\n'.join(unitask_descriptions)
    dataset_description = (
        'A Multitask dataset based on CuBERT datasets below:\n' +
        single_description)

    super().__init__(
        *args,
        dataset_name=('cubert_multitask_rebalanced_dataset'
                      if rebalance_skew else 'cubert_multitask_dataset'),
        dataset_description=(dataset_description +
                             ('\n\nREBALANCED' if rebalance_skew else '')),
        stage_1_dir=stage_1_dir,
        configuration=configuration,
        urls={},
        git_url={},
        **kwargs)
    self.dataset_extracted_dir = os.path.join(self.raw_data_dir, 'json')

    # Will be filled after download.
    self.tokenizer = None
    self.subword_text_encoder = None

  def unitask_dataset_classes(self):
    return self._UNITASK_DATSETS

  def download_dataset(self):
    """As per superclass."""
    for dataset in self.unitask_datasets.values():
      print(f'Downloading {dataset.dataset_name}')
      dataset.download_dataset()

  def get_all_raw_data_paths(self):
    """Get paths to all raw data. Abusing type to return dataset type too."""
    data_paths = []
    for acronym, dataset in self.unitask_datasets.items():
      for path in dataset.get_all_raw_data_paths():
        # Since there's no way to return metadata with each path -- in
        # particular, the unitask name -- we combine the path and the type
        # into a json-formatted string, which we return as a path. This isn't
        # pretty and could be improved with more thought.
        task_path = {
            TASK_TYPE_FIELD_NAME: acronym,
            PATH_FIELD_NAME: path}
        task_path_json = json.dumps(task_path)
        data_paths.append(task_path_json)
    return data_paths

  def raw_data_paths_to_raw_data_do_fn(self):
    """As per superclass."""
    return JsonExtractor(
        random_split_fn=self.get_random_split,
        use_random_split=bool(self.user_defined_split_range),
        unitask_duplication=self.unitask_duplication,
        random_seed=self.seed)

  def raw_data_to_graph_to_output_example(
      self, raw_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert raw data to the unified GraphToOutputExample data structure.

    This extends the graphination done by the unitask dataset, by adding
    a final task node, holding the task acronym. This allows the model to know
    which task it is supposed to be performing given an input. We stick to the
    flat CuBERT graphination, but we could have made this extra node a super
    node connected to all token nodes.

    Given that CuBERT graphinators trim their examples to fix the maximum
    graph size, we take away one from the graph size budget, so we can have
    space for the final task node.

    Args:
      raw_data: As per superclass. However, we assume that the dictionary
        also has a 'task' field, created by the JsonExtractor below.

    Raises:
      GraphToOutputExampleNotValidError if the GraphToOutputExample is not
      valid.

    Returns:
      A dictionary with keys 'split' and 'GraphToOutputExample'. Values are the
      split(train/validation/test) the data belongs to, and the
      GraphToOutputExample instance.
    """
    unitask_acronym = raw_data[TASK_TYPE_FIELD_NAME]
    split = raw_data['split']
    # Reserving the last token for the task acronym.
    max_graph_size = self.configuration.max_graph_sizes[split] - 1

    data = raw_data['data']
    dataset = self.unitask_datasets[unitask_acronym]
    graph_to_output_example = dataset.data_to_graph_to_output_example(
        data, max_graph_size, split)

    if unitask_acronym != 'VMR' and not graph_to_output_example:
      # We should never get None for anything other than VMR. VMR may generate
      # impossible examples that the downstream filter would not catch, so
      # we delete them here.
      raise AssertionError('Failed to produce a G2OE for raw record '
                           f'{raw_data}')

    if graph_to_output_example:
      # Append the acronym as the last node, with a different type.
      length = len(graph_to_output_example.get_nodes())
      graph_to_output_example.add_node(
          node_id=length,
          node_type=TASK_TYPE_FIELD_NAME,
          node_label=unitask_acronym)
      graph_to_output_example.add_additional_field(
          field_name=TASK_TYPE_FIELD_NAME, field_value=unitask_acronym)

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
  """Class to read the CuBERT Exception Classification json files."""

  def __init__(self, random_split_fn, use_random_split: bool,
               unitask_duplication: Mapping[str, float],
               random_seed: int = 0) -> None:
    self.random_split_fn = random_split_fn
    self.use_random_split = use_random_split
    self.unitask_duplication = unitask_duplication
    self.dataset_seed = random_seed
    self.my_random = random.Random()  # Used for example duplication.

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

  def _open_path(self, path: str):
    return open(path)

  def process(self, file_path: str) -> Iterable[Mapping[str, Any]]:
    """Function to read each json file.

    Args:
      file_path: Originally path to a raw data file. It's actually a json-
        encoded object containing the path and the task acronym.

    Yields:
      A dictionary with 'split' and 'data' as keys. The value of the 'split'
      field is the split (train/valid/test) that the data belongs to. The value
      of the 'data' is the parsed raw json data. Also adding the task acronym
      in there.
    """
    task_path = json.loads(file_path)
    task_acronym = task_path[TASK_TYPE_FIELD_NAME]
    actual_path = task_path[PATH_FIELD_NAME]
    split = (
        self.random_split_fn()
        if self.use_random_split else self._get_split(actual_path))
    with self._open_path(actual_path) as f:
      if split == constants.TRAIN_SPLIT_NAME and self.unitask_duplication:
        # We only do rebalancing for the train split.
        # We decide how much to duplicate for each line separately, but we
        # want to always make the same decision for a given line. So
        # we use the basename (don't care about full path, since it might
        # change), line count, and line content as seed. We also use the
        # dataset seed.
        duplication = self.unitask_duplication[task_acronym]
        full_copies = int(duplication)
        partial_copy_probability = duplication - full_copies
        basename = os.path.basename(actual_path)
        for line_index, line in enumerate(f):
          self.my_random.seed(
              f'{self.dataset_seed}.{basename}_{line_index}_{line}')
          json_data = json.loads(line)
          extra_copy = int(
              self.my_random.uniform(0.0, 1.0) < partial_copy_probability)
          copies_to_make = full_copies + extra_copy
          for _ in range(copies_to_make):
            yield {
                'split': split,
                'data': json_data,
                TASK_TYPE_FIELD_NAME: task_acronym
            }
      else:
        for line in f:
          json_data = json.loads(line)
          yield {
              'split': split,
              'data': json_data,
              TASK_TYPE_FIELD_NAME: task_acronym
          }
