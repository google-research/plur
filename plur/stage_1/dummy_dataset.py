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

"""Classes for converting the dummy dataset to a PLUR dataset."""
import string

import apache_beam as beam
from plur.stage_1.plur_dataset import Configuration
from plur.stage_1.plur_dataset import PlurDataset
from plur.utils.graph_to_output_example import GraphToOutputExample
from plur.utils.graph_to_output_example import GraphToOutputExampleNotValidError


class DummyDataset(PlurDataset):
  """DummyDataset that contains random data, it is used for testing PlurDataset."""
  _URLS = {}
  _GIT_URL = {}
  _DATASET_NAME = 'dummy_dataset'
  _DATASET_DESCRIPTION = """\
  This dataset is only used for test the data generation process, all data are
  generated randomly.
  """

  def __init__(self,
               stage_1_dir,
               configuration: Configuration = Configuration(),
               transformation_funcs=(),
               filter_funcs=(),
               user_defined_split_range=(),
               num_shards=1000,
               seed=0,
               num_random_graph=1000,
               min_node_per_graph=100,
               max_node_per_graph=1000,
               deduplicate=False):
    self.num_random_graph = num_random_graph
    self.min_node_per_graph = min_node_per_graph
    self.max_node_per_graph = max_node_per_graph
    super().__init__(self._DATASET_NAME, self._URLS, self._GIT_URL,
                     self._DATASET_DESCRIPTION, stage_1_dir,
                     transformation_funcs=transformation_funcs,
                     filter_funcs=filter_funcs,
                     user_defined_split_range=user_defined_split_range,
                     num_shards=num_shards, seed=seed,
                     configuration=configuration, deduplicate=deduplicate)

  def download_dataset(self):
    """All data are generated on the fly, so we 'pass' here."""
    pass

  def get_all_raw_data_paths(self):
    """All data are generated on the fly, only return a dummy value."""
    return ['/unused_value/']

  def raw_data_paths_to_raw_data_do_fn(self):
    """Use RandomDataGenerator to generate random graphs."""
    return RandomDataGenerator(self.num_random_graph,
                               super().get_random_split,
                               self._generate_random_graph_to_output_example)

  def _generate_random_graph_to_output_example(self):
    """Generate a random GraphToOutputExample.

    The input graph is a chain where the node type are uppercase ASCII and the
    node label are lowercase ASCII. All nodes are connected to the previous node
    with lower node id to form a chain. The output is a combination of all
    possible output types (token/pointer/class).

    Returns:
      A random GraphToOutputExample.
    """
    graph_to_output_example = GraphToOutputExample()

    for i in range(self.random.randint(self.min_node_per_graph,
                                       self.max_node_per_graph)):
      node_type = self.random.choice(string.ascii_uppercase)
      node_label = self.random.choice(string.ascii_lowercase)
      graph_to_output_example.add_node(
          i, node_type, node_label, is_repair_candidate=(i % 2 == 0))

    for i in range(len(graph_to_output_example.get_nodes()) - 1):
      graph_to_output_example.add_edge(i, i+1, 'NEXT_NODE')

    graph_to_output_example.add_token_output('foo')
    graph_to_output_example.add_pointer_output(
        self.random.choice(
            list(range(len(graph_to_output_example.get_nodes())))))
    graph_to_output_example.add_class_output('0')

    return graph_to_output_example

  def raw_data_to_graph_to_output_example(self, raw_data):
    """Return raw data as it is."""
    split = raw_data['split']
    # The raw data is already a GraphToOutputExample instance.
    graph_to_output_example = raw_data['data']

    for transformation_fn in self.transformation_funcs:
      graph_to_output_example = transformation_fn(graph_to_output_example)

    if not graph_to_output_example.check_if_valid():
      raise GraphToOutputExampleNotValidError(
          'Invalid GraphToOutputExample found.')

    for filter_fn in self.filter_funcs:
      if not filter_fn(graph_to_output_example):
        graph_to_output_example = None
        break

    return {'split': split, 'GraphToOutputExample': graph_to_output_example}


class RandomDataGenerator(beam.DoFn):
  """Class that generates a random GraphToOutputExample."""

  def __init__(self, num_random_graph, random_split_fn,
               generate_random_graph_to_output_example_fn):
    self.num_random_graph = num_random_graph
    self.random_split_fn = random_split_fn
    self._generate_random_graph_to_output_example_fn = (
        generate_random_graph_to_output_example_fn)

  def process(self, _):
    for _ in range(self.num_random_graph):
      yield {'split': self.random_split_fn(),
             'data': self._generate_random_graph_to_output_example_fn()}
