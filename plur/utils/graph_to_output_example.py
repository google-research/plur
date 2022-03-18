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

"""Definition of GraphToOutputExample, it represents one sample in PlurDataset.

GraphToOutputExample is a data structure that unifies all data that PLUR reads,
it is basically a dictionary with fields 'nodes', 'edges' and 'output'. 'nodes'
represents the nodes in the graph, 'edges' represents the edges in the graph,
and 'output' represents the output, which can be a token, a pointer, or a class.
All the information are stored inside dictionaries because it is easier to store
them in JSON format.

The 'nodes' field:
  The 'nodes' field contains a list of 'node' dictionaries. Each 'node' must
  have values in 'id', 'type' and 'label' fields. The 'id' field is the node id
  and must be unique among all 'nodes', 'id' must also be zero based numbering.
  The 'type' field is a string that is the node's type and the 'label' field is
  a string that is the node's label. 'node' can contain additional fields
  that are specified in kwargs, but they are not required for processing
  GraphToOutputExample in PlurDataset.

The 'edges' field:
  The 'edges' field contains a list of 'edge' dictionaries. Each 'edge' must
  have values in 'src', 'dst' and 'edge_type' fields. The 'src' field is the
  source node id of the edge. The 'dst' field is the destination node id of the
  edge. The 'edge_type' field is a string that is the edge's type. The tuple
  (src, dst, edge_type) must be unique among all 'edges'. 'edge' can contain
  additional fields that are specified in kwargs, but they are not required for
  processing GraphToOutputExample in PlurDataset.

The 'output' field:
  The 'output' field contains a list of 'token', 'pointer' and 'class'
  dictionaries. The 'token' dictionary has field 'token' and the value is the
  expected output token. The 'pointer' dictionary has field 'pointer' and the
  value is the expected pointed node id. The 'class' dictionary has field
  'class' and the value is the expected output class.

Having 'nodes' and 'edges' allows GraphToOutputExample to represent a graph
input. In the 'output' field, having 'token', 'pointer' and 'class' fields
allows GraphToOutputExample to represent tasks such as fault localization
(using 'pointer'), vulnerability classification (using 'class'), automatic
program repair (using 'token' and 'pointer') etc.
"""
import hashlib
from typing import Mapping, Sequence, Union

from plur.utils import constants
from plur.utils import util
import typing_extensions


class DataDict(typing_extensions.TypedDict):
  nodes: Sequence[Mapping[str, Union[int, str]]]
  edges: Sequence[Mapping[str, Union[int, str]]]
  output: Sequence[Mapping[str, Union[int, str]]]
  provenance: str


class GraphToOutputExample():
  """The unified data structure, see above for more information."""

  def __init__(self):
    self._data = DataDict(
        nodes=[],
        edges=[],
        output=[],
        provenance='',
    )
    self._unique_nodes = set()
    self._unique_edges = set()

  def __repr__(self) -> str:
    return (f'nodes: {self._data["nodes"]}\n'
            f'edges: {self._data["edges"]}\n'
            f'output: {self._data["output"]}')

  def add_node(self, node_id: int, node_type: str, node_label: str, **kwargs):
    """Add node to GraphToOutputExample.

    Args:
      node_id: id of the node.
      node_type: node type.
      node_label: node label.
      **kwargs: A dictionary of additional key and value that are added to the
        node dictionary.

    Raises:
      ValueError if the node_id is duplicate.
    """
    node = {
        'id': node_id,
        'type': util.escaped_str(node_type),
        'label': util.escaped_str(node_label)
    }
    node.update(kwargs)

    if node_id in self._unique_nodes:
      raise ValueError('Duplicate node_id {}'.format(node_id))
    else:
      self._unique_nodes.add(node_id)
    self._data['nodes'].append(node)

  def add_edge(self, src_id: int, dst_id: int, edge_type: str, **kwargs):
    """Add bidirectional edges to GraphToOutputExample."""
    reversed_edge_type = 'REVERSED_' + edge_type
    self.add_unidirectional_edge(src_id, dst_id, edge_type, **kwargs)
    self.add_unidirectional_edge(dst_id, src_id, reversed_edge_type, **kwargs)

  def add_unidirectional_edge(self, src_id: int, dst_id: int, edge_type: str,
                              **kwargs):
    """Add edge to GraphToOutputExample.

    Args:
      src_id: source node id of the edge.
      dst_id: destination node id of the edge.
      edge_type: edge type.
      **kwargs: A dictionary of additional key and value that are added to the
        edge dictionary.

    Raises:
      ValueError if the (src_id, dst_id, edge_type) tuple is duplicate.
    """
    edge = {
        'src': src_id,
        'dst': dst_id,
        'type': util.escaped_str(edge_type)
    }
    edge.update(kwargs)
    if (src_id, dst_id, edge_type) in self._unique_edges:
      raise ValueError('Duplicate edge {} {} {}'.format(
          src_id, dst_id, edge_type))
    else:
      self._unique_edges.add((src_id, dst_id, edge_type))
    self._data['edges'].append(edge)

  def add_token_output(self, token: str):
    self._data['output'].append({
        'token': util.escaped_str(token)
    })

  def add_pointer_output(self, node_id: int):
    self._data['output'].append({
        'pointer': node_id
    })

  def add_class_output(self, class_name: str):
    self._data['output'].append({
        'class': util.escaped_str(class_name)
    })

  def set_provenance(self, provenance: str):
    self._data['provenance'] = provenance

  def set_data(self, data):
    """Set the data attribute.

    It is usually called when reading GraphToOutputExample from disk. Otherwise
    use the add_* function to construct the data.

    Args:
      data: The graph dictionary, containing nodes, edges and output.

    Raises:
      ValueError if there are duplicate nodes or edges.
    """
    # For backward compatibility.
    if 'provenance' not in data:
      data['provenance'] = ''
    self._data = data
    self._unique_nodes = set()
    self._unique_edges = set()
    for node in self.get_nodes():
      if node['id'] in self._unique_nodes:
        raise ValueError('Duplicate node id {}'.format(node['id']))
      else:
        self._unique_nodes.add(node['id'])

    for edge in self.get_edges():
      if (edge['src'], edge['dst'], edge['type']) in self._unique_edges:
        raise ValueError('Duplicate edge {} {} {}'.format(
            edge['src'], edge['dst'], edge['type']))

  def get_data(self):
    return self._data

  def get_nodes(self):
    self._data['nodes'].sort(key=lambda x: x['id'])
    return self._data['nodes']

  def get_node_types(self):
    return [node['type'] for node in self.get_nodes()]

  def get_node_labels(self):
    return [node['label'] for node in self.get_nodes()]

  def get_edges(self):
    return self._data['edges']

  def get_edge_types(self):
    return [edge['type'] for edge in self.get_edges()]

  def get_num_edge_types(self):
    edge_type_set = set()
    for edge_type in self.get_edge_types():
      edge_type_set.add(edge_type)
    return len(edge_type_set)

  def get_output(self):
    return self._data['output']

  def get_output_as_tokens(self, append_done=True, include_pointer_id=False):
    """Get the output as tokens.

    Convert the output to tokens. For outputs are are not tokens, we convert
    them into tokens. This can be used for debugging purposes, ie. print
    the output for this specific GraphToOutputExample. It can also be used to
    build the output token vocabulary.

    Args:
      append_done: Boolean indicating if we should append the output token
        with a constants.DONE_TOKEN at the end.
      include_pointer_id: Boolean indicating when converting pointer to token,
        if we should include the pointed node id. For example if the pointer
        is pointed to node 3, then include_pointer_id=False will be 'POINTER',
        and include_pointer_id=True will be 'POINTER(3)'

    Returns:
      A list of tokens representing the output.
    """
    output_tokens = []
    for output in self._data['output']:
      if 'token' in output:
        output_tokens.append(output['token'])
      elif 'pointer' in output:
        if include_pointer_id:
          output_tokens.append(constants.POINTER_TOKEN + '({})'.format(
              output['pointer']))
        else:
          output_tokens.append(constants.POINTER_TOKEN)
      elif 'class' in output:
        output_tokens.append(constants.CLASS_TOKEN_PREFIX + output['class'])
    if append_done:
      output_tokens.append(constants.DONE_TOKEN)
    return output_tokens

  def get_output_tokens_and_index(self):
    """Get output that are token and their indices."""
    output_tokens_and_index = []
    for index, output in enumerate(self._data['output']):
      if 'token' in output:
        output_tokens_and_index.append((index, output['token']))
    return output_tokens_and_index

  def get_output_pointers_and_index(self):
    """Get output that are pointer and their indices."""
    output_pointers_and_index = []
    for index, output in enumerate(self._data['output']):
      if 'pointer' in output:
        output_pointers_and_index.append((index, output['pointer']))
    return output_pointers_and_index

  def get_output_class_and_index(self):
    """Get output that are class and their indices."""
    output_class_and_index = []
    for index, output in enumerate(self._data['output']):
      if 'class' in output:
        output_class_and_index.append((index, output['class']))
    return output_class_and_index

  def add_additional_field(self, field_name, field_value):
    self._data[field_name] = field_value

  def get_field_by_name(self, field_name):
    return self._data[field_name]

  def get_provenance(self) -> str:
    return self._data['provenance']

  def check_if_valid(self) -> bool:
    """Check if self is a valid instance of GraphToOutputExample.

    This checks that all mandatory fields, such as 'nodes', 'edges' and
    'output', exists and has the right type, along with other checks. This is
    intended to be used when the user defines their own transformation on the
    GraphToOutputExample and wants to check that the transformation returns a
    valid GraphToOutputExample instance. This can also be used when the data
    in the dataset is not in graph format, and we create our own nodes and
    edges.

    Returns:
      A boolean indicating if self is a valid GraphToOutputExample.
    """
    if not isinstance(self._data, dict):
      return False

    if 'nodes' not in self._data:
      return False
    elif not isinstance(self._data['nodes'], list):
      return False

    if 'edges' not in self._data:
      return False
    elif not isinstance(self._data['edges'], list):
      return False

    if 'output' not in self._data:
      return False
    elif not isinstance(self._data['output'], list):
      return False

    node_ids_list = []
    for node in self._data['nodes']:
      if 'id' not in node:
        return False
      elif not isinstance(node['id'], int):
        return False

      if 'type' not in node:
        return False
      elif not isinstance(node['type'], str):
        return False

      if 'label' not in node:
        return False
      elif not isinstance(node['label'], str):
        return False

      node_ids_list.append(node['id'])

    # Checks if there are duplicate node ids.
    node_ids_set = set(node_ids_list)
    if len(node_ids_list) != len(node_ids_set):
      return False

    # This checks that all node ids are in [0, n], where n the number of nodes.
    if sorted(node_ids_set) != list(range(max(node_ids_set)+1)):
      return False

    edge_list = []
    for edge in self._data['edges']:
      if 'src' not in edge:
        return False
      elif not isinstance(edge['src'], int):
        return False
      elif edge['src'] not in node_ids_set:
        return False

      if 'dst' not in edge:
        return False
      elif not isinstance(edge['dst'], int):
        return False
      elif edge['dst'] not in node_ids_set:
        return False

      if 'type' not in edge:
        return False
      elif not isinstance(edge['type'], str):
        return False

      edge_list.append((edge['src'], edge['dst'], edge['type']))

    # Checks if there are duplicate edges.
    edge_set = set(edge_list)
    if len(edge_list) != len(edge_set):
      return False

    return True

  def compute_hash(self) -> int:
    """Compute a hash for GraphToOutputExample."""
    # This is not implemented as a __hash__() function because the object is
    # mutable.
    # We define a canonicalized representation for GraphToOutputExample by
    # sorting the nodes, edges and outputs.
    nodes = self.get_nodes()  # get_nodes() already sorts the nodes.
    # Sort the edges
    edges = self.get_edges()
    edges = sorted(edges, key=lambda e: (e['src'], e['dst'], e['type']))
    # The output is order depedent, therefore we don't have to sort them.
    outputs = self.get_output()

    # The canonicalized representation is the contactnated string. Nodes,
    # edges and outputs should only contain primitive types, therefore calling
    # str() on them should be ok.
    canonicalized_string = str(nodes) + str(edges) + str(outputs)
    return int(hashlib.md5(canonicalized_string.encode('utf-8')).hexdigest(),
               16)


class GraphToOutputExampleNotValidError(Exception):
  """Exception to be thrown if check_if_valid fails."""
  pass
