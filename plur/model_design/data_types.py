# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type definitions for various PLUR data types.

Also includes some utilities for type annotations and runtime checking.
"""
import dataclasses
import enum
import inspect
import typing
from typing import Callable, Mapping, NewType, Optional, TYPE_CHECKING, Tuple, Type, TypeVar, Union, cast

import flax
import jax
import numpy as np

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


@dataclasses.dataclass
class Graph2TocopoPaddingSpec:
  num_nodes_per_graph: int
  num_edge_types: int
  output_length: int
  num_input_timesteps: int = 1
  num_time_edge_types: int = 1
  node_position_dim: int = 1


class BaseNDArray(np.ndarray):
  dtype: Optional[np.dtype] = None
  shape_names: str = ''


# Allow functions that masquerade as type names, as in the implementation of
# typing.NewType, for example.

if typing.TYPE_CHECKING:

  # At static type-checking time, just treat all NDArrays as np.ndarray.
  def NDArrayType(unused_arg1, unused_arg2):
    return BaseNDArray

else:

  # At runtime, we make a separate type for each type + shape combination.
  def NDArrayType(numeric_type, shape_names):
    """Wrapper around typing.NewType to support N-dimensional arrays.

    We add additional annotations about the data type (dtype) and shape of the
    arrays, to support runtime type and shape checking.

    Implementation mimics that of typing.NewType.

    Args:
      numeric_type: Type of elements of the array.
      shape_names: String with a character naming each dimension in order.
        E.g., 'ehh' for a num_edge_types x num_hiddens x num_hiddens tensor.

    Returns:
      A new type specific to the shape and type of the array.
    """
    dtype = np.dtype(numeric_type)
    new_type_name = 'NDArray_{}_{}'.format(dtype, shape_names)
    new_type = NewType(new_type_name, np.ndarray)
    new_type.dtype = dtype
    new_type.shape_names = shape_names
    return new_type



# Abbreviations used for shape annotations in comments. These are also chosen
# to be consistent with named dimensions in einsum operations inside the models.
# * b: Number of graphs in a batch.
# * t: Number of input timesteps per sample.
# * n: Number of attention heads.
# * v: Number of nodes in a graph.
# * e: Number of edge types.
# * s: Number of subtokens in a node. TODO: Simplify all to S=1?
# * p: Number of positional indices per node.
# * h: Number of hidden dimensions.

# For node-related data and model parameters.
NDArrayIntB = NDArrayType(np.int, 'b')
NDArrayIntBV = NDArrayType(np.int, 'bv')
NDArrayIntBTV = NDArrayType(np.int, 'btv')
NDArrayIntBTVP = NDArrayType(np.int, 'btvp')
NDArrayIntBTVS = NDArrayType(np.int, 'btvs')
NDArrayFloatBTVSH = NDArrayType(np.float32, 'btvsh')
# Without time axis, for decoders that use only the final encoded state.
NDArrayFloatB = NDArrayType(np.float32, 'b')
NDArrayFloatBVH = NDArrayType(np.float32, 'bvh')
NDArrayFloatBTVH = NDArrayType(np.float32, 'btvh')
NDArrayFloatBTNVH = NDArrayType(np.float32, 'btnvh')
NDArrayObjBTV = NDArrayType(np.object, 'btv')

# For edge-related data and model parameters.
NDArrayBoolBTEVV = NDArrayType(np.bool, 'btevv')
NDArrayBoolBVETT = NDArrayType(np.bool, 'bvett')
NDArrayFloatEH = NDArrayType(np.float32, 'eh')
NDArrayFloatEHH = NDArrayType(np.float32, 'ehh')

# For pointer candidates.
NDArrayBoolBTV = NDArrayType(np.bool, 'btv')

# For target outputs.
# * o: Length of output Tocopo sequence.
# * u: Output vocab size
NDArrayIntBO = NDArrayType(np.int, 'bo')
NDArrayBoolBO = NDArrayType(np.bool, 'bo')
NDArrayBoolBOV = NDArrayType(np.bool, 'bov')
NDArrayBoolBNOV = NDArrayType(np.bool, 'bnov')
NDArrayBoolBTNOV = NDArrayType(np.bool, 'btnov')
NDArrayBoolBOO = NDArrayType(np.bool, 'boo')  # For causal self-attention.
NDArrayFloatBOU = NDArrayType(np.float32, 'bou')
NDArrayFloatBOV = NDArrayType(np.float32, 'bov')
NDArrayFloatBNOV = NDArrayType(np.float32, 'bnov')
NDArrayFloatBV = NDArrayType(np.float32, 'bv')
NDArrayFloatBOH = NDArrayType(np.float32, 'boh')
NDArrayFloatBNOH = NDArrayType(np.float32, 'bnoh')
NDArrayObjB = NDArrayType(np.object, 'b')
NDArrayObjBO = NDArrayType(np.object, 'bo')

# For encoder-decoder attention.
NDArrayFloatBTOH = NDArrayType(np.float32, 'btoh')
NDArrayFloatBTNOH = NDArrayType(np.float32, 'btnoh')
NDArrayFloatBTOV = NDArrayType(np.float32, 'btov')
NDArrayFloatBTNOV = NDArrayType(np.float32, 'btnov')
NDArrayBoolBTOV = NDArrayType(np.bool, 'btov')

# For decoder parameters.
NDArrayFloatHU = NDArrayType(np.float32, 'hu')

# For target losses.
NDArrayFloatBO = NDArrayType(np.float32, 'bo')


###############################################################################
#
#  Data formats
#
###############################################################################
@dataclasses.dataclass(frozen=True)
class BaseDataclass:
  _HAS_DYNAMIC_ATTRIBUTES = True  # For type-checking dynamic attribute access.


# BatchedTrainGraphNodeData contains all necessary arrays of graph nodes for
# training.  It is decorated with @flax_dataclass to make it safe
# to pass to Jax.
@flax_dataclass
class BatchedTrainGraphNodeData(BaseDataclass):
  """All arrays from graph nodes used in training."""
  token_ids: NDArrayIntBTVS
  type_ids: NDArrayIntBTV
  token_positions: Optional[NDArrayIntBTVP] = None
  # Allowed pointer candidates.
  pointer_candidates: Optional[NDArrayBoolBTV] = None


# BatchedEvalGraphNodeData contains all necessary arrays of graph nodes for
# testing. It is decorated with @dataclasses.dataclass to show that
# it should not be used in training.
@dataclasses.dataclass(frozen=True)
class BatchedEvalGraphNodeData(BaseDataclass):
  """All arrays from graph nodes used in testing."""
  node_texts: NDArrayObjBTV


# BatchedTrainGraphEdgeData contains all necessary arrays of graph edges for
# training. It is decorated with @flax_dataclass to make it safe to pass to Jax.
@flax_dataclass
class BatchedTrainGraphEdgeData(BaseDataclass):
  """All arrays from graph edges used in training."""
  edges: NDArrayBoolBTEVV
  time_edges: Optional[NDArrayBoolBVETT] = None


# BatchedTocopoTargetData contains all necessary arrays of tocopo target for
# training and testing. It is decorated with @dataclasses.dataclass to signal
# that it should not be used for training. It stores all arrays when tensorizing
# the tocopo target.
@dataclasses.dataclass(frozen=True)
class BatchedTocopoTargetData(BaseDataclass):
  """All arrays from tocopo target used in training and testing."""
  token_ids: NDArrayIntBO
  is_target_copy: NDArrayBoolBOV
  is_target_pointer: NDArrayBoolBOV
  tokens: NDArrayObjBO


# BatchedTrainTocopoTargetData contains all necessary arrays of tocopo target
# for training. The fields are a subset of BatchedTocopoTargetData fields, which
# are used during training. It is decorated with @flax_dataclass to make it safe
# to pass to Jax.
@flax_dataclass
class BatchedTrainTocopoTargetData(BaseDataclass):
  """All arrays from tocopo target used in training."""
  token_ids: NDArrayIntBO
  is_target_copy: NDArrayBoolBOV
  is_target_pointer: NDArrayBoolBOV


# BatchedEvalTocopoTargetData contains all necessary arrays of tocopo target
# for testing. The fields are a subset of BatchedTocopoTargetData fields, which
# are used during testing. It is decorated with @dataclasses.dataclass to show
# that it should not be used in training.
@dataclasses.dataclass(frozen=True)
class BatchedEvalTocopoTargetData(BaseDataclass):
  """All arrays from tocopo target used in testing."""
  tokens: NDArrayObjBO


@flax_dataclass
class BatchedTrainTocopoData(BaseDataclass):
  node_data: BatchedTrainGraphNodeData
  edge_data: BatchedTrainGraphEdgeData
  target_data: Optional[BatchedTrainTocopoTargetData]


@dataclasses.dataclass(frozen=True)
class BatchedEvalTocopoData(BaseDataclass):
  node_data: BatchedEvalGraphNodeData
  target_data: BatchedEvalTocopoTargetData
  provenance: Optional[NDArrayObjB] = None


@flax_dataclass
class BatchedTocopoLogits(BaseDataclass):
  token_logits: NDArrayFloatBOU
  copy_logits: NDArrayFloatBOV
  pointer_logits: NDArrayFloatBOV


@dataclasses.dataclass(frozen=True)
class TrainingConfiguration(BaseDataclass):
  """Configuration details for training."""
  loss: Callable[[BatchedTocopoLogits, BatchedTrainTocopoTargetData], float]
  # Although we'd like to type the arguments, Callable does not support optional
  # or keyword arguments.
  accuracy: Callable[..., Tuple[int, int, int, int]]
  optimizer: 'typing.Any'
  num_training_steps: int
  valid_steps: int
  train_logging_steps: int
  max_validation_batches: int
  exp_dir: str
  checkpoint_dir: str
  checkpoint_every_n_steps: int


@dataclasses.dataclass(frozen=True)
class EvaluationConfiguration(BaseDataclass):
  """Configuration details for evaluation."""
  eval_flush_steps: int
  checkpoint_dir: str
  evaluation_dir: str
  evaluation_mode: str
  loss: Optional[Callable[[BatchedTocopoLogits, BatchedTrainTocopoTargetData],
                          float]] = None
  # Although we'd like to type the arguments, Callable does not support optional
  # or keyword arguments.
  accuracy: Optional[Callable[..., Tuple[int, int, int, int]]] = None


# Integer enumeration for the tocopo (token, copy and pointer) kinds. The
# integers (0, 1 and 2) are the indices of token, copy and pointer logits
# when we sample predictions in training._sample_prediction().
class TocopoKind(enum.IntEnum):
  TOKEN = 0
  COPY = 1
  POINTER = 2


###############################################################################
#
#  Utility for generating test data of the right type and shape.
#
###############################################################################
DataOrArray = TypeVar('DataOrArray', BaseDataclass, BaseNDArray)
def get_array_value_shape(array_value: DataOrArray,
                          array_type: Type[DataOrArray]):
  """Gets a description of the shape of data.

  Handles dataclasses and NDArrayTypes. If the inputs are a dataclass, then the
  result is returned as a dataclass with the same hierarchical structure, but
  shape tuples as the leaf values.

  Args:
    array_value: A dataclass or NDArrayType with concrete data values.
    array_type: A dataclass type or NDArrayType.

  Returns:
    A shape tuple or a dataclass with the same structure as `array_value` but
    with shape tuples at the leaf values.
  """
  if dataclasses.is_dataclass(array_type):
    dataclass_array_value = cast(BaseDataclass, array_value)
    dataclass_array_type = cast(Type[BaseDataclass], array_type)
    return dataclass_array_type(**{
        field.name: get_array_value_shape(dataclass_array_value[field.name],
                                          field.type)
        for field in dataclasses.fields(dataclass_array_type)
    })
  else:
    nd_array_value = array_value  # type: np.ndarray
    return nd_array_value.shape


def get_array_value_dtype(array_value: DataOrArray,
                          array_type: Type[DataOrArray]):
  """Gets the dtype associated with a BaseDataclass or NDArrayType.

  We abuse BaseDataclass a bit here, saying that the dtype of a BaseDataclass is
  a dataclass with the same hierarchical structure as BaseDataclass but with
  `np.dtype` as the value of the leaf objects.

  Args:
    array_value: A dataclass or NDArrayType with concrete data values.
    array_type: A dataclass type or NDArrayType.

  Returns:
    A dtype or a dataclass with the same structure as `array_value` but with
    dtypes at the leaf values.
  """
  if dataclasses.is_dataclass(array_type):
    dataclass_array_value = cast(BaseDataclass, array_value)
    dataclass_array_type = cast(Type[BaseDataclass], array_type)
    return dataclass_array_type(**{
        field.name: get_array_value_dtype(dataclass_array_value[field.name],
                                          field.type)
        for field in dataclasses.fields(dataclass_array_type)
    })
  else:
    nd_array_value = array_value  # type: np.ndarray
    return nd_array_value.dtype


class NDArrayGenerator:
  """Generates numpy NDArrays with consistent sizes for each named dimension.

  Also supports some basic consistency checks for use in dynamically checking
  shapes and types of NDArrayType arrays.
  """

  def __init__(self, fixed_sizes: Optional[Mapping[str, int]] = None, seed=0):
    self.dimension_sizes = {}   # Mapping from dimension name to size.
    self.random_state = np.random.RandomState(seed)
    if fixed_sizes is not None:
      self.dimension_sizes.update(fixed_sizes)

  def get_shape(self,
                array_type: Union[Type[BaseDataclass], Type[BaseNDArray]]):
    """Gets the shape associated with a BaseDataclass or np.ndarray.

    We abuse BaseDataclass a bit here, saying that the shape of a BaseDataclass
    is a dataclass with the same hierarchical structure as BaseDataclass but
    with shape tuples as the value of the leaf objects.

    Args:
      array_type: A dataclass type or NDArrayType.

    Returns:
      A shape tuple or a dataclass with the same structure as `array_type`
      instances but with shape tuples at the leaf values.
    """
    if dataclasses.is_dataclass(array_type):
      dataclass_array_type = array_type  # type: Type[BaseDataclass]
      return dataclass_array_type(**{
          field.name: self.get_shape(field.type)
          for field in dataclasses.fields(dataclass_array_type)
      })
    else:
      nd_array_type = array_type  # type: Type[BaseNDArray]
      return tuple(self.dimension_sizes[name]
                   for name in nd_array_type.shape_names)

  def isinstance(self, array_value, array_type):
    """Checks if `ndarray` is of type `array_type`.

    If `ndarray` is a BaseDataclass of type `array_type`, then we recursively
    check whether all fields are of the right type and shape.

    Args:
      array_value: A BaseDataclass instance or NDArrayType instance.
      array_type: A BaseDataclass type or NDArrayType type.

    Returns:
      True if `array_value` is recursively of the shape and type expected of a
      `array_type` instance.
    """
    if dataclasses.is_dataclass(array_type):
      for field in dataclasses.fields(array_type):
        if (not hasattr(array_value, field.name) or
            not self.isinstance(getattr(array_value, field.name), field.type)):
          return False
      return True

    return (array_value.dtype == array_type.dtype and
            np.all(array_value.shape == self.get_shape(array_type)))

  def random(self, array_type: Union[Type[BaseDataclass], Type[BaseNDArray]]):
    """Generates a random array with the given shape and type."""
    if dataclasses.is_dataclass(array_type):
      # TODO: enable type hints for Optional typed fields.

      return array_type(
          **{
              field.name: self.random(field.type)
              for field in dataclasses.fields(array_type)
              if field.name not in ('time_edges', 'token_positions',
                                    'pointer_candidates')
          })

    nd_array_type = array_type  # type: Type[BaseNDArray]
    for name in nd_array_type.shape_names:
      if name not in self.dimension_sizes:
        self.dimension_sizes[name] = self.random_state.randint(5, 20)

    shape = self.get_shape(nd_array_type)
    if nd_array_type.dtype == np.int:
      return self.random_state.randint(0, 5, size=shape)
    elif nd_array_type.dtype == np.bool:
      return self.random_state.rand(*shape) > .5
    else:
      return self.random_state.randn(*shape).astype(nd_array_type.dtype)


def execute_with_random_arguments(layer, array_generator):
  """Executes a layer object with a random array of the right shape and type.

  Args:
    layer: A model object that implements the forward pass of the model via the
      __call__ method. Must be annotated with argument and return value types
      using the types in this module.
    array_generator: A NDArrayGenerator used to generate random arrays with
      consistent shapes.

  Returns:
    A tuple of
    * the output of layer.__call__, and
    * the expected output type according to the return value annotation on
      layer.__call__.
  """
  argspec = inspect.getfullargspec(layer.__call__)
  if argspec.args[0] != 'self': raise ValueError('Did not pass in a layer.')

  type_hints = typing.get_type_hints(layer.__call__)
  # Exclude `rng` from this treatment.
  args = [arg for arg in argspec.args[1:] if arg != 'rng']
  random_arg_data = [array_generator.random(type_hints[arg]) for arg in args]

  output = layer(*random_arg_data, rng=jax.random.PRNGKey(0))
  return output, type_hints['return']


def dynamic_shapes_are_consistent(layer, array_generator):
  """Checks that a layer produces outputs of the right shape and type.

  Assumes all arguments are positional (as typical in layers).

  Args:
    layer: A class that defines a neural network layer. Must implement the
      forward calculation via the __call__ method and have have NDArrayType
      hints for all arguments and return types.
    array_generator: An NDArrayGenerator.

  Returns:
    A boolean indicating if the result of executing the layer with random input
    arrays produces outputs consistent with the type annotations.
  """

  result, expected_return_type = (
      execute_with_random_arguments(layer, array_generator))

  return array_generator.isinstance(result, expected_return_type)
