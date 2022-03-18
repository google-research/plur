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

"""JAX layers for models."""

import dataclasses
import typing
from typing import List, Optional, TYPE_CHECKING, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from plur.model_design import data_types as dt
import trax


flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)

_START_TOKEN_ID = 0

# A large negative value that leads to a probability close to zero when
# exponentiated.
_LOG_OF_SMALL_VALUE = -1e9


@flax_dataclass
class Dense():
  """Wrapper around trax.layers.core.Dense."""

  # Parameters.
  weights: List[jnp.ndarray]

  # Additional fields that are not parameters.
  _layer: trax.layers.core.Dense = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng, input_dim: int, output_dim: int, use_bias=True):
    layer = trax.layers.core.Dense(output_dim, use_bias=use_bias)
    dummy_data = np.zeros((1, input_dim))
    layer.init(dummy_data, rng=rng)
    return Dense(weights=layer.weights, _layer=layer)

  def __call__(self, inputs, rng: jnp.ndarray):
    outputs, _ = self._layer.pure_fn(inputs, self.weights, self._layer.state,
                                     rng)
    return outputs


@flax_dataclass
class GRUCell():
  """Wrapper around trax.layers.rnn.GRUCell."""

  # Parameters.
  weights: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

  # Additional fields that are not parameters.
  _layer: trax.layers.rnn.GRUCell = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng, hidden_dim: int):
    layer = trax.layers.rnn.GRUCell(hidden_dim)
    dummy_data = np.zeros((1, hidden_dim))
    layer.init([dummy_data, dummy_data], rng=rng)
    return GRUCell(weights=layer.weights, _layer=layer)

  def __call__(self, *inputs, rng: jnp.ndarray):
    outputs, _ = self._layer.pure_fn(inputs, self.weights, self._layer.state,
                                     rng)
    # Extract the first element only; a call to trax GRUCell.forward(..)
    # will return (new_gru_state, new_gru_state).
    return outputs[0]


@flax_dataclass
class LayerNorm():
  """Wrapper around trax.layers.normalization.LayerNorm."""

  # Parameters.
  weights: Tuple[jnp.ndarray, jnp.ndarray]

  # Additional fields that are not parameters.
  _layer: trax.layers.normalization.LayerNorm = (
      flax.struct.field(pytree_node=False))

  @staticmethod
  def create(rng, hidden_dim: int):
    layer = trax.layers.normalization.LayerNorm()
    dummy_data = np.zeros((1, hidden_dim), dtype=np.float32)
    layer.init(dummy_data, rng=rng)
    return LayerNorm(weights=layer.weights, _layer=layer)

  def __call__(self, inputs, rng: jnp.ndarray):
    outputs, _ = self._layer.pure_fn(inputs, self.weights, self._layer.state,
                                     rng)
    return outputs


@flax_dataclass
class Dropout():
  """Wrapper around trax.layers.core.Dropout."""

  # Parameters.
  weights: Tuple[()]

  # Additional fields that are not parameters.
  _layer: trax.layers.core.Dropout = (
      flax.struct.field(pytree_node=False))

  @staticmethod
  def create(rng, deterministic: bool, dropout_rate: float, shared_axes=None):
    dropout_mode = 'train' if not deterministic else 'test'
    layer = trax.layers.core.Dropout(
        rate=dropout_rate, shared_axes=shared_axes, mode=dropout_mode)
    layer.init(None, rng=rng)
    return Dropout(weights=tuple(), _layer=layer)

  def __call__(self, inputs, rng: jnp.ndarray):
    outputs, _ = self._layer.pure_fn(inputs, self.weights, self._layer.state,
                                     rng)
    return outputs


@flax_dataclass
class PositionalEncoding():
  """A simpler-to-modify re-implementation of positional embeddings.

  There shouldn't be a major difference between this and
  the trax version, but this implementation doesn't support caching or
  other extra features in trax, which makes the implementation simpler. Here
  we have also multiplied `pos` by 2 pi in the sine and cosine calculations, so
  that positional embeddings have a bit more variability.
  """
  # TODO: update the Trax version with position reindexing if/when
  # that version becomes more broadly used.

  # Positional encodings are sinusoidal functions at different frequencies.
  # We want to use the same signal during the entire training
  # process so we want to turn off learning for these params. We do this by
  # using them along with jax.lax.stop_gradient.
  weights: jnp.ndarray
  null_position_embedding: Optional[jnp.ndarray]  #  Weight for absent tokens.
  per_position_dim: int = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng,
             max_length: int,
             hidden_dim: int,
             position_dim: int = 1):
    """Static initializer for PositionalEncoding module.

    Args:
      rng: Random number generator.
      max_length: maximum value of position to be encoded. Usually this is the
        max length of the input sequence.
      hidden_dim: Size of hidden dimension.
      position_dim: Number of positional indicators per token.

    Returns:
      module: A `PositionalEncoding` module object.
    """
    if hidden_dim % 2 != 0:
      raise ValueError(
          'Positional encoding layer should have even number of dimensions. '
          'Got: {} dimensions.'.format(hidden_dim))

    # max_length x 1
    pos = 1 + jnp.arange(max_length)[:, jnp.newaxis]

    # 1 x hidden_dim / 2
    i = jnp.arange(hidden_dim // 2)[jnp.newaxis, :]

    # max_length x hidden_dim / 2
    sines = jnp.sin(2 * np.pi * pos / (max_length ** (i * 2.0 / hidden_dim)))
    cosines = jnp.cos(2 * np.pi * pos / (max_length ** (i * 2.0 / hidden_dim)))

    # max_length x hidden_dim
    weights = jnp.concatenate([sines, cosines], axis=-1)

    # The weight for 'absent' tokens, if any. Useful in session modeling.
    null_position_embedding = jax.random.normal(rng, (hidden_dim,))
    per_position_dim = int(jnp.ceil(hidden_dim / position_dim))
    return PositionalEncoding(
        weights=weights,
        null_position_embedding=null_position_embedding,
        per_position_dim=per_position_dim)

  def __call__(
      self,
      rng: jnp.ndarray,
      inputs: dt.BaseNDArray,
      token_positions: Optional[dt.NDArrayIntBTVP] = None) -> dt.BaseNDArray:
    """Adds positional embeddings to `inputs`."""
    sequence_length = inputs.shape[-2]
    weights = jax.lax.stop_gradient(self.weights)
    # If token positions are given, make sure that we allocate the correct
    # positional encodings based on those.
    if token_positions is not None:
      result = inputs + self.reorder_positional_encodings(
          weights, token_positions)
    else:
      positional_encodings = jax.lax.dynamic_slice_in_dim(
          weights, 0, sequence_length)
      result = inputs + positional_encodings
    return result

  def reorder_positional_encodings(
      self,
      base_positional_encodings: dt.BaseNDArray,
      token_positions: dt.NDArrayIntBTVP) -> dt.NDArrayFloatBTVH:
    """Scatters positional encodings based on token positions."""
    all_positions = jnp.split(
        token_positions, token_positions.shape[-1], axis=-1)
    all_encodings = []
    # Each individual positional encoding will be clipped to its fraction of the
    # overall dimension, ceiled. The stacked result is additionally clipped to
    # the base encoding dimension.
    for positions in all_positions:  # 'positions' is of shape BTV1.
      encoding_at_positions = base_positional_encodings[jnp.squeeze(
          positions, axis=-1)]
      position_encoding = jnp.where(positions < 0, self.null_position_embedding,
                                    encoding_at_positions)
      all_encodings.append(
          jax.lax.dynamic_slice_in_dim(
              position_encoding, 0, self.per_position_dim, axis=-1))

    combined_encoding = jnp.concatenate(all_encodings, axis=-1)
    return jax.lax.dynamic_slice_in_dim(
        combined_encoding, 0, base_positional_encodings.shape[-1], axis=-1)


def divide_into_heads(
    num_heads: int,
    queries: dt.NDArrayFloatBTOH,
    keys: dt.NDArrayFloatBTVH,
    values: dt.NDArrayFloatBTVH,
    mask_structure: Optional[dt.NDArrayBoolBTOV] = None
) -> Tuple[dt.NDArrayFloatBTNVH, dt.NDArrayFloatBTNOH, dt.NDArrayFloatBTNOH,
           Optional[dt.NDArrayBoolBTNOV]]:
  """Reshapes the provided tensors to permit multi-head attention.

  Args:
    num_heads: The number of parallel attention heads to produce.
    queries: Attention query vectors.
    keys: Attention key vectors.
    values: Attention value vectors.
    mask_structure: Sparsity mask for attention.

  Returns:
    The queries, keys, values, and (if provided) mask_structure tensors reshaped
      to allow multi-headed attention.
  """

  def reshape_with_heads(tensor):
    # Splits the attention dimension into heads and moves the heads axis to the
    # second position, so it can be treated as a batch dimension.
    curr_shape = tensor.shape  # Shape: BTVH.
    hidden_dim = curr_shape[-1]
    dim_per_head = hidden_dim // num_heads
    new_shape = curr_shape[:-1] + (num_heads, dim_per_head)
    with_heads = jnp.reshape(tensor, new_shape)  # BTV + (heads, H/heads)
    # Move the 'heads' axis before the sequence dim to attend over sequence.
    return jnp.moveaxis(with_heads, -2, -3)  # BT(heads)V(H/heads).

  queries = reshape_with_heads(queries)
  keys = reshape_with_heads(keys)
  values = reshape_with_heads(values)
  if mask_structure is not None:
    mask_structure = mask_structure[..., None, :, :]  # Add heads dimension.
  return queries, keys, values, mask_structure


def mask_attention_weights(
    mask_structure: Optional[Union[dt.NDArrayBoolBTOV, dt.NDArrayBoolBTNOV]],
    queries: Union[dt.NDArrayFloatBTOH,
                   dt.NDArrayFloatBTNOH], keys: Union[dt.NDArrayFloatBTVH,
                                                      dt.NDArrayFloatBTNVH]
) -> Union[dt.NDArrayFloatBTOV, dt.NDArrayFloatBTNOV]:
  """Attention op that only calculates attention weights.

  This operation is similar to the "scaled dot product attention" op in the
  "Attention is all you need" paper - https://arxiv.org/pdf/1706.03762.pdf.
  The output is a set of unnormalized similarity scores between query and key
  vectors. Supports both single-head and multi-head attention by abstracting
  over one or more leading "batch" dimensions.

  # TODO: fix ascii-art for multi-dimensional case.

         attention scores
                ^
                | [BTOV]
          +-----------+
          |   Mask    |
          +-----------+
                ^
                | [BTOV]
           +---------+
           |  Scale  |
           +---------+
                ^
                | [BTOV]
         +--------------+
         |    MatMul    |
         +--------------+
           ^          ^
           |          |
           +          +
           Q          K
         [BTOH]      [BTVH]

  Note: inputs may have multiple leading batch dimensions, such as multiple
  attention heads. This does not affect the attention computation.

  There are no learnable weights in this layer. Therefore it is not decorated
  with @flax_dataclass.

  Args:
    mask_structure: Sparsity mask for attention.
    queries: Attention query vectors.
    keys: Attention key vectors.

  Returns:
    attention_scores: Unnormalized attention scores.
  """
  hidden_dim = queries.shape[-1]
  attention_scores = (
      jnp.einsum('...vh,...th->...tv', keys, queries) / jnp.sqrt(hidden_dim))

  if mask_structure is not None:
    # Use appropriate value for elements that should not be attended to.
    attention_scores = jnp.where(mask_structure, attention_scores,
                                 _LOG_OF_SMALL_VALUE)

  return attention_scores


def mask_attention_probs(
    mask_structure: Optional[Union[dt.NDArrayBoolBTOV, dt.NDArrayBoolBTNOV]],
    queries: Union[dt.NDArrayFloatBTOH,
                   dt.NDArrayFloatBTNOH], keys: Union[dt.NDArrayFloatBTVH,
                                                      dt.NDArrayFloatBTNVH]
) -> Union[dt.NDArrayFloatBTOV, dt.NDArrayFloatBTNOV]:
  """Normalizes attention scores to return a probability.

  Take softmax along keys dimension in order to get a probability. This can
  semantically mean the probability that a query vector is similar to a key
  vector. If `mask_structure` is all False across dimension==2, return the all
  zeros vector for that slice (so it's not in the simplex).

  Note that the same `mask_structure` is used in `mask_attention_weights` (to
  mask out pre-softmax) and at the end of this function (to mask out
  post-softmax).

  There are no learnable weights in this layer. Therefore it is not decorated
  with @flax_dataclass.

  Args:
    mask_structure: Sparsity mask for attention.
    queries: Attention query vectors.
    keys: Attention key vectors.

  Returns:
    attention_probs: Attention probabilities between keys and vectors. If all
    elements of a dimension==2 slice are disallowed by `mask_structure`, return
    all zeros for those elements.
  """
  attention_scores = mask_attention_weights(mask_structure, queries, keys)
  # Take softmax over keys dimensions.
  attention_probs = jnp.exp(
      attention_scores -
      jax.scipy.special.logsumexp(attention_scores, axis=-1, keepdims=True))

  if mask_structure is not None:
    attention_probs = jnp.where(mask_structure, attention_probs, 0.0)

  return attention_probs


@flax_dataclass
class MaskedAttention():
  """Attention layer using sparsity mask.

  Computes an attention vector averaging over value vectors using attention
  probabilities between key and query pairs. Similar to the Multiheaded
  attention layer in the "Attention is all you needed" paper -
  https://arxiv.org/pdf/1706.03762.pdf.

                      attention vectors
                               ^
                               |[BTH]
                         +------------+
                         |   MatMul   |
                         +--+------+--+
                            ^      ^
                    +-------+      +-------+
                    |[BTV]                 |
       +------------------------+          |
       |  mask_attention_probs  |          |
       |       +---------+      |          |
       |       | softmax |      |          |
       |       +---------+      |          |
       +------------------------+          |
                    ^                      |[BVH]
                    | [BTV]                |
      +-------------+------------+         |
      |  mask_attention_weights  |         |
      +----+--------------+------+         |
           ^              ^                |
           |[BTH]         |[BVH]           |
       +--------+     +--------+      +----+---+
       | Linear |     | Linear |      | Linear |
       +---+----+     +---+----+      +----+---+
           ^              ^                ^
           |              |                |
           +              +                +
           Q              K                V
         [BTH]          [BVH]            [BVH]


  Note: Unlike the `mask_attention_prob` function, this layer contains
  learnable weights which transform key, query and dense layers through Dense
  layers.
  """
  # Parameters.
  query_dense: Dense
  key_dense: Dense
  value_dense: Dense
  project_dense: Dense
  dropout_layer: Dropout
  num_heads: int = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng,
             hidden_dim: int,
             num_heads: int = 1,
             use_bias=True,
             deterministic: bool = False,
             dropout_rate: float = 0.1):
    q_rng, k_rng, v_rng, p_rng, dropout_rng = jax.random.split(rng, 5)
    return MaskedAttention(
        query_dense=Dense.create(
            q_rng, hidden_dim, hidden_dim, use_bias=use_bias),
        key_dense=Dense.create(
            k_rng, hidden_dim, hidden_dim, use_bias=use_bias),
        value_dense=Dense.create(
            v_rng, hidden_dim, hidden_dim, use_bias=use_bias),
        project_dense=Dense.create(
            p_rng, hidden_dim, hidden_dim, use_bias=use_bias),
        dropout_layer=Dropout.create(
            dropout_rng,
            deterministic=deterministic,
            dropout_rate=dropout_rate,
            shared_axes=[-2]),
        num_heads=num_heads)

  def __call__(self, mask_structure: Union[dt.NDArrayBoolBTOV, None],
               query_hiddens: dt.NDArrayFloatBTOH,
               key_hiddens: dt.NDArrayFloatBTVH,
               value_hiddens: dt.NDArrayFloatBTVH,
               rng: jnp.ndarray) -> dt.NDArrayFloatBTOH:
    (rng_query, rng_keys, rng_values, rng_project,
     rng_dropout) = jax.random.split(rng, 5)
    queries = self.query_dense(query_hiddens, rng_query)
    keys = self.key_dense(key_hiddens, rng_keys)
    values = self.value_dense(value_hiddens, rng_values)

    # Divide into num_heads, if doing multi-head attention.
    if self.num_heads > 1:
      query_shape = queries.shape  # Preserve for the final output reshape.
      queries, keys, values, mask_structure = divide_into_heads(
          self.num_heads, queries, keys, values, mask_structure)

    attention_probs = mask_attention_probs(mask_structure, queries, keys)
    attention_probs = self.dropout_layer(attention_probs, rng_dropout)

    result = jnp.einsum('...vh,...tv->...th', values, attention_probs)
    if self.num_heads > 1:
      # Shifts the heads dimension before the sequence dimension:
      # [batch, time, seq, heads, dim] -> [batch, time, heads, seq, dim]
      result = jnp.reshape(jnp.moveaxis(result, -3, -2), query_shape)

    return self.project_dense(result, rng_project)


@flax_dataclass
class BiasedMaskedAttention():
  """Relation-Biased Attention layer using sparsity mask.

  Computes an attention vector averaging over value vectors using attention
  probabilities between key and query pairs, with learned (scalar) bias weights
  attuned to relations indicated in the input. Otherwise identical to
  MaskedAttention.
  """
  # Parameters.
  query_dense: Dense
  key_dense: Dense
  value_dense: Dense
  project_dense: Dense
  bias_embeddings: jnp.ndarray
  bias_scalar: jnp.ndarray
  dropout_layer: Dropout
  num_heads: int = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng,
             hidden_dim: int,
             num_edge_types: int,
             num_heads: int = 1,
             use_bias=True,
             deterministic: bool = False,
             dropout_rate: float = 0.1):
    """Static initializer for BiasedMaskedAttention class."""
    q_rng, k_rng, v_rng, p_rng, b_rng, dropout_rng = jax.random.split(rng, 6)
    query_dense = Dense.create(q_rng, hidden_dim, hidden_dim, use_bias=use_bias)
    key_dense = Dense.create(k_rng, hidden_dim, hidden_dim, use_bias=use_bias)
    value_dense = Dense.create(v_rng, hidden_dim, hidden_dim, use_bias=use_bias)
    project_dense = Dense.create(
        p_rng, hidden_dim, hidden_dim, use_bias=use_bias)
    bias_embeddings = jax.random.normal(b_rng, (num_edge_types, hidden_dim))
    bias_scalar = jax.random.normal(b_rng, (hidden_dim,))

    return BiasedMaskedAttention(
        query_dense=query_dense,
        key_dense=key_dense,
        value_dense=value_dense,
        project_dense=project_dense,
        bias_embeddings=bias_embeddings,
        bias_scalar=bias_scalar,
        dropout_layer=Dropout.create(
            dropout_rng,
            deterministic=deterministic,
            dropout_rate=dropout_rate,
            shared_axes=[-2]),
        num_heads=num_heads)

  # Enabling rematerialization unconditionally under the assumption targeted
  # model sizes require it. For reference, the attention weights of a model
  # might require 8GiB of memory (the product of 8 examples_per_batch,
  # 16 layers, 16 heads, 1024**2 sequence length, 4 bytes per float).
  # Note the rematerialization could be tighter by restricting it only to the
  # attention matrix.
  @jax.remat
  def __call__(self, mask_structure: Union[dt.NDArrayBoolBTOV, None],
               query_hiddens: dt.NDArrayFloatBTOH,
               key_hiddens: dt.NDArrayFloatBTVH,
               value_hiddens: dt.NDArrayFloatBTVH,
               edge_data: Union[dt.NDArrayBoolBTEVV, dt.NDArrayBoolBVETT],
               rng: jnp.ndarray) -> dt.NDArrayFloatBTOH:
    (rng_query, rng_keys, rng_values, rng_project,
     rng_dropout) = jax.random.split(rng, 5)
    queries = self.query_dense(query_hiddens, rng_query)
    keys = self.key_dense(key_hiddens, rng_keys)
    values = self.value_dense(value_hiddens, rng_values)

    # Divide into num_heads, if doing multi-head attention.
    if self.num_heads > 1:
      query_shape = queries.shape  # Preserve for the final output reshape.
      queries, keys, values, mask_structure = divide_into_heads(
          self.num_heads, queries, keys, values, mask_structure)

    attention_scores = mask_attention_weights(mask_structure, queries, keys)
    # Combine embedding and scalar projection to minimize memory footprint.
    bias_value = jnp.einsum('...ewv,eh,h->...wv', edge_data,
                            self.bias_embeddings, self.bias_scalar)
    keys_scalar = jnp.sum(keys, -1)
    if self.num_heads > 1:
      bias_value = bias_value[..., None, :, :]
    attention_scores += jnp.einsum('...w,...wv->...wv', keys_scalar, bias_value)

    # Take softmax over keys dimensions.
    attention_probs = jnp.exp(
        attention_scores -
        jax.scipy.special.logsumexp(attention_scores, axis=-1, keepdims=True))

    if mask_structure is not None:
      attention_probs = jnp.where(mask_structure, attention_probs, 0.0)

    attention_probs = self.dropout_layer(attention_probs, rng_dropout)

    result = jnp.einsum('...vh,...tv->...th', values, attention_probs)
    if self.num_heads > 1:
      # Shifts the heads dimension before the sequence dimension:
      # [batch, time, seq, heads, dim] -> [batch, time, heads, seq, dim]
      result = jnp.reshape(jnp.moveaxis(result, -3, -2), query_shape)
    return self.project_dense(result, rng_project)


@flax_dataclass
class FFNBlock():
  """Feedforward network block used by Transformer.

  Transforms its inputs through two dense layers, with the intermediate
  representation generally substantially wider than the input and output.
  Additionally, the first dense layer is ReLU activate, and final states are
  layer-normalized (optionally after a residual addition of the inputs).
  """

  # Parameters.
  dense_layer_1: Dense
  dense_layer_2: Dense
  layer_norm: LayerNorm
  dropout_layer_1: Dropout
  dropout_layer_2: Dropout
  dropout_layer_residual: Dropout

  @staticmethod
  def create(rng,
             in_hidden_dim: int,
             out_hidden_dim: int,
             mid_dim: Optional[int] = None,
             deterministic: bool = False,
             dropout_rate: float = 0.1):
    """Creates a FeedForward block with two layers and layer-norm."""
    if mid_dim is None:
      mid_dim = out_hidden_dim
    dense1_rng, dense2_rng, layer_norm_rng, dropout_rng, rng = jax.random.split(
        rng, 5)

    dense_layer_1 = Dense.create(dense1_rng, in_hidden_dim, mid_dim)
    dense_layer_2 = Dense.create(dense2_rng, mid_dim, out_hidden_dim)
    layer_norm = LayerNorm.create(layer_norm_rng, out_hidden_dim)

    (dropout_rng_1, dropout_rng_2,
     dropout_rng_residual) = jax.random.split(dropout_rng, 3)
    dropout_layer_1 = Dropout.create(
        dropout_rng_1, deterministic=deterministic, dropout_rate=dropout_rate)
    dropout_layer_2 = Dropout.create(
        dropout_rng_2, deterministic=deterministic, dropout_rate=dropout_rate)
    dropout_layer_residual = Dropout.create(
        dropout_rng_residual,
        deterministic=deterministic,
        dropout_rate=dropout_rate)

    return FFNBlock(
        dense_layer_1=dense_layer_1,
        dense_layer_2=dense_layer_2,
        layer_norm=layer_norm,
        dropout_layer_1=dropout_layer_1,
        dropout_layer_2=dropout_layer_2,
        dropout_layer_residual=dropout_layer_residual)

  def __call__(self,
               hiddens: dt.NDArrayFloatBTOH,
               rng: jnp.ndarray,
               add_residual: bool = False) -> dt.NDArrayFloatBTOH:
    (rng_dense1, rng_dense2, rng_layer_norm, rng_dropout1, rng_dropout2,
     rng_dropout_residual) = (
         jax.random.split(rng, 6))

    # Pre-normalization.
    intermediate = self.layer_norm(hiddens, rng_layer_norm)

    intermediate = self.dense_layer_1(intermediate, rng_dense1)
    intermediate = jax.nn.relu(intermediate)
    intermediate = self.dropout_layer_1(intermediate, rng_dropout1)
    messages = self.dense_layer_2(intermediate, rng_dense2)
    messages = self.dropout_layer_2(messages, rng_dropout2)

    if add_residual:
      messages += self.dropout_layer_residual(hiddens, rng_dropout_residual)

    return messages


@flax_dataclass
class AttentionBlock():
  """A block with Attention and LayerNorm.

              output_hiddens[BOH]
                     ^
                     |
              +-------------+
              |  LayerNorm  |
              +-------------+
                     ^
                     |
                  +-----+
       +--------->+ Sum |
       |          +-----+
       |             ^
       |             | messages[BOH]
       |   +-------------------+
       |   |                   |
       |   |  MaskedAttention  |
       |   |                   |
       |   +-------------------+
       |             ^
       +-------------+
                     +
               output_hiddens
                    [BOH]

  Component used both in output_to_output and input_to_output attention
  message passing.
  """

  # Parameters.
  attention_layer: Union[MaskedAttention, BiasedMaskedAttention]
  dropout_layer: Dropout
  # Note there is an opportunity for specializing self-attention to use a single
  # `LayerNorm` layer.
  query_norm_layer: LayerNorm
  key_norm_layer: LayerNorm
  use_relational_bias: bool = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng,
             hidden_dim: int,
             num_transformer_attention_heads: int = 1,
             use_relational_bias: bool = False,
             num_edge_types: Optional[int] = None,
             use_bias=True,
             deterministic: bool = False,
             dropout_rate: float = 0.1):
    """Returns an attention block with layer norm and optional relational bias.

    If used, the relational bias is inserted by the BiasedMaskedAttention layer
    (instead of the default, MaskedAttention), which uses the available edges
    in the input to tune the key-value attention towards known relations (e.g.,
    data-flow, syntactic relations) through a learned component, as described in
    (Hellendoorn et al., ICLR'20).

    Args:
      rng: Random number generator.
      hidden_dim: The input and output dimensionality of the attention.
      num_transformer_attention_heads: The number of attention heads to use.
        If 1 (default), computes regular, single-headed attention.
      use_relational_bias: Whether to leverage edge data in the attention.
      num_edge_types: The number of distinct edges in the input.
      use_bias: Whether to add bias in the query/key/value computation.
      deterministic: Whether to run this layer deterministically, i.e., without
        dropout.
      dropout_rate: The fraction of nodes to disable when using dropout.
    """
    (attention_rng, query_norm_layer_rng, key_norm_layer_rng,
     dropout_rng) = jax.random.split(rng, 4)

    if use_relational_bias:
      attention_layer = BiasedMaskedAttention.create(
          attention_rng,
          hidden_dim,
          num_edge_types,
          num_transformer_attention_heads,
          use_bias=use_bias,
          deterministic=deterministic,
          dropout_rate=dropout_rate)
    else:
      attention_layer = MaskedAttention.create(
          attention_rng,
          hidden_dim,
          num_transformer_attention_heads,
          use_bias=use_bias,
          deterministic=deterministic,
          dropout_rate=dropout_rate)
    dropout_layer = Dropout.create(
        dropout_rng, deterministic=deterministic, dropout_rate=dropout_rate)
    return AttentionBlock(
        attention_layer=attention_layer,
        dropout_layer=dropout_layer,
        query_norm_layer=LayerNorm.create(query_norm_layer_rng, hidden_dim),
        key_norm_layer=LayerNorm.create(key_norm_layer_rng, hidden_dim),
        use_relational_bias=use_relational_bias)

  def __call__(
      self,
      mask_structure: Union[dt.NDArrayBoolBOO, dt.NDArrayBoolBTOV, None],
      query_hiddens: dt.NDArrayFloatBTOH,
      key_hiddens: Union[dt.NDArrayFloatBTOH, dt.NDArrayFloatBTVH],
      rng: jnp.ndarray,
      edge_data: Optional[Union[dt.NDArrayBoolBTEVV,
                                dt.NDArrayBoolBVETT]] = None
  ) -> dt.NDArrayFloatBTOH:
    (rng_attention, rng_dropout, rng_query_layer_norm,
     rng_key_layer_norm) = jax.random.split(rng, 4)

    # Pre-normalization.
    normalized_query_hiddens = self.query_norm_layer(query_hiddens,
                                                     rng_query_layer_norm)
    normalized_key_hiddens = self.key_norm_layer(key_hiddens,
                                                 rng_key_layer_norm)

    if self.use_relational_bias:
      attention_layer = typing.cast(BiasedMaskedAttention, self.attention_layer)
      messages = attention_layer(
          mask_structure=mask_structure,
          query_hiddens=normalized_query_hiddens,
          key_hiddens=normalized_key_hiddens,
          value_hiddens=normalized_key_hiddens,
          edge_data=edge_data,
          rng=rng_attention)
    else:
      attention_layer = typing.cast(MaskedAttention, self.attention_layer)
      messages = attention_layer(
          mask_structure=mask_structure,
          query_hiddens=normalized_query_hiddens,
          key_hiddens=normalized_key_hiddens,
          value_hiddens=normalized_key_hiddens,
          rng=rng_attention)

    messages = self.dropout_layer(messages, rng_dropout)

    # Residual.
    # TODO: dropout on residual path?
    result = query_hiddens + messages

    return result


@flax_dataclass
class InputEmbeddingModule():
  """Module holding parameters that allows embedding node tokens and types.

  Token and type embeddings are generated by looking up embeddings from
  a table of embedding parameters but the respective token and type ids.

                         input embeddings
                              [BTVH]
                                ^
                                |
                              +-+-+
                              |Sum|
                              ++-++
                               | |
                +--------------+ +---------------+
                |                                |
                |[BTVH]                           |[BTVH]
    +----------------------+         +---------------------+
    |    token_embedding   |         |   type_embedding    |
    | [token_vocab_size,H] |         | [type_vocab_size,H] |
    |                      |         |                     |
    +-----------+----------+         +-----------+---------+
                ^                                ^
                |                                |
            token_ids                        type_ids
              [BTVS]                            [BTV]

  Note: The order of the nodes matter as they affect the positional encoding.
  We might experiment with different graph orderings.
  """

  token_embeddings: jnp.ndarray
  type_name_embeddings: jnp.ndarray
  positional_encoding_layer: PositionalEncoding
  _pad_token_id: int = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng,
             max_input_length: int,
             token_vocab_size: int,
             type_vocab_size: int,
             hidden_dim: int,
             pad_token_id: int,
             position_dim: int = 1):
    """Creates a node embedding module."""
    position_rng, token_rng, type_rng = jax.random.split(rng, 3)
    scale = 1.0  # / hidden_dim
    token_embeddings = scale * (
        jax.random.normal(token_rng, (token_vocab_size, hidden_dim)))
    type_name_embeddings = scale * (
        jax.random.normal(type_rng, (type_vocab_size, hidden_dim)))
    positional_encoding_layer = PositionalEncoding.create(
        position_rng,
        max_length=max_input_length,
        hidden_dim=hidden_dim,
        position_dim=position_dim)
    return InputEmbeddingModule(
        token_embeddings=token_embeddings,
        type_name_embeddings=type_name_embeddings,
        positional_encoding_layer=positional_encoding_layer,
        _pad_token_id=pad_token_id)

  def __call__(
      self, input_data: dt.BatchedTrainGraphNodeData,
      rng: jnp.ndarray) -> Tuple[dt.NDArrayFloatBTVH, dt.NDArrayFloatBTVH]:

    # BTVS. Used to zero out embeddings for padding sub-tokens.
    subtoken_mask = input_data.token_ids != self._pad_token_id

    # Sum token and node type embeddings.
    embeddings = (
        self._embed_tokens(input_data.token_ids, subtoken_mask,
                           input_data.token_positions, rng) +
        self._embed_types(input_data.type_ids))

    # BTV1, marks which tokens are entirely padding. Returned for use elsewhere.
    whole_token_mask = jnp.any(subtoken_mask, axis=-1, keepdims=True)

    return embeddings, whole_token_mask

  def _embed_tokens(self, token_ids: dt.NDArrayIntBTVS,
                    subtoken_mask: dt.NDArrayIntBTVS,
                    token_positions: Optional[dt.NDArrayIntBTVP],
                    rng: jnp.ndarray) -> dt.NDArrayFloatBTVH:
    """Embeds subtokens and aggregates into token embeddings."""
    # BTVS -> BTVSH -> BTVH
    subtoken_embeddings = self.token_embeddings[(token_ids,)]
    subtoken_counts = jnp.sum(subtoken_mask, axis=-1, keepdims=True)  # BTV1
    # Effectively mean-pool over non-padding subtokens.
    token_embeddings = jnp.sum(
        subtoken_embeddings * subtoken_mask[..., np.newaxis], axis=-2)
    # TODO: Would be better to use a 'where' on (subtoken_counts
    # == 0) instead, but that introduces NaNs.
    token_embeddings = token_embeddings / (1e-5 + subtoken_counts)  # BTVH

    # Add positional encodings to token_embeddings
    # BTVH -> BTVH. Assign to a local variable so that it is picked up by
    # (neural) debuggers.
    result = self.positional_encoding_layer(rng, token_embeddings,
                                            token_positions)
    return result

  def _embed_types(self, type_ids: dt.NDArrayIntBTV) -> dt.NDArrayFloatBTVH:
    """Embed node type names."""
    # Assign to a local variable so that it is picked up by (neural) debuggers.
    result = self.type_name_embeddings[(type_ids,)]
    return result


@flax_dataclass
class OutputEmbeddingModule():
  """Module generating embeddings for output targets.

  Output embeddings are a combination of token, copy and pointer embeddings.
  Each corresponding to different ways in which the target could be generated.
  It's possible that for some cases there may not be a copy or a pointer that
  could generate the target. In such cases, the embedding should be filtered
  out by the `is_target_copy` and `is_target_pointer` masks respectively.

                                 ^
                                 | output embeddings
                                 |    [BTH]
                               +-+-+
                               |sum|
                               +-+-+
                                 |
              +-------------------------------------+
              |[BTH]             |[BTH]             |[BTH]
     +----------------+     +--------+    +------------------+
     | copy_attention |     | Linear |    | token_embeddings |
     +----+--------+--+     +---+----+    +---------+--------+
          |        |            |                   |
          +        |         +--+---+               +
  is_target_copy   |         |select|            targets
      [BTV]        |         ++---+-+
                   |          |   |
                   +----------+   +
                   |            is_pointer_copy
                   +                [BTV]
                input hiddens
                   [BVH]
  """
  # Model parameters
  token_embeddings: jnp.ndarray
  # TODO: Should also embed positions.
  copy_attention_layer: MaskedAttention
  pointer_dense: Dense
  positional_encoding_layer: PositionalEncoding
  start_token_embedding: jnp.ndarray

  @staticmethod
  def create(rng, max_output_length: int, output_vocab_size: int,
             hidden_dim: int):
    """Creates an output tocopo sequence embedding module.

    Args:
      rng: Random number generator.
      max_output_length: Maximum possible length of output sequence.
      output_vocab_size: Vocab size of output tokens.
      hidden_dim: Size of hidden dimension.

    Returns:
      Module embedding output sequence.
    """
    (token_rng, position_rng, copy_rng, pointer_rng) = jax.random.split(rng, 4)
    scale = 1.0 / hidden_dim
    token_embeddings = scale * (
        jax.random.normal(token_rng, (output_vocab_size, hidden_dim)))

    copy_attention_layer = MaskedAttention.create(
        copy_rng, hidden_dim=hidden_dim)
    pointer_dense = Dense.create(
        pointer_rng, input_dim=hidden_dim, output_dim=hidden_dim)
    positional_encoding_layer = PositionalEncoding.create(
        position_rng, max_length=max_output_length, hidden_dim=hidden_dim)

    rng, start_token_rng = jax.random.split(rng, 2)
    start_token_embedding = scale * jax.random.normal(start_token_rng,
                                                      (1, hidden_dim))
    return OutputEmbeddingModule(
        token_embeddings=token_embeddings,
        copy_attention_layer=copy_attention_layer,
        pointer_dense=pointer_dense,
        positional_encoding_layer=positional_encoding_layer,
        start_token_embedding=start_token_embedding)

  def __call__(self, target_data: dt.BatchedTrainTocopoTargetData,
               input_hiddens: dt.NDArrayFloatBVH,
               rng: jnp.ndarray) -> dt.NDArrayFloatBOH:
    rng_token, rng_copy, rng_pointer = jax.random.split(rng, 3)
    token_embeddings = self._embed_tokens(target_data.token_ids, rng_token)

    copy_embeddings = self._embed_copies(target_data.is_target_copy,
                                         token_embeddings, input_hiddens,
                                         rng_copy)

    pointer_embeddings = self._embed_pointers(target_data.is_target_pointer,
                                              input_hiddens, rng_pointer)

    unshifted_tocopo_embeddings = (
        token_embeddings + copy_embeddings + pointer_embeddings)
    b, _, h = unshifted_tocopo_embeddings.shape

    # Embeddings to use as input need to be shifted so we're not leaking targets
    # to the future.
    tocopo_embeddings = jnp.concatenate([
        self.start_token_embedding * jnp.ones((b, 1, h)),
        unshifted_tocopo_embeddings[:, :-1, :]
    ], axis=1)
    return tocopo_embeddings

  def _embed_tokens(self, token_ids: dt.NDArrayIntBO,
                    rng: jnp.ndarray) -> dt.NDArrayFloatBOH:
    """Embed tokens from partial output edit script."""
    #  BO -> BOH
    token_embeddings = self.token_embeddings[(token_ids,)]
    # Add positional encodings to token_embeddings
    # BOH -> BOH
    result = self.positional_encoding_layer(rng, token_embeddings)
    return result

  def _embed_copies(self, is_target_copy: dt.NDArrayBoolBOV,
                    output_hiddens: dt.NDArrayFloatBOH,
                    input_hiddens: dt.NDArrayFloatBVH,
                    rng: jnp.ndarray) -> dt.NDArrayFloatBOH:
    """Embed copies."""
    output_hiddens = self.copy_attention_layer(
        mask_structure=is_target_copy,
        query_hiddens=output_hiddens,
        key_hiddens=input_hiddens,
        value_hiddens=input_hiddens,
        rng=rng)
    return output_hiddens

  def _embed_pointers(self, is_target_pointer: dt.NDArrayBoolBOV,
                      input_hiddens: dt.NDArrayFloatBVH,
                      rng: jnp.ndarray) -> dt.NDArrayFloatBOH:
    output_hiddens = jnp.einsum('btv,bvh->bth', is_target_pointer,
                                input_hiddens)
    return self.pointer_dense(output_hiddens, rng)
