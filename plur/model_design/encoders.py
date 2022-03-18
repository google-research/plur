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

"""Encoders."""
import dataclasses
from typing import Optional, TYPE_CHECKING, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from plur.model_design import layers
import plur.model_design.data_types as dt
from plur.model_design.model_configs import ModelConfig

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


@flax_dataclass
class GGNN():
  """Gated Graph Neural Network.

  Runs multiple rounds of message passing and gates node hiddens at each step
  using a GRUCell.
  """

  # Model parameters.
  edge_kernels: dt.NDArrayFloatEHH
  edge_biases: dt.NDArrayFloatEH
  gru_cell: layers.GRUCell
  layer_norms: Tuple[layers.LayerNorm]
  # Only used when also using temporal message passing.
  time_edge_kernels: dt.NDArrayFloatEHH
  time_edge_biases: dt.NDArrayFloatEH

  # Additional fields that are not parameters.
  _num_steps: int = flax.struct.field(pytree_node=False)
  _use_temporal_messages: bool = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng: jnp.ndarray,
             num_steps: int,
             num_edge_types: int,
             hidden_dim: int,
             use_temporal_messages: bool = False,
             num_time_edge_types: Optional[int] = None):
    """Creates a GGNN module and initializes parameters."""
    rng, kernels_rng, bias_rng, gru_rng = jax.random.split(rng, 4)
    # Use Glorot normalization.
    edge_kernels = jax.random.normal(
        kernels_rng,
        (num_edge_types, hidden_dim, hidden_dim)) / jnp.sqrt(hidden_dim)
    edge_biases = 1e-5 * jax.random.normal(bias_rng,
                                           (num_edge_types, hidden_dim))

    if use_temporal_messages:
      rng, time_kernels_rng, time_bias_rng = jax.random.split(rng, 3)
      time_edge_kernels = jax.random.normal(
          time_kernels_rng,
          (num_time_edge_types, hidden_dim, hidden_dim)) / jnp.sqrt(hidden_dim)
      time_edge_biases = 1e-5 * jax.random.normal(
          time_bias_rng, (num_time_edge_types, hidden_dim))
    else:
      time_edge_kernels = time_edge_biases = None

    layer_norms = []
    for _ in range(num_steps + 1):
      rng, layer_norm_rng = jax.random.split(rng, 2)
      layer_norms.append(layers.LayerNorm.create(layer_norm_rng, hidden_dim))

    return GGNN(
        edge_kernels=edge_kernels,
        edge_biases=edge_biases,
        gru_cell=layers.GRUCell.create(gru_rng, hidden_dim),
        layer_norms=tuple(layer_norms),
        _num_steps=num_steps,
        _use_temporal_messages=use_temporal_messages,
        time_edge_kernels=time_edge_kernels,
        time_edge_biases=time_edge_biases)

  def __call__(self, node_hiddens: dt.NDArrayFloatBTVH,
               edge_data: dt.BatchedTrainGraphEdgeData,
               rng: jnp.ndarray) -> dt.NDArrayFloatBTVH:
    rng, rng_layer_norm = jax.random.split(rng)
    node_hiddens = self.layer_norms[0](node_hiddens, rng_layer_norm)
    for i in range(self._num_steps):
      messages = self._messages(node_hiddens, edge_data.edges)
      if self._use_temporal_messages:
        # Move time axis to penultimate position to use temporal attention.
        node_hiddens_temporal = jnp.moveaxis(node_hiddens, 2, 1)  # BTVH -> BVTH
        temporal_messages = self._messages(
            node_hiddens_temporal, edge_data.time_edges, is_temporal=True)
        messages += jnp.moveaxis(temporal_messages, 2, 1)  # BVTH -> BTVH

      rng, rng_update, rng_layer_norm = jax.random.split(rng, 3)
      node_hiddens = self._update(node_hiddens, messages, rng_update)
      node_hiddens = self.layer_norms[i + 1](node_hiddens, rng_layer_norm)
    return node_hiddens

  def _messages(self,
                node_hiddens: dt.NDArrayFloatBTVH,
                edges: Union[dt.NDArrayBoolBTEVV, dt.NDArrayBoolBVETT],
                is_temporal: bool = False) -> dt.NDArrayFloatBTVH:
    """Transforms per-node hiddens into per-node incoming messages."""
    # Einsum indices:
    # w: sending node
    # v: receiving node
    # h: initial hidden dimension
    # i: transformed hidden dimension
    # e: edge type
    dense_component = jnp.einsum(
        '...wh,ehi,...ewv->...vi', node_hiddens,
        self.time_edge_kernels if is_temporal else self.edge_kernels, edges)

    # Could be combined into a single einsum. Expanded for readability.
    incoming_edge_counts = jnp.einsum('...ewv->...ev', edges.astype(jnp.int32))
    bias_component = jnp.einsum(
        '...ev,ei->...vi', incoming_edge_counts,
        self.time_edge_biases if is_temporal else self.edge_biases)

    return dense_component + bias_component

  def _update(self, node_hiddens: dt.NDArrayFloatBTVH,
              node_messages: dt.NDArrayFloatBTVH,
              rng: jnp.ndarray) -> dt.NDArrayFloatBTVH:
    """Uses aggregated messages to update node state."""
    return self.gru_cell(node_hiddens, node_messages, rng=rng)


@flax_dataclass
class GGNNGraphEncoder():
  """Encodes input graphs to a set of per-node hiddens.

                         node hiddens
                            [BVH]
                              ^
                              |
                      +-------+------+
                      |              |
                      |     GGNN     |
                      |              |
                      +---+------+---+
                          ^      ^
                   +------+      +-----+
    node embeddings|                   |
             [BVH] |                   |
      +------------+---------+         |
      | InputEmbeddingModule |         |
      +------------+---------+         |
                   ^                   |
                   |                   |
                   +                   +
               node_data           edge_data

  """

  # Model parameters.
  node_embedder: layers.InputEmbeddingModule
  ggnn: GGNN
  initial_final_merger: layers.Dense

  @staticmethod
  def create(rng: jnp.ndarray, config: ModelConfig):
    """Creates a GGNNGraphEncoder module and initializes parameters."""
    embedder_rng, ggnn_rng, dense_rng = jax.random.split(rng, 3)
    return GGNNGraphEncoder(
        node_embedder=layers.InputEmbeddingModule.create(
            embedder_rng,
            config.max_input_length,
            config.token_vocab_size,
            config.type_vocab_size,
            config.hidden_dim,
            config.node_text_pad_token_id,
            position_dim=config.node_position_dim),
        initial_final_merger=layers.Dense.create(dense_rng,
                                                 2 * config.hidden_dim,
                                                 config.hidden_dim),
        ggnn=GGNN.create(
            ggnn_rng,
            config.num_input_propagation_steps,
            config.num_edge_types,
            config.hidden_dim,
            use_temporal_messages=config.model_temporal_relations,
            num_time_edge_types=config.num_time_edge_types))

  def __call__(self, batched_data: dt.BatchedTrainTocopoData,
               rng: jnp.ndarray) -> dt.NDArrayFloatBTVH:
    """Returns node states after embedding and processing by a GGNN.

    Zeroes out padding token states, and re-uses the node embeddings after the
    GGNN through an additional dense layer.

    Args:
      batched_data: batched node and edge data.
      rng: One-time use PRNG.
    """
    rng_embedder, rng_ggnn, rng_ifm = jax.random.split(rng, 3)
    node_embeddings, node_mask = self.node_embedder(batched_data.node_data,
                                                    rng_embedder)
    node_hiddens = self.ggnn(node_embeddings, batched_data.edge_data, rng_ggnn)
    node_hiddens *= node_mask
    initial_final_hiddens = jnp.concatenate([node_embeddings, node_hiddens],
                                            axis=-1)
    node_hiddens = self.initial_final_merger(initial_final_hiddens, rng_ifm)

    return node_hiddens


@flax_dataclass
class TransformerEncoder():
  """Encodes input graphs to a set of per-node hiddens."""

  # Model parameters.
  node_embedder: layers.InputEmbeddingModule
  attention_blocks: Tuple[layers.AttentionBlock]
  ffn_blocks: Tuple[layers.FFNBlock]

  # Additional fields that are not parameters.
  _jax2tf_compatible: bool = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng: jnp.ndarray, config: ModelConfig):
    """Creates a new Transformer-style encoder with an embedding layer."""
    attention_blocks = []
    ffn_blocks = []
    for _ in range(config.num_input_propagation_steps):
      rng, attention_rng, ffn_rng = jax.random.split(rng, 3)
      attention_blocks.append(
          layers.AttentionBlock.create(
              attention_rng,
              config.hidden_dim,
              config.num_transformer_attention_heads,
              config.use_relational_bias,
              config.num_edge_types,
              use_bias=False,
              deterministic=config.deterministic,
              dropout_rate=config.dropout_rate))
      ffn_blocks.append(
          layers.FFNBlock.create(
              ffn_rng,
              config.hidden_dim,
              config.hidden_dim,
              mid_dim=config.ff_dim,
              deterministic=config.deterministic,
              dropout_rate=config.dropout_rate))

    rng, embedder_rng = jax.random.split(rng, 2)
    node_embedder = layers.InputEmbeddingModule.create(
        embedder_rng,
        config.max_input_length,
        config.token_vocab_size,
        config.type_vocab_size,
        config.hidden_dim,
        config.node_text_pad_token_id,
        position_dim=config.node_position_dim)
    return TransformerEncoder(
        node_embedder=node_embedder,
        attention_blocks=tuple(attention_blocks),
        ffn_blocks=tuple(ffn_blocks),
        _jax2tf_compatible=config.jax2tf_compatible)

  def __call__(self, batched_data: dt.BatchedTrainTocopoData,
               rng: jnp.ndarray) -> dt.NDArrayFloatBTVH:
    rng, rng_embedder = jax.random.split(rng)
    node_embeddings, node_mask = self.node_embedder(batched_data.node_data,
                                                    rng_embedder)
    if self._jax2tf_compatible:
      # TF does not currently support general_dot product on 'bool', 'int8', or
      # 'int16'. Upcasting to 'int32'.
      i_node_mask = node_mask.astype('int32')
      i_attention_mask = jnp.einsum('...ux,...vx->...uv', i_node_mask,
                                    i_node_mask)
      attention_mask = i_attention_mask.astype('bool')
    else:
      attention_mask = jnp.einsum('...ux,...vx->...uv', node_mask, node_mask)
    node_hiddens = node_embeddings
    for attention, ffn in zip(self.attention_blocks, self.ffn_blocks):
      rng, rng_attention, rng_ffn = jax.random.split(rng, 3)
      node_hiddens = attention(
          mask_structure=attention_mask,
          query_hiddens=node_hiddens,
          key_hiddens=node_hiddens,
          rng=rng_attention,
          edge_data=batched_data.edge_data.edges)
      node_hiddens = ffn(node_hiddens, rng=rng_ffn, add_residual=True)
      node_hiddens *= node_mask
    return node_hiddens
