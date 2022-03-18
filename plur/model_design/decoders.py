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

"""Decoder generating Tocopo outputs."""

import dataclasses
from typing import Optional, TYPE_CHECKING, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np

from plur.model_design import layers
import plur.model_design.data_types as dt
from plur.model_design.model_configs import ModelConfig

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


# A large negative value that leads to a probability close to zero when
# exponentiated.
_LOG_OF_SMALL_VALUE = -1e9


@flax_dataclass
class TocopoOutputHead():
  """Final prediction layer in a tocopo decoder.

  Generates tocopo (token, copy and pointer logits). Attends over input node
  hiddens for the copy and pointer logits.

          token logits       copy logits               pointer logits
                ^                 ^                          ^
                |[BOU]            |[BOV]                     |[BOV]
                |                 |                          |
                |     +-----------+------------+  +----------+-------------+
                |     | mask attention weights |  | mask attention weights |
                |     +----+------+------------+  +-----+----+-------------+
                |          ^      ^                     ^    ^
                |          |      |                     |    |
                |          |      |                     |    |
                |          |      |                     |    |
                |          |      |                     |    |
                |          |      |                     |    |
  input hiddens |          |      |                     |    |
      [BVH]   +---+        |    +---+                   |    |
  +-----------+ | +--------+----+ | +-------------------+    |
                |                 |                          |
                |[BOH]            |[BOH]                     |[BOH]
           +--------+         +--------+                 +--------+
           | Linear |         | Linear |                 | Linear |
           +----+---+         +---+----+                 +---+----+
                ^                 ^                          ^
                |                 |                          |
                +--------------------------------------------+
                                  |
                                  +
                                output hiddens
                                [BOH]
  """

  # Model parameters
  token_output_layer: layers.Dense
  copy_output_layer: layers.Dense
  pointer_output_layer: layers.Dense
  # This is a model config variable and jax shouldn't run grad over it.
  _use_pointer_candidate_masking: bool = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng: jnp.ndarray,
             hidden_dim: int,
             output_vocab_size: int,
             use_pointer_candidate_masking: bool = False):
    """Static builder of final layer of tocopo logits."""
    token_rng, copy_rng, pointer_rng = jax.random.split(rng, 3)

    token_output_layer = layers.Dense.create(token_rng, hidden_dim,
                                             output_vocab_size)
    copy_output_layer = layers.Dense.create(copy_rng, hidden_dim, hidden_dim)
    pointer_output_layer = layers.Dense.create(pointer_rng, hidden_dim,
                                               hidden_dim)
    return TocopoOutputHead(
        token_output_layer=token_output_layer,
        copy_output_layer=copy_output_layer,
        pointer_output_layer=pointer_output_layer,
        _use_pointer_candidate_masking=use_pointer_candidate_masking)

  # Add input hiddens, copy and pointer layer here.
  def __call__(
      self,
      output_hiddens: dt.NDArrayFloatBOH,
      input_hiddens: dt.NDArrayFloatBVH,
      rng: jnp.ndarray,
      pointer_candidates: Optional[dt.NDArrayBoolBTV] = None,
      out_to_input_node_mask: Optional[dt.NDArrayBoolBTOV] = None
  ) -> dt.BatchedTocopoLogits:
    rng_token, rng_copy, rng_pointer = jax.random.split(rng, 3)

    # Token output head.
    token_logits = self.token_output_layer(output_hiddens, rng_token)

    # Copy output head.
    copy_output_hiddens = self.copy_output_layer(output_hiddens, rng_copy)
    copy_attn_weights = layers.mask_attention_weights(
        mask_structure=out_to_input_node_mask,
        queries=copy_output_hiddens,
        keys=input_hiddens)

    # Pointer output head.
    pointer_output_hiddens = self.pointer_output_layer(output_hiddens,
                                                       rng_pointer)
    pointer_attn_weights = layers.mask_attention_weights(
        mask_structure=out_to_input_node_mask,
        queries=pointer_output_hiddens,
        keys=input_hiddens)

    pointer_logits = self._mask_pointer_candidates(pointer_attn_weights,
                                                   pointer_candidates)

    tocopo_logits = dt.BatchedTocopoLogits(
        token_logits=dt.NDArrayFloatBOU(token_logits),
        copy_logits=dt.NDArrayFloatBOV(copy_attn_weights),
        pointer_logits=pointer_logits)
    return tocopo_logits

  def _mask_pointer_candidates(
      self, pointer_logits: dt.NDArrayFloatBOV,
      pointer_candidates: dt.NDArrayBoolBTV) -> dt.NDArrayFloatBOV:
    """Apply a mask over pointer candidates disabling select candidates."""
    if self._use_pointer_candidate_masking:
      # Flatten across time dimension.
      # BTV -> B(TV)
      batch_size, unused_num_time_steps, unused_num_nodes = (
          pointer_candidates.shape)
      mask_structure = jnp.reshape(pointer_candidates, [batch_size, -1])
      # Setup for Broadcast. B(TV) -> B1(TV)
      mask_structure = mask_structure[:, jnp.newaxis]
      pointer_logits = jnp.where(mask_structure, pointer_logits,
                                 _LOG_OF_SMALL_VALUE)

    return pointer_logits


@flax_dataclass
class TocopoDecoder():
  """Decode tocopo logits.

  Consumes input hiddens and target outputs to generate tocopo logits.

          token logits   copy logits   pointer logits
                  ^             ^               ^
             [BOU]|             |[BOV]          |[BOV]
                  +-----------+ | +-------------+
                              | | |
                        +-----+-+-+--------+
              +-------->+ TocopoOutputHead |
              |         +-------+----------+
              |                 ^
              |                 |
              |   +------------------------------+
              |   |             |output hiddens  |
              |   |             |[BOH]           |
              |   |    +----------------+        |
              |   |    |   in_to_out    |        |
              +------->+ AttentionBlock |        |
              |   |    +----------------+        |
  node hiddens|   |             ^output hiddens  | X num_output_prop_steps
     [BVH]    |   |             |[BOH]           |
  +-----------+   |    +----------------+        |
              |   |    |   out_to_out   |        |
              |   |    | AttentionBlock |        |
              |   |    +--------+-------+        |
              |   |             ^                |
              |   |             |                |
              |   +------------------------------+
              |                 |output hiddens
              |                 |[BOH]
              |      +-----------------------+
              +----->+ OutputEmbeddingModule |
                     +----------+------------+
                                ^
                                +
                           target_data
  """
  # Model parameters
  in_normalizer: layers.LayerNorm
  output_embedder: layers.OutputEmbeddingModule
  out_to_out_attention_blocks: Tuple[layers.AttentionBlock]
  in_to_out_attention_blocks: Tuple[layers.AttentionBlock]
  ffn_blocks: Tuple[layers.FFNBlock]
  out_normalizer: layers.LayerNorm

  tocopo_output_head: TocopoOutputHead

  # Additional fields that are not parameters.
  _num_output_propagation_steps: int = flax.struct.field(pytree_node=False)
  _input_pad_token_id: int = flax.struct.field(pytree_node=False)

  @staticmethod
  def create(rng: jnp.ndarray, config: ModelConfig):
    """Static builder of model decoding tocopo sequences."""

    # With pre-normalization, we need to normalize the outputs of the encoder
    # and decoder stacks. The normalizer for the encoder's output
    # (`in_normalizer`) is only used on paths that do not already perform
    # normalization (such as those with attention blocks). Avoiding
    # normalization on those paths is the reason we do not simply normalize the
    # output of the encoder inside the encoder.
    rng, in_normalizer_rng, out_normalizer_rng = jax.random.split(rng, 3)
    in_normalizer = layers.LayerNorm.create(in_normalizer_rng,
                                            config.hidden_dim)
    out_normalizer = layers.LayerNorm.create(out_normalizer_rng,
                                             config.hidden_dim)

    rng, output_rng, embedding_rng = jax.random.split(rng, 3)

    output_embedder = layers.OutputEmbeddingModule.create(
        embedding_rng,
        max_output_length=config.max_output_length,
        output_vocab_size=config.output_vocab_size,
        hidden_dim=config.hidden_dim)
    tocopo_output_head = TocopoOutputHead.create(
        output_rng,
        hidden_dim=config.hidden_dim,
        output_vocab_size=config.output_vocab_size,
        use_pointer_candidate_masking=config.use_pointer_candidate_masking)

    out_to_out_attention_blocks = []
    in_to_out_attention_blocks = []
    ffn_blocks = []

    for _ in range(config.num_output_propagation_steps):
      rng, out_to_out_rng, in_to_out_rng, ffn_rng = jax.random.split(rng, 4)
      out_to_out_attention_blocks.append(
          layers.AttentionBlock.create(
              out_to_out_rng,
              hidden_dim=config.hidden_dim,
              num_transformer_attention_heads=(
                  config.num_transformer_attention_heads),
              deterministic=config.deterministic,
              dropout_rate=config.dropout_rate))
      in_to_out_attention_blocks.append(
          layers.AttentionBlock.create(
              in_to_out_rng,
              hidden_dim=config.hidden_dim,
              num_transformer_attention_heads=(
                  config.num_transformer_attention_heads),
              deterministic=config.deterministic,
              dropout_rate=config.dropout_rate))
      ffn_blocks.append(
          layers.FFNBlock.create(ffn_rng, config.hidden_dim,
                                 config.hidden_dim, config.ff_dim,
                                 deterministic=config.deterministic,
                                 dropout_rate=config.dropout_rate))

    return TocopoDecoder(
        in_normalizer=in_normalizer,
        output_embedder=output_embedder,
        tocopo_output_head=tocopo_output_head,
        out_to_out_attention_blocks=tuple(out_to_out_attention_blocks),
        in_to_out_attention_blocks=tuple(in_to_out_attention_blocks),
        ffn_blocks=tuple(ffn_blocks),
        out_normalizer=out_normalizer,
        _num_output_propagation_steps=config.num_output_propagation_steps,
        _input_pad_token_id=config.node_text_pad_token_id)

  def __call__(self, batched_data: dt.BatchedTrainTocopoData,
               input_hiddens: dt.NDArrayFloatBVH,
               rng: jnp.ndarray) -> dt.BatchedTocopoLogits:
    rng, rng_in_normalizer, rng_embedder = jax.random.split(rng, 3)

    input_hiddens_normalized = self.in_normalizer(input_hiddens,
                                                  rng_in_normalizer)

    output_embeddings = self.output_embedder(batched_data.target_data,
                                             input_hiddens_normalized,
                                             rng_embedder)

    # Create the attention masks
    # TODO: Consider caching these masks.
    _, output_length, _ = output_embeddings.shape

    # Generates a lower triangular adjacency matrix. A[i,j]=1 => Output j is
    # allowed to attend on input i. This particular structure lets us do causal
    # self-attention. Also note that the targets have already been shifted
    # by 1 by this time, such that we are not leaking output labels at
    # prediction time.
    #
    # output_length = 4
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    causal_out_to_out_mask = jnp.tril(
        jnp.ones((output_length, output_length), dtype=np.bool))
    # Expand first dimension to broadcast along batch_size dimension.
    causal_out_to_out_mask = causal_out_to_out_mask[jnp.newaxis, ...]

    input_node_mask = jnp.any(
        batched_data.node_data.token_ids != self._input_pad_token_id, axis=-1,
        keepdims=False)  # BTV
    batch_size, temporal_dim, num_nodes = input_node_mask.shape
    input_node_mask = input_node_mask.reshape(
        batch_size, 1, temporal_dim * num_nodes)  # B1(TV)
    in_to_out_mask = jnp.tile(input_node_mask, [1, output_length, 1])  # BO(TV)

    # Alternate between each out_to_out and in_to_out attn block for
    # num_output_propagation steps, passing on the output_hiddens to the next
    # block. The attention blocks have their own layer normalization, so we
    # pass the hiddens unnormalized. Note this could be improved to avoid the
    # duplication.
    output_hiddens = output_embeddings
    for out_to_out_block, in_to_out_block, ffn_block in zip(
        self.out_to_out_attention_blocks, self.in_to_out_attention_blocks,
        self.ffn_blocks):
      rng, out_to_out_rng, in_to_out_rng, ffn_rng = jax.random.split(rng, 4)
      output_hiddens = in_to_out_block(
          mask_structure=in_to_out_mask,
          query_hiddens=output_hiddens,
          key_hiddens=input_hiddens,
          rng=in_to_out_rng)
      output_hiddens = out_to_out_block(
          mask_structure=causal_out_to_out_mask,
          query_hiddens=output_hiddens,
          key_hiddens=output_hiddens,
          rng=out_to_out_rng)

      output_hiddens = ffn_block(output_hiddens, add_residual=True, rng=ffn_rng)

    rng, rng_out_normalizer = jax.random.split(rng)
    output_hiddens_normalized = self.out_normalizer(output_hiddens,
                                                    rng_out_normalizer)

    return self.tocopo_output_head(
        output_hiddens=output_hiddens_normalized,
        input_hiddens=input_hiddens_normalized,
        rng=rng,
        pointer_candidates=batched_data.node_data.pointer_candidates,
        out_to_input_node_mask=in_to_out_mask)


@flax_dataclass
class PointerOutputHead():
  """Simple class that converts node hiddens into predicted pointer logits.

  Applies attention to node hiddens with a learned query vector in order to
  turn
  """

  query_vector: jnp.ndarray

  @staticmethod
  def create(rng: jnp.ndarray, hidden_dim):
    return PointerOutputHead(query_vector=jax.random.normal(rng, (hidden_dim,)))

  def __call__(self, node_hiddens):
    attention_scores = jnp.einsum('h,bvh->bv', self.query_vector,
                                  node_hiddens)
    pointer_logits = (
        attention_scores -
        jax.scipy.special.logsumexp(attention_scores, axis=-1, keepdims=True))

    return pointer_logits
