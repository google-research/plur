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

"""End-to-end models comprising encoders and decoders."""

import dataclasses
from typing import TYPE_CHECKING

import flax
import jax
import jax.numpy as jnp

from plur.model_design import data_types as dt
from plur.model_design import decoders
from plur.model_design import encoders
from plur.model_design.model_configs import ModelConfig

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


@flax_dataclass
class GGNN2Tocopo():
  """Model mapping graph inputs to Tocopo output sequences.

    Combines the GGNNGraphEncoder module and the TocopoDecoder to compute
    tocopo logits.

     token logits   copy logits   pointer logits
           ^             ^               ^
      [BTU]|             |[BTV]          |[BTV]
           +---------+   |   +-----------+
                     |   |   |
                +----+---+---+----+
                |                 |
                |  TocopoDecoder  |
                |                 |
                +---+---------+---+
                    ^         ^
                    |         |
              +-----+         +-------+
              | node hiddens          |
              | [BVH]                 |
    +---------+----------+            |
    |                    |            |
    |  GGNNGraphEncoder  |            |
    |                    |            |
    +---+-------------+--+            |
        ^             ^               |
        |             |               |
        +             +               +
    node_data     edge_data      target_data
  """
  # Model parameters
  graph_encoder: encoders.GGNNGraphEncoder
  tocopo_decoder: decoders.TocopoDecoder

  @staticmethod
  def create(rng: jnp.ndarray, config: ModelConfig):
    """Static builder of GGNN2Tocopo."""
    graph_encoder_rng, tocopo_decoder_rng = jax.random.split(rng, 2)
    graph_encoder = encoders.GGNNGraphEncoder.create(graph_encoder_rng, config)
    tocopo_decoder = decoders.TocopoDecoder.create(tocopo_decoder_rng, config)
    return GGNN2Tocopo(graph_encoder=graph_encoder,
                       tocopo_decoder=tocopo_decoder)

  def __call__(self, batched_data: dt.BatchedTrainTocopoData,
               rng: jnp.ndarray) -> dt.BatchedTocopoLogits:
    rng_encoder, rng_decoder = jax.random.split(rng)
    node_hiddens = self.graph_encoder(batched_data, rng_encoder)

    # Flatten time axis for decoder.
    batch_dim = node_hiddens.shape[0]
    hidden_dim = node_hiddens.shape[-1]
    node_hiddens = jax.numpy.reshape(node_hiddens, [batch_dim, -1, hidden_dim])

    tocopo_logits = self.tocopo_decoder(batched_data, node_hiddens, rng_decoder)
    return tocopo_logits


@flax_dataclass
class Transformer2Tocopo():
  """Model mapping graph inputs to Tocopo output sequences.

  Uses a transformer model for the encoder, which only leverages edge structure
  if use_relational_bias = True, in which case it uses GREAT-style attention.
  """
  # Model parameters
  encoder: encoders.TransformerEncoder
  decoder: decoders.TocopoDecoder

  @staticmethod
  def create(rng: jnp.ndarray, config: ModelConfig):
    """Static builder of Transformer2Tocopo."""
    encoder_rng, decoder_rng = jax.random.split(rng, 2)
    encoder = encoders.TransformerEncoder.create(encoder_rng, config)
    decoder = decoders.TocopoDecoder.create(decoder_rng, config)
    return Transformer2Tocopo(encoder=encoder, decoder=decoder)

  def __call__(self, batched_data: dt.BatchedTrainTocopoData,
               rng: jnp.ndarray) -> dt.BatchedTocopoLogits:
    rng_encoder, rng_decoder = jax.random.split(rng)
    node_hiddens = self.encoder(batched_data, rng_encoder)

    # Flatten time axis for decoder.
    batch_dim = node_hiddens.shape[0]
    hidden_dim = node_hiddens.shape[-1]
    node_hiddens = jax.numpy.reshape(node_hiddens, [batch_dim, -1, hidden_dim])

    tocopo_logits = self.decoder(batched_data, node_hiddens, rng_decoder)
    return tocopo_logits
