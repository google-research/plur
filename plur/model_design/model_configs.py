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

"""Contains model configuration(s)."""
import dataclasses
from typing import Optional, TYPE_CHECKING

import flax

flax_dataclass = (
    flax.struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass)


@flax_dataclass
class ModelConfig:
  """A general model configuration, using optional values for non-essentials."""
  hidden_dim: int
  deterministic: bool
  dropout_rate: float
  max_input_length: Optional[int]
  num_input_timesteps: Optional[int]
  node_position_dim: Optional[int]
  model_temporal_relations: Optional[bool]
  num_input_propagation_steps: Optional[int]
  token_vocab_size: Optional[int]
  node_text_pad_token_id: Optional[int]
  ff_dim: Optional[int]
  num_transformer_attention_heads: Optional[int]
  num_edge_types: Optional[int]
  num_time_edge_types: Optional[int]
  use_relational_bias: Optional[bool]
  max_output_length: Optional[int]
  type_vocab_size: Optional[int]
  output_vocab_size: Optional[int]
  num_output_propagation_steps: Optional[int]
  use_pointer_candidate_masking: Optional[bool]
  # Adds workarounds to prevent use of some JAX operations not in TF.
  jax2tf_compatible: Optional[bool]


def create_model_config(is_training,
                        hidden_dim,
                        max_input_length=None,
                        num_input_timesteps=None,
                        model_temporal_relations=True,
                        node_position_dim=1,
                        num_input_propagation_steps=None,
                        token_vocab_size=None,
                        node_text_pad_token_id=None,
                        dropout_rate: float = 0.1,
                        ff_dim=None,
                        num_transformer_attention_heads=1,
                        num_edge_types=None,
                        num_time_edge_types=None,
                        use_relational_bias=False,
                        max_output_length=None,
                        type_vocab_size=None,
                        output_vocab_size=None,
                        num_output_propagation_steps=None,
                        use_pointer_candidate_masking=False,
                        jax2tf_compatible=None):
  """Returns a modeling configuration, either for training or evaluation."""
  # Assign conventional default to the intermediate dimension in a FF-layer
  # block, if not specified.
  if ff_dim is None:
    ff_dim = 4 * hidden_dim
  if num_input_timesteps is None:
    num_input_timesteps = 1
  # No point modeling temporal relations with a single timestep.
  if num_input_timesteps == 1:
    model_temporal_relations = False
  if num_transformer_attention_heads is None:
    num_transformer_attention_heads = 1
  if jax2tf_compatible is None:
    jax2tf_compatible = False
  return ModelConfig(
      hidden_dim=hidden_dim,
      max_input_length=max_input_length,
      num_input_timesteps=num_input_timesteps,
      model_temporal_relations=model_temporal_relations,
      node_position_dim=node_position_dim,
      num_input_propagation_steps=num_input_propagation_steps,
      token_vocab_size=token_vocab_size,
      node_text_pad_token_id=node_text_pad_token_id,
      deterministic=not is_training,
      dropout_rate=dropout_rate,
      ff_dim=ff_dim,
      num_transformer_attention_heads=num_transformer_attention_heads,
      num_edge_types=num_edge_types,
      num_time_edge_types=num_time_edge_types,
      use_relational_bias=use_relational_bias,
      max_output_length=max_output_length,
      type_vocab_size=type_vocab_size,
      output_vocab_size=output_vocab_size,
      num_output_propagation_steps=num_output_propagation_steps,
      use_pointer_candidate_masking=use_pointer_candidate_masking,
      jax2tf_compatible=jax2tf_compatible)


def get_train_config(hidden_dim,
                     max_input_length=None,
                     num_input_timesteps=None,
                     model_temporal_relations=True,
                     node_position_dim=1,
                     num_input_propagation_steps=None,
                     token_vocab_size=None,
                     node_text_pad_token_id=None,
                     num_transformer_attention_heads=None,
                     num_edge_types=None,
                     num_time_edge_types=None,
                     use_relational_bias=False,
                     max_output_length=None,
                     type_vocab_size=None,
                     output_vocab_size=None,
                     num_output_propagation_steps=None,
                     use_pointer_candidate_masking=False,
                     jax2tf_compatible=None,
                     dropout_rate: float = 0.1):
  """Returns a model config for training, which uses drop-out."""
  return create_model_config(
      is_training=True,
      hidden_dim=hidden_dim,
      max_input_length=max_input_length,
      num_input_timesteps=num_input_timesteps,
      model_temporal_relations=model_temporal_relations,
      node_position_dim=node_position_dim,
      num_input_propagation_steps=num_input_propagation_steps,
      token_vocab_size=token_vocab_size,
      node_text_pad_token_id=node_text_pad_token_id,
      dropout_rate=dropout_rate,
      num_transformer_attention_heads=num_transformer_attention_heads,
      num_edge_types=num_edge_types,
      num_time_edge_types=num_time_edge_types,
      use_relational_bias=use_relational_bias,
      max_output_length=max_output_length,
      type_vocab_size=type_vocab_size,
      output_vocab_size=output_vocab_size,
      num_output_propagation_steps=num_output_propagation_steps,
      use_pointer_candidate_masking=use_pointer_candidate_masking,
      jax2tf_compatible=jax2tf_compatible)


def get_eval_config(hidden_dim,
                    max_input_length=None,
                    num_input_timesteps=None,
                    model_temporal_relations=True,
                    node_position_dim=1,
                    num_input_propagation_steps=None,
                    token_vocab_size=None,
                    node_text_pad_token_id=None,
                    num_transformer_attention_heads=None,
                    num_edge_types=None,
                    num_time_edge_types=None,
                    use_relational_bias=False,
                    max_output_length=None,
                    type_vocab_size=None,
                    output_vocab_size=None,
                    num_output_propagation_steps=None,
                    use_pointer_candidate_masking=False,
                    jax2tf_compatible=None,
                    dropout_rate: float = 0.1):
  """Returns a model config for evaluating, which disables drop-out."""
  return create_model_config(
      is_training=False,
      hidden_dim=hidden_dim,
      max_input_length=max_input_length,
      num_input_timesteps=num_input_timesteps,
      model_temporal_relations=model_temporal_relations,
      node_position_dim=node_position_dim,
      num_input_propagation_steps=num_input_propagation_steps,
      token_vocab_size=token_vocab_size,
      node_text_pad_token_id=node_text_pad_token_id,
      dropout_rate=dropout_rate,
      num_transformer_attention_heads=num_transformer_attention_heads,
      num_edge_types=num_edge_types,
      num_time_edge_types=num_time_edge_types,
      use_relational_bias=use_relational_bias,
      max_output_length=max_output_length,
      type_vocab_size=type_vocab_size,
      output_vocab_size=output_vocab_size,
      num_output_propagation_steps=num_output_propagation_steps,
      use_pointer_candidate_masking=use_pointer_candidate_masking,
      jax2tf_compatible=jax2tf_compatible)
