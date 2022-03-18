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

"""Code defining the interface to data."""

import abc
from typing import Text

from plur.model_design import data_generation


class DataManager(abc.ABC):
  """An adapter class to allow access to PLUR data."""

  def __init__(self):
    pass

  @property
  def train_data_generator_fn(self):
    return self._train_data_generator_fn# bind-properties

  @property
  def train_data_generator(self):
    return self._train_data_generator# bind-properties

  @property
  def valid_data_generator_fn(self):
    return self._valid_data_generator_fn# bind-properties

  @property
  def test_data_generator_fn(self):
    return self._test_data_generator_fn# bind-properties

  @property
  def padding_spec(self):
    return self._padding_spec# bind-properties

  @property
  def input_encoder(self):
    raise NotImplementedError

  @property
  def output_encoder(self):
    return self._output_encoder# bind-properties

  @property
  @abc.abstractmethod
  def token_vocab_size(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def type_vocab_size(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def output_vocab_size(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def node_text_pad_token_id(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def output_oov_token_id(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def output_pad_token_id(self):
    raise NotImplementedError


class PlurDataManager(DataManager):
  """A PLUR specific data manager."""

  def __init__(self, data_dir: Text, batch_size_per_host: int,
               drop_remainder: bool):
    super().__init__()
    (self._train_data_generator_fn, self._valid_data_generator_fn,
     self._test_data_generator_fn, self._padding_spec, self._output_encoder) = (
         data_generation.get_plur_data_generator_and_padding_spec(
             data_dir, batch_size_per_host, drop_remainder))
    self._train_data_generator = self._train_data_generator_fn()

  @property
  def token_vocab_size(self):
    return self._train_data_generator.node_label_vocab_size

  @property
  def type_vocab_size(self):
    return self._train_data_generator.node_type_vocab_size

  @property
  def output_vocab_size(self):
    return self._train_data_generator.output_token_vocab_size

  @property
  def node_text_pad_token_id(self):
    return self._train_data_generator.node_label_pad_id

  @property
  def output_oov_token_id(self):
    return self._train_data_generator.output_token_oov_id

  @property
  def output_pad_token_id(self):
    return self._train_data_generator.output_token_pad_id
