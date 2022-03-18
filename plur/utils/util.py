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

"""PLUR utility functions."""

import json
import os
import shutil

# This env variable disables error from GitPython if git is not found in the
# path. This can happen when running beam in distributed settings, and git
# is not installed (it is also not needed in that case).
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


import apache_beam as beam
import git
from plur.utils import constants



def escaped_str(s: str):
  """Escape all special characters in the string.

  Some special characters can have surprising effect, such as the newline
  character. For example it is common that the vocabulary file have one line
  per token, but if the token contain a newline character, it will be counted
  as two tokens and mess up the loading of vocabulary files. Therefore we use
  this function to escape all special characters when reading strings in the
  wild.

  Args:
    s: A string.

  Returns:
    A string where all the special characters are escaped.
  """
  # We use [1:-1] to remove the quote added by repr()
  return repr(s)[1:-1]


def graph_to_output_example_is_not_none(graph_to_output_example_dict):
  return graph_to_output_example_dict['GraphToOutputExample'] is not None


def filter_split(data, split):
  return data['split'] == split


def add_field_to_metadata(metadata, fieldname, value, distinguishing_text=''):
  metadata = (
      metadata
      | f'Add {fieldname} to metadata {distinguishing_text}' >> beam.ParDo(
          AddDictField(fieldname), value))
  return metadata


def filter_impossible_tfexample(tfexample_feature_dict,
                                output_token_vocab_dict):
  output_oov_token_id = output_token_vocab_dict[constants.OOV_TOKEN]
  for index, output_token_id in enumerate(
      tfexample_feature_dict[constants.KEY_OUTPUT_TOKEN_IDS]):
    # If the output token is OOV and can not be copied from input.
    if (output_token_id == output_oov_token_id and
        index not in tfexample_feature_dict[constants.KEY_COPY_OUTPUT_INDICES]):
      return False
  return True


def safe_division(numerator, denominator):
  if denominator == 0:
    return 0.0
  else:
    return numerator / denominator


def check_need_to_extract(files_to_extract, extracted_dir, filename):
  """Check if we need to (re)extract the compressed file.

  If extracted_dir exists, ie. the compressed file 'filename' is already
  extracted, we ask the user if she wants to reextract it. Otherwise we add
  filename into the list files_to_extract.

  Args:
    files_to_extract: A list of compressed filesnames that should be extracted.
    extracted_dir: Directory that 'filename' should extract to.
    filename: The filename of the compressed file.
  Returns:
    A list of filenames that should be extracted. If extracted_dir does not
    exist, or if the user wants to reextract it, the list is
    files_to_extract + [filename]. Otherwise the returned list is
    the input argument files_to_extract.
  """
  if os.path.exists(extracted_dir):
    re_extract = input(
        '{} already exists, delete it and re-extract?(y/n): '.format(
            extracted_dir))
    if re_extract.lower() == 'y':
      shutil.rmtree(extracted_dir)
      files_to_extract.append(filename)
  else:
    files_to_extract.append(filename)
  return files_to_extract


class AddDictField(beam.DoFn):

  def __init__(self, fieldname):
    self.fieldname = fieldname

  def process(self, metadata, value):
    metadata[self.fieldname] = value
    yield metadata


class JsonCoder(beam.coders.Coder):

  def encode(self, x):
    return json.dumps(x).encode('utf-8')

  def decode(self, x):
    return json.loads(x).decode('utf-8')


class GitProgress(git.RemoteProgress):

  def update(self, op_code, cur_count, max_count=None, message=''):
    print(self._cur_line, end='\r')
