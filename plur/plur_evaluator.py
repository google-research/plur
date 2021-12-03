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

"""Functions for evaluating predictions on PLUR datasets.
"""
from absl import app
from absl import flags

from plur.eval.code2seq_eval import Code2seqEval
from plur.eval.convattn_eval import ConvattnEval
from plur.eval.cubert_exception_classification_eval import CuBertExceptionClassificationEval
from plur.eval.cubert_function_docstring_classification_eval import CuBertFunctionDocstringClassificationEval
from plur.eval.cubert_swapped_operand_classification_eval import CuBertSwappedOperandClassificationEval
from plur.eval.cubert_variable_misuse_classification_eval import CuBertVariableMisuseClassificationEval
from plur.eval.cubert_variable_misuse_repair_eval import CuBertVariableMisuseRepairEval
from plur.eval.cubert_variable_misuse_repair_unpointed_eval import CuBertVariableMisuseRepairUnpointedEval
from plur.eval.cubert_wrong_operator_classification_eval import CuBertWrongOperatorClassificationEval
from plur.eval.funcom_eval import FuncomEval
from plur.eval.great_var_misuse_eval import GreatVarMisuseEval
from plur.eval.great_var_misuse_unpointed_eval import GreatVarMisuseUnpointedEval
from plur.eval.hoppity_single_ast_diff_eval import HoppitySingleAstDiffEval
from plur.eval.manysstubs4j_eval import Manysstubs4jEval
from plur.eval.retrieve_and_edit_eval import RetrieveAndEditEval

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'dataset_name',
    'code2seq_dataset',
    (
        'code2seq_dataset',
        'convattn_dataset',
        'cubert_exception_classification_dataset',
        'cubert_variable_misuse_classification_dataset',
        'cubert_variable_misuse_repair_dataset',
        'cubert_variable_misuse_repair_unpointed_dataset',
        'cubert_variable_misuse_repair_nocopy_dataset',
        'cubert_variable_misuse_repair_unpointed_nocopy_dataset',
        'cubert_swapped_operand_classification_dataset',
        'cubert_function_docstring_classification_dataset',
        'cubert_wrong_operator_classification_dataset',
        'funcom_dataset',
        'great_var_misuse_dataset',
        'great_var_misuse_unpointed_dataset',
        'hoppity_single_ast_diff_dataset',
        'manysstubs4j_dataset',
        'ogb_code_dataset',
        'retrieve_and_edit_dataset',
    ),
    'Name of the dataset to evaluate.')
flags.DEFINE_string('prediction_file_pattern', '/tmp/predictions.txt',
                    'Glob file pattern containing the predictions.')
flags.DEFINE_string('target_file_pattern', '/tmp/targets.txt',
                    'Glob file pattern containing the ground truths.')
flags.DEFINE_integer('top_n_beam_search', 1,
                     'Only evaluate top n predictions of beam search.')


def get_eval_class(dataset_name):
  """Get the dataset class based on dataset_name."""
  if dataset_name == 'code2seq_dataset':
    return Code2seqEval
  elif dataset_name == 'convattn_dataset':
    return ConvattnEval
  elif dataset_name == 'funcom_dataset':
    return FuncomEval
  elif dataset_name == 'great_var_misuse_dataset':
    return GreatVarMisuseEval
  elif dataset_name == 'great_var_misuse_unpointed_dataset':
    return GreatVarMisuseUnpointedEval
  elif dataset_name == 'cubert_exception_classification_dataset':
    return CuBertExceptionClassificationEval
  elif dataset_name == 'cubert_variable_misuse_classification_dataset':
    return CuBertVariableMisuseClassificationEval
  elif dataset_name == 'cubert_variable_misuse_repair_dataset':
    return CuBertVariableMisuseRepairEval
  elif dataset_name == 'cubert_variable_misuse_repair_unpointed_dataset':
    return CuBertVariableMisuseRepairUnpointedEval
  elif dataset_name == 'cubert_variable_misuse_repair_nocopy_dataset':
    return CuBertVariableMisuseRepairEval
  elif dataset_name == 'cubert_variable_misuse_repair_unpointed_nocopy_dataset':
    return CuBertVariableMisuseRepairUnpointedEval
  elif dataset_name == 'cubert_swapped_operand_classification_dataset':
    return CuBertSwappedOperandClassificationEval
  elif dataset_name == 'cubert_function_docstring_classification_dataset':
    return CuBertFunctionDocstringClassificationEval
  elif dataset_name == 'cubert_wrong_operator_classification_dataset':
    return CuBertWrongOperatorClassificationEval
  elif dataset_name == 'hoppity_single_ast_diff_dataset':
    return HoppitySingleAstDiffEval
  elif dataset_name == 'ogb_code_dataset':
    # The ogb_code_dataset eval function is the same as convattn_dataset.
    return ConvattnEval
  elif dataset_name == 'manysstubs4j_dataset':
    return Manysstubs4jEval
  elif dataset_name == 'retrieve_and_edit_dataset':
    return RetrieveAndEditEval
  else:
    raise ValueError('{} is not supported.'.format(dataset_name))


def evaluate_predictions(dataset_name, prediction_file_pattern,
                         target_file_pattern, top_n_beam_search):
  """Evaluate the top n predictions per target and return the metrics as string.

  Args:
    dataset_name: The name of the dataset, must be one of the names in flag
      enum for dataset_name.
    prediction_file_pattern: The glob file pattern containing the predictions.
      It should be formatted as 1 line per prediction.
    target_file_pattern: The glob file parttern containing the targets.
      It should be formatted as 1 line per target.
    top_n_beam_search: Only evaluate top n predictions of each target.

  Returns:
    The evaluation class for the dataset specified by dataset_name.
  """
  eval_class = get_eval_class(dataset_name)
  return eval_class(prediction_file_pattern, target_file_pattern,
                    top_n_beam_search)


def main(_):
  evaluator = evaluate_predictions(FLAGS.dataset_name,
                                   FLAGS.prediction_file_pattern,
                                   FLAGS.target_file_pattern,
                                   FLAGS.top_n_beam_search)
  print(evaluator.get_metric_as_string())


if __name__ == '__main__':
  app.run(main)
