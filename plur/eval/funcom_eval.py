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

"""Compute metrics used in Funcom dataset.
"""
import collections

from nltk.translate.bleu_score import corpus_bleu
from plur.eval import util as eval_util
from plur.eval.plur_eval import PlurEval


class FuncomEval(PlurEval):
  """Eval class for Funcom dataset.

  We use the same evaluation function as: LeClair, Alexander, et al. 'Improved
  code summarization via a graph neural network.' arXiv preprint
  arXiv:2004.02843 (2020), since Funcom is just a dataset paper. However, it
  should be noted that the authors of 'Improved code summarization via a graph
  neural network' did not use the original funcom dataset. They have generated
  ASTs from the Funcom dataset and used that as input.
  """

  def compute_metric_for_one_target(self):
    """pass here since we can use corpus_bleu on all predictions."""
    pass

  def evaluate_once(self,
                    grouped_prediction_lines,
                    target_lines,
                    highest_n_gram_bleu=4):
    """Evaluate the predictions and compute BLEU scores.

    Args:
      grouped_prediction_lines: predicted lines from eval step.
      target_lines: ground truth lines from eval step.
      highest_n_gram_bleu: BLEU n-gram score we use to choose the highest
        BLEU1, BLEU2, BLEU3 and BLEU4 score. Must be an integer in
        [1, 2, 3, 4].

    Returns:
      A tuple of BLEU1, BLEU2, BLEU3 and BLEU4 scores. For example BLEU4 is the
      BLEU score when calculating with 4-grams.
    """
    assert 1 <= highest_n_gram_bleu <= 4

    # The list of references to corpus_bleu. It is expected to have
    # type list[list[list[str]]].
    target_lines_tokens = [
        [line.split(' ')] for line in target_lines
    ]

    # highest_bleu_scores stores the BLEU scores for different n-grams.
    # highest_bleu_scores[1] is the BLEU score with unigram and
    # highest_bleu_scores[2] is the BLEU score with bigram and etc.
    # highest_bleu_scores[0] will always be zero.
    # We only compute BLEU score up till 4-gram to compare against the BLEU
    # scores in 'Improved code summarization via a graph neural network'.
    highest_bleu_scores = [0, 0, 0, 0, 0]
    for rank in range(self.top_n):
      # The list of hypotheses to corpus_bleu. It is expected to have
      # type list[list[str]] .
      prediction_lines_tokens = [
          beam_lines[rank].split(' ') for beam_lines in grouped_prediction_lines
      ]

      # Compute BLEU score for the current prediction rank.
      # NLTK has a potential bug when setting auto_reweight=True.
      # The reweight is calculated like this:
      #
      # if auto_reweigh:
      #   if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
      #     weights = (1 / hyp_lengths,) * hyp_lengths
      #
      # But the hyp_lengths is calculated as:
      #
      # for references, hypothesis in zip(list_of_references, hypotheses):
      #   for i, _ in enumerate(weights, start=1):
      #     p_i = modified_precision(references, hypothesis, i)
      #     p_numerators[i] += p_i.numerator
      #     p_denominators[i] += p_i.denominator
      #
      #   hyp_len = len(hypothesis)
      #   hyp_lengths += hyp_len
      #   ref_lengths += closest_ref_length(references, hyp_len)
      #
      # The hyp_lengths is sum of all hypotheses lengths, instead of being the
      # maximum length of all hypotheses
      current_bleu_score = [
          0,  # Always 0.
          corpus_bleu(target_lines_tokens, prediction_lines_tokens,  # 1-gram.
                      weights=(1,)) * 100,
          corpus_bleu(target_lines_tokens, prediction_lines_tokens,  # 2-grams.
                      weights=(0.5, 0.5)) * 100,
          corpus_bleu(target_lines_tokens, prediction_lines_tokens,  # 3-grams.
                      weights=(1/3, 1/3, 1/3)) * 100,
          corpus_bleu(target_lines_tokens, prediction_lines_tokens,  # 4-grams.
                      weights=(0.25, 0.25, 0.25, 0.25)) * 100
      ]

      # Choose the highest BLEU score based on the highest_n_gram_bleu argument.
      if current_bleu_score[highest_n_gram_bleu] > highest_bleu_scores[
          highest_n_gram_bleu]:
        highest_bleu_scores = current_bleu_score

    metrics = collections.OrderedDict()
    metrics_format = collections.OrderedDict()

    metrics['bleu_1'] = highest_bleu_scores[1]
    metrics_format['bleu_1'] = '{:.5f}'
    metrics['bleu_2'] = highest_bleu_scores[2]
    metrics_format['bleu_2'] = '{:.5f}'
    metrics['bleu_3'] = highest_bleu_scores[3]
    metrics_format['bleu_3'] = '{:.5f}'
    metrics['bleu_4'] = highest_bleu_scores[4]
    metrics_format['bleu_4'] = '{:.5f}'

    return eval_util.Results(
        total=len(target_lines), metrics=metrics, metrics_format=metrics_format)

  def get_metric_as_string(self):
    return str(self.evaluate())
