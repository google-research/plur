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

"""Code for recording measurements."""

from absl import flags

from plur.model_design import data_types as dt
from plur.model_design import metrics
import tensorflow as tf


FLAGS = flags.FLAGS

METRIC_LOSS = 'Loss'
METRIC_ELEMENT_ACCURACY = 'ElAcc'
METRIC_SEQUENCE_ACCURACY = 'SeqAcc'
METRIC_POINTER_ACCURACY = 'PoAcc'
METRIC_POINTER_SEQUENCE_ACCURACY = 'PoSeqAcc'
METRIC_TOCO_ACCURACY = 'ToCoAcc'
METRIC_TOCO_SEQUENCE_ACCURACY = 'ToCoSeqAcc'
METRIC_TOKEN_RATIO = 'ToRatio'
METRICS = (METRIC_LOSS, METRIC_ELEMENT_ACCURACY, METRIC_SEQUENCE_ACCURACY,
           METRIC_POINTER_ACCURACY, METRIC_POINTER_SEQUENCE_ACCURACY,
           METRIC_TOCO_ACCURACY, METRIC_TOCO_SEQUENCE_ACCURACY,
           METRIC_TOKEN_RATIO)


class SplitMeasurementRecorder():
  """A class for recording a data split's measurements."""

  def __init__(self, split: str, summary_writer: tf.summary.SummaryWriter):
    self._split = split
    self._summary_writer = summary_writer

  def record_measurements(self, step: int, loss: float,
                          accuracy: metrics.AccuracyMetrics):
    """Records training split measurements."""
    measurements = {
        METRIC_LOSS: loss,
        METRIC_ELEMENT_ACCURACY: accuracy.get_element_accuracy(),
        METRIC_SEQUENCE_ACCURACY: accuracy.get_seq_accuracy(),
        METRIC_POINTER_ACCURACY: accuracy.get_pointer_accuracy(),
        METRIC_POINTER_SEQUENCE_ACCURACY: accuracy.get_pointer_seq_accuracy(),
        METRIC_TOCO_ACCURACY: accuracy.get_toco_accuracy(),
        METRIC_TOCO_SEQUENCE_ACCURACY: accuracy.get_toco_seq_accuracy(),
        METRIC_TOKEN_RATIO: accuracy.get_token_prediction_ratio(),
    }

    with self._summary_writer.as_default():
      for metric_name, value in measurements.items():
        tf.summary.scalar(f'{self._split}{metric_name}', value, step=step)
      self._summary_writer.flush()


class MeasurementRecorder(object):
  """A class for recording measurements about model's training."""

  def __init__(self, model_train_args: dt.TrainingConfiguration):
    """Initializes a MeasurementRecorder.

    Args:
      model_train_args: Training configuration.
    """
    # Tensforflow event logging.
    self._event_writer = tf.summary.create_file_writer(model_train_args.exp_dir)

    self._train_recorder = SplitMeasurementRecorder('Train', self._event_writer)
    self._validation_recorder = SplitMeasurementRecorder(
        'Valid', self._event_writer)

  def record_train_measurements(self, step: int, loss: float,
                                accuracy: metrics.AccuracyMetrics):
    """Records training split measurements."""
    self._train_recorder.record_measurements(step, loss, accuracy)

  def record_validation_measurements(self, step: int, loss: float,
                                     accuracy: metrics.AccuracyMetrics):
    """Records validation split measurements."""
    self._validation_recorder.record_measurements(step, loss, accuracy)
