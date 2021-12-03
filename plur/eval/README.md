# PLUR evaluation

The PLUR evaluation reports metrics that are measured in the state-of-the-art paper trained on the respective dataset. The PLUR evaluation works in offline mode, meaning that it evaluates the predictions that are already stored on disk.

## Usage

To evaluate predictions, for example on the code2seq dataset, you can run:

```bash
python3 plur_evaluator.py --dataset_name=code2seq_dataset --target_file=/tmp/code2seq_dataset/targets.txt --prediction_file=/tmp/code2seq_dataset/predictions.txt
```

It will extract the predictions and targets, then compute precision, recall and F1 score (metrics used in the code2seq paper).

## Concrete example

Here is a concrete example on how to run the code2seq evaluation. First we generate some random predictions and targets.

```bash
$ echo $'code2seq eval\nfoo bar' > /tmp/predictions.txt
$ cat /tmp/predictions.txt
code2seq eval
foo bar
$ echo $'code2seq eval test\nhello world' > /tmp/targets.txt
$ cat /tmp/targets.txt
code2seq eval test
hello world
```

Then we evaluate the predictions:

```bash
$ python3 plur_evaluator.py --dataset_name=convattn_dataset --prediction_file=/tmp/predictions.txt --target_file=/tmp/targets.txt
Precision: 0.5, Recall: 0.4, F1-score: 0.4444444444444445
```

## File format

plur_evaluator.py expects the targets and predictions to be stored on disk in a certain format. The target file should have 1 line for each ground truth target, and the prediction file should have 1 line for each prediction. The order of the lines in the prediction file should match the lines in the target file. For example:

```
targets.txt:
FIRST TARGET
SECOND TARGET
THIRD TARGET
...

predictions.txt:
PREDICTION FOR THE FIRST TARGET
PREDICTION FOR THE SECOND TARGET
PREDICTION FOR THE THIRD TARGET
...
```

For some datasets, we also support multiple predictions per target. In this case, the predictions for the same target should be separated by a tab (\t). For example:

```
target.txt:
FIRST TARGET
SECOND TARGET
THIRD TARGET
...

predictions.txt:
FIRST PREDICTION FOR THE FIRST TARGET\tSECOND PREDICTION FOR THE FIRST TARGET
FIRST PREDICTION FOR THE SECOND TARGET\tSECOND PREDICTION FOR THE SECOND TARGET
FIRST PREDICTION FOR THE THIRD TARGET\tSECOND PREDICTION FOR THE THIRD TARGET
...
```

Multiple predictions are separated by tabs (\t) because we are sure that they are not part of the target/prediction. If characters like `;` or `|` are used to separate predictions, we cannot be sure that `;` or `|` in the line are separating the predictions, or they are part of the prediction.

## State-of-the-art results

Please see the PLUR paper ([Chen et al., 2021](https://papers.nips.cc/paper/2021/hash/c2937f3a1b3a177d2408574da0245a19-Abstract.html)) for comparisons to state-of-the-art results at the time of publication.
