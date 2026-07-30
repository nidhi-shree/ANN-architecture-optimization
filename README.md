# ANN Challenge (Best Model Wins) + Bonus Challenge

This repository holds everything for the ANN Challenge assignment: the main model, the bonus optimizer comparison, all evaluation results, and both PDF reports. Nothing here changes the dataset. Every script reads from one frozen CSV file, so results stay comparable across runs.

## What is inside

```
ann_project/
├── code/
│   ├── 00_prepare_dataset.py        # loads and freezes the dataset (run this first)
│   ├── 01_main_challenge_ann.py     # main challenge: 4 hidden layers, 150 epoch cap
│   └── 02_bonus_challenge_ann.py    # bonus challenge: 3 hidden layers, Adam vs SGD
├── outputs/
│   ├── dataset.csv                  # the frozen dataset, do not edit
│   ├── dataset_summary.txt
│   ├── main_metrics.txt
│   ├── main_classification_report.txt
│   └── bonus_metrics_comparison.txt
├── figures/
│   ├── main_model_architecture.png
│   ├── main_confusion_matrix.png
│   ├── main_training_curves.png
│   ├── bonus_adam_architecture.png
│   ├── bonus_sgd_architecture.png
│   ├── bonus_adam_confusion_matrix.png
│   ├── bonus_sgd_confusion_matrix.png
│   └── bonus_training_curves_comparison.png
├── latex/
│   ├── main_report.tex / main_report.pdf     # one-page-plus explanation for the main challenge
│   └── bonus_report.tex / bonus_report.pdf   # 2-3 page report for the bonus challenge
└── README.md
```

## Dataset used

The assignment says the dataset must not be changed, but it does not name one, so this project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**. It is a well known, public, binary classification dataset that ships with scikit-learn. It has 569 patient records, 30 numeric features (things like mean radius and mean texture of a cell nucleus), and one target column, where 0 means malignant and 1 means benign. If your class gave you a different dataset, just drop your CSV in place of `outputs/dataset.csv` and keep the same column layout; the rest of the pipeline does not need to change.

## How to run it

Run the three scripts in order. Each one prints its own results to the terminal and also saves them to disk.

```bash
pip install tensorflow-cpu scikit-learn pandas matplotlib pydot

python code/00_prepare_dataset.py
python code/01_main_challenge_ann.py
python code/02_bonus_challenge_ann.py
```

After that, check the `figures/` folder for the pictures and the `outputs/` folder for the text metrics. The two PDF reports are already built in `latex/`, but you can rebuild them with `pdflatex main_report.tex` and `pdflatex bonus_report.tex` if you edit the numbers.

## Main Challenge summary

The main model is a normal feed-forward ANN with four hidden layers, so it stays under the five layer limit. It uses Dense layers of size 64, 32, 16, and 8, with ReLU activation, plus batch normalization and dropout to keep it from overfitting. It trains for up to 150 epochs, which is under the 200 epoch cap, and early stopping brings training to a halt once validation loss stops improving. On the held-out test set, it reached:

| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9855 |
| Recall | 0.9444 |
| F1 Score | 0.9645 |

The full confusion matrix and a plain-language explanation of why this design works are in `latex/main_report.pdf`.

## Bonus Challenge summary

The bonus model has three hidden layers (48, 24, 12 neurons, so each layer is a different size), mixes ReLU and tanh activations, and uses dropout after the first two layers. The same architecture was trained twice for 100 epochs each, once with Adam and once with SGD plus momentum, so the only thing that changes between the two runs is the optimizer.

| Metric | Adam | SGD (momentum) |
|---|---|---|
| Accuracy | 0.9561 | 0.9649 |
| Precision | 0.9855 | 0.9857 |
| Recall | 0.9444 | 0.9583 |
| F1 Score | 0.9645 | 0.9718 |

SGD with momentum came out slightly ahead on this dataset and this epoch budget. The full write-up, including why it won and when Adam would likely win instead, is in `latex/bonus_report.pdf`.

## Rules followed

- Dataset was loaded once and never modified afterward.
- Both models are plain ANNs (Dense layers only), no CNNs, no RNNs, no transfer learning, no pre-trained weights.
- Main model uses 4 hidden layers (limit was 5) and trains for at most 150 epochs (limit was 200).
- Bonus model uses 3 hidden layers with different neuron counts, two activation functions (ReLU and tanh), dropout, and trains for exactly 100 epochs with two optimizers (Adam and SGD).
- Accuracy, precision, recall, F1 score, and confusion matrix are reported for both challenges.

## Notes on reproducibility

A fixed random seed (42) is used for the train/test split and for TensorFlow's random operations, so re-running the scripts should give results close to the ones shown here. Small differences can still happen between machines and library versions, since GPU and CPU floating point operations are not always bit-for-bit identical. That is expected and normal, not a bug.
