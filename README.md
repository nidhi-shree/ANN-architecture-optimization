# ANN Architecture Optimization

## Dataset used

This project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**. It is a well known, public, binary classification dataset that ships with scikit-learn. It has 569 patient records, 30 numeric features (things like mean radius and mean texture of a cell nucleus), and one target column, where 0 means malignant and 1 means benign. 

## How to run it

```bash
pip install tensorflow-cpu scikit-learn pandas matplotlib pydot

python code/00_prepare_dataset.py
python code/01_main_challenge_ann.py
python code/02_bonus_challenge_ann.py
```

## Main Challenge summary

The main model is a normal feed-forward ANN with four hidden layers, so it stays under the five layer limit. It uses Dense layers of size 64, 32, 16, and 8, with ReLU activation, plus batch normalization and dropout to keep it from overfitting. It trains for up to 150 epochs, which is under the 200 epoch cap, and early stopping brings training to a halt once validation loss stops improving. On the held-out test set, it reached:

| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9855 |
| Recall | 0.9444 |
| F1 Score | 0.9645 |


## Bonus Challenge summary

The bonus model has three hidden layers (48, 24, 12 neurons, so each layer is a different size), mixes ReLU and tanh activations, and uses dropout after the first two layers. The same architecture was trained twice for 100 epochs each, once with Adam and once with SGD plus momentum, so the only thing that changes between the two runs is the optimizer.

| Metric | Adam | SGD (momentum) |
|---|---|---|
| Accuracy | 0.9561 | 0.9649 |
| Precision | 0.9855 | 0.9857 |
| Recall | 0.9444 | 0.9583 |
| F1 Score | 0.9645 | 0.9718 |
ns are not always bit-for-bit identical. That is expected and normal, not a bug.
