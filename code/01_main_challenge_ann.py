"""
01_main_challenge_ann.py

ANN Challenge (Best Model Wins) - Main Deliverable Script
-----------------------------------------------------------
Rules followed:
    - Dataset is NOT changed (loaded as-is from outputs/dataset.csv,
      which was frozen by 00_prepare_dataset.py).
    - ANN only (a plain feed-forward Dense network, no CNN/RNN tricks).
    - Maximum 5 hidden layers  -> this model uses 4 hidden layers.
    - Maximum 200 epochs       -> this model trains for 150 epochs
      (EarlyStopping is used, so the real number of epochs run can be
      lower, but the cap passed to .fit() never exceeds 200).
    - No transfer learning, no pre-trained weights.

Outputs:
    figures/main_model_architecture.png   -> architecture screenshot
    outputs/main_metrics.txt              -> accuracy / precision / recall / f1
    figures/main_confusion_matrix.png     -> confusion matrix heatmap
    outputs/main_classification_report.txt
    figures/main_training_curves.png      -> loss/accuracy curves
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "/home/claude/ann_project/outputs/dataset.csv"
FIG_DIR = "/home/claude/ann_project/figures"
OUT_DIR = "/home/claude/ann_project/outputs"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y


def build_model(input_dim):
    """
    Architecture (4 hidden layers, well under the 5 layer cap):

        Input(30)
          -> Dense(64, relu) + BatchNorm
          -> Dense(32, relu) + Dropout(0.3)
          -> Dense(16, relu) + Dropout(0.2)
          -> Dense(8,  relu)
          -> Dense(1,  sigmoid)   [output layer]

    Reasoning for these choices is written up in the one-page report
    (report_main_challenge.pdf).
    """
    model = keras.Sequential(name="Main_Challenge_ANN")
    model.add(layers.Input(shape=(input_dim,), name="input_layer"))

    model.add(layers.Dense(64, activation="relu",
                            kernel_initializer="he_normal", name="hidden_1"))
    model.add(layers.BatchNormalization(name="batchnorm_1"))

    model.add(layers.Dense(32, activation="relu",
                            kernel_initializer="he_normal", name="hidden_2"))
    model.add(layers.Dropout(0.3, name="dropout_1"))

    model.add(layers.Dense(16, activation="relu",
                            kernel_initializer="he_normal", name="hidden_3"))
    model.add(layers.Dropout(0.2, name="dropout_2"))

    model.add(layers.Dense(8, activation="relu",
                            kernel_initializer="he_normal", name="hidden_4"))

    model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    model.summary()

    # Save architecture screenshot (deliverable #2)
    keras.utils.plot_model(
        model,
        to_file=f"{FIG_DIR}/main_model_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        dpi=150,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,          # <= 200 epoch cap, satisfies the rule
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---- Evaluation ----
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["malignant", "benign"])

    print("\n=== MAIN CHALLENGE RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print(report)

    with open(f"{OUT_DIR}/main_metrics.txt", "w") as f:
        f.write("Main ANN Challenge - Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"Epochs actually run: {len(history.history['loss'])}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    with open(f"{OUT_DIR}/main_classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Main Challenge - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/main_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss over epochs"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy over epochs"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/main_training_curves.png", dpi=150)
    plt.close(fig)

    print("\nAll outputs saved to figures/ and outputs/ folders.")


if __name__ == "__main__":
    main()
