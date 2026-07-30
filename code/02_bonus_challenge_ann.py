"""
02_bonus_challenge_ann.py

Bonus Challenge - Design Your Own ANN
--------------------------------------
Constraints satisfied:
    - At least 3 hidden layers                 -> 3 hidden layers used
    - Different number of neurons in each layer -> 48, 24, 12
    - At least two different activation funcs   -> relu + tanh
    - Dropout used to fight overfitting          -> Dropout after layer 1 & 2
    - Two optimizers compared                    -> Adam vs SGD (same architecture)
    - Train for 100 epochs                       -> epochs=100, no early stopping,
                                                     so both runs get an equal,
                                                     fair comparison
    - Same dataset as the main challenge (not changed)

Outputs (per optimizer, "adam" and "sgd"):
    figures/bonus_{opt}_architecture.png
    figures/bonus_{opt}_confusion_matrix.png
    figures/bonus_training_curves_comparison.png (both optimizers together)
    outputs/bonus_metrics_comparison.txt
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
DATA_PATH = "/home/claude/ann_project/outputs/dataset.csv"
FIG_DIR = "/home/claude/ann_project/figures"
OUT_DIR = "/home/claude/ann_project/outputs"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y


def build_model(input_dim, optimizer_name):
    """
    Fixed architecture used for BOTH optimizer runs, so the comparison
    is fair (only the optimizer changes):

        Input(30)
          -> Dense(48, relu)
          -> Dropout(0.3)
          -> Dense(24, tanh)
          -> Dropout(0.2)
          -> Dense(12, relu)
          -> Dense(1, sigmoid)
    """
    model = keras.Sequential(name=f"Bonus_ANN_{optimizer_name}")
    model.add(layers.Input(shape=(input_dim,), name="input_layer"))

    model.add(layers.Dense(48, activation="relu",
                            kernel_initializer="he_normal", name="hidden_1_relu"))
    model.add(layers.Dropout(0.3, name="dropout_1"))

    model.add(layers.Dense(24, activation="tanh",
                            kernel_initializer="glorot_uniform", name="hidden_2_tanh"))
    model.add(layers.Dropout(0.2, name="dropout_2"))

    model.add(layers.Dense(12, activation="relu",
                            kernel_initializer="he_normal", name="hidden_3_relu"))

    model.add(layers.Dense(1, activation="sigmoid", name="output_layer"))

    if optimizer_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=0.001)
    else:
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_experiment(optimizer_name, X_train, X_test, y_train, y_test):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(X_train.shape[1], optimizer_name)

    keras.utils.plot_model(
        model,
        to_file=f"{FIG_DIR}/bonus_{optimizer_name}_architecture.png",
        show_shapes=True, show_layer_names=True, show_layer_activations=True, dpi=150,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,          # fixed at 100 as required, no early stopping
        batch_size=32,
        verbose=2,
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["malignant", "benign"]),
        "history": history.history,
    }

    # Confusion matrix figure
    cm = metrics["cm"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens" if optimizer_name == "adam" else "Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Bonus Challenge ({optimizer_name.upper()}) - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bonus_{optimizer_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    return metrics


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    for opt_name in ["adam", "sgd"]:
        print(f"\n========== Training with {opt_name.upper()} ==========")
        results[opt_name] = run_experiment(opt_name, X_train, X_test, y_train, y_test)

    # ---- Combined training curve comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for opt_name, style in [("adam", "-"), ("sgd", "--")]:
        h = results[opt_name]["history"]
        axes[0].plot(h["val_loss"], style, label=f"{opt_name} val_loss")
        axes[1].plot(h["val_accuracy"], style, label=f"{opt_name} val_acc")
    axes[0].set_title("Validation Loss: Adam vs SGD"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Validation Accuracy: Adam vs SGD"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/bonus_training_curves_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Metrics comparison text file ----
    with open(f"{OUT_DIR}/bonus_metrics_comparison.txt", "w") as f:
        f.write("Bonus Challenge - Adam vs SGD Comparison\n")
        f.write("=" * 45 + "\n\n")
        for opt_name in ["adam", "sgd"]:
            m = results[opt_name]
            f.write(f"--- Optimizer: {opt_name.upper()} ---\n")
            f.write(f"Accuracy : {m['accuracy']:.4f}\n")
            f.write(f"Precision: {m['precision']:.4f}\n")
            f.write(f"Recall   : {m['recall']:.4f}\n")
            f.write(f"F1 Score : {m['f1']:.4f}\n")
            f.write(f"Confusion Matrix:\n{m['cm']}\n")
            f.write(f"{m['report']}\n\n")

    print("\n=== SUMMARY ===")
    for opt_name in ["adam", "sgd"]:
        m = results[opt_name]
        print(f"{opt_name.upper():5s} -> acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
