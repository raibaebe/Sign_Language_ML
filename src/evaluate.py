"""Model evaluation: accuracy, precision/recall/F1, confusion matrix."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset, LABEL_TO_LETTER, NUM_CLASSES
from src.preprocessing import preprocess_pipeline
from src.model import load_trained_model


def evaluate(model_path="models/trained_model.h5", data_dir="data",
             save_plots=True, output_dir="models"):
    """Evaluate model on test set, print metrics, plot confusion matrix."""
    model = load_trained_model(model_path)
    dataset = load_dataset(data_dir)
    X_test, y_test = preprocess_pipeline(dataset["test_images"], dataset["test_labels"])

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = [LABEL_TO_LETTER[i] for i in range(NUM_CLASSES)]

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class metrics
    accuracy = np.trace(cm) / cm.sum()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n{'':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    print("-" * 32)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{class_names[i]:>6s} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {int(cm[i].sum()):>6d}")

    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"\nSaved to {path}")

    plt.show()
    return {"accuracy": accuracy, "confusion_matrix": cm}


def plot_learning_curve(history, save_path="models/learning_curve.png"):
    """Plot training and validation accuracy/loss over epochs."""
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, history["accuracy"], "b-o", markersize=4, label="Training")
    ax1.plot(epochs, history["val_accuracy"], "r-o", markersize=4, label="Validation")
    ax1.set_title("Model Accuracy", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history["loss"], "b-o", markersize=4, label="Training")
    ax2.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Validation")
    ax2.set_title("Model Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Learning Curve", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"\nLearning curve saved to {save_path}")

    plt.show()
    return fig
