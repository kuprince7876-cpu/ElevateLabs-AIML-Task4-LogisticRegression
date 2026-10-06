import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

def load_and_save_data():
    """Load Breast Cancer Wisconsin Dataset and save locally."""
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target  # 1 = Benign, 0 = Malignant
    
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/breast_cancer.csv", index=False)
    print("Dataset successfully saved to dataset/breast_cancer.csv")
    print(f"Dataset Shape: {df.shape}\n")
    return df, cancer.target_names

def plot_sigmoid():
    """Plot the sigmoid function curve."""
    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(7, 4))
    plt.plot(z, sigmoid, color='blue', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    plt.title("Sigmoid Activation Function")
    plt.xlabel("Log-Odds / Z-score")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sigmoid_curve.png")
    plt.show()

def train_and_evaluate():
    df, target_names = load_and_save_data()
    
    # Feature & Target Split
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Initialization and Training
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions & Probabilities
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics Evaluation
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    print("=" * 50)
    print("MODEL EVALUATION METRICS (Default Threshold = 0.5)")
    print("=" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_auc_curve.png')
    plt.show()

    # Threshold Tuning Example
    print("=" * 50)
    print("THRESHOLD TUNING COMPARISON")
    print("=" * 50)
    thresholds = [0.3, 0.5, 0.7]
    for th in thresholds:
        y_custom_pred = (y_probs >= th).astype(int)
        p = precision_score(y_test, y_custom_pred)
        r = recall_score(y_test, y_custom_pred)
        print(f"Threshold: {th} | Precision: {p:.4f} | Recall: {r:.4f}")

if __name__ == "__main__":
    plot_sigmoid()
    train_and_evaluate()
