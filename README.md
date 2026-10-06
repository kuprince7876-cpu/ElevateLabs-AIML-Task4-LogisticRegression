# ElevateLabs-AIML-Task4-LogisticRegression

# Elevate Labs AI & ML Internship - Task 4: Classification with Logistic Regression

## Overview
This repository implements binary classification using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**. It covers feature standardization, model training, confusion matrix visualization, ROC-AUC evaluation, threshold tuning, and sigmoid curve demonstration.

---

## Solutions to Interview Questions

### 1. How does logistic regression differ from linear regression?
* **Output Range:** Linear regression predicts continuous numeric values ($-\infty \text{ to } +\infty$), whereas Logistic Regression predicts probabilities bounded between $0$ and $1$.
* **Activation Function:** Logistic regression applies the non-linear **Sigmoid Function** to the linear combination of inputs.
* **Loss Function:** Linear regression optimizes **Mean Squared Error (MSE)** using Ordinary Least Squares, while Logistic Regression optimizes **Binary Cross-Entropy Loss (Log Loss)** using maximum likelihood estimation.

### 2. What is the sigmoid function?
The sigmoid function maps any real-valued input $z$ into a value between 0 and 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
In logistic regression, $z = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n$. The output represents the probability that a given sample belongs to the positive class.

### 3. What is precision vs recall?
* **Precision:** Out of all positive predictions, how many were correctly positive?
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity):** Out of all actual positive cases, how many were correctly identified?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **Trade-off:** Increasing the decision threshold improves precision but reduces recall, and vice-versa.

### 4. What is the ROC-AUC curve?
* **ROC Curve:** Plots the **True Positive Rate (TPR / Recall)** against the **False Positive Rate (FPR)** at various probability decision thresholds.
* **AUC (Area Under Curve):** Quantifies overall model performance across all thresholds ($1.0$ is perfect performance, $0.5$ represents random guessing).

### 5. What is the confusion matrix?
A $2 \times 2$ table evaluating classification performance by contrasting actual vs predicted labels:
* **True Positives (TP):** Correctly predicted positive samples.
* **True Negatives (TN):** Correctly predicted negative samples.
* **False Positives (FP):** Incorrectly predicted positive samples (Type I Error).
* **False Negatives (FN):** Incorrectly predicted negative samples (Type II Error).

### 6. What happens if classes are imbalanced?
* The model becomes biased toward predicting the majority class, yielding high accuracy but poor precision/recall for the minority class.
* **Mitigation:** Resampling (SMOTE, undersampling), adjusting class weights (`class_weight='balanced'`), or using metrics like F1-Score, PR-AUC, and ROC-AUC instead of raw accuracy.

### 7. How do you choose the threshold?
* The default threshold is $0.5$.
* **Medical / Critical Applications (High Recall needed):** Lower the threshold (e.g., $0.3$) to minimize False Negatives.
* **Spam Detection (High Precision needed):** Raise the threshold (e.g., $0.7$) to avoid classifying legitimate messages as spam (minimizing False Positives).
* Optimal threshold can be determined using Precision-Recall curves or ROC curves (Youden's J statistic).

### 8. Can logistic regression be used for multi-class problems?
Yes. Logistic regression extends to multi-class problems via:
* **One-vs-Rest (OvR):** Trains $N$ separate binary classifiers (one per class).
* **Multinomial Logistic Regression (Softmax Regression):** Generalizes logistic regression to multi-class outputs natively using the Softmax function.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/ElevateLabs-AIML-Task4-LogisticRegression.git
   cd ElevateLabs-AIML-Task4-LogisticRegression
