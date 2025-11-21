# Code to simulate a dataset similar to the paper and compare a knowledge-based Bayesian-style model
# with data-driven models.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    roc_curve, auc, brier_score_loss, roc_auc_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------ 1) Simulate dataset ------------------
N = 2000  
prevalence = 217/363  
base_logit = np.log(prevalence / (1 - prevalence))

features = ["CCP_pos", "RhF_pos", "Stiffness_30m", "Symmetric_swelling",
            "ESR_high", "Skin_psoriasis", "Smoking", "Female", "Age_over_50"]

p_feature_given_RA = {
    "CCP_pos": 0.65, "RhF_pos": 0.55, "Stiffness_30m": 0.60,
    "Symmetric_swelling": 0.55, "ESR_high": 0.50, "Skin_psoriasis": 0.05,
    "Smoking": 0.35, "Female": 0.70, "Age_over_50": 0.4
}

p_feature_given_Other = {
    "CCP_pos": 0.08, "RhF_pos": 0.12, "Stiffness_30m": 0.30,
    "Symmetric_swelling": 0.25, "ESR_high": 0.20, "Skin_psoriasis": 0.25,
    "Smoking": 0.30, "Female": 0.60, "Age_over_50": 0.45
}

y = np.random.binomial(1, prevalence, size=N)
X = np.zeros((N, len(features)), dtype=int)

for i, fname in enumerate(features):
    X[:, i] = np.where(
        y == 1,
        np.random.binomial(1, p_feature_given_RA[fname], size=N),
        np.random.binomial(1, p_feature_given_Other[fname], size=N)
    )

df = pd.DataFrame(X, columns=features)
df['RA'] = y

print("\nSynthetic Dataset Preview:")
print(df.head())

# ------------------ 2) Knowledge-based scoring model ------------------
kb_weights = {
    "intercept": base_logit - 0.2,
    "CCP_pos": 2.0, "RhF_pos": 1.3, "Stiffness_30m": 0.9,
    "Symmetric_swelling": 0.8, "ESR_high": 0.6,
    "Skin_psoriasis": -1.2, "Smoking": 0.2,
    "Female": 0.15, "Age_over_50": -0.1
}

def kb_predict_proba(df_in):
    logit = np.full(df_in.shape[0], kb_weights["intercept"])
    for f, w in kb_weights.items():
        if f != "intercept":
            logit += df_in[f] * w
    return 1 / (1 + np.exp(-logit))

df['KB_prob'] = kb_predict_proba(df)
threshold = prevalence
df['KB_pred'] = (df['KB_prob'] >= threshold).astype(int)

# ------------------ 3) Data-driven models ------------------
X_mat = df[features].values
y_vec = df['RA'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_mat, y_vec, test_size=0.3, stratify=y_vec, random_state=1
)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_pred = (lr_prob >= threshold).astype(int)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
bnb_prob = bnb.predict_proba(X_test)[:, 1]
bnb_pred = (bnb_prob >= threshold).astype(int)

# ------------------ 4) Evaluation ------------------
def evaluate(y_true, prob, pred, name):
    auc_score = roc_auc_score(y_true, prob)
    brier = brier_score_loss(y_true, prob)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    cm = confusion_matrix(y_true, pred)

    print(f"\n====== {name} ======")
    print("Accuracy:", acc)
    print("AUROC:", auc_score)
    print("Brier Score:", brier)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("Confusion Matrix:\n", cm)

    return auc_score, brier, acc

print("\nModel comparison:")
kb_scores = evaluate(y_test, kb_predict_proba(pd.DataFrame(X_test, columns=features)),
                     (kb_predict_proba(pd.DataFrame(X_test, columns=features)) >= threshold).astype(int),
                     "Knowledge-Based Model")

lr_scores = evaluate(y_test, lr_prob, lr_pred, "Logistic Regression (Data-driven)")

bnb_scores = evaluate(y_test, bnb_prob, bnb_pred, "BernoulliNB (Data-driven)")

# ------------------ 5) ROC plots ------------------
def plot_roc(y_true, prob, title):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_val = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

plot_roc(y_test, kb_predict_proba(pd.DataFrame(X_test, columns=features)),
         "ROC - Knowledge-Based Model")

plot_roc(y_test, lr_prob, "ROC - Logistic Regression")

plot_roc(y_test, bnb_prob, "ROC - BernoulliNB")

print("\nDone.")
