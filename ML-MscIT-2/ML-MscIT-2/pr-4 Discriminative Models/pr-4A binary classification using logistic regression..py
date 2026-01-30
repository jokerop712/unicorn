from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    auc
)
import matplotlib.pyplot as plt


# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    random_state=1
)


# Train-test split
trainX, testX, trainy, testy = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=2
)


# No-skill predictions
ns_probs = [0 for _ in range(len(testy))]


# Logistic Regression model
model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
model.fit(trainX, trainy)


# Predict probabilities
lr_probs = model.predict_proba(testX)[:, 1]


# ROC AUC
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)

print(f"No Skill: ROC AUC={ns_auc:.3f}")
print(f"Logistic: ROC AUC={lr_auc:.3f}")


# ROC Curve
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Precision-Recall Curve
yhat = model.predict(testX)

lr_precision, lr_recall, _ = precision_recall_curve(
    testy,
    lr_probs
)

lr_f1 = f1_score(testy, yhat)
lr_pr_auc = auc(lr_recall, lr_precision)

print(f"Logistic: F1={lr_f1:.3f} PR-AUC={lr_pr_auc:.3f}")


# Plot Precision-Recall Curve
no_skill = sum(testy) / len(testy)

plt.plot(
    [0, 1],
    [no_skill, no_skill],
    linestyle='--',
    label='No Skill'
)

plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
