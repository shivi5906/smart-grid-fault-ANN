import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
 
# ── Load ──────────────────────────────────────────────────
model     = tf.keras.models.load_model("models/ann_model.keras")
threshold = pickle.load(open("models/threshold.pkl", "rb"))
scaler    = pickle.load(open("models/scaler.pkl", "rb"))
 
X_test = pd.read_csv("data/processed/X_test.csv").values
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
 
print(f"Test samples : {len(X_test)}")
print(f"Threshold    : {threshold:.4f}  (tuned for recall ≥ 0.90)")
 
# ── Predict ───────────────────────────────────────────────
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= threshold).astype(int)
 
# ── Console report ────────────────────────────────────────
print("\n── Classification Report ────────────────────────────")
print(classification_report(y_test, y_pred,
                             target_names=['stable', 'unstable']))
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Avg Prec : {average_precision_score(y_test, y_prob):.4f}")
 
# ── Confusion matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Missed faults (FN): {fn}  ← keep this low")
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
 
# 1. Confusion matrix
ax = axes[0]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['stable','unstable'])
ax.set_yticklabels(['stable','unstable'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black',
                fontsize=13, fontweight='bold')
 
# 2. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score   = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.4f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
 
# 3. Precision-Recall curve
prec, rec, thresholds_pr = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec[:-1], prec[:-1], lw=2)
axes[2].axvline(x=y_pred[y_test==1].mean(), color='red',
                linestyle='--', label=f'threshold={threshold:.2f}')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve')
axes[2].legend()
 
plt.suptitle('Model Evaluation — Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/10_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
 
# ── Threshold sensitivity table ───────────────────────────
print("\n── Threshold Sensitivity ────────────────────────────")
print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FN':>6}")
for t in np.arange(0.20, 0.55, 0.05):
    yp = (y_prob >= t).astype(int)
    from sklearn.metrics import recall_score, precision_score, f1_score
    r  = recall_score(y_test, yp, zero_division=0)
    p  = precision_score(y_test, yp, zero_division=0)
    f1 = f1_score(y_test, yp, zero_division=0)
    fn = int(((y_test==1) & (yp==0)).sum())
    marker = ' ◄' if abs(t - threshold) < 0.026 else ''
    print(f"{t:>10.2f} {r:>8.4f} {p:>10.4f} {f1:>8.4f} {fn:>6}{marker}")
 
print("\nEvaluation done ✅")