import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
 
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
 
# ── Load data ─────────────────────────────────────────────
FEATURE_NAMES = pickle.load(open("models/feature_names.pkl", "rb"))
class_weight  = pickle.load(open("models/class_weight.pkl", "rb"))
 
X_train = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\X_train.csv").values
X_val   = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\X_val.csv").values
X_test  = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\X_test.csv").values
y_train = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\y_train.csv").values.ravel()
y_val   = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\y_val.csv").values.ravel()
y_test  = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\data\processed\y_test.csv").values.ravel()
 
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Class weight: {class_weight}")
 
# ── Model ─────────────────────────────────────────────────
def build_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
 
    x = tf.keras.layers.Dense(128)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
 
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
 
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
 
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)
 
model = build_model(X_train.shape[1])
model.summary()
 
# ── Compile ───────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc'),
    ]
)
 
# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_recall', mode='max',
        patience=8, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras", monitor='val_recall',
        mode='max', save_best_only=True, verbose=1
    ),
]
 
# ── Train ─────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)
 
# ── Training curves ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric, title in zip(axes,
    ['loss', 'recall', 'auc'],
    ['Loss', 'Recall', 'AUC']):
    ax.plot(history.history[metric],       label='train')
    ax.plot(history.history[f'val_{metric}'], label='val')
    ax.set_title(title); ax.legend(); ax.set_xlabel('Epoch')
plt.suptitle('Training History', fontsize=13)
plt.tight_layout()
plt.savefig('plots/07_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
 
# ── Threshold tuning (recall-optimized) ──────────────────
y_prob = model.predict(X_val).ravel()
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
 
# Find threshold where recall >= 0.90 with best precision
target_recall = 0.90
valid = [(t, p, r) for p, r, t in zip(precisions, recalls, thresholds)
         if r >= target_recall]
 
if valid:
    best_t, best_p, best_r = max(valid, key=lambda x: x[1])
else:
    # fallback: closest to target recall
    idx = np.argmin(np.abs(recalls[:-1] - target_recall))
    best_t = thresholds[idx]
    best_p, best_r = precisions[idx], recalls[idx]
 
print(f"\nThreshold tuning (target recall >= {target_recall}):")
print(f"  Threshold : {best_t:.4f}")
print(f"  Recall    : {best_r:.4f}")
print(f"  Precision : {best_p:.4f}")
 
# PR curve plot
plt.figure(figsize=(8, 5))
plt.plot(recalls[:-1], precisions[:-1], lw=2)
plt.axvline(best_r, color='red', linestyle='--', label=f'threshold={best_t:.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation)')
plt.legend(); plt.tight_layout()
plt.savefig('plots/08_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()
 
# ── Final evaluation on test set ──────────────────────────
y_pred_prob = model.predict(X_test).ravel()
y_pred      = (y_pred_prob >= best_t).astype(int)
 
print("\n── Test Set Evaluation ──────────────────────────────")
print(classification_report(y_test, y_pred,
                             target_names=['stable', 'unstable']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['stable','unstable'])
ax.set_yticklabels(['stable','unstable'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Test Set')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('plots/09_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
 
# ── Save model + threshold ────────────────────────────────
model.save("models/ann_model.keras")
pickle.dump(float(best_t), open("models/threshold.pkl", "wb"))
 
print(f"\nSaved: models/ann_model.keras")
print(f"Saved: models/threshold.pkl  (threshold={best_t:.4f})")
print("Model training done ✅")