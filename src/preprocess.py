import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
 
# ── Load ──────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\shiva\OneDrive\Desktop\projects 2.0\ANN smart grid\smart-grid-fault-ANN\dataset\smart_grid_stability_augmented.csv")
 
# Drop regression target if present (leakage risk)
if 'stab' in df.columns:
    df.drop(columns=['stab'], inplace=True)
 
# ── Encode ────────────────────────────────────────────────
df['stabf'] = df['stabf'].map({'stable': 0, 'unstable': 1})
 
X = df.drop('stabf', axis=1)
y = df['stabf']
 
FEATURE_NAMES = X.columns.tolist()
 
# ── Class weight (recall-biased, for model.fit) ───────────
n_stable   = (y == 0).sum()
n_unstable = (y == 1).sum()
class_weight = {0: 1.0, 1: round(n_stable / n_unstable, 4)}
print(f"Class weight: {class_weight}")
 
# ── Split: 70 / 15 / 15 ──────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
 
# ── Scale ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
 
# ── Save ──────────────────────────────────────────────────
pickle.dump(scaler,       open("models/scaler.pkl", "wb"))
pickle.dump(class_weight, open("models/class_weight.pkl", "wb"))
pickle.dump(FEATURE_NAMES, open("models/feature_names.pkl", "wb"))
 
pd.DataFrame(X_train, columns=FEATURE_NAMES).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_val,   columns=FEATURE_NAMES).to_csv("data/processed/X_val.csv",   index=False)
pd.DataFrame(X_test,  columns=FEATURE_NAMES).to_csv("data/processed/X_test.csv",  index=False)
 
y_train.reset_index(drop=True).to_csv("data/processed/y_train.csv", index=False)
y_val.reset_index(drop=True).to_csv("data/processed/y_val.csv",     index=False)
y_test.reset_index(drop=True).to_csv("data/processed/y_test.csv",   index=False)
 
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Features: {len(FEATURE_NAMES)}")
print("Preprocessing done ✅")