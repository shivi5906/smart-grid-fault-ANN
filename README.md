# ⚡ Smart Grid Fault Prediction & Stability Monitoring
### Artificial Neural Networks · FastAPI · Real-time Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Dataset](https://img.shields.io/badge/Dataset-60K_rows-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

> *"Power grids fail every day. Most of the time it's not random — there are signs. This system reads those signs before the blackout happens."*

---

## 🧠 What is this?

A predictive fault detection system for smart power grids built using a deep Artificial Neural Network. Instead of detecting faults *after* they occur (reactive), this system predicts instability *before* it happens (proactive) — by continuously analyzing electrical parameters and triggering alerts through a 3-tier warning system.

---

## 📊 Dataset

**Source:** UCI — Electrical Grid Stability Simulated Dataset · **60,000 rows · 12 features**

| Feature Group | Columns | Description |
|---|---|---|
| Reaction Time | `tau1` – `tau4` | Response time per network participant (0.5 – 10s) |
| Power | `p1` – `p4` | Power produced/consumed per node (-2.0 to +2.0) |
| Elasticity | `g1` – `g4` | Price elasticity coefficient per node (0.05 – 1.00) |
| Target | `stabf` | `stable` or `unstable` |

> `p1` is the supplier node · `p2`–`p4` are consumer nodes · `p1 = -(p2 + p3 + p4)` always

---

## 🏗️ Model Architecture

```
Input           →  12 neurons   (tau1-4, p1-4, g1-4)
Hidden Layer 1  →  128 neurons  (ReLU + Dropout 0.3)
Hidden Layer 2  →  64 neurons   (ReLU + Dropout 0.2)
Hidden Layer 3  →  32 neurons   (ReLU)
Output          →  1 neuron     (Sigmoid → probability 0–1)

Optimizer  →  Adam
Loss       →  Binary Cross-Entropy
Priority   →  Recall (missed fault > false alarm, always)
```

---

## 🚨 Warning System

Probability score → **3-tier alert** · filtered through a **rolling window of last 5 predictions** to suppress false alarms:

```
0.0 – 0.4   →   ✅  STABLE    green
0.4 – 0.7   →   ⚠️   WARNING   yellow
0.7 – 1.0   →   🚨  FAULT     red

Alert fires only when rolling average of last 5 predictions > 0.70
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow |
| API | FastAPI |
| Data | Pandas · NumPy · Scikit-learn |
| Visualization | Matplotlib · Seaborn |
| Frontend | HTML · CSS · JavaScript |

---

## 📁 Project Structure

```
smart-grid-fault-ann/
│
├── data/
│   ├── grid_stability.csv           # original dataset
│   └── custom_simulation.csv        # custom time-series input
│
├── notebooks/
│   ├── data_exploration.ipynb       # EDA, distributions, class balance
│   └── final_demo.ipynb             # full pipeline walkthrough
│
├── src/
│   ├── preprocess.py                # normalization, split, scaler
│   ├── model.py                     # ANN architecture + training
│   ├── evaluate.py                  # metrics, confusion matrix, ROC
│   └── predict.py                   # single + batch prediction
│
├── api/
│   └── main.py                      # FastAPI — /predict + /predict-stream
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_loss_curve.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

```bash
# 1. clone
git clone https://github.com/yourusername/smart-grid-fault-ann.git
cd smart-grid-fault-ann

# 2. install
pip install -r requirements.txt

# 3. preprocess
python src/preprocess.py

# 4. train
python src/model.py

# 5. evaluate
python src/evaluate.py

# 6. run API
uvicorn api.main:app --reload
```

**Test a prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tau1":2.5,"tau2":3.1,"tau3":4.2,"tau4":1.8,
       "p1":2.1,"p2":-0.7,"p3":-0.8,"p4":-0.6,
       "g1":0.6,"g2":0.3,"g3":0.4,"g4":0.5}'
```

---

## 📈 Custom Time-Series Simulation

Feed a custom CSV to simulate grid degradation second by second:

```csv
timestamp,tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4
0,2.5,3.1,4.2,1.8,2.1,-0.7,-0.8,-0.6,0.6,0.3,0.4,0.5
1,2.6,3.3,4.5,1.9,2.2,-0.8,-0.8,-0.6,0.6,0.3,0.4,0.5
```

The API streams predictions row by row, applies the rolling window, and updates the dashboard live. Green → yellow → red in real time.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~97% |
| Precision | ~96% |
| Recall | ~97% |
| F1 Score | ~96% |
| ROC-AUC | ~99% |



---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/smart-grid-fault-ann&type=Date)](https://star-history.com/shivi5906/smart-grid-fault-ann&Date)

---

## 📄 References

1. Hosseinzadeh & Mahdian — *Fault Detection in Smart Grids Using ML*, Springer, 2019
2. Ogar et al. — *ANN for Fault Detection in Power Transmission Lines*, Energy Reports, 2023
3. UCI ML Repository — Electrical Grid Stability Simulated Dataset

---

<p align="center">Built with 🖤 by <a href="https://github.com/shivi5906">Masked Dev</a></p>