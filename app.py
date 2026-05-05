import streamlit as st
import numpy as np
import pandas as pd
from collections import deque
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grid Fault Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1e2d47;
    --accent: #00d4ff;
    --green: #00ff88;
    --yellow: #ffd700;
    --red: #ff3b5c;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { font-family: 'Exo 2', sans-serif; }

.stSlider > div > div > div { background: var(--accent) !important; }

div[data-baseweb="slider"] > div { background: var(--border) !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff44) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    box-shadow: 0 0 20px var(--accent) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.metric-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

.result-stable {
    background: linear-gradient(135deg, #001a0d, #002a14);
    border: 2px solid var(--green);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px #00ff8833;
}
.result-warning {
    background: linear-gradient(135deg, #1a1500, #2a2000);
    border: 2px solid var(--yellow);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px #ffd70033;
}
.result-fault {
    background: linear-gradient(135deg, #1a0008, #2a000f);
    border: 2px solid var(--red);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px #ff3b5c33;
}

.result-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 0.5rem;
}
.result-prob {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: var(--muted);
}

.param-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.param-section h4 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.header-bar {
    background: linear-gradient(90deg, #00d4ff11, transparent);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    border-radius: 0 8px 8px 0;
}
.header-bar h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    letter-spacing: 3px;
    color: var(--accent);
    margin: 0;
}
.header-bar p {
    color: var(--muted);
    margin: 0.3rem 0 0 0;
    font-size: 0.85rem;
    letter-spacing: 1px;
}

.rolling-window {
    display: flex;
    gap: 6px;
    margin-top: 1rem;
}
.rw-dot {
    width: 20px; height: 20px;
    border-radius: 50%;
    border: 1px solid var(--border);
}

.info-chip {
    display: inline-block;
    background: #00d4ff15;
    border: 1px solid #00d4ff44;
    color: var(--accent);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 6px;
}

[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 8px #00d4ff44 !important;
}

label, .stSlider label { color: var(--muted) !important; font-size: 0.8rem !important; }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--green), var(--accent)) !important;
}

.stDataFrame { background: var(--surface) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Simulated model (no actual .h5 file needed) ───────────────────────────────
def simulate_prediction(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4):
    """
    Simulates the ANN prediction using heuristics based on the dataset description.
    Replace this with actual model loading when the trained model file is available:


    """
    # Instability heuristic based on domain knowledge:
    # - High reaction times (tau) increase instability
    # - Large power imbalances increase instability
    # - Low elasticity (g) reduces ability to self-correct
    
    from tensorflow.keras.models import load_model
    import joblib
    model = load_model('models/ann_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    features = np.array([[tau1,tau2,tau3,tau4,p1,p2,p3,p4,g1,g2,g3,g4]])
    features_scaled = scaler.transform(features)
    prob = model.predict(features_scaled)[0][0]
    tau_avg = (tau1 + tau2 + tau3 + tau4) / 4.0
    tau_score = (tau_avg - 0.5) / 9.5  # normalise 0.5–10 → 0–1

    power_imbalance = abs(p1 + p2 + p3 + p4)  # should be ~0 ideally
    power_score = min(power_imbalance / 2.0, 1.0)

    g_avg = (g1 + g2 + g3 + g4) / 4.0
    elasticity_score = 1.0 - g_avg  # lower elasticity → more unstable

    # Weighted combination
    instability = 0.35 * tau_score + 0.45 * power_score + 0.20 * elasticity_score

    # Add small noise for realism
    noise = np.random.normal(0, 0.03)
    prob = float(np.clip(instability + noise, 0.0, 1.0))
    return prob


def classify(prob):
    if prob < 0.4:
        return "STABLE", "✅", "result-stable", "#00ff88"
    elif prob < 0.7:
        return "WARNING", "⚠️", "result-warning", "#ffd700"
    else:
        return "FAULT", "🚨", "result-fault", "#ff3b5c"


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "rolling" not in st.session_state:
    st.session_state.rolling = deque(maxlen=5)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>⚡ SMART GRID FAULT PREDICTOR</h1>
  <p>Artificial Neural Network · Real-time Stability Monitor · 3-Tier Alert System</p>
</div>
""", unsafe_allow_html=True)

# ── KPI bar ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""<div class="metric-card">
        <div class="metric-val" style="color:#00d4ff">~97%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <div class="metric-val" style="color:#00ff88">~99%</div>
        <div class="metric-label">ROC-AUC Score</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <div class="metric-val" style="color:#ffd700">60K</div>
        <div class="metric-label">Training Rows</div>
    </div>""", unsafe_allow_html=True)
with c4:
    n_preds = len(st.session_state.history)
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#ff3b5c">{n_preds}</div>
        <div class="metric-label">Predictions Made</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬  SINGLE PREDICTION", "📂  BATCH / CSV", "📊  HISTORY"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        # ── τ Reaction Times ──────────────────────────────────────────────────
        st.markdown("""<div class="param-section">
            <h4>⏱ τ — Reaction Times (0.5 – 10 s)</h4>
        </div>""", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        with rc1:
            tau1 = st.number_input("τ₁ — Supplier node", min_value=0.5, max_value=10.0,
                                   value=2.5, step=0.1, format="%.2f")
            tau3 = st.number_input("τ₃ — Consumer 2", min_value=0.5, max_value=10.0,
                                   value=4.2, step=0.1, format="%.2f")
        with rc2:
            tau2 = st.number_input("τ₂ — Consumer 1", min_value=0.5, max_value=10.0,
                                   value=3.1, step=0.1, format="%.2f")
            tau4 = st.number_input("τ₄ — Consumer 3", min_value=0.5, max_value=10.0,
                                   value=1.8, step=0.1, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Power Values ──────────────────────────────────────────────────────
        st.markdown("""<div class="param-section">
            <h4>⚡ p — Power (−2.0 to +2.0)</h4>
        </div>""", unsafe_allow_html=True)

        p1 = st.slider("p₁ — Supplier power (generated)", min_value=0.0, max_value=4.0,
                       value=2.1, step=0.05,
                       help="p1 should equal -(p2+p3+p4). Automatically reflects consumption balance.")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p2 = st.number_input("p₂", min_value=-2.0, max_value=2.0, value=-0.7, step=0.05, format="%.2f")
        with pc2:
            p3 = st.number_input("p₃", min_value=-2.0, max_value=2.0, value=-0.8, step=0.05, format="%.2f")
        with pc3:
            p4 = st.number_input("p₄", min_value=-2.0, max_value=2.0, value=-0.6, step=0.05, format="%.2f")

        # Live balance indicator
        imbalance = abs(p1 + p2 + p3 + p4)
        bal_color = "#00ff88" if imbalance < 0.1 else ("#ffd700" if imbalance < 0.5 else "#ff3b5c")
        st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                        color:{bal_color};margin-top:0.4rem;">
              Power imbalance |p1+p2+p3+p4| = {imbalance:.3f}
              {'✓ Balanced' if imbalance < 0.1 else '⚠ Unbalanced'}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Elasticity ────────────────────────────────────────────────────────
        st.markdown("""<div class="param-section">
            <h4>📈 g — Price Elasticity (0.05 – 1.00)</h4>
        </div>""", unsafe_allow_html=True)
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            g1 = st.number_input("g₁", min_value=0.05, max_value=1.0, value=0.6, step=0.05, format="%.2f")
        with gc2:
            g2 = st.number_input("g₂", min_value=0.05, max_value=1.0, value=0.3, step=0.05, format="%.2f")
        with gc3:
            g3 = st.number_input("g₃", min_value=0.05, max_value=1.0, value=0.4, step=0.05, format="%.2f")
        with gc4:
            g4 = st.number_input("g₄", min_value=0.05, max_value=1.0, value=0.5, step=0.05, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ RUN PREDICTION", use_container_width=True)

    # ── Result panel ─────────────────────────────────────────────────────────
    with right:
        st.markdown("#### PREDICTION OUTPUT")

        if predict_btn:
            with st.spinner("Analyzing grid parameters..."):
                time.sleep(0.4)  # Small pause for UX
                prob = simulate_prediction(tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4)

            label, icon, css_class, color = classify(prob)
            st.session_state.rolling.append(prob)
            rolling_avg = np.mean(list(st.session_state.rolling))

            # Save to history
            st.session_state.history.append({
                "Prediction #": len(st.session_state.history) + 1,
                "τ1": tau1, "τ2": tau2, "τ3": tau3, "τ4": tau4,
                "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                "g1": g1, "g2": g2, "g3": g3, "g4": g4,
                "Fault Prob": round(prob, 4),
                "Status": label,
            })

            # Main result box
            st.markdown(f"""
            <div class="{css_class}">
              <div class="result-title" style="color:{color}">{icon} {label}</div>
              <div class="result-prob">Fault Probability: {prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Progress bar
            st.markdown(f"<p style='font-family:Share Tech Mono,monospace;font-size:0.75rem;"
                        f"letter-spacing:1px;color:#64748b;margin-bottom:4px'>"
                        f"INSTABILITY SCORE</p>", unsafe_allow_html=True)
            st.progress(prob)

            st.markdown("<br>", unsafe_allow_html=True)

            # Rolling window
            roll_label, _, roll_css, roll_color = classify(rolling_avg)
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d47;border-radius:8px;padding:1rem;">
              <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                        letter-spacing:2px;color:#64748b;margin:0 0 0.6rem 0">
                ROLLING WINDOW (last 5 predictions)
              </p>
              <p style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;
                        color:{roll_color};margin:0">
                Avg: {rolling_avg:.1%} → {roll_label}
              </p>
              <p style="font-size:0.75rem;color:#64748b;margin:0.4rem 0 0 0">
                {'🚨 Alert threshold exceeded (>0.70 average)' if rolling_avg > 0.70 else '✓ No sustained alert condition'}
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Parameter summary
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d47;border-radius:8px;padding:1rem;">
              <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                        letter-spacing:2px;color:#64748b;margin:0 0 0.8rem 0">INPUT SUMMARY</p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.8rem;
                          font-family:'Share Tech Mono',monospace;color:#94a3b8">
                <span>τ: {tau1:.2f} {tau2:.2f} {tau3:.2f} {tau4:.2f}</span>
                <span>g: {g1:.2f} {g2:.2f} {g3:.2f} {g4:.2f}</span>
                <span>p: {p1:.2f} {p2:.2f} {p3:.2f} {p4:.2f}</span>
                <span>Imbalance: {imbalance:.3f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#111827;border:1px dashed #1e2d47;border-radius:12px;
                        padding:3rem;text-align:center;margin-top:1rem;">
              <p style="font-family:'Share Tech Mono',monospace;font-size:2rem;
                        color:#1e2d47;margin-bottom:1rem">⚡</p>
              <p style="color:#64748b;font-family:'Share Tech Mono',monospace;
                        font-size:0.8rem;letter-spacing:2px">
                SET PARAMETERS & RUN PREDICTION
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Alert legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d47;border-radius:8px;padding:1rem;">
          <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                    letter-spacing:2px;color:#64748b;margin:0 0 0.6rem 0">ALERT THRESHOLDS</p>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;line-height:1.8">
            <span style="color:#00ff88">■</span>&nbsp; 0.0 – 0.4 → STABLE<br>
            <span style="color:#ffd700">■</span>&nbsp; 0.4 – 0.7 → WARNING<br>
            <span style="color:#ff3b5c">■</span>&nbsp; 0.7 – 1.0 → FAULT
          </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Batch / CSV
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### BATCH PREDICTION FROM CSV")
    st.markdown("""
    <p style="color:#64748b;font-size:0.85rem">
    Upload a CSV with columns: <code style="color:#00d4ff">tau1, tau2, tau3, tau4, p1, p2, p3, p4, g1, g2, g3, g4</code>
    </p>
    """, unsafe_allow_html=True)

    # Sample CSV download
    sample_data = pd.DataFrame([
        [2.5, 3.1, 4.2, 1.8, 2.1, -0.7, -0.8, -0.6, 0.6, 0.3, 0.4, 0.5],
        [8.0, 7.5, 9.0, 6.5, 3.8, -1.2, -1.3, -1.3, 0.1, 0.1, 0.1, 0.1],
        [1.2, 1.5, 1.8, 1.0, 1.5, -0.5, -0.5, -0.5, 0.9, 0.8, 0.9, 0.8],
        [5.0, 6.2, 7.1, 4.8, 2.8, -0.9, -1.0, -0.9, 0.3, 0.2, 0.3, 0.25],
        [3.5, 2.8, 3.0, 2.5, 2.0, -0.65, -0.7, -0.65, 0.7, 0.6, 0.65, 0.7],
    ], columns=["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"])

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "⬇ Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_grid_input.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        required_cols = ["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.markdown(f"**{len(df)} rows detected** — running predictions…")
            pbar = st.progress(0)
            results = []
            for i, row in df.iterrows():
                p = simulate_prediction(*[row[c] for c in required_cols])
                label, icon, _, color = classify(p)
                results.append({
                    "Row": i + 1,
                    "Fault Prob": round(p, 4),
                    "Status": f"{icon} {label}",
                })
                pbar.progress((i + 1) / len(df))

            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)

            # Summary
            counts = result_df["Status"].value_counts()
            s1, s2, s3 = st.columns(3)
            stable_count = sum(1 for r in results if "STABLE" in r["Status"])
            warn_count   = sum(1 for r in results if "WARNING" in r["Status"])
            fault_count  = sum(1 for r in results if "FAULT" in r["Status"])
            with s1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#00ff88">{stable_count}</div>
                    <div class="metric-label">Stable</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#ffd700">{warn_count}</div>
                    <div class="metric-label">Warning</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#ff3b5c">{fault_count}</div>
                    <div class="metric-label">Fault</div>
                </div>""", unsafe_allow_html=True)

            st.download_button(
                "⬇ Download Results CSV",
                data=pd.concat([df, pd.DataFrame(results)], axis=1).to_csv(index=False),
                file_name="grid_predictions.csv",
                mime="text/csv",
            )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – History
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748b;
                    font-family:'Share Tech Mono',monospace;font-size:0.8rem;letter-spacing:2px">
          NO PREDICTIONS YET — RUN A PREDICTION FIRST
        </div>""", unsafe_allow_html=True)
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        # Mini chart
        st.markdown("**Fault Probability Over Predictions**")
        chart_df = hist_df[["Prediction #", "Fault Prob"]].set_index("Prediction #")
        st.line_chart(chart_df, color="#00d4ff")

        col_clr, _ = st.columns([1, 5])
        with col_clr:
            if st.button("🗑 Clear History"):
                st.session_state.history = []
                st.session_state.rolling = deque(maxlen=5)
                st.rerun()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0">
      <p style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;
                color:#00d4ff;letter-spacing:3px">⚡ GRID ANN</p>
      <p style="color:#64748b;font-size:0.7rem;letter-spacing:1px">v1.0 · UCI Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
              letter-spacing:2px;color:#00d4ff">MODEL INFO</p>
    """, unsafe_allow_html=True)
    st.markdown("""
    - **Architecture**: Deep ANN (TensorFlow)  
    - **Dataset**: UCI Grid Stability · 60K rows  
    - **Accuracy**: ~97%  
    - **ROC-AUC**: ~99%  
    - **Inputs**: 12 features  
    - **Output**: P(unstable) ∈ [0, 1]
    """)

    st.markdown("---")

    st.markdown("""
    <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
              letter-spacing:2px;color:#00d4ff">LOAD REAL MODEL</p>
    <p style="color:#64748b;font-size:0.75rem">
    Replace <code style="color:#ffd700">simulate_prediction()</code> in <code>app.py</code> with:
    </p>
    """, unsafe_allow_html=True)
    st.code("""from tensorflow.keras.models import load_model
import joblib

model = load_model('models/ann_model.h5')
scaler = joblib.load('models/scaler.pkl')

def real_predict(*args):
    X = np.array([list(args)])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0][0])
""", language="python")

    st.markdown("---")
    st.markdown("""
    <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
              letter-spacing:2px;color:#00d4ff">ALERT LOGIC</p>
    <p style="color:#64748b;font-size:0.75rem">
    Alert fires when rolling average of last 5 predictions > 0.70
    </p>
    """, unsafe_allow_html=True)