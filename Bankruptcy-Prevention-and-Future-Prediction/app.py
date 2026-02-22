import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# -----------------------------------------------------------
# 🔧 Streamlit Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="💼 Bankruptcy Risk Analyzer",
    page_icon="📊",
    layout="wide"
)   

# 🌙 Dark UI Styling
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #E0E0E0; }
h1, h2, h3, h4 { color: #00E6A8; }
.stTabs [role="tablist"] button { background-color: #1C1F26; color: #E0E0E0; border-radius: 8px; margin-right: 4px; }
.stTabs [role="tablist"] button[data-baseweb="tab"]:hover { background-color: #00E6A830; color: white; }
.stTabs [role="tablist"] button[data-baseweb="tab"][aria-selected="true"] { background-color: #00E6A8; color: black; }
.stButton>button { background-color: #00E6A8; color: black; border-radius: 6px; font-size: 16px; font-weight: 600; padding: 0.6rem 1.4rem; }
.stButton>button:hover { background-color: #03c997; color: white; }
.metric-card { background-color: #1E222B; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 0 8px rgba(0,0,0,0.4); }
.metric-value { font-size: 26px; font-weight: bold; margin-top: 5px; }
.metric-label { font-size: 14px; color: #A0A0A0; }
.result-banner { border-radius: 20px; padding: 3rem 2rem; text-align: center; margin-top: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 Load Model, Scaler, Metadata
# -----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("models/best_model_knn.pkl", "rb"))
    scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
    metadata = pickle.load(open("models/model_metadata.pkl", "rb"))
    return model, scaler, metadata

try:
    model, scaler, metadata = load_artifacts()
    features = metadata["features"]
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# -----------------------------------------------------------
# 🔍 Prediction Helper
# -----------------------------------------------------------
def predict_with_model(df_input):
    scaled = scaler.transform(df_input)
    preds = model.predict(scaled)
    prob_non_bankruptcy = np.clip(model.predict_proba(scaled)[:, 1], 0.01, 0.99)
    prob_bankruptcy = 1 - prob_non_bankruptcy
    return preds, prob_bankruptcy, prob_non_bankruptcy

# -----------------------------------------------------------
# 🧮 Feature Engineering (internal only)
# -----------------------------------------------------------
def create_features(df):
    df_featured = df.copy()
    try:
        for col in ["industrial_risk", "management_risk", "operating_risk"]:
            if col in df_featured.columns:
                df_featured[col] = 1 - df_featured[col]

        rcols = ["industrial_risk", "management_risk", "operating_risk"]
        fcols = ["financial_flexibility", "credibility", "competitiveness"]
        df_featured["financial_health_score"] = df_featured[fcols].mean(axis=1)
        df_featured["management_impact_score"] = df_featured["management_risk"] / (
            df_featured["financial_flexibility"] + df_featured["credibility"] + 1)
        df_featured["risk_stability_ratio"] = (
            df_featured["financial_flexibility"] + df_featured["credibility"]) / (
            df_featured["management_risk"] + 1)
        df_featured["risk_volatility"] = df_featured[rcols].std(axis=1)
        weights = {"financial_flexibility": 0.4, "credibility": 0.3, "competitiveness": 0.3}
        df_featured["financial_stability"] = sum(df_featured[c] * w for c, w in weights.items())
        df_featured["risk_financial_ratio"] = df_featured[rcols].mean(axis=1) / (
            df_featured[fcols].mean(axis=1) + 1)
        df_featured["management_financial_risk"] = df_featured["management_risk"] / (
            df_featured["financial_flexibility"] + 0.1)
        df_featured["operational_sustainability"] = (
            (df_featured["financial_flexibility"] + df_featured["competitiveness"]) / 2
        ) * (1 - df_featured["operating_risk"])
        df_featured["compound_risk"] = ((df_featured[rcols] > 0.7).sum(axis=1) / len(rcols))
        df_featured["financial_x_management"] = (
            df_featured["financial_health_score"] * df_featured["management_risk"])
        df_featured["risk_x_operational"] = (
            df_featured["risk_volatility"] * df_featured["operating_risk"])
    except Exception:
        pass
    return df_featured

# -----------------------------------------------------------
# 🎯 Risk Dashboard
# -----------------------------------------------------------
def display_risk_dashboard(probability):
    if probability >= 0.7:
        level, color, emoji, bg = "HIGH RISK", "#FF4B4B", "🔴", "#8B2E2E"
    elif probability >= 0.4:
        level, color, emoji, bg = "MEDIUM RISK", "#FFA500", "🟠", "#7A4C0F"
    else:
        level, color, emoji, bg = "LOW RISK", "#00E6A8", "🟢", "#145B42"

    st.markdown(f"""
    <div class="result-banner" style="background:linear-gradient(180deg,{bg},#1E1E1E);">
        <h2 style="font-size:2rem; color:white;">📊 Analysis Results</h2>
        <h1 style="color:{color}; font-size:2.5rem;">{emoji} {level}</h1>
        <p style="color:#FFFFFF; font-size:1.3rem;">{probability*100:.1f}% Bankruptcy Probability</p>
        <p style="color:#BBBBBB; font-size:1rem;">{100 - probability*100:.1f}% Success Probability</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧭 Sidebar
# -----------------------------------------------------------
st.sidebar.title("📘 Model Details")
st.sidebar.success(f"Model: {metadata.get('model_name','KNN')}")
st.sidebar.write(f"Type: {metadata.get('model_type','Classifier')}")
st.sidebar.write(f"Trained on: {metadata.get('training_date','Unknown')}")
st.sidebar.markdown("### Performance Metrics")
for k, v in metadata.get("performance", {}).items():
    st.sidebar.metric(label=k.capitalize(), value=round(v, 4))
st.sidebar.markdown("---")
st.sidebar.caption("💡 Use tabs below for manual or batch prediction.")

# -----------------------------------------------------------
# 🚀 Main Interface
# -----------------------------------------------------------
st.title("💼 Bankruptcy Risk Analyzer")
st.markdown("Predict company **bankruptcy probability** using your trained KNN model. Input manually or upload datasets for bulk analysis.")

tab1, tab2 = st.tabs(["Manual Entry", "Upload Dataset"])

# -----------------------------------------------------------
# 🧾 Manual Entry
# -----------------------------------------------------------
with tab1:
    st.subheader("Enter Company Risk Levels")
    options = {"0.0 - Low": 0.0, "0.5 - Medium": 0.5, "1.0 - High": 1.0}

    base_features = [
        "industrial_risk", "management_risk", "financial_flexibility",
        "credibility", "competitiveness", "operating_risk"
    ]

    cols = st.columns(3)
    inputs = {}

    for i, feature in enumerate(base_features):
        with cols[i % 3]:
            label = feature.replace("_", " ").capitalize()
            choice = st.selectbox(label, list(options.keys()), index=1)
            inputs[feature] = options[choice]

    if st.button("🔍 Analyze Bankruptcy Risk", use_container_width=True):
        df = pd.DataFrame([inputs])
        df = create_features(df)
        df_input = df.reindex(columns=features, fill_value=0)
        preds, prob_bankruptcy, prob_non_bankruptcy = predict_with_model(df_input)
        display_risk_dashboard(prob_bankruptcy[0])

# -----------------------------------------------------------
# 📂 Batch Upload Prediction
# -----------------------------------------------------------
with tab2:
    st.subheader("Upload Dataset for Batch Prediction")
    file = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        try:
            df_raw = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
            st.success(f"✅ Loaded {df_raw.shape[0]} records.")
            st.dataframe(df_raw.head())

            if not all(f in df_raw.columns for f in features):
                st.info("🧩 Creating additional features for compatibility...")
                df_raw = create_features(df_raw)

            df_input = df_raw.reindex(columns=features, fill_value=0)

            if st.button("⚙️ Run Model Prediction", use_container_width=True):
                with st.spinner("Predicting bankruptcy risk using trained model..."):
                    preds, prob_bankruptcy, prob_non_bankruptcy = predict_with_model(df_input)

                    results = pd.DataFrame({
                        "Prediction": np.where(preds == 1, "Non-Bankruptcy", "Bankruptcy"),
                        "Bankruptcy Probability (%)": np.round(prob_bankruptcy * 100, 2),
                        "Success Probability (%)": np.round(prob_non_bankruptcy * 100, 2)
                    })

                    # ✅ Show only original base columns + prediction results
                    base_features = [
                        "industrial_risk", "management_risk", "financial_flexibility",
                        "credibility", "competitiveness", "operating_risk"
                    ]
                    df_display = df_raw[base_features].copy() if all(f in df_raw.columns for f in base_features) else df_raw.copy()
                    output = pd.concat([df_display, results], axis=1)

                    st.success("✅ Prediction completed successfully!")
                    st.dataframe(output.head(10))

                    avg_prob = np.mean(prob_bankruptcy)
                    display_risk_dashboard(avg_prob)

                    csv = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Predictions (CSV)",
                        csv,
                        file_name=f"bankruptcy_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
