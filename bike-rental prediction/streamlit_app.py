import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike · Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System & CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Nunito:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #05050f;
    color: #dde0f0;
}

.stApp {
    background: radial-gradient(ellipse 80% 50% at 20% -10%, #1a1040 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 110%, #0d2240 0%, transparent 60%),
                #05050f;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0820 0%, #060618 100%) !important;
    border-right: 1px solid rgba(120,100,255,0.15) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
section[data-testid="stSidebar"] * { color: #c0bde8 !important; }
section[data-testid="stSidebar"] label {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6e6a9a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 400 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(120,100,255,0.2) !important;
    border-radius: 8px !important;
}

.main .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 1300px;
}

#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(120,100,255,0.18) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
div[data-testid="metric-container"] label {
    color: #6e6a9a !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 26px !important;
    color: #e8e4ff !important;
    font-weight: 500 !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: #6e6a9a !important;
    padding: 10px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    text-transform: uppercase !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
    background: rgba(167,139,250,0.06) !important;
}

hr { border-color: rgba(120,100,255,0.1) !important; }

.stDownloadButton > button, .stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    background: linear-gradient(135deg, #5b3cf5 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(91,60,245,0.4) !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(120,100,255,0.25) !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.5) !important;
    background: rgba(167,139,250,0.04) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(120,100,255,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 1.5rem;
    border-bottom: 1px solid rgba(120,100,255,0.12);
    margin-bottom: 1.8rem;
}
.brand-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #5b3cf5, #38bdf8);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 20px rgba(91,60,245,0.35);
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #e0d9ff 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5b5880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

.pred-card {
    background: linear-gradient(135deg, rgba(91,60,245,0.12) 0%, rgba(56,189,248,0.06) 100%);
    border: 1px solid rgba(120,100,255,0.25);
    border-radius: 20px;
    padding: 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(91,60,245,0.15), transparent 70%);
    border-radius: 50%;
}
.pred-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 88px;
    font-weight: 500;
    line-height: 1;
    background: linear-gradient(135deg, #c4b5fd 0%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pred-unit {
    font-family: 'Nunito', sans-serif;
    font-size: 13px;
    color: #5b5880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}
.pred-badge {
    display: inline-block;
    margin-top: 14px;
    padding: 6px 16px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.badge-vlow  { background: rgba(56,189,248,0.12);  color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }
.badge-mod   { background: rgba(250,204,21,0.12);   color: #fbbf24; border: 1px solid rgba(250,204,21,0.25); }
.badge-high  { background: rgba(249,115,22,0.12);   color: #fb923c; border: 1px solid rgba(249,115,22,0.25); }
.badge-vhigh { background: rgba(239,68,68,0.12);    color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

.insight-chip {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(120,100,255,0.12);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #b0adc8;
    line-height: 1.45;
}
.chip-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5b5880;
    margin-bottom: 10px;
}

.sb-brand {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.sb-mode-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.sb-mode-single { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.sb-mode-bulk   { background: rgba(56,189,248,0.15);  color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }

.bulk-stat {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(120,100,255,0.15);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}
.bulk-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 500;
    color: #c4b5fd;
    line-height: 1;
}
.bulk-stat-lbl {
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5b5880;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

.template-card {
    background: rgba(56,189,248,0.04);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.template-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 14px;
    color: #67e8f9;
    margin-bottom: 4px;
}
.template-desc {
    font-size: 12px;
    color: #5b5880;
    line-height: 1.5;
}

.stAlert { border-radius: 12px !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(120,100,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Artifacts ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path('artifacts')
    if not base.exists():
        return None, None, None
    model  = joblib.load(base / 'model.pkl')
    scaler = joblib.load(base / 'scaler.pkl')
    cols   = joblib.load(base / 'feature_columns.pkl')
    return model, scaler, cols

model, scaler, feature_columns = load_artifacts()


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(hr, weekday, temp, atemp, hum, windspeed,
                   season, yr, mnth, holiday, workingday, weathersit):
    is_weekend = int(weekday in [0, 6])
    if hr <= 6:    time_of_day = 'night'
    elif hr <= 12: time_of_day = 'morning'
    elif hr <= 18: time_of_day = 'afternoon'
    else:          time_of_day = 'evening'

    scaled = scaler.transform([[temp, atemp, hum, windspeed]])[0]
    t, at, h, w = scaled
    temp_humidity = t * h

    raw = {
        'hr': hr, 'weekday': weekday,
        'temp': t, 'atemp': at, 'hum': h, 'windspeed': w,
        'is_weekend': is_weekend, 'temp_humidity': temp_humidity,
        'season': season, 'yr': yr, 'mnth': mnth,
        'holiday': holiday, 'workingday': workingday,
        'weathersit': weathersit, 'time_of_day': time_of_day,
    }
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=[
        'season', 'yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'time_of_day'
    ], drop_first=True)
    return df.reindex(columns=feature_columns, fill_value=0)


def build_features_bulk(row):
    return build_features(
        hr=int(row['hr']), weekday=int(row['weekday']),
        temp=float(row['temp']), atemp=float(row['atemp']),
        hum=float(row['hum']), windspeed=float(row['windspeed']),
        season=str(row['season']), yr=int(row['yr']),
        mnth=int(row['mnth']), holiday=int(row['holiday']),
        workingday=int(row['workingday']), weathersit=int(row['weathersit']),
    )


def demand_tier(count):
    if count < 50:   return ("Very Low",  "badge-vlow",  "◈")
    if count < 150:  return ("Moderate",  "badge-mod",   "◉")
    if count < 300:  return ("High",      "badge-high",  "●")
    return                  ("Very High", "badge-vhigh", "⬤")


def demand_color(count):
    if count < 50:   return "#38bdf8"
    if count < 150:  return "#fbbf24"
    if count < 300:  return "#fb923c"
    return                  "#f87171"


# ── Template CSV ───────────────────────────────────────────────────────────────
TEMPLATE_COLS = ['hr','weekday','temp','atemp','hum','windspeed',
                 'season','yr','mnth','holiday','workingday','weathersit']

def make_template():
    sample = pd.DataFrame([
        [8,  1, 0.44, 0.44, 0.80, 0.19, 'fall',   1, 9,  0, 1, 1],
        [17, 3, 0.60, 0.59, 0.55, 0.25, 'summer', 1, 7,  0, 1, 2],
        [10, 6, 0.30, 0.30, 0.75, 0.10, 'spring', 0, 4,  0, 0, 3],
        [22, 0, 0.20, 0.20, 0.90, 0.05, 'winter', 0, 1,  1, 0, 2],
        [13, 4, 0.70, 0.68, 0.40, 0.35, 'fall',   1, 10, 0, 1, 1],
    ], columns=TEMPLATE_COLS)
    return sample.to_csv(index=False).encode()


# ── Artifact guard ─────────────────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-icon">🚲</div>
        <div>
            <div class="brand-title">Bike</div>
            <div class="brand-sub">Demand Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error("**Artifacts not found.** Run the notebook `Bike_Sharing_Demand_Prediction.ipynb` end-to-end to generate the `artifacts/` folder, then restart.")
    st.code("jupyter nbconvert --to notebook --execute Bike_Sharing_Demand_Prediction.ipynb\nstreamlit run streamlit_app.py", language="bash")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-brand">🚲 Bike</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Prediction Mode",
        options=["Single Prediction", "Bulk Prediction"],
        label_visibility="collapsed",
    )

    chip_class = "sb-mode-single" if mode == "Single Prediction" else "sb-mode-bulk"
    chip_label = "⊙ Single Mode" if mode == "Single Prediction" else "⊞ Bulk Mode"
    st.markdown(f'<div class="sb-mode-chip {chip_class}">{chip_label}</div>', unsafe_allow_html=True)

    st.markdown("---")

    if mode == "Single Prediction":
        st.markdown('<div class="section-label">⏰ Time & Date</div>', unsafe_allow_html=True)
        hr      = st.slider("Hour of Day",              0, 23, 8)
        weekday = st.slider("Weekday  (0=Sun · 6=Sat)", 0,  6, 2)
        mnth    = st.slider("Month",                    1, 12, 6)
        yr      = st.selectbox("Year", options=[0, 1],
                    format_func=lambda x: "2011" if x == 0 else "2012")

        st.markdown("---")
        st.markdown('<div class="section-label">🌤 Weather</div>', unsafe_allow_html=True)
        temp      = st.slider("Temperature (norm.)",     0.0, 1.0, 0.50, 0.01)
        atemp     = st.slider("Feels-like Temp (norm.)", 0.0, 1.0, 0.50, 0.01)
        hum       = st.slider("Humidity (norm.)",        0.0, 1.0, 0.60, 0.01)
        windspeed = st.slider("Wind Speed (norm.)",      0.0, 1.0, 0.20, 0.01)
        weathersit = st.selectbox("Weather Condition", options=[1,2,3,4],
            format_func=lambda x: {
                1: "☀️ Clear / Few Clouds",
                2: "🌥️ Mist / Cloudy",
                3: "🌧️ Light Rain / Snow",
                4: "⛈️ Heavy Rain / Ice",
            }[x])
        season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])

        st.markdown("---")
        st.markdown('<div class="section-label">📅 Day Type</div>', unsafe_allow_html=True)
        holiday    = st.selectbox("Holiday?",     [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        workingday = st.selectbox("Working Day?", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
    else:
        st.markdown("""
        <div style="font-size:12px; color:#6e6a9a; line-height:1.8;">
        Upload a <b style="color:#c4b5fd">CSV file</b> with the required columns.<br>
        Download the template to get started.<br><br>
        <b style="color:#67e8f9">Required columns:</b><br>
        <span style="font-family:'JetBrains Mono',monospace; font-size:10px;">
        hr · weekday · temp · atemp<br>
        hum · windspeed · season · yr<br>
        mnth · holiday · workingday · weathersit
        </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="brand-header">
    <div class="brand-icon">🚲</div>
    <div>
        <div class="brand-title">Bike · Demand Predictor</div>
        <div class="brand-sub">Hourly bike rental forecast · ML-powered</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Prediction":

    tab_pred, tab_hourly = st.tabs(["  📊  Prediction  ", "  📉  24-Hour Forecast  "])

    features   = build_features(hr, weekday, temp, atemp, hum, windspeed,
                                season, yr, mnth, holiday, workingday, weathersit)
    prediction = max(0, int(round(float(model.predict(features)[0]))))
    tier_label, tier_class, tier_dot = demand_tier(prediction)

    time_label = (
        "Night 🌙" if hr <= 6 else
        "Morning ☀️" if hr <= 12 else
        "Afternoon 🌤️" if hr <= 18 else
        "Evening 🌆"
    )
    day_label = "Weekend" if weekday in [0, 6] else "Weekday"
    month_names = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── TAB: Prediction ────────────────────────────────────────────────────────
    with tab_pred:
        col_main, col_side = st.columns([1.1, 1], gap="large")

        with col_main:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-number">{prediction}</div>
                <div class="pred-unit">Estimated Rentals This Hour</div>
                <div>
                    <span class="pred-badge {tier_class}">{tier_dot} {tier_label} Demand</span>
                    &nbsp;
                    <span class="pred-badge badge-vlow">{time_label}</span>
                    &nbsp;
                    <span class="pred-badge badge-mod">{day_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🌡️ Temperature", f"{temp * 41:.0f} °C")
            m2.metric("💧 Humidity",    f"{hum * 100:.0f}%")
            m3.metric("💨 Wind Speed",  f"{windspeed * 67:.0f} km/h")
            m4.metric("🗓️ Month",       month_names[mnth])

        with col_side:
            st.markdown('<div class="section-label">🔍 Smart Insights</div>', unsafe_allow_html=True)

            insights = []
            if hr in [7, 8, 9]:
                insights.append(("⏰", "Morning rush hour — commuter demand is at peak."))
            elif hr in [17, 18, 19]:
                insights.append(("⏰", "Evening rush hour — high return-trip demand."))
            elif hr <= 5:
                insights.append(("🌙", "Late night / early morning — very low demand expected."))

            if weekday in [0, 6]:
                insights.append(("🏖️", "Weekend pattern — leisure rides dominate, higher variability."))
            else:
                insights.append(("💼", "Weekday — consistent commuter-driven demand."))

            if season == "fall":
                insights.append(("🍂", "Fall is the peak season for bike rentals."))
            elif season == "spring":
                insights.append(("🌸", "Spring sees lower demand — mild temps, less cycling."))
            elif season == "summer":
                insights.append(("☀️", "Summer brings steady leisure and commute demand."))
            elif season == "winter":
                insights.append(("❄️", "Winter suppresses demand — cold conditions deter riders."))

            if weathersit == 1:
                insights.append(("☀️", "Clear skies — optimal cycling conditions."))
            elif weathersit == 2:
                insights.append(("🌥️", "Misty/cloudy — moderate impact on ridership."))
            elif weathersit == 3:
                insights.append(("🌧️", "Light rain/snow — demand is significantly lower."))
            elif weathersit == 4:
                insights.append(("⛈️", "Severe weather — expect very few rentals."))

            if hum > 0.85:
                insights.append(("💦", "Very high humidity — discomfort may deter cyclists."))
            if temp > 0.75:
                insights.append(("🔥", "High temperature — heat may reduce cycling comfort."))
            if holiday == 1:
                insights.append(("🎉", "Public holiday — overall demand tends to be lower."))
            if not insights:
                insights.append(("✅", "Typical conditions — moderate demand expected."))

            for icon, text in insights[:6]:
                st.markdown(f"""
                <div class="insight-chip">
                    <span class="chip-icon">{icon}</span>
                    <span>{text}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB: 24-Hour Forecast ──────────────────────────────────────────────────
    with tab_hourly:
        st.markdown('<div class="section-label">📉 Full-Day Demand Forecast — Fixed Conditions, All Hours</div>', unsafe_allow_html=True)

        hourly_preds = []
        for h in range(24):
            f = build_features(h, weekday, temp, atemp, hum, windspeed,
                               season, yr, mnth, holiday, workingday, weathersit)
            hourly_preds.append(max(0, int(round(float(model.predict(f)[0])))))

        hours      = list(range(24))
        bar_colors = [demand_color(v) for v in hourly_preds]

        fig, ax = plt.subplots(figsize=(13, 4))
        fig.patch.set_facecolor('#07071a')
        ax.set_facecolor('#07071a')

        ax.bar(hours, hourly_preds, color=bar_colors, width=0.72,
               zorder=3, edgecolor='none', linewidth=0)
        ax.bar([hr], [hourly_preds[hr]], color=bar_colors[hr], width=0.72,
               zorder=4, edgecolor='#ffffff', linewidth=2.0)
        ax.axvline(hr, color='#ffffff', linewidth=1, linestyle='--', alpha=0.2, zorder=5)
        ax.text(hr, hourly_preds[hr] + max(hourly_preds) * 0.03,
                str(hourly_preds[hr]), ha='center', va='bottom',
                fontsize=9, color='#ffffff', fontweight='bold', fontfamily='monospace')

        ax.set_xlim(-0.6, 23.6)
        ax.set_ylim(0, max(hourly_preds) * 1.18 if max(hourly_preds) > 0 else 10)
        ax.set_xlabel("Hour of Day", color='#4a4870', fontsize=10, labelpad=8)
        ax.set_ylabel("Predicted Rentals", color='#4a4870', fontsize=10, labelpad=8)
        ax.tick_params(colors='#4a4870', length=0)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}h" for h in hours], fontsize=7.5, color='#4a4870')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color='#12122a', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        patches = [
            mpatches.Patch(color="#38bdf8", label="Very Low  (<50)"),
            mpatches.Patch(color="#fbbf24", label="Moderate  (50–149)"),
            mpatches.Patch(color="#fb923c", label="High  (150–299)"),
            mpatches.Patch(color="#f87171", label="Very High  (≥300)"),
        ]
        leg = ax.legend(handles=patches, loc='upper left', frameon=True,
                        framealpha=0.15, edgecolor='#2a2a4a',
                        fontsize=8.5, labelcolor='#9090b0')
        leg.get_frame().set_facecolor('#0a0a20')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Hourly Breakdown Table</div>', unsafe_allow_html=True)

        period = []
        for h in hours:
            if h <= 6:    period.append("🌙 Night")
            elif h <= 12: period.append("☀️ Morning")
            elif h <= 18: period.append("🌤️ Afternoon")
            else:         period.append("🌆 Evening")

        summary_df = pd.DataFrame({
            "Hour":              [f"{h:02d}:00" for h in hours],
            "Period":            period,
            "Predicted Rentals": hourly_preds,
            "Demand Tier":       [demand_tier(v)[0] for v in hourly_preds],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=300)

        st.download_button(
            "⬇  Download 24-Hour Forecast CSV",
            data=summary_df.to_csv(index=False).encode(),
            file_name="hourly_forecast.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# BULK PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-label">⊞ Bulk Prediction — Upload CSV File</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="template-card">
        <div class="template-title">📄 CSV Template</div>
        <div class="template-desc">
            Download the template with sample rows and correct column names.<br>
            Fill in your data following the format, then upload below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "⬇  Download Template CSV",
            data=make_template(),
            file_name="bike_prediction_template.csv",
            mime="text/csv",
        )
    with col_info:
        with st.expander("📋 Column reference", expanded=False):
            ref = pd.DataFrame({
                "Column":  TEMPLATE_COLS,
                "Type":    ["int","int","float","float","float","float","str","int","int","int","int","int"],
                "Values":  [
                    "0–23", "0–6 (0=Sun)", "0.0–1.0", "0.0–1.0",
                    "0.0–1.0", "0.0–1.0",
                    "spring/summer/fall/winter", "0=2011 / 1=2012",
                    "1–12", "0=No / 1=Yes", "0=No / 1=Yes", "1–4",
                ],
            })
            st.dataframe(ref, use_container_width=True, hide_index=True)

    st.markdown("---")

    uploaded = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        label_visibility="visible",
    )

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)

            missing_cols = [c for c in TEMPLATE_COLS if c not in raw_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: **{', '.join(missing_cols)}**. Check your CSV against the template.")
                st.stop()

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#5b5880;">FILE LOADED</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#a78bfa;">
                    {uploaded.name} &nbsp;·&nbsp; {len(raw_df):,} rows
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("👁️ Preview uploaded data", expanded=False):
                st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

            with st.spinner("Running predictions…"):
                preds, errors = [], []
                for _, row in raw_df.iterrows():
                    try:
                        feat = build_features_bulk(row)
                        pred = max(0, int(round(float(model.predict(feat)[0]))))
                        preds.append(pred)
                        errors.append(None)
                    except Exception as e:
                        preds.append(None)
                        errors.append(str(e))

            result_df = raw_df.copy()
            result_df['predicted_rentals'] = preds
            result_df['demand_tier'] = [
                demand_tier(p)[0] if p is not None else "Error" for p in preds
            ]

            valid_preds = [p for p in preds if p is not None]
            err_count   = sum(1 for e in errors if e is not None)

            # ── Summary stats ──────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">📊 Prediction Summary</div>', unsafe_allow_html=True)

            s1, s2, s3, s4, s5 = st.columns(5)
            stats = [
                (f"{len(raw_df):,}",                                    "Total Rows"),
                (f"{int(np.mean(valid_preds)):,}" if valid_preds else "—", "Avg Rentals"),
                (f"{int(np.max(valid_preds)):,}"  if valid_preds else "—", "Peak Rentals"),
                (f"{int(np.sum(valid_preds)):,}"  if valid_preds else "—", "Total Demand"),
                (str(err_count),                                          "Errors"),
            ]
            err_color = "#f87171" if err_count > 0 else "#4ade80"
            for col, (val, lbl) in zip([s1,s2,s3,s4,s5], stats):
                color = err_color if lbl == "Errors" else "#c4b5fd"
                with col:
                    st.markdown(f"""
                    <div class="bulk-stat">
                        <div class="bulk-stat-val" style="color:{color};">{val}</div>
                        <div class="bulk-stat-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts ─────────────────────────────────────────────────────────
            chart1, chart2 = st.columns(2, gap="large")

            with chart1:
                st.markdown('<div class="section-label">Distribution of Predicted Rentals</div>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                fig2.patch.set_facecolor('#07071a')
                ax2.set_facecolor('#07071a')
                n, bins, patches_hist = ax2.hist(valid_preds, bins=30,
                    color='#5b3cf5', edgecolor='none', alpha=0.85)
                for patch, left in zip(patches_hist, bins[:-1]):
                    mid = left + (bins[1] - bins[0]) / 2
                    patch.set_facecolor(demand_color(mid))
                    patch.set_alpha(0.85)
                ax2.set_xlabel("Predicted Rentals", color='#4a4870', fontsize=9)
                ax2.set_ylabel("Frequency", color='#4a4870', fontsize=9)
                ax2.tick_params(colors='#4a4870', length=0)
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                ax2.yaxis.grid(True, color='#12122a', linewidth=0.6)
                ax2.axvline(np.mean(valid_preds), color='#a78bfa', linewidth=1.5,
                            linestyle='--', label=f"Mean: {np.mean(valid_preds):.0f}")
                leg2 = ax2.legend(fontsize=8, labelcolor='#9090b0',
                                  framealpha=0.15, edgecolor='#2a2a4a',
                                  facecolor='#0a0a20')
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            with chart2:
                st.markdown('<div class="section-label">Demand Tier Breakdown</div>', unsafe_allow_html=True)
                tier_counts  = result_df['demand_tier'].value_counts()
                tier_order   = ["Very Low", "Moderate", "High", "Very High"]
                tier_clr_map = {"Very Low":"#38bdf8","Moderate":"#fbbf24",
                                "High":"#fb923c","Very High":"#f87171"}
                tc     = {t: tier_counts.get(t, 0) for t in tier_order}
                labels = [k for k, v in tc.items() if v > 0]
                vals   = [v for v in tc.values() if v > 0]
                colors_pie = [tier_clr_map[l] for l in labels]

                fig3, ax3 = plt.subplots(figsize=(6, 3.5))
                fig3.patch.set_facecolor('#07071a')
                ax3.set_facecolor('#07071a')
                wedges, texts, autotexts = ax3.pie(
                    vals, labels=None, colors=colors_pie,
                    autopct='%1.1f%%', startangle=140, pctdistance=0.78,
                    wedgeprops=dict(width=0.52, edgecolor='#07071a', linewidth=2)
                )
                for at in autotexts:
                    at.set_color('#ffffff')
                    at.set_fontsize(9)
                ax3.legend(wedges, [f"{l}  ({tc[l]:,})" for l in labels],
                           loc='center left', bbox_to_anchor=(0.88, 0.5),
                           fontsize=8, labelcolor='#9090b0',
                           framealpha=0.15, edgecolor='#2a2a4a',
                           facecolor='#0a0a20')
                ax3.set_title("Demand Tiers", color='#4a4870', fontsize=10, pad=10)
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close()

            # ── Hourly avg from bulk data ──────────────────────────────────────
            if result_df['hr'].nunique() > 1:
                st.markdown('<div class="section-label">Average Predicted Rentals by Hour</div>', unsafe_allow_html=True)
                hourly_avg = (result_df.groupby('hr')['predicted_rentals']
                              .mean().reset_index())

                fig4, ax4 = plt.subplots(figsize=(13, 3))
                fig4.patch.set_facecolor('#07071a')
                ax4.set_facecolor('#07071a')
                ax4.bar(hourly_avg['hr'], hourly_avg['predicted_rentals'],
                        color=[demand_color(v) for v in hourly_avg['predicted_rentals']],
                        width=0.7, zorder=3)
                ax4.set_xlabel("Hour", color='#4a4870', fontsize=9)
                ax4.set_ylabel("Avg Predicted Rentals", color='#4a4870', fontsize=9)
                ax4.tick_params(colors='#4a4870', length=0)
                for spine in ax4.spines.values():
                    spine.set_visible(False)
                ax4.yaxis.grid(True, color='#12122a', linewidth=0.6, zorder=0)
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=True)
                plt.close()

            # ── Results table ──────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Full Results Table</div>', unsafe_allow_html=True)
            show_cols = ['hr','weekday','season','mnth','temp','hum',
                         'weathersit','predicted_rentals','demand_tier']
            show_cols = [c for c in show_cols if c in result_df.columns]
            st.dataframe(result_df[show_cols], use_container_width=True,
                         hide_index=True, height=320)

            st.download_button(
                "⬇  Download Full Results CSV",
                data=result_df.to_csv(index=False).encode(),
                file_name="bike_demand_predictions.csv",
                mime="text/csv",
            )

            if err_count > 0:
                with st.expander(f"⚠️ {err_count} row(s) had errors"):
                    err_df = pd.DataFrame({
                        'row':   [i for i, e in enumerate(errors) if e],
                        'error': [e for e in errors if e],
                    })
                    st.dataframe(err_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"**Failed to process file:** {e}")

    else:
        st.markdown("""
        <div style="
            text-align:center; padding: 60px 20px;
            border: 2px dashed rgba(120,100,255,0.12);
            border-radius: 20px; margin-top: 20px;
        ">
            <div style="font-size:48px; margin-bottom:16px;">📂</div>
            <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:700;
                        color:#3d3a5c; margin-bottom:8px;">
                No file uploaded yet
            </div>
            <div style="font-size:13px; color:#2e2b48; line-height:1.7;">
                Download the template above, fill in your data,<br>
                then upload the CSV to get instant predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
