import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock price Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f5f7ff !important; color: #1a1f36 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0ebff 100%) !important;
        border-right: 2px solid #ddd0ff !important;
    }
    [data-testid="stHeader"] { background-color: #f5f7ff !important; }
    .app-header {
        background: linear-gradient(135deg, #ffffff 0%, #f8f0ff 40%, #f0f8ff 100%);
        border: 1.5px solid #cdb8ff; border-top: 5px solid #7b2ff7;
        border-radius: 18px; padding: 26px 34px; margin-bottom: 18px;
        box-shadow: 0 6px 32px rgba(123,47,247,0.10);
    }
    .app-header h1 {
        background: linear-gradient(90deg, #7b2ff7 0%, #e91e8c 40%, #ff6b35 70%, #ffb800 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-size: 2.4rem; font-weight: 800; margin: 0;
    }
    .app-header p { color: #7a6aaa; margin: 8px 0 0 0; font-size: 0.92rem; font-weight: 500; }
    .ticker-strip {
        background: linear-gradient(90deg, #fdf0ff, #f0f4ff, #fff8f0);
        border: 1.5px solid #ddd0ff; border-left: 5px solid #e91e8c;
        border-radius: 10px; padding: 12px 22px; margin-bottom: 22px;
        color: #6a3a9a; font-size: 0.83rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .big-signal {
        font-size: 1.5rem; font-weight: 800; padding: 20px 30px;
        border-radius: 16px; text-align: center; margin: 20px 0; letter-spacing: 0.04em;
    }
    .buy  { background: linear-gradient(135deg,#eafff4,#d5ffe8); border:2.5px solid #00b84a; color:#005c24; box-shadow:0 6px 28px rgba(0,184,74,0.20); }
    .hold { background: linear-gradient(135deg,#fffce8,#fff3c0); border:2.5px solid #e6aa00; color:#7a5500; box-shadow:0 6px 28px rgba(230,170,0,0.20); }
    .sell { background: linear-gradient(135deg,#fff0f0,#ffd6d6); border:2.5px solid #e60026; color:#8a0018; box-shadow:0 6px 28px rgba(230,0,38,0.20); }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg,#ffffff,#faf4ff) !important;
        border: 1.5px solid #ddd0ff !important; border-top: 4px solid #7b2ff7 !important;
        border-radius: 16px !important; padding: 18px !important;
        box-shadow: 0 4px 18px rgba(123,47,247,0.09) !important;
    }
    [data-testid="stMetricLabel"] { color:#9a7acc !important; font-size:0.72rem !important; text-transform:uppercase !important; letter-spacing:0.08em !important; font-weight:700 !important; }
    [data-testid="stMetricValue"] { color:#1a1036 !important; font-weight:800 !important; font-size:1.3rem !important; }
    [data-testid="stTabs"] { border-bottom: 2.5px solid #ddd0ff !important; }
    [data-testid="stTabs"] button { background:transparent !important; color:#b0a0cc !important; border:none !important; font-weight:700 !important; font-size:0.93rem !important; padding:10px 22px !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#7b2ff7 !important; border-bottom:3px solid #7b2ff7 !important; }
    .stButton > button { background:linear-gradient(135deg,#f0e8ff,#e8efff) !important; color:#5a1fd4 !important; border:1.5px solid #c4a0ff !important; border-radius:10px !important; font-weight:700 !important; transition:all 0.2s !important; }
    .stButton > button:hover { background:linear-gradient(135deg,#7b2ff7,#9b4fff) !important; color:#ffffff !important; border-color:#7b2ff7 !important; box-shadow:0 4px 18px rgba(123,47,247,0.28) !important; transform:translateY(-1px) !important; }
    .preset-row .stButton > button { background:linear-gradient(135deg,#f5f0ff,#ede4ff) !important; color:#7b2ff7 !important; border:2px solid #c4a0ff !important; border-radius:10px !important; font-weight:800 !important; font-size:0.95rem !important; }
    .preset-row .stButton > button:hover { background:linear-gradient(135deg,#7b2ff7,#e91e8c) !important; color:#ffffff !important; border-color:transparent !important; box-shadow:0 4px 18px rgba(123,47,247,0.35) !important; transform:translateY(-2px) !important; }
    .forecast-btn .stButton > button { background:linear-gradient(135deg,#7b2ff7,#e91e8c) !important; color:#ffffff !important; border:none !important; border-radius:14px !important; font-size:1.08rem !important; font-weight:800 !important; padding:14px 36px !important; box-shadow:0 8px 28px rgba(123,47,247,0.35) !important; width:100% !important; }
    .forecast-btn .stButton > button:hover { background:linear-gradient(135deg,#9b4fff,#ff4a9b) !important; box-shadow:0 10px 36px rgba(123,47,247,0.50) !important; transform:translateY(-2px) !important; }
    .predict-btn .stButton > button { background:linear-gradient(135deg,#00aaff,#0066cc) !important; color:#ffffff !important; border:none !important; border-radius:12px !important; font-weight:800 !important; padding:12px 30px !important; box-shadow:0 6px 22px rgba(0,170,255,0.30) !important; width:100% !important; }
    .predict-btn .stButton > button:hover { background:linear-gradient(135deg,#0066cc,#004499) !important; transform:translateY(-2px) !important; }
    [data-testid="stSlider"] > div > div > div { background:linear-gradient(90deg,#c4a0ff,#7b2ff7) !important; }
    [data-testid="stSlider"] > div > div > div > div { background:#7b2ff7 !important; border:3px solid #ffffff !important; box-shadow:0 2px 10px rgba(123,47,247,0.40) !important; }
    [data-testid="stSelectbox"] > div > div, [data-testid="stTextArea"] textarea {
        background:#ffffff !important; border:1.5px solid #c4a0ff !important;
        color:#1a1f36 !important; border-radius:10px !important;
    }
    [data-testid="stFileUploader"] { background:linear-gradient(135deg,#fdfaff,#f8f2ff) !important; border:2px dashed #c4a0ff !important; border-radius:14px !important; }
    [data-testid="stDataFrame"] { border:1.5px solid #ddd0ff !important; border-radius:12px !important; box-shadow:0 2px 14px rgba(123,47,247,0.07) !important; overflow:hidden !important; }
    [data-testid="stDownloadButton"] button { background:linear-gradient(135deg,#eafff4,#d5ffe8) !important; color:#005c24 !important; border:2px solid #00b84a !important; border-radius:10px !important; font-weight:700 !important; width:100% !important; }
    [data-testid="stDownloadButton"] button:hover { background:linear-gradient(135deg,#00b84a,#009e3a) !important; color:#ffffff !important; }
    [data-testid="stAlert"] { border-radius:12px !important; border-left-width:4px !important; }
    code { background:#f0e8ff !important; color:#7b2ff7 !important; border-radius:5px !important; padding:2px 7px !important; font-weight:600 !important; }
    h2, h3 { background:linear-gradient(90deg,#7b2ff7,#e91e8c); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; font-weight:800 !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#7b2ff7 !important; -webkit-text-fill-color:#7b2ff7 !important; }
    [data-testid="stSidebar"] .stMarkdown p { color:#6a4a9a !important; font-weight:500 !important; }
    .rel-badge { display:inline-block; padding:5px 16px; border-radius:24px; font-size:0.82rem; font-weight:700; margin:8px 0; }
    .rel-high { background:#eafff4; color:#005c24; border:2px solid #00b84a; }
    .rel-mod  { background:#fffce8; color:#7a5500; border:2px solid #e6aa00; }
    .rel-low  { background:#fff5e8; color:#7a3800; border:2px solid #e67800; }
    .rel-spec { background:#fff0f0; color:#8a0018; border:2px solid #e60026; }
    .info-box { background:linear-gradient(135deg,#f8f4ff,#f4f8ff); border:1.5px solid #ddd0ff; border-left:4px solid #7b2ff7; border-radius:12px; padding:16px 20px; margin:12px 0; color:#3a2a6a; font-size:0.9rem; line-height:1.7; }
    .info-box strong { color:#7b2ff7; }
    .date-badge { display:inline-block; background:linear-gradient(135deg,#f0e8ff,#e8f0ff); border:1.5px solid #c4a0ff; border-radius:10px; padding:8px 18px; font-weight:700; color:#5a1fd4; font-size:0.95rem; margin:6px 0; }
    .price-pill { display:inline-block; background:linear-gradient(135deg,#f0e8ff,#e8f0ff); border:1.5px solid #c4a0ff; border-radius:22px; padding:5px 16px; font-weight:700; color:#5a1fd4; font-size:0.96rem; margin:2px 4px; }
    hr { border-color:#e8e0ff !important; margin:20px 0 !important; }
    .disclaimer { font-size:0.73rem; color:#9a8ab5; border-top:1px solid #e8e0ff; padding-top:10px; margin-top:14px; }
    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:#f5f7ff; }
    ::-webkit-scrollbar-thumb { background:linear-gradient(#7b2ff7,#e91e8c); border-radius:4px; }
    [data-testid="stCaptionContainer"] p { color:#9a8ab5 !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>📈 Stock price Predictor</h1>
    <p>⚡ LSTM Neural Network &nbsp;•&nbsp; Forecasts from your CSV's Last Date
       &nbsp;•&nbsp; Buy / Hold / Sell Signals &nbsp;•&nbsp; NSE India &amp; Yahoo Finance</p>
</div>
<div class="ticker-strip">
    🟢 MODEL ACTIVE &nbsp;•&nbsp; LSTM + MinMaxScaler &nbsp;•&nbsp; Sequence: 60 days
    &nbsp;•&nbsp; Forecast starts from your dataset's last date &nbsp;•&nbsp; 1 Day → 365 Days
</div>
""", unsafe_allow_html=True)

# ─── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model("model/lstm_model.keras")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model/scaler: {e}")
    model_loaded = False

SEQUENCE_LENGTH = 60

# ─── Helpers ───────────────────────────────────────────────────────────────────
def clean_price(val):
    if isinstance(val, str):
        val = val.replace('"', '').replace(',', '').strip()
    return float(val)

def extract_close_prices(df):
    cols = [c.strip().upper() for c in df.columns]
    df.columns = cols

    if   'CLOSE' in cols and 'SERIES' in cols:    source, close_col = "NSE India",    'CLOSE'
    elif 'CLOSE' in cols and 'ADJ CLOSE' in cols: source, close_col = "Yahoo Finance", 'CLOSE'
    elif 'CLOSE' in cols:                          source, close_col = "Generic",       'CLOSE'
    else: return None, None, None, "No 'Close' column found in your CSV."

    date_col = next((c for c in cols if c in ('DATE', 'TIMESTAMP') or 'DATE' in c), None)
    last_date = None

    try:
        prices = df[close_col].apply(clean_price).dropna().tolist()

        if source == "NSE India":
            prices = prices[::-1]
            if date_col:
                try:
                    last_date = pd.to_datetime(df[date_col].dropna().iloc[0], dayfirst=True)
                except: pass
        else:
            if date_col:
                try:
                    last_date = pd.to_datetime(df[date_col].dropna().iloc[-1], dayfirst=False)
                except: pass

    except Exception as e:
        return None, None, None, f"Could not parse Close prices: {e}"

    return prices, source, last_date, None


def predict_price(prices_list):
    arr = np.array(prices_list[-SEQUENCE_LENGTH:]).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, SEQUENCE_LENGTH, 1)
    pred_scaled = model.predict(X, verbose=0)
    return float(scaler.inverse_transform(pred_scaled)[0][0])


def run_forecast_with_progress(prices, n_days, prog=None):
    future, window = [], list(prices[-SEQUENCE_LENGTH:])
    for i in range(n_days):
        p = predict_price(window)
        future.append(p)
        window.append(p); window.pop(0)
        if prog and (i + 1) % max(1, n_days // 30) == 0:
            pct = min(int((i + 1) / n_days * 100), 100)
            prog.progress(pct, text=f"⏳ Forecasting day {i+1} of {n_days}…")
    if prog: prog.empty()
    return future


def signal_html(ret):
    if ret > 10:
        return '<div class="big-signal buy">▲ &nbsp; STRONG BUY &nbsp;—&nbsp; Bullish Trend Forecast &nbsp; ▲</div>'
    elif ret > 0:
        return '<div class="big-signal hold">◆ &nbsp; HOLD &nbsp;—&nbsp; Modest Gains Expected &nbsp; ◆</div>'
    else:
        return '<div class="big-signal sell">▼ &nbsp; SELL / AVOID &nbsp;—&nbsp; Bearish Trend Forecast &nbsp; ▼</div>'


def reliability_badge_html(n):
    if n <= 7:    return '<span class="rel-badge rel-high">🟢 High Reliability (1–7 days)</span>'
    elif n <= 30: return '<span class="rel-badge rel-mod">🟡 Moderate Reliability (7–30 days)</span>'
    elif n <= 90: return '<span class="rel-badge rel-low">🟠 Low — Directional Only (30–90 days)</span>'
    else:         return '<span class="rel-badge rel-spec">🔴 Speculative — Educational Only (90–365 days)</span>'


def apply_light_chart(fig, ax):
    fig.patch.set_facecolor("#fafbff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#7a6aaa", labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor("#ddd0ff")
    ax.grid(True, alpha=0.45, color="#ede8ff", linestyle="--")
    ax.legend(facecolor="#fdfaff", edgecolor="#ddd0ff", labelcolor="#3a2a6a", fontsize=9, framealpha=0.97)



# ─── Simple next-day prediction chart ─────────────────────────────────────────
def plot_single(hist, predicted, currency, start_date=None):
    dp = hist[-60:]
    fig, ax = plt.subplots(figsize=(13, 4))
    x = list(range(len(dp)))

    ax.plot(x, dp, color="#7b2ff7", linewidth=2, label="Historical Close", zorder=3)
    ax.fill_between(x, dp, min(dp), alpha=0.08, color="#7b2ff7")

    delta = predicted - dp[-1]
    pct   = (delta / dp[-1]) * 100
    dot_color = "#00b84a" if delta >= 0 else "#e60026"
    pred_x = len(dp)
    ax.plot([pred_x - 1, pred_x], [dp[-1], predicted],
            color=dot_color, linewidth=1.8, linestyle="--", zorder=4)
    ax.scatter(pred_x, predicted, color=dot_color, s=180, zorder=6,
               edgecolors="#fff", linewidths=2,
               label=f"Predicted: {currency}{predicted:,.2f}  ({pct:+.2f}%)")

    ax.set_ylabel(f"Price ({currency})", color="#7a6aaa", fontsize=9)
    title = "Next-Day Prediction"
    if start_date:
        title += f"  |  {(start_date + timedelta(days=1)).strftime('%d %b %Y')}"
    ax.set_title(title, color="#7b2ff7", fontsize=12, fontweight="bold", pad=10)
    apply_light_chart(fig, ax)
    fig.tight_layout()
    return fig


# ─── Simple forecast chart ─────────────────────────────────────────────────────
def plot_forecast(hist, future, currency, n_days, start_date=None):
    lc = hist[-1]
    fu_arr = np.array(future)
    is_bullish = future[-1] >= lc
    fc = "#00a854" if is_bullish else "#e60026"

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(n_days)

    # Forecast line + fill
    ax.plot(x, fu_arr, color=fc, linewidth=2.2, label=f"{n_days}-Day Forecast", zorder=3)
    ax.fill_between(x, fu_arr, lc, alpha=0.10, color=fc)

    # Last close reference line
    ax.axhline(lc, color="#7b2ff7", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Last Close {currency}{lc:,.2f}")

    # Mark only peak and trough dots — no overlapping text boxes
    pk = max(future); pk_i = future.index(pk)
    tr = min(future); tr_i = future.index(tr)
    price_range = max(fu_arr) - min(fu_arr) if max(fu_arr) != min(fu_arr) else lc * 0.01

    ax.scatter(pk_i, pk, color="#00b84a", s=80, zorder=5, edgecolors="#fff", linewidths=1.5)
    ax.scatter(tr_i, tr, color="#e60026", s=80, zorder=5, edgecolors="#fff", linewidths=1.5)

    # Simple text labels — positioned safely above/below with fixed offset
    y_top = max(fu_arr.max(), lc) + price_range * 0.18
    y_bot = min(fu_arr.min(), lc) - price_range * 0.18
    ax.text(pk_i, y_top, f"Peak\n{currency}{pk:,.0f}", ha="center", va="bottom",
            fontsize=8, color="#005c24", fontweight="bold")
    ax.text(tr_i, y_bot, f"Trough\n{currency}{tr:,.0f}", ha="center", va="top",
            fontsize=8, color="#8a0018", fontweight="bold")

    # X-axis date ticks
    n_ticks = min(7, n_days)
    tick_idx = np.linspace(0, n_days - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    if start_date:
        dates = [start_date + timedelta(days=int(i) + 1) for i in tick_idx]
        ax.set_xticklabels([d.strftime("%d %b \'%y") for d in dates],
                           rotation=30, ha="right", fontsize=8, color="#7a6aaa")
    else:
        ax.set_xticklabels([f"Day +{i+1}" for i in tick_idx],
                           rotation=30, ha="right", fontsize=8, color="#7a6aaa")

    # Final return label on right side
    final_ret = ((future[-1] - lc) / lc) * 100
    ax.text(n_days - 1, future[-1],
            f"  {'+' if final_ret >= 0 else ''}{final_ret:.1f}%",
            va="center", fontsize=9, fontweight="bold", color=fc)

    start_str = start_date.strftime("%d %b %Y") if start_date else "Start"
    end_str   = (start_date + timedelta(days=n_days)).strftime("%d %b %Y") if start_date else f"Day +{n_days}"
    ax.set_ylabel(f"Price ({currency})", color="#7a6aaa", fontsize=9)
    ax.set_title(f"📈  {n_days}-Day Forecast  |  {start_str}  →  {end_str}",
                 color="#7b2ff7", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-0.5, n_days - 0.5)
    apply_light_chart(fig, ax)
    fig.tight_layout()
    return fig

# ─── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model("model/lstm_model.keras")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model/scaler: {e}")
    model_loaded = False

SEQUENCE_LENGTH = 60

# ─── Helpers ───────────────────────────────────────────────────────────────────
def clean_price(val):
    if isinstance(val, str):
        val = val.replace('"', '').replace(',', '').strip()
    return float(val)

def extract_close_prices(df):
    cols = [c.strip().upper() for c in df.columns]
    df.columns = cols

    if   'CLOSE' in cols and 'SERIES' in cols:    source, close_col = "NSE India",    'CLOSE'
    elif 'CLOSE' in cols and 'ADJ CLOSE' in cols: source, close_col = "Yahoo Finance", 'CLOSE'
    elif 'CLOSE' in cols:                          source, close_col = "Generic",       'CLOSE'
    else: return None, None, None, "No 'Close' column found in your CSV."

    date_col = next((c for c in cols if c in ('DATE', 'TIMESTAMP') or 'DATE' in c), None)
    last_date = None

    try:
        prices = df[close_col].apply(clean_price).dropna().tolist()

        if source == "NSE India":
            prices = prices[::-1]
            if date_col:
                try:
                    last_date = pd.to_datetime(df[date_col].dropna().iloc[0], dayfirst=True)
                except: pass
        else:
            if date_col:
                try:
                    last_date = pd.to_datetime(df[date_col].dropna().iloc[-1], dayfirst=False)
                except: pass

    except Exception as e:
        return None, None, None, f"Could not parse Close prices: {e}"

    return prices, source, last_date, None


def predict_price(prices_list):
    arr = np.array(prices_list[-SEQUENCE_LENGTH:]).reshape(-1, 1)
    scaled = scaler.transform(arr)
    X = scaled.reshape(1, SEQUENCE_LENGTH, 1)
    pred_scaled = model.predict(X, verbose=0)
    return float(scaler.inverse_transform(pred_scaled)[0][0])


def run_forecast_with_progress(prices, n_days, prog=None):
    future, window = [], list(prices[-SEQUENCE_LENGTH:])
    for i in range(n_days):
        p = predict_price(window)
        future.append(p)
        window.append(p); window.pop(0)
        if prog and (i + 1) % max(1, n_days // 30) == 0:
            pct = min(int((i + 1) / n_days * 100), 100)
            prog.progress(pct, text=f"⏳ Forecasting day {i+1} of {n_days}…")
    if prog: prog.empty()
    return future
# ─── Session state ─────────────────────────────────────────────────────────────
if "n_days" not in st.session_state: st.session_state.n_days = 30
if "m_days" not in st.session_state: st.session_state.m_days = 7

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    currency   = st.selectbox("Currency symbol", ["₹", "$", "€", "£", "¥"], index=0)
    show_chart = st.toggle("Show price chart", value=True)
    st.markdown("---")
    st.markdown("**📂 Supported Formats**")
    st.markdown("✅ Yahoo Finance CSV"); st.markdown("✅ NSE India CSV")
    st.markdown("---")
    st.markdown("**📡 Reliability Guide**")
    st.markdown('<span style="color:#00a854;font-weight:800">🟢 1–7 days</span> — High', unsafe_allow_html=True)
    st.markdown('<span style="color:#c98a00;font-weight:800">🟡 7–30 days</span> — Moderate', unsafe_allow_html=True)
    st.markdown('<span style="color:#c85a00;font-weight:800">🟠 30–90 days</span> — Low', unsafe_allow_html=True)
    st.markdown('<span style="color:#cc0020;font-weight:800">🔴 90–365 days</span> — Speculative', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📊 Signal Logic**")
    st.markdown('<span style="background:#eafff4;color:#005c24;font-weight:800;padding:2px 10px;border-radius:8px">▲ BUY</span> &nbsp; Return > 10%', unsafe_allow_html=True)
    st.markdown('<span style="background:#fffce8;color:#7a5500;font-weight:800;padding:2px 10px;border-radius:8px">◆ HOLD</span> &nbsp; Return 0–10%', unsafe_allow_html=True)
    st.markdown('<span style="background:#fff0f0;color:#8a0018;font-weight:800;padding:2px 10px;border-radius:8px">▼ SELL</span> &nbsp; Return < 0%', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🧠 Model: LSTM Neural Network")
    st.caption("📐 Scaler: MinMaxScaler")
    st.caption("🔢 Sequence: 60 days")
    st.markdown('<p class="disclaimer">⚠️ For educational use only.<br>This is NOT financial advice.</p>', unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂  Upload CSV & Predict", "🔢  Manual Input", "📖  How to Use"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Historical Stock Data CSV")

    col_a, col_b = st.columns(2)
    col_a.markdown("**✅ Yahoo Finance format**")
    col_a.code("Date, Open, High, Low, Close, Adj Close, Volume\n2023-12-29, 1400, 1450, 1390, 1428.80, ...")
    col_b.markdown("**✅ NSE India format**")
    col_b.code('DATE, SERIES, OPEN, HIGH, LOW, CLOSE, ...\n29-Dec-2023, EQ, ..., "1,428.80"')

    uploaded_file = st.file_uploader("📁 Drop your CSV file here", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            prices, source, last_date, error = extract_close_prices(df)

            if error:
                st.error(error)
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📋 Format", source)
                c2.metric("📊 Records", str(len(prices)))
                c3.metric("🕐 Latest Close", f"{currency}{prices[-1]:,.2f}")
                if last_date:
                    c4.metric("📅 Last Date in CSV", last_date.strftime("%d %b %Y"))
                else:
                    c4.metric("📅 Last Date in CSV", "Not detected")

                # Info box
                if last_date:
                    forecast_end = last_date + timedelta(days=365)
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>📅 Forecast starts from your dataset's last date:</strong>
                    <span class="date-badge">🗓 {last_date.strftime("%d %b %Y")}</span>
                    &nbsp;→&nbsp;
                    <span class="date-badge">🗓 {forecast_end.strftime("%d %b %Y")}</span>
                    (if you select 1 Year / 365 days)<br><br>
                    The model uses the last <strong>60 closing prices</strong> from your CSV as input
                    and predicts forward from <strong>{last_date.strftime("%d %b %Y")}</strong>.
                    All forecast dates in the table and chart will be relative to this date — not today's date.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Date column not detected — forecast dates will be shown as Day +1, Day +2, etc.")

                st.markdown("**Last 5 closing prices (oldest → newest):**")
                pills = " &nbsp;<span style='color:#e91e8c;font-size:1.1rem;font-weight:800'>→</span>&nbsp; ".join(
                    [f'<span class="price-pill">{currency}{p:,.2f}</span>' for p in prices[-5:]]
                )
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("")

                if len(prices) < SEQUENCE_LENGTH:
                    st.error(f"Need at least {SEQUENCE_LENGTH} records. Got {len(prices)}.")
                else:
                    # ── Next-day prediction ───────────────────────────────────
                    st.markdown("---")
                    st.subheader("🎯 Next-Day Prediction")
                    if last_date:
                        next_day = last_date + timedelta(days=1)
                        st.caption(f"Predicting price for: **{next_day.strftime('%d %b %Y')}** (1 day after your last CSV date)")

                    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
                    predict_clicked = st.button("🔍 Predict Next Day Price", key="predict_btn")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if predict_clicked:
                        if not model_loaded:
                            st.error("Model not loaded.")
                        else:
                            with st.spinner("Running LSTM prediction…"):
                                predicted = predict_price(prices)
                            lp = prices[-1]; delta = predicted - lp; pct = (delta / lp) * 100
                            arrow = "🔺" if delta >= 0 else "🔻"
                            r1, r2, r3 = st.columns(3)
                            r1.metric("📌 Last Close",      f"{currency} {lp:,.2f}")
                            r2.metric("🎯 Predicted Price", f"{currency} {predicted:,.2f}", delta=f"{arrow} {pct:+.2f}%")
                            r3.metric("💰 Change",          f"{currency} {delta:+,.2f}",    delta=f"{pct:+.2f}%")
                            if show_chart:
                                # Enhanced chart with annotated predicted price
                                st.pyplot(plot_single(prices, predicted, currency, last_date))

                    # ── Multi-day / 1-Year forecast ────────────────────────────
                    st.markdown("---")
                    st.subheader("🔮 Multi-Day / 1-Year Forecast")

                    if last_date:
                        st.markdown(
                            f"Forecast will run from "
                            f"**{last_date.strftime('%d %b %Y')}** "
                            f"(your CSV's last date) forward.",
                        )

                    st.markdown("**⚡ Quick Select:**")
                    st.markdown('<div class="preset-row">', unsafe_allow_html=True)
                    pb1, pb2, pb3, pb4, pb5 = st.columns(5)
                    with pb1:
                        if st.button("1W", key="b1w"): st.session_state.n_days = 7;   st.rerun()
                    with pb2:
                        if st.button("1M", key="b1m"): st.session_state.n_days = 30;  st.rerun()
                    with pb3:
                        if st.button("3M", key="b3m"): st.session_state.n_days = 90;  st.rerun()
                    with pb4:
                        if st.button("6M", key="b6m"): st.session_state.n_days = 180; st.rerun()
                    with pb5:
                        if st.button("1Y", key="b1y"): st.session_state.n_days = 365; st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                    n_days = st.slider(
                        "Or drag to choose days:",
                        min_value=1, max_value=365,
                        value=st.session_state.n_days,
                        key="n_days"
                    )
                    if n_days != st.session_state.n_days:
                        st.session_state.n_days = n_days

                    nd = st.session_state.n_days

                    if last_date:
                        forecast_end_date = last_date + timedelta(days=nd)
                        st.markdown(
                            f"<span class='date-badge'>📅 {last_date.strftime('%d %b %Y')}</span>"
                            f" &nbsp;<span style='color:#e91e8c;font-weight:800;font-size:1.2rem'>→</span>&nbsp; "
                            f"<span class='date-badge'>📅 {forecast_end_date.strftime('%d %b %Y')}</span>"
                            f" &nbsp;&nbsp; <span style='color:#7b2ff7;font-weight:700'>({nd} days)</span>",
                            unsafe_allow_html=True
                        )

                    st.markdown(reliability_badge_html(nd), unsafe_allow_html=True)

                    if nd > 90:
                        st.warning(
                            "⚠️ **Long-Range Warning:** LSTM compounds errors over time. "
                            "Forecasts beyond 90 days are directional trends only — not precise price targets."
                        )

                    st.markdown('<div class="forecast-btn">', unsafe_allow_html=True)
                    forecast_clicked = st.button(f"🚀 Forecast Next {nd} Days", key="forecast_btn")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if forecast_clicked:
                        if not model_loaded:
                            st.error("Model not loaded.")
                        else:
                            prog = st.progress(0, text=f"⏳ Starting {nd}-day forecast from {last_date.strftime('%d %b %Y') if last_date else 'last date'}…")
                            future = run_forecast_with_progress(prices, nd, prog)

                            lc = prices[-1]; ret = ((future[-1] - lc) / lc) * 100
                            avg = np.mean(future); peak = np.max(future)
                            trough = np.min(future); vol = np.std(future)
                            peak_day = future.index(peak) + 1
                            trough_day = future.index(trough) + 1

                            st.markdown(signal_html(ret), unsafe_allow_html=True)

                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("📌 Last Close",      f"{currency} {lc:,.2f}")
                            m2.metric("📈 Peak Price",      f"{currency} {peak:,.2f}",   f"{((peak-lc)/lc)*100:+.2f}% (Day {peak_day})")
                            m3.metric("📉 Trough Price",    f"{currency} {trough:,.2f}", f"{((trough-lc)/lc)*100:+.2f}% (Day {trough_day})")
                            m4.metric("📊 Avg Forecast",    f"{currency} {avg:,.2f}")
                            m5.metric(f"🎯 {nd}D Return",   f"{ret:+.2f}%",              f"Vol ±{currency}{vol:,.2f}")

                            if show_chart:
                                # Enhanced dual-panel forecast chart
                                st.pyplot(plot_forecast(prices, future, currency, nd, last_date))

                            # ── Forecast table with REAL DATES from CSV ────────
                            st.markdown("**📋 Forecast Table** *(sampled ~30 rows)*")
                            step = max(1, nd // 30)
                            base_date = last_date if last_date else datetime.today()
                            rows = [
                                {
                                    "Day":  f"Day +{i+1}",
                                    "Date": (base_date + timedelta(days=i+1)).strftime("%d %b %Y"),
                                    f"Predicted ({currency})": f"{currency} {future[i]:,.2f}",
                                    "Change vs Last Close": f"{((future[i]-lc)/lc)*100:+.2f}%",
                                    "Signal": "🔺 UP" if future[i] > lc else "🔻 DOWN",
                                }
                                for i in range(0, nd, step)
                            ]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)

                            full_rows = [
                                {
                                    "Day":  f"Day +{i+1}",
                                    "Date": (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                                    f"Predicted_{currency}": round(p, 2),
                                    "Change_pct": round(((p - lc) / lc) * 100, 2),
                                    "Signal": "UP" if p > lc else "DOWN",
                                }
                                for i, p in enumerate(future)
                            ]
                            csv_out = pd.DataFrame(full_rows).to_csv(index=False).encode()
                            st.download_button(
                                f"⬇️ Download Full {nd}-Day Forecast CSV",
                                csv_out, f"forecast_{nd}days.csv", "text/csv"
                            )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Manual Input
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔢 Enter Last 60 Closing Prices Manually")
    st.caption("✅ Correct: 1428.80 &nbsp;&nbsp;&nbsp; ❌ Wrong: 1,428.80")

    user_input = st.text_area(
        "Comma-separated values (oldest → newest):",
        height=180, placeholder="1400.50, 1410.20, 1415.00, 1428.80, ..."
    )

    manual_date_input = st.date_input(
        "📅 Last date of your prices (optional — for accurate forecast dates):",
        value=datetime.today()
    )

    st.markdown("**⚡ Quick Select:**")
    st.markdown('<div class="preset-row">', unsafe_allow_html=True)
    mb1, mb2, mb3, mb4, mb5 = st.columns(5)
    with mb1:
        if st.button("1W", key="m1w"): st.session_state.m_days = 7;   st.rerun()
    with mb2:
        if st.button("1M", key="m1m"): st.session_state.m_days = 30;  st.rerun()
    with mb3:
        if st.button("3M", key="m3m"): st.session_state.m_days = 90;  st.rerun()
    with mb4:
        if st.button("6M", key="m6m"): st.session_state.m_days = 180; st.rerun()
    with mb5:
        if st.button("1Y", key="m1y"): st.session_state.m_days = 365; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    manual_days = st.slider("Forecast days:", 1, 365, value=st.session_state.m_days, key="m_days")
    if manual_days != st.session_state.m_days:
        st.session_state.m_days = manual_days

    st.markdown(reliability_badge_html(st.session_state.m_days), unsafe_allow_html=True)
    if st.session_state.m_days > 90:
        st.warning("⚠️ Long-range forecasts show directional trends only.")

    st.markdown('<div class="forecast-btn">', unsafe_allow_html=True)
    manual_run = st.button("🚀 Predict & Forecast", key="manual_run")
    st.markdown('</div>', unsafe_allow_html=True)

    if manual_run:
        if not model_loaded:
            st.error("Model not loaded.")
        elif not user_input.strip():
            st.warning("Please enter closing prices.")
        else:
            try:
                prices_m = [float(x.strip()) for x in user_input.split(",") if x.strip()]
                if len(prices_m) != SEQUENCE_LENGTH:
                    st.error(f"Got {len(prices_m)} values — need exactly {SEQUENCE_LENGTH}.")
                else:
                    manual_base = datetime.combine(manual_date_input, datetime.min.time())
                    with st.spinner("Predicting…"):
                        predicted = predict_price(prices_m)
                    last = prices_m[-1]; delta = predicted - last; pct = (delta / last) * 100
                    arrow = "🔺" if delta >= 0 else "🔻"
                    r1, r2, r3 = st.columns(3)
                    r1.metric("📌 Last Price",       f"{currency} {last:,.2f}")
                    r2.metric("🎯 Next-Day Predict", f"{currency} {predicted:,.2f}", delta=f"{arrow} {pct:+.2f}%")
                    r3.metric("💰 Change",           f"{currency} {delta:+,.2f}")

                    if show_chart:
                        st.pyplot(plot_single(prices_m, predicted, currency, manual_base))

                    if st.session_state.m_days > 1:
                        md = st.session_state.m_days
                        prog2 = st.progress(0, text=f"⏳ Forecasting {md} days…")
                        future = run_forecast_with_progress(prices_m, md, prog2)
                        ret = ((future[-1] - last) / last) * 100
                        st.markdown(signal_html(ret), unsafe_allow_html=True)
                        fm1, fm2, fm3, fm4 = st.columns(4)
                        fm1.metric(f"🎯 {md}D Return", f"{ret:+.2f}%")
                        fm2.metric("📈 Peak",   f"{currency} {max(future):,.2f}", f"{((max(future)-last)/last)*100:+.2f}%")
                        fm3.metric("📉 Trough", f"{currency} {min(future):,.2f}", f"{((min(future)-last)/last)*100:+.2f}%")
                        fm4.metric("📊 Avg",    f"{currency} {np.mean(future):,.2f}")
                        if show_chart:
                            st.pyplot(plot_forecast(prices_m, future, currency, md, manual_base))
            except ValueError:
                st.error("Invalid input — use plain numbers without commas.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – How to Use
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📖 How to Use Stock price Predictor")
    st.markdown("""
    ### ✅ Your Analysis (Data 2000–2023 → Forecast from last date)

    | Step | What Happens |
    |------|-------------|
    | Upload CSV (2000–2023) | App reads all historical prices |
    | Detects last date | e.g. **29 Dec 2023** |
    | **Full history chart** | Shown immediately with ATH, ATL, 200-day MA, volatility |
    | Forecast 365 days | Predicts **30 Dec 2023 → 29 Dec 2024** |
    | Table & chart dates | All shown relative to your **CSV's last date** ✅ |

    ---

    ### 📊 Charts Explained

    **Full History Overview** *(shown on upload)*
    - Purple = price above 200-day MA (bullish zone)
    - Pink = price below 200-day MA (bearish zone)
    - 🟢 Green dot = All-Time High (ATH)
    - 🔴 Red dot = All-Time Low (ATL)
    - Bottom panel = 30-day rolling volatility

    **Next-Day Prediction Chart**
    - Colour-coded line (purple = up candles, pink = down candles)
    - Annotated callout box on predicted price point
    - Dashed connector from last close to prediction

    **Multi-Day Forecast Chart**
    - ±5% outer confidence band
    - ±2% inner confidence band
    - Peak 🟢 and Trough 🔴 annotated on chart
    - Bottom panel = daily return % vs last close

    ---

    ### 📂 How to Download Your CSV

    **Yahoo Finance:** `finance.yahoo.com` → Search stock → Historical Data → Download

    **NSE India:** `nseindia.com` → Search stock → Historical Data → Download

    ---

    ### ⚠️ Reliability
    | Window | Level | Notes |
    |--------|-------|-------|
    | 1–7 days    | 🟢 High       | Best accuracy |
    | 7–30 days   | 🟡 Moderate   | Good trend |
    | 30–90 days  | 🟠 Low        | Direction only |
    | 90–365 days | 🔴 Speculative| Educational only |

    ---
    > ⚠️ **This is NOT financial advice.** Educational use only.
    """)