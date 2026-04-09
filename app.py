import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import json
import requests
import resend
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="Smart Delivery AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg:    #080d18;
    --bg1:   #0d1525;
    --card:  #0f1c2e;
    --border:#1a2d45;
    --accent:#00c8f0;
    --a2:    #6d28d9;
    --a3:    #f59e0b;
    --tx:    #e8f0fe;
    --txm:   #7a8fad;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: var(--bg) !important;
    color: var(--tx) !important;
}
.main .block-container { padding: 1.8rem 2.2rem; max-width: 1380px; }

section[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--tx) !important; }

.topbar {
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--a2), var(--a3), var(--accent));
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
    border-radius: 0 0 4px 4px;
    margin-bottom: 1.8rem;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:300%} }

.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 25px; font-weight: 700; color: var(--tx);
    letter-spacing: -0.5px; line-height: 1.25; margin: 0 0 4px 0;
}
.page-title span { color: var(--accent); }
.page-sub { font-size: 12px; color: var(--txm); }

.sh {
    font-family: 'Space Mono', monospace;
    font-size: 10px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--accent);
    display: flex; align-items: center; gap: 10px;
    margin: 1.8rem 0 1rem 0;
}
.sh::after { content:''; flex:1; height:1px; background:var(--border); }

.kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 22px; position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeUp 0.5s ease both;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,200,240,0.07); }
.kpi::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--accent), var(--a2));
}
.kpi-label { font-size:10px; font-weight:600; letter-spacing:1.8px; text-transform:uppercase; color:var(--txm); margin-bottom:8px; }
.kpi-value { font-family:'Space Mono',monospace; font-size:26px; font-weight:700; color:var(--tx); line-height:1; }

.pred-card {
    background: linear-gradient(135deg, #0a1628 0%, #0f1e38 60%, #100d28 100%);
    border: 1px solid #1e3050; border-radius: 20px; padding: 36px 30px;
    text-align: center; position: relative; overflow: hidden;
    animation: fadeUp 0.4s ease both;
}
.pred-card::after {
    content:''; position:absolute; top:-80px; right:-80px;
    width:240px; height:240px;
    background: radial-gradient(circle, rgba(0,200,240,0.07) 0%, transparent 70%);
    pointer-events:none;
}
.pred-label { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:var(--txm); margin-bottom:12px; font-weight:600; }
.pred-eta   { font-family:'Space Mono',monospace; font-size:72px; font-weight:700; color:var(--accent); line-height:1; text-shadow:0 0 40px rgba(0,200,240,0.22); }
.pred-unit  { font-size:20px; color:var(--txm); font-family:'Space Mono',monospace; }
.pred-insight { margin-top:20px; padding:13px 16px; background:rgba(0,200,240,0.05); border:1px solid rgba(0,200,240,0.12); border-radius:10px; font-size:13px; color:#a0b4cc; line-height:1.65; }
.pred-pills { margin-top:16px; display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; text-align:left; }
.pred-pill  { background:#0d1828; padding:11px 13px; border-radius:10px; border:1px solid var(--border); }
.pred-pill-label { font-size:9px; color:var(--txm); letter-spacing:1.2px; text-transform:uppercase; margin-bottom:4px; }
.pred-pill-val   { font-family:'Space Mono',monospace; font-size:15px; color:var(--tx); }

.ai-wrap { animation: fadeUp 0.4s ease both; }
.ai-card {
    border-radius: 14px; padding: 22px 24px;
    border: 1px solid; margin-bottom: 12px;
}
.ai-card.good     { background: #061a0f; border-color: #0d4a2a; }
.ai-card.warning  { background: #1a1200; border-color: #4a3000; }
.ai-card.critical { background: #1a0808; border-color: #5a1010; }
.ai-card.neutral  { background: #0d1828; border-color: var(--border); }

.ai-card * { color: var(--tx) !important; }
.ai-stitle  { color: var(--txm) !important; font-size:10px; font-weight:700; letter-spacing:1.8px; text-transform:uppercase; margin-bottom:7px; }
.ai-text    { font-size:14px !important; line-height:1.75 !important; }

.ai-badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    margin-bottom:14px; font-family:'Space Mono',monospace;
}
.badge-good     { background:#0d4a2a; color:#10b981; }
.badge-warning  { background:#4a3000; color:#f59e0b; }
.badge-critical { background:#5a1010; color:#ef4444; }

.copilot-resp {
    background: #0d1828; border:1px solid var(--border);
    border-radius:14px; padding:20px 22px; margin-top:14px;
    animation:fadeUp 0.35s ease both;
}
.copilot-resp * { color: var(--tx) !important; }

.sb-brand { font-family:'Space Mono',monospace; font-size:15px; font-weight:700; color:var(--accent) !important; letter-spacing:2px; text-transform:uppercase; padding:0.8rem 0 0.4rem 0; border-bottom:1px solid var(--border); margin-bottom:1.2rem; }
.sb-sub   { font-size:10px; color:var(--txm) !important; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:1.4rem; }

.stButton > button {
    background: linear-gradient(135deg, #00c8f0, #6d28d9) !important;
    color: white !important; border: none !important; border-radius: 9px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 1.2px !important; text-transform: uppercase !important;
    padding: 11px 20px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 18px rgba(0,200,240,0.12) !important;
}
.stButton > button:hover  { opacity:0.82 !important; transform:translateY(-1px) !important; }
.stButton > button:active { transform:translateY(0) !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:var(--card) !important; border-radius:12px !important;
    padding:4px !important; gap:2px !important; border:1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:transparent !important; color:var(--txm) !important;
    border-radius:8px !important; font-family:'Space Mono',monospace !important;
    font-size:11px !important; font-weight:700 !important;
    letter-spacing:0.5px !important; padding:8px 16px !important; border:none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,rgba(0,200,240,0.12),rgba(109,40,217,0.12)) !important;
    color:var(--accent) !important; border:1px solid rgba(0,200,240,0.18) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background:#0d1828 !important; border:1px solid var(--border) !important;
    color:var(--tx) !important; border-radius:8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color:var(--accent) !important;
    box-shadow:0 0 0 3px rgba(0,200,240,0.1) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color:var(--txm) !important; font-size:12px !important; font-weight:600 !important;
}

[data-baseweb="select"] > div { background:#0d1828 !important; border-color:var(--border) !important; }
[data-baseweb="select"] * { color:var(--tx) !important; background:#0d1828 !important; }
[data-baseweb="tag"] { background:rgba(0,200,240,0.12) !important; }

[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:12px !important; }
[data-testid="stJson"] { background:#0d1828 !important; border:1px solid var(--border) !important; border-radius:10px !important; }
[data-testid="stJson"] * { color:var(--tx) !important; }

[data-testid="metric-container"] {
    background:#0d1828 !important; border:1px solid var(--border) !important;
    border-radius:12px !important; padding:14px 18px !important;
}
[data-testid="metric-container"] label { color:var(--txm) !important; font-size:11px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:var(--tx) !important; font-family:'Space Mono',monospace !important;
}

[data-testid="stAlert"] {
    border-radius:10px !important; border-left-width:3px !important;
    background:#0d1828 !important;
}
[data-testid="stAlert"] p { color:var(--tx) !important; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

@keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn { from{opacity:0} to{opacity:1} }
.fade-in { animation:fadeIn 0.5s ease both; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

CITY_MAP = {
    "BANG":"Bangalore","INDORE":"Indore","COIM":"Coimbatore",
    "CHEN":"Chennai","HYD":"Hyderabad","RANCH":"Ranchi",
    "MYS":"Mysore","DELHI":"Delhi","KOLKATA":"Kolkata",
    "PUNE":"Pune","MUMBAI":"Mumbai","AHMD":"Ahmedabad"
}
ORDER_MAP   = {"Buffet":0,"Drinks":1,"Meal":2,"Snack":3}
VEHICLE_MAP = {"Bicycle":0,"Electric Scooter":1,"Motorcycle":2,"Scooter":3}
COLORS      = ["#00c8f0","#6d28d9","#f59e0b","#10b981","#ef4444","#ec4899"]

def extract_city(did):
    did = str(did).upper()
    for k, v in CITY_MAP.items():
        if k in did: return v
    return "Other"

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(13,21,37,0.8)",
    font=dict(family="Sora", color="#7a8fad", size=11),
    xaxis=dict(gridcolor="#1a2d45", linecolor="#1a2d45", zerolinecolor="#1a2d45"),
    yaxis=dict(gridcolor="#1a2d45", linecolor="#1a2d45", zerolinecolor="#1a2d45"),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2d45"),
    title_font=dict(color="#e8f0fe", size=13, family="Sora"),
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        m = joblib.load(os.path.join(base, "delivery_time_model.joblib"))
        f = joblib.load(os.path.join(base, "model_features.joblib"))
    except FileNotFoundError:
        with open(os.path.join(base, "delivery_time_model.pkl"), "rb") as fh: m = pickle.load(fh)
        with open(os.path.join(base, "model_features.pkl"),       "rb") as fh: f = pickle.load(fh)
    return m, f

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "Delivery_Dataset.csv")
    if not os.path.exists(p): return None

    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Delivery Time_taken(min)": "Delivery_Time_min"}, inplace=True)

    df["distance_km"] = df.apply(lambda r: haversine(
        r["Restaurant_latitude"],      r["Restaurant_longitude"],
        r["Delivery_location_latitude"], r["Delivery_location_longitude"]
    ), axis=1)

    df = df[df["distance_km"] <= 50].copy()
    df["City"]            = df["Delivery_person_ID"].apply(extract_city)
    df["Type_of_vehicle"] = df["Type_of_vehicle"].str.strip().str.title()
    df["Type_of_order"]   = df["Type_of_order"].str.strip().str.title()
    return df

model, FEATURES = load_model()
df_full = load_data()

# ─────────────────────────────────────────────
# ENV / API KEYS
# ─────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM    = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

# ─────────────────────────────────────────────
# GROQ / LLAMA CLIENT
# ─────────────────────────────────────────────
_GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
_GROQ_MODEL  = "llama-3.3-70b-versatile"

def _llama(prompt):
    response = _GROQ_CLIENT.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# AI DECISION
# ─────────────────────────────────────────────
def get_ai_decision(signals):
    if not GROQ_API_KEY:
        return {"status":"ERROR","reason":"Groq API key not configured.",
                "immediate_action":"Add GROQ_API_KEY to .env",
                "long_term_recommendation":"Set up environment variables."}

    delayed_pct = signals.get("delayed_pct", 0)
    avg_eta     = signals.get("avg_delivery_time_min", 0)

    # Rule-based classification — Llama writes narrative, not the verdict
    if delayed_pct > 20 or avg_eta > 35:
        forced_status = "CRITICAL"
    elif delayed_pct > 10 or avg_eta > 28:
        forced_status = "WARNING"
    else:
        forced_status = "GOOD"

    prompt = f"""You are an elite operations intelligence engine for a food delivery SaaS platform used by operations managers and C-suite executives. Your analysis must be data-driven, specific, and immediately actionable — never generic.

## Live Operations Snapshot
{json.dumps(signals, indent=2)}

## Classification
Status is already determined as: {forced_status}
Do NOT change or override this status.

## Your Task
Analyze the snapshot above and return a JSON object with exactly these fields:

1. "status" → Must be exactly: "{forced_status}"

2. "reason" → 3 sentences MAX. Must:
   - Reference AT LEAST 2 specific numbers from the snapshot
   - Identify the single biggest bottleneck (city, vehicle type, or order type) using the breakdown data provided
   - State the business impact (customer satisfaction, revenue risk, or retention)
   - Sound like a McKinsey ops analyst wrote it, not a chatbot

3. "immediate_action" → 1 sentence. Must:
   - Be executable within 24 hours by an ops manager
   - Name a specific city or vehicle type pulled from the breakdown data
   - Start with an action verb (e.g., "Reallocate", "Flag", "Dispatch", "Audit")

4. "long_term_recommendation" → 1 sentence. Must:
   - Be a strategic initiative (weeks/months horizon)
   - Reference a specific pattern visible in the breakdown data
   - Mention expected outcome (e.g., "reduce delayed_pct by ~X%")

## Hard Rules
- Return STRICT JSON ONLY — no markdown, no explanation outside JSON
- Every claim must trace back to a number in the snapshot
- No filler phrases like "it is important to", "we recommend considering", "as shown"
- If delayed_pct > 15, the reason MUST call it out as a retention risk
- Use the city-level and vehicle-level breakdown data — do not give generic advice

Return format:
{{"status":"","reason":"","immediate_action":"","long_term_recommendation":""}}"""

    try:
        t = _llama(prompt).replace("```json","").replace("```","").strip()
        start, end = t.find("{"), t.rfind("}") + 1
        if start != -1 and end > start:
            t = t[start:end]
        result = json.loads(t)
        result["status"] = forced_status  # enforce — never let model override
        return result
    except Exception as e:
        return {"status": forced_status, "reason": str(e),
                "immediate_action": "Check Groq API key and response format.",
                "long_term_recommendation": "Add response validation and retry logic."}


# ─────────────────────────────────────────────
# COPILOT
# ─────────────────────────────────────────────
def ask_copilot(question, signals):
    if not GROQ_API_KEY:
        return "Groq API key not configured."

    prompt = f"""You are an expert operations analyst for a food delivery platform. You have access to real-time operational data.

## Live Operations Data
{json.dumps(signals, indent=2)}

## Question from Operations Manager
{question}

## Instructions
- Answer in 3-5 sentences MAX
- You MUST cite at least 2 specific numbers from the data in your answer
- If the question is about a specific city or vehicle type, pull the exact figure from the breakdown data
- Give a concrete recommendation at the end of your answer
- Do not use filler phrases like "Great question" or "Based on the data provided"
- Write like a senior analyst briefing a VP — direct, confident, specific"""

    try:
        return _llama(prompt)
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────
def send_report_email(to_email: str, decision: dict, signals: dict):
    if not RESEND_API_KEY:
        return "Error: Resend API key not configured."

    c_map = {"GOOD":"#10b981","WARNING":"#f59e0b","CRITICAL":"#ef4444","ERROR":"#6b7280"}
    c = c_map.get(decision.get("status","ERROR"), "#6b7280")
    s = decision.get("status","N/A")

    # Build city breakdown rows
    city_rows = ""
    for city, eta in signals.get("avg_eta_by_city", {}).items():
        delay = signals.get("delay_rate_by_city", {}).get(city, 0)
        delay_color = "#ef4444" if delay > 20 else "#f59e0b" if delay > 10 else "#10b981"
        city_rows += f"""
        <tr>
          <td style="padding:10px 14px;color:#e8f0fe;font-size:13px;border-bottom:1px solid #1a2d45;">{city}</td>
          <td style="padding:10px 14px;color:#00c8f0;font-family:monospace;font-size:13px;border-bottom:1px solid #1a2d45;">{eta} min</td>
          <td style="padding:10px 14px;font-family:monospace;font-size:13px;border-bottom:1px solid #1a2d45;color:{delay_color};">{delay}%</td>
        </tr>"""

    # Build vehicle breakdown rows
    vehicle_rows = ""
    for vehicle, eta in signals.get("avg_eta_by_vehicle", {}).items():
        vehicle_rows += f"""
        <tr>
          <td style="padding:10px 14px;color:#e8f0fe;font-size:13px;border-bottom:1px solid #1a2d45;">{vehicle}</td>
          <td style="padding:10px 14px;color:#00c8f0;font-family:monospace;font-size:13px;border-bottom:1px solid #1a2d45;">{eta} min</td>
        </tr>"""

    html = f"""
<div style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto;background:#080d18;padding:36px;border-radius:18px;border:1px solid #1a2d45;">
  <div style="height:3px;background:linear-gradient(90deg,#00c8f0,#6d28d9,#f59e0b);border-radius:3px;margin-bottom:28px;"></div>

  <h1 style="color:#e8f0fe;font-size:22px;margin:0 0 5px 0;font-family:monospace;letter-spacing:-0.5px;">Delivery Performance Report</h1>
  <p style="color:#7a8fad;font-size:12px;margin:0 0 22px 0;">Smart Delivery AI Platform &nbsp;·&nbsp; Auto-generated alert</p>

  <div style="display:inline-block;background:{c}22;color:{c};padding:5px 18px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;border:1px solid {c}55;margin-bottom:26px;font-family:monospace;">{s}</div>

  <!-- KPI row -->
  <div style="display:flex;gap:12px;margin-bottom:22px;flex-wrap:wrap;">
    <div style="flex:1;min-width:120px;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;padding:16px 18px;">
      <div style="color:#7a8fad;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Avg ETA</div>
      <div style="color:#00c8f0;font-family:monospace;font-size:22px;font-weight:700;">{signals.get('avg_delivery_time_min','—')} min</div>
    </div>
    <div style="flex:1;min-width:120px;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;padding:16px 18px;">
      <div style="color:#7a8fad;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Delayed &gt;35min</div>
      <div style="color:#ef4444;font-family:monospace;font-size:22px;font-weight:700;">{signals.get('delayed_pct','—')}%</div>
    </div>
    <div style="flex:1;min-width:120px;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;padding:16px 18px;">
      <div style="color:#7a8fad;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Avg Rating</div>
      <div style="color:#10b981;font-family:monospace;font-size:22px;font-weight:700;">{signals.get('avg_partner_rating','—')}</div>
    </div>
    <div style="flex:1;min-width:120px;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;padding:16px 18px;">
      <div style="color:#7a8fad;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">Total Orders</div>
      <div style="color:#e8f0fe;font-family:monospace;font-size:22px;font-weight:700;">{signals.get('total_deliveries','—'):,}</div>
    </div>
  </div>

  <!-- Analysis -->
  <div style="background:#0d1525;border:1px solid #1a2d45;border-radius:12px;padding:20px;margin-bottom:14px;">
    <p style="color:#7a8fad;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 10px 0;">AI Analysis</p>
    <p style="color:#e8f0fe;font-size:14px;line-height:1.75;margin:0;">{decision.get('reason','')}</p>
  </div>

  <!-- Immediate Action -->
  <div style="background:#0d1525;border:1px solid #1a2d45;border-left:3px solid #f59e0b;border-radius:12px;padding:20px;margin-bottom:14px;">
    <p style="color:#7a8fad;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 10px 0;">Immediate Action (24h)</p>
    <p style="color:#e8f0fe;font-size:14px;line-height:1.75;margin:0;">{decision.get('immediate_action','')}</p>
  </div>

  <!-- Long-Term -->
  <div style="background:#0d1525;border:1px solid #1a2d45;border-left:3px solid #00c8f0;border-radius:12px;padding:20px;margin-bottom:22px;">
    <p style="color:#7a8fad;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 10px 0;">Long-Term Strategy</p>
    <p style="color:#e8f0fe;font-size:14px;line-height:1.75;margin:0;">{decision.get('long_term_recommendation','')}</p>
  </div>

  <!-- City Breakdown Table -->
  {'<div style="margin-bottom:22px;"><p style="color:#7a8fad;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 10px 0;">City Breakdown</p><table style="width:100%;border-collapse:collapse;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;overflow:hidden;"><thead><tr><th style="padding:10px 14px;text-align:left;color:#7a8fad;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;border-bottom:1px solid #1a2d45;">City</th><th style="padding:10px 14px;text-align:left;color:#7a8fad;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;border-bottom:1px solid #1a2d45;">Avg ETA</th><th style="padding:10px 14px;text-align:left;color:#7a8fad;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;border-bottom:1px solid #1a2d45;">Delay Rate</th></tr></thead><tbody>' + city_rows + '</tbody></table></div>' if city_rows else ''}

  <!-- Vehicle Breakdown Table -->
  {'<div style="margin-bottom:22px;"><p style="color:#7a8fad;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 10px 0;">Vehicle Breakdown</p><table style="width:100%;border-collapse:collapse;background:#0d1525;border:1px solid #1a2d45;border-radius:12px;overflow:hidden;"><thead><tr><th style="padding:10px 14px;text-align:left;color:#7a8fad;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;border-bottom:1px solid #1a2d45;">Vehicle</th><th style="padding:10px 14px;text-align:left;color:#7a8fad;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;border-bottom:1px solid #1a2d45;">Avg ETA</th></tr></thead><tbody>' + vehicle_rows + '</tbody></table></div>' if vehicle_rows else ''}

  <hr style="border:none;border-top:1px solid #1a2d45;margin:0 0 18px 0;"/>
  <p style="color:#3a4a5c;font-size:11px;margin:0;">Powered by Gradient Boosting ML · Llama 3.3 70B · Smart Delivery AI</p>
</div>"""

    try:
        resend.Emails.send({
            "from":    f"Delivery AI <{RESEND_FROM}>",
            "to":      [to_email],
            "subject": f"[{s}] Delivery Performance Alert — Action Required",
            "html":    html
        })
        return f"Report sent to {to_email}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">DeliveryAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Smart Operations Platform</div>', unsafe_allow_html=True)

    if df_full is not None:
        all_cities = sorted(df_full["City"].unique().tolist())
        selected_cities = st.multiselect(
            "Filter by City", options=all_cities,
            default=all_cities[:5] if len(all_cities) > 5 else all_cities
        )
        df_filtered = df_full[df_full["City"].isin(selected_cities)] if selected_cities else df_full

        st.markdown("---")
        st.markdown('<div style="font-size:10px;color:#7a8fad;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">Live Stats</div>', unsafe_allow_html=True)
        st.metric("Total Deliveries",  f"{len(df_filtered):,}")
        st.metric("Avg ETA",           f"{df_filtered['Delivery_Time_min'].mean():.1f} min")
        st.metric("Delayed (>35 min)", f"{(df_filtered['Delivery_Time_min']>35).mean()*100:.1f}%")
        st.metric("Avg Distance",      f"{df_filtered['distance_km'].mean():.1f} km")
    else:
        selected_cities = []
        df_filtered = None
        st.warning("Delivery_Dataset.csv not found.")

    st.markdown("---")
    st.markdown('<p style="font-size:10px;color:#2a3a4c;text-align:center;line-height:1.6;">Gradient Boosting · Llama 3.3 70B via Groq<br>Streamlit · Plotly</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="fade-in" style="margin-bottom:1.8rem;">
  <div class="page-title">Smart <span>Delivery</span> AI Platform</div>
  <div class="page-sub">ETA prediction &nbsp;&middot;&nbsp; AI analytics &nbsp;&middot;&nbsp; Business insights &nbsp;&middot;&nbsp; 45,593 real delivery records</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Predict ETA", "Dataset Explorer", "Analytics", "AI Copilot"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT ETA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sh">Delivery Parameters</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        age          = st.slider("Partner Age", 18, 60, 30)
        rating       = st.slider("Partner Rating", 1.0, 5.0, 4.5, step=0.1)
        distance     = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        order_type   = st.selectbox("Type of Order",   ["Buffet","Drinks","Meal","Snack"], index=2)
        vehicle_type = st.selectbox("Type of Vehicle", ["Bicycle","Electric Scooter","Motorcycle","Scooter"], index=2)
        predict_btn  = st.button("Predict Delivery Time", use_container_width=True)

    with right:
        if predict_btn:
            X_in = pd.DataFrame(
                [[age, rating, distance, ORDER_MAP[order_type], VEHICLE_MAP[vehicle_type]]],
                columns=FEATURES
            )
            pred = model.predict(X_in)[0]
            if   pred < 25: insight = "Fast delivery expected. High-rated partner and short distance — optimal conditions."
            elif pred < 35: insight = "Moderate delivery time. Consider a higher-rated partner if available."
            else:           insight = "Longer ETA expected. Distance or vehicle type is the primary delay factor."

            st.markdown(f"""
<div class="pred-card">
  <div class="pred-label">Estimated Delivery Time</div>
  <div class="pred-eta">{pred:.1f}<span class="pred-unit"> min</span></div>
  <div class="pred-insight">{insight}</div>
  <div class="pred-pills">
    <div class="pred-pill">
      <div class="pred-pill-label">Rating</div>
      <div class="pred-pill-val">{rating}</div>
    </div>
    <div class="pred-pill">
      <div class="pred-pill-label">Distance</div>
      <div class="pred-pill-val">{distance} km</div>
    </div>
    <div class="pred-pill">
      <div class="pred-pill-label">Vehicle</div>
      <div class="pred-pill-val" style="font-size:13px;">{vehicle_type}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="pred-card" style="opacity:0.4;">
  <div class="pred-label">Estimated Delivery Time</div>
  <div class="pred-eta" style="color:#1a2d45;">--<span class="pred-unit"> min</span></div>
  <div class="pred-insight">Set parameters on the left and hit Predict.</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:2rem;">Model Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({
            "Feature":    ["Partner Rating","Distance","Partner Age","Vehicle Type","Order Type"],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(color=COLORS[:len(fi)], line=dict(color="rgba(0,0,0,0)"))
        ))
        fig_fi.update_layout(**PLOT_BASE, height=210)
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ══════════════════════════════════════════════
with tab2:
    if df_filtered is None:
        st.error("Delivery_Dataset.csv not found.")
    else:
        st.markdown('<div class="sh">Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "Total Records",  f"{len(df_filtered):,}"),
            (c2, "Cities",         str(df_filtered['City'].nunique())),
            (c3, "Avg ETA",        f"{df_filtered['Delivery_Time_min'].mean():.1f} min"),
            (c4, "Avg Distance",   f"{df_filtered['distance_km'].mean():.1f} km"),
        ]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:1.6rem;">Raw Data</div>', unsafe_allow_html=True)
        default_cols = ["Delivery_person_ID","Delivery_person_Age","Delivery_person_Ratings",
                        "distance_km","Type_of_order","Type_of_vehicle","Delivery_Time_min","City"]
        show_cols = st.multiselect("Columns", options=df_filtered.columns.tolist(), default=default_cols)
        n_rows    = st.slider("Rows to show", 10, 200, 50, step=10)
        st.dataframe(df_filtered[show_cols].head(n_rows).reset_index(drop=True),
                     use_container_width=True, height=420)
        st.markdown(f'<p style="color:#3a4a5c;font-size:11px;margin-top:6px;">Showing {n_rows} of {len(df_filtered):,} rows &nbsp;·&nbsp; GPS outliers removed (distance > 50 km)</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════
with tab3:
    if df_filtered is None:
        st.error("Delivery_Dataset.csv not found.")
    else:
        st.markdown('<div class="sh">Performance Metrics</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        delayed = (df_filtered["Delivery_Time_min"] > 35).mean() * 100
        fast    = (df_filtered["Delivery_Time_min"] < 25).mean() * 100
        for col, lbl, val in [
            (k1, "Avg ETA",         f"{df_filtered['Delivery_Time_min'].mean():.1f} min"),
            (k2, "Delayed >35 min", f"{delayed:.1f}%"),
            (k3, "Fast <25 min",    f"{fast:.1f}%"),
            (k4, "Avg Rating",      f"{df_filtered['Delivery_person_Ratings'].mean():.2f}"),
            (k5, "Median Distance", f"{df_filtered['distance_km'].median():.1f} km"),
        ]:
            with col:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        r1a, r1b = st.columns(2, gap="medium")
        with r1a:
            fig = px.histogram(df_filtered, x="Delivery_Time_min", nbins=30,
                               title="Delivery Time Distribution",
                               color_discrete_sequence=[COLORS[0]])
            fig.update_layout(**PLOT_BASE, height=290)
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with r1b:
            ca = df_filtered.groupby("City")["Delivery_Time_min"].mean().reset_index().sort_values("Delivery_Time_min", ascending=False)
            fig = px.bar(ca, x="City", y="Delivery_Time_min", title="Avg ETA by City",
                         color="Delivery_Time_min",
                         color_continuous_scale=["#00c8f0","#6d28d9","#ef4444"])
            fig.update_layout(**PLOT_BASE, height=290, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        r2a, r2b = st.columns(2, gap="medium")
        with r2a:
            samp = df_filtered.sample(min(3000, len(df_filtered)), random_state=42)
            fig = px.scatter(samp, x="distance_km", y="Delivery_Time_min",
                             color="Type_of_vehicle", title="Distance vs Delivery Time",
                             color_discrete_sequence=COLORS, opacity=0.45)
            fig.update_layout(**PLOT_BASE, height=310)
            fig.update_traces(marker_size=4)
            st.plotly_chart(fig, use_container_width=True)
        with r2b:
            fig = px.box(df_filtered, x="Type_of_vehicle", y="Delivery_Time_min",
                         title="Delivery Time by Vehicle",
                         color="Type_of_vehicle", color_discrete_sequence=COLORS)
            fig.update_layout(**PLOT_BASE, height=310, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        r3a, r3b = st.columns(2, gap="medium")
        with r3a:
            samp2 = df_filtered.sample(min(2000, len(df_filtered)), random_state=7)
            fig = px.scatter(samp2, x="Delivery_person_Ratings", y="Delivery_Time_min",
                             color="City", title="Partner Rating vs Delivery Time",
                             color_discrete_sequence=COLORS, opacity=0.45)
            fig.update_layout(**PLOT_BASE, height=290)
            fig.update_traces(marker_size=4)
            st.plotly_chart(fig, use_container_width=True)
        with r3b:
            oa = df_filtered.groupby("Type_of_order")["Delivery_Time_min"].mean().reset_index()
            fig = px.pie(oa, values="Delivery_Time_min", names="Type_of_order",
                         title="Avg ETA by Order Type",
                         color_discrete_sequence=COLORS, hole=0.52)
            fig.update_layout(**PLOT_BASE, height=290)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — AI COPILOT
# ══════════════════════════════════════════════
with tab4:
    if df_filtered is not None:
        signals = {
            "total_deliveries":      int(len(df_filtered)),
            "avg_delivery_time_min": round(float(df_filtered["Delivery_Time_min"].mean()), 2),
            "delayed_pct":           round(float((df_filtered["Delivery_Time_min"] > 35).mean() * 100), 2),
            "fast_pct":              round(float((df_filtered["Delivery_Time_min"] < 25).mean() * 100), 2),
            "avg_partner_rating":    round(float(df_filtered["Delivery_person_Ratings"].mean()), 2),
            "avg_distance_km":       round(float(df_filtered["distance_km"].mean()), 2),
            "cities_covered":        int(df_filtered["City"].nunique()),
            "low_rated_partners_pct": round(float((df_filtered["Delivery_person_Ratings"] < 4.0).mean() * 100), 2),
            "avg_eta_by_vehicle":    df_filtered.groupby("Type_of_vehicle")["Delivery_Time_min"].mean().round(2).to_dict(),
            "avg_eta_by_city":       df_filtered.groupby("City")["Delivery_Time_min"].mean().round(2).to_dict(),
            "avg_eta_by_order_type": df_filtered.groupby("Type_of_order")["Delivery_Time_min"].mean().round(2).to_dict(),
            "delay_rate_by_city":    df_filtered.groupby("City")["Delivery_Time_min"].apply(lambda x: round((x > 35).mean() * 100, 2)).to_dict(),
        }
    else:
        signals = {"total_deliveries":1000,"avg_delivery_time_min":27.3,
                   "delayed_pct":8.2,"avg_partner_rating":4.3}

    col_l, col_r = st.columns([1.15, 0.85], gap="large")

    # ── LEFT: Analysis + Email ──
    with col_l:
        st.markdown('<div class="sh">AI Performance Analysis</div>', unsafe_allow_html=True)
        analyze_btn = st.button("Run AI Analysis", use_container_width=True)

        if "ai_decision" not in st.session_state:
            st.session_state["ai_decision"] = None

        if analyze_btn:
            with st.spinner("Analyzing..."):
                st.session_state["ai_decision"] = get_ai_decision(signals)

        decision = st.session_state["ai_decision"]

        if decision:
            status    = decision.get("status", "ERROR")
            card_cls  = {"GOOD":"good","WARNING":"warning","CRITICAL":"critical"}.get(status, "neutral")
            badge_cls = {"GOOD":"badge-good","WARNING":"badge-warning","CRITICAL":"badge-critical"}.get(status, "badge-warning")

            st.markdown(f"""
<div class="ai-wrap">
  <div class="ai-card {card_cls}">
    <span class="ai-badge {badge_cls}">{status}</span>
    <div class="ai-stitle">Analysis</div>
    <div class="ai-text">{decision.get('reason','')}</div>
  </div>
  <div class="ai-card {card_cls}">
    <div class="ai-stitle">Immediate Action</div>
    <div class="ai-text">{decision.get('immediate_action','')}</div>
  </div>
  <div class="ai-card {card_cls}">
    <div class="ai-stitle">Long-Term Strategy</div>
    <div class="ai-text">{decision.get('long_term_recommendation','')}</div>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="sh" style="margin-top:1.4rem;">Send Report via Email</div>', unsafe_allow_html=True)
            email_input = st.text_input(
                "Recipient Email",
                placeholder="manager@company.com",
                help="Full report with city & vehicle breakdowns delivered to this address"
            )
            send_btn = st.button("Send Report", use_container_width=True)

            if send_btn:
                if not email_input or "@" not in email_input:
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner(f"Sending to {email_input}..."):
                        result = send_report_email(email_input, decision, signals)
                    if "Error" in result:
                        st.error(result)
                    else:
                        st.success(result)
        else:
            st.markdown("""
<div class="ai-card neutral" style="text-align:center;padding:40px 24px;opacity:0.6;">
  <div class="ai-stitle" style="margin-bottom:10px;">No analysis yet</div>
  <div class="ai-text">Click <strong>Run AI Analysis</strong> to get Llama insights on the filtered dataset.</div>
</div>""", unsafe_allow_html=True)

    # ── RIGHT: Copilot ──
    with col_r:
        st.markdown('<div class="sh">Operations Copilot</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px;color:#7a8fad;margin-bottom:14px;">Ask Llama anything about your delivery data.</p>', unsafe_allow_html=True)

        for q in ["Why are deliveries slow today?",
                  "Which vehicle type causes most delays?",
                  "How can we reduce avg delivery time?",
                  "Which city needs most attention?"]:
            if st.button(q, key=f"qb_{q}", use_container_width=True):
                st.session_state["copilot_q"] = q

        user_q  = st.text_input(
            "Or ask your own question",
            value=st.session_state.get("copilot_q", ""),
            placeholder="e.g. Why are Bangalore deliveries slower?",
            key="copilot_input"
        )
        ask_btn = st.button("Ask AI", use_container_width=True)

        if ask_btn and user_q:
            with st.spinner("Thinking..."):
                answer = ask_copilot(user_q, signals)
            st.markdown(f"""
<div class="copilot-resp">
  <div class="ai-stitle" style="color:#00c8f0 !important;">Response</div>
  <div class="ai-text">{answer}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:1.4rem;">Live Context Signals</div>', unsafe_allow_html=True)
        st.json(signals)