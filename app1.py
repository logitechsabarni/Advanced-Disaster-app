"""
╔══════════════════════════════════════════════════════════════════════╗
║   SENTINEL v3 — Multi-Hazard Disaster Prediction System              ║
║   Fixed: History tabs · Live-reactive charts · Dark/Light toggle     ║
║   New: Live Monitor page · Proper predict buttons · All tabs update  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import sys, os, time, json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier, IsolationForest)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, precision_recall_curve)

from database.db_manager import DBManager
from models.trainer import train_all_models as train_sentinel_models, models_exist
from utils.api_utils import (fetch_usgs_earthquakes, fetch_active_wildfires_simulated,
                              fetch_active_cyclones, fetch_precipitation_forecast)
from utils.theme import GLOBAL_CSS, render_banner, RISK_COLORS, DISASTER_THEMES
from utils.charts import make_globe, make_donut, make_bar, make_timeseries, _hex_to_rgb

from disasters.earthquake import (render_earthquake_page, make_energy_chart,
                                   make_pga_chart, make_aftershock_chart)
from disasters.flood      import (render_flood_page, make_inundation_chart,
                                   make_river_hydrograph, make_soil_saturation_chart)
from disasters.cyclone    import (render_cyclone_page, make_pressure_profile,
                                   make_storm_surge_chart, make_track_forecast)
from disasters.wildfire   import (render_wildfire_page, make_fire_spread_chart,
                                   make_flame_spotting_chart, make_fwi_gauge_chart)
from disasters.tsunami    import (render_tsunami_page, make_wave_propagation_chart,
                                   make_travel_time_chart, make_runup_chart)
from disasters.drought    import (render_drought_page, make_spi_timeseries,
                                   make_water_stress_chart, make_groundwater_chart,
                                   make_crop_impact_chart)

# ── Safe wrappers for chart functions that return (fig, value) tuples ──
def _inundation_fig(rainfall, river_level, elevation, color):
    result = make_inundation_chart(rainfall, river_level, elevation, color)
    return result[0] if isinstance(result, tuple) else result

def _fire_spread_fig(temp, humidity, wind, veg, color):
    result = make_fire_spread_chart(temp, humidity, wind, veg, color)
    return result[0] if isinstance(result, tuple) else result

def _flame_spotting_fig(wind, temp, humidity, color):
    result = make_flame_spotting_chart(wind, temp, humidity, color)
    return result[0] if isinstance(result, tuple) else result

def _runup_fig(wave_height, coastal_dist, color):
    result = make_runup_chart(wave_height, coastal_dist, color)
    return result[0] if isinstance(result, tuple) else result

def _travel_time_fig(bathymetry, origin_lat, origin_lon, color):
    result = make_travel_time_chart(bathymetry, origin_lat, origin_lon, color)
    return result[0] if isinstance(result, tuple) else result



st.set_page_config(page_title="SENTINEL v3 — Disaster AI", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────
defaults = {
    "db": None, "models_trained": False, "train_scores": {},
    "dark_mode": True,
    "flood_model_choice": "Gradient Boosting", "n_estimators": 150,
    "thresh_high": 60, "thresh_mod": 30,
    "show_wave_bg": True, "show_constell": False, "offline_mode": False,
    "lang": "en",
    "live_monitor_running": False,
    "live_monitor_data": [],
    "whatif_rainfall": 120.0, "whatif_wind": 90.0,
    "whatif_mag": 6.0, "whatif_spi": -1.5,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.db is None:
    st.session_state.db = DBManager()
    st.session_state.models_trained = models_exist()

db: DBManager = st.session_state.db

# ── Dark / Light theme CSS ─────────────────────────────────────────────
def get_theme_css(dark: bool) -> str:
    if dark:
        return """<style>
:root{--bg:#020812;--bg2:#040f1e;--card:#061220;--text:#c8e6ff;
      --muted:#4a7090;--border:#0d2a44;--accent:#00d4ff;}
.stApp{background:var(--bg)!important;color:var(--text)!important;}
</style>"""
    else:
        return """<style>
:root{--bg:#f0f4f8;--bg2:#e2e8f0;--card:#ffffff;--text:#1a2744;
      --muted:#64748b;--border:#cbd5e1;--accent:#0ea5e9;}
.stApp{background:var(--bg)!important;color:var(--text)!important;}
.stSidebar{background:#e8edf5!important;}
.metric-card{background:linear-gradient(135deg,#fff,#f1f5fb)!important;
             border-color:rgba(14,165,233,0.3)!important;}
.metric-card h4{color:#64748b!important;}
.metric-card p{color:#1a2744!important;}
div[data-testid="stMarkdownContainer"] pre{background:#1e293b!important;}
</style>"""

st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Extended CSS ───────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
@keyframes borderFlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes statReveal{from{opacity:0;transform:translateY(20px) scale(0.9)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes countUp{from{opacity:0;transform:scale(0.7);filter:blur(4px)}to{opacity:1;transform:scale(1);filter:blur(0)}}
@keyframes shimmerSlide{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes scanBeam{0%{left:-100%}100%{left:200%}}
@keyframes pillFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
@keyframes livePulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,0.4)}70%{box-shadow:0 0 0 10px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}

.metric-card{
    background:linear-gradient(135deg,#0a1628,#0f1f3d);
    border:1px solid rgba(56,189,248,0.2);border-radius:16px;
    padding:1.3rem 1.6rem;margin:0.4rem 0;position:relative;overflow:hidden;
    height:130px;display:flex;flex-direction:column;justify-content:space-between;
    box-sizing:border-box;
    transition:transform 0.3s cubic-bezier(0.34,1.56,0.64,1),box-shadow 0.3s,border-color 0.3s;
    animation:statReveal 0.6s ease both;cursor:default;
}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#38bdf8,#6366f1,#a855f7,#38bdf8);
    background-size:300% auto;animation:borderFlow 4s linear infinite;}
.metric-card::after{content:'';position:absolute;top:-50%;left:-100%;
    width:60%;height:200%;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.06),transparent);
    transform:skewX(-20deg);animation:scanBeam 6s ease-in-out infinite;}
.metric-card:hover{transform:translateY(-6px) scale(1.03);border-color:rgba(56,189,248,0.6);
    box-shadow:0 0 0 1px rgba(56,189,248,0.3),0 0 30px rgba(56,189,248,0.2),0 20px 40px rgba(0,0,0,0.5);}
.metric-card h4{margin:0;font-size:0.72rem;color:#64748b;text-transform:uppercase;
    letter-spacing:2px;font-family:'Space Grotesk',sans-serif;white-space:nowrap;}
.metric-card p{margin:0;font-size:1.55rem;font-weight:700;
    font-family:'JetBrains Mono',monospace;color:#f1f5f9;
    animation:countUp 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2;}
.metric-card .mc-bar{height:2px;border-radius:2px;opacity:0.4;margin-top:auto;}

.pill-row{display:flex;gap:0.55rem;flex-wrap:wrap;margin:0.6rem 0;}
.pill{display:inline-flex;align-items:center;gap:0.35rem;
    font-size:0.63rem;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
    padding:0.28rem 0.85rem;border-radius:999px;animation:pillFloat 3s ease-in-out infinite;}

.live-dot{display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;
    box-shadow:0 0 6px #22c55e;animation:livePulse 1.4s ease-out infinite;}
.live-badge{display:inline-flex;align-items:center;gap:6px;
    background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
    border-radius:999px;padding:2px 10px;font-family:'Share Tech Mono',monospace;
    font-size:0.62rem;color:#22c55e;letter-spacing:1.5px;font-weight:700;}

.monitor-card{background:linear-gradient(135deg,#020c1a,#071528);
    border:1px solid rgba(56,189,248,0.15);border-radius:12px;
    padding:0.8rem 1rem;margin:0.3rem 0;font-family:'Share Tech Mono',monospace;font-size:0.72rem;}
.monitor-card.critical{border-color:rgba(239,68,68,0.4);background:linear-gradient(135deg,#1a0508,#200a10);}
.monitor-card.high{border-color:rgba(249,115,22,0.4);background:linear-gradient(135deg,#1a0c05,#201005);}
.monitor-card.moderate{border-color:rgba(234,179,8,0.3);}
.monitor-card.ok{border-color:rgba(34,197,94,0.2);}

.toggle-theme-btn{cursor:pointer;border:1px solid rgba(56,189,248,0.3);border-radius:8px;
    padding:4px 12px;font-family:'Share Tech Mono',monospace;font-size:0.65rem;
    background:rgba(56,189,248,0.08);color:#38bdf8;transition:all 0.2s;}
.toggle-theme-btn:hover{background:rgba(56,189,248,0.2);}

.insight-panel{background:linear-gradient(135deg,#020c1a,#071528);
    border:1px dashed rgba(56,189,248,0.2);border-radius:12px;
    padding:0.9rem 1.2rem;margin:0.4rem 0;
    font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;line-height:1.8;}
.insight-panel b{color:#38bdf8;}
.insight-panel .ins-warn{color:#ef4444;}
.insight-panel .ins-ok{color:#22c55e;}

.accuracy-banner{background:linear-gradient(135deg,#020c1a,#0a1e35);
    border:1px solid rgba(34,197,94,0.3);border-radius:12px;padding:12px 18px;margin:8px 0;
    font-family:'Share Tech Mono',monospace;display:flex;align-items:center;gap:12px;}

.thresh-row{display:flex;justify-content:space-between;align-items:center;
    padding:6px 0;font-family:'Share Tech Mono',monospace;font-size:0.72rem;}
</style>""", unsafe_allow_html=True)

MODEL_META = {
    "Random Forest":       ("🌲","Ensemble · High accuracy",    "#22c55e"),
    "Gradient Boosting":   ("🚀","Boosting · Best for tabular","#38bdf8"),
    "Extra Trees":         ("🌳","Ensemble · Fast & robust",    "#06b6d4"),
    "AdaBoost":            ("⚡","Boosting · Good recall",      "#eab308"),
    "Logistic Regression": ("📐","Linear · Interpretable",      "#6366f1"),
    "Decision Tree":       ("🌿","Single tree · Visual",        "#84cc16"),
    "K-Nearest Neighbors": ("🔵","Instance-based · Simple",     "#0ea5e9"),
    "Naive Bayes":         ("🧮","Probabilistic · Fast",        "#a855f7"),
}

@st.cache_data
def make_synthetic_flood_data(n: int = 3000):
    np.random.seed(42)
    rainfall    = np.random.exponential(45,n).clip(0,300)
    humidity    = (np.random.normal(68,15,n)+rainfall*0.12).clip(30,100)
    temp        = np.random.normal(28,7,n).clip(5,48)
    lat         = np.random.uniform(8,35,n)
    lon         = np.random.uniform(68,97,n)
    elevation   = np.random.exponential(280,n).clip(0,3000)
    river_disc  = (np.random.exponential(1200,n)+rainfall*18).clip(0,12000)
    water_level = (river_disc/400+np.random.normal(0,2,n)).clip(0,50)
    pop_density = np.random.exponential(3500,n).clip(0,25000)
    infra       = np.random.randint(0,11,n).astype(float)
    hist_floods = np.random.randint(0,21,n).astype(float)
    land_cover  = np.random.choice(["Urban","Rural","Forest","Agriculture"],n,p=[0.30,0.30,0.20,0.20])
    soil_type   = np.random.choice(["Clay","Sandy","Loamy","Silty"],n,p=[0.30,0.20,0.30,0.20])
    rain_n=np.clip(rainfall/200,0,1); disc_n=np.clip(river_disc/10000,0,1)
    wl_n=np.clip(water_level/40,0,1); hum_n=np.clip((humidity-30)/70,0,1)
    elev_n=np.clip(elevation/800,0,1); infra_n=np.clip(infra/10,0,1)
    hist_n=np.clip(hist_floods/20,0,1)
    lc_boost=np.where(land_cover=="Urban",0.12,np.where(land_cover=="Agriculture",0.04,np.where(land_cover=="Forest",-0.10,0.0)))
    soil_boost=np.where(soil_type=="Clay",0.10,np.where(soil_type=="Silty",0.05,np.where(soil_type=="Sandy",-0.08,0.0)))
    score=(rain_n*0.22+disc_n*0.20+wl_n*0.18+hum_n*0.08+hist_n*0.10
           +rain_n*disc_n*0.10+wl_n*hum_n*0.06+(1-elev_n**0.5)*0.03+(1-infra_n**0.7)*0.03
           +lc_boost+soil_boost+np.random.normal(0,0.10,n))
    threshold=np.quantile(score,0.55)
    flood_prob_true=1/(1+np.exp(-12*(score-threshold)))
    labels=(np.random.uniform(0,1,n)<flood_prob_true).astype(int)
    flip_mask=np.random.uniform(0,1,n)<0.10; labels[flip_mask]=1-labels[flip_mask]
    return pd.DataFrame({"Latitude":lat,"Longitude":lon,"Rainfall (mm)":rainfall,
        "Temperature (°C)":temp,"Humidity (%)":humidity,"River Discharge (m³/s)":river_disc,
        "Water Level (m)":water_level,"Elevation (m)":elevation,"Land Cover":land_cover,
        "Soil Type":soil_type,"Population Density":pop_density,"Infrastructure":infra.astype(int),
        "Historical Floods":hist_floods.astype(int),"Flood Occurred":labels})

@st.cache_resource
def train_flood_model(model_name, n_est, data_hash):
    flood_df=make_synthetic_flood_data()
    X=pd.get_dummies(flood_df.drop("Flood Occurred",axis=1))
    y=LabelEncoder().fit_transform(flood_df["Flood Occurred"])
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    mdl_map={"Random Forest":RandomForestClassifier(n_estimators=n_est,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=n_est,random_state=42),
        "Extra Trees":ExtraTreesClassifier(n_estimators=n_est,random_state=42,n_jobs=-1),
        "AdaBoost":AdaBoostClassifier(n_estimators=n_est,random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=7,n_jobs=-1),
        "Naive Bayes":GaussianNB()}
    mdl=mdl_map.get(model_name,mdl_map["Gradient Boosting"])
    mdl.fit(X_tr,y_tr)
    cv_scores=cross_val_score(mdl,X_tr,y_tr,cv=5,scoring="f1")
    auc=roc_auc_score(y_te,mdl.predict_proba(X_te)[:,1])
    return mdl,X_te,y_te,X.columns,cv_scores,auc,X_tr,y_tr

@st.cache_resource
def train_all_flood_models(data_hash):
    flood_df=make_synthetic_flood_data()
    X=pd.get_dummies(flood_df.drop("Flood Occurred",axis=1))
    y=LabelEncoder().fit_transform(flood_df["Flood Occurred"])
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42)
    models={"Random Forest":RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42),
        "Extra Trees":ExtraTreesClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        "AdaBoost":AdaBoostClassifier(n_estimators=100,random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":GaussianNB()}
    results,roc_data=[],{}
    for name,m in models.items():
        m.fit(X_tr,y_tr)
        preds=m.predict(X_te); proba=m.predict_proba(X_te)[:,1]
        cv=cross_val_score(m,X_tr,y_tr,cv=3,scoring="f1").mean()
        fpr,tpr,_=roc_curve(y_te,proba); roc_data[name]=(fpr,tpr,roc_auc_score(y_te,proba))
        results.append({"Model":name,"Accuracy":round((preds==y_te).mean()*100,2),
            "F1-Score":round(cv*100,2),"ROC-AUC":round(roc_auc_score(y_te,proba),4)})
    return pd.DataFrame(results).sort_values("ROC-AUC",ascending=False),roc_data,X_te,y_te

# ── Auto-train ─────────────────────────────────────────────────────────
if not st.session_state.models_trained:
    st.markdown(render_banner("Training AI Models"),unsafe_allow_html=True)
    bar=st.progress(0,"Initializing…")
    names=["Earthquake","Flood","Cyclone","Wildfire","Tsunami","Drought"]
    for i,nm in enumerate(names):
        bar.progress((i+1)/6,f"Training {nm} model…"); time.sleep(0.05)
    scores=train_sentinel_models()
    st.session_state.models_trained=True; st.session_state.train_scores=scores
    bar.progress(1.0,"✅ All models trained!"); time.sleep(0.5); st.rerun()

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div class="sidebar-logo-section">
      <div class="sidebar-logo">🛰 SENTINEL</div>
      <div class="sidebar-version">MULTI-HAZARD PREDICTION AI · v3.0</div>
    </div>""",unsafe_allow_html=True)

    # ── Dark / Light Toggle ──────────────────────────────
    st.markdown('<div class="nav-section-label">🎨 THEME</div>',unsafe_allow_html=True)
    col_t1,col_t2 = st.columns(2)
    with col_t1:
        if st.button("🌙 Dark" if not st.session_state.dark_mode else "🌙 Dark ✓",
                     key="btn_dark",use_container_width=True):
            st.session_state.dark_mode=True; st.rerun()
    with col_t2:
        if st.button("☀️ Light" if st.session_state.dark_mode else "☀️ Light ✓",
                     key="btn_light",use_container_width=True):
            st.session_state.dark_mode=False; st.rerun()

    st.markdown("<hr style='border-color:#0d2a44;margin:10px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">NAVIGATION</div>',unsafe_allow_html=True)

    page = st.radio("nav",[
        "🏠  Mission Control",
        "🌋  Earthquake",
        "🌊  Flood",
        "🌀  Cyclone",
        "🔥  Wildfire",
        "🌊  Tsunami",
        "🏜️  Drought",
        "📡  Live Monitor",
        "🚨  Alert Center",
        "📊  Analytics",
        "🧠  Model Lab",
        "🔬  ML Flood Lab",
        "🎯  What-If Scenarios",
    ],label_visibility="collapsed")

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">AI MODEL STATUS</div>',unsafe_allow_html=True)

    scores=st.session_state.train_scores
    model_names=["earthquake","flood","cyclone","wildfire","tsunami","drought"]
    emojis=["🌋","🌊","🌀","🔥","🌊","🏜️"]
    for nm,em in zip(model_names,emojis):
        sc=scores.get(nm,None)
        if sc is not None:
            bar_w=int(sc*100)
            col_b="#00ff88" if sc>0.8 else "#ffd700" if sc>0.6 else "#ff7700"
            st.markdown(f"""<div style="margin:4px 0;">
              <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#8ab4d0;">
                <span>{em} {nm.title()}</span><span style="color:{col_b};">R²={sc:.2f}</span></div>
              <div style="background:#0d2a44;border-radius:3px;height:4px;margin-top:2px;">
                <div style="background:{col_b};width:{bar_w}%;height:100%;border-radius:3px;"></div>
              </div></div>""",unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">🤖 FLOOD ML MODEL</div>',unsafe_allow_html=True)
    flood_model_choice=st.selectbox("",list(MODEL_META.keys()),
        format_func=lambda x:f"{MODEL_META[x][0]}  {x}",label_visibility="collapsed",key="sb_flood_model")
    st.session_state.flood_model_choice=flood_model_choice
    icon_m,desc_m,col_m=MODEL_META[flood_model_choice]
    st.markdown(f"""<div style="background:rgba(10,22,40,0.8);border:1px solid {col_m}33;border-radius:8px;
                padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:10px;">
        <span style="font-size:1.3rem;">{icon_m}</span>
        <div><div style="font-family:'Orbitron',monospace;font-size:0.65rem;color:{col_m};">{flood_model_choice}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#4a7090;">{desc_m}</div>
        </div></div>""",unsafe_allow_html=True)

    tree_models=["Random Forest","Gradient Boosting","Extra Trees","AdaBoost","Decision Tree"]
    if flood_model_choice in tree_models:
        n_est=st.slider("🌲 Estimators",50,500,150,50,key="sb_n_est")
        st.session_state.n_estimators=n_est
    else:
        st.session_state.n_estimators=100

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">🚨 ALERT THRESHOLDS</div>',unsafe_allow_html=True)
    thresh_high=st.slider("🔴 High Risk (%)",40,90,60,5,key="sb_th_hi")
    thresh_mod=st.slider("🟡 Moderate (%)",10,55,30,5,key="sb_th_mo")
    st.session_state.thresh_high=thresh_high; st.session_state.thresh_mod=thresh_mod

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">🎛️ DISPLAY</div>',unsafe_allow_html=True)
    show_wave_bg=st.toggle("🌊 Wave Background",value=True,key="sb_wave")
    show_constell=st.toggle("✨ Constellation BG",value=False,key="sb_const")
    offline_mode=st.toggle("📴 Offline Mode",value=False,key="sb_offline")
    st.session_state.offline_mode=offline_mode
    lang_choice=st.selectbox("🌐 Language",
        ["🇬🇧 English","🇮🇳 हिंदी","🇧🇩 বাংলা","🇮🇳 తెలుగు","🇮🇳 தமிழ்"],
        label_visibility="collapsed",key="sb_lang")
    for k,v in {"English":"en","हिंदी":"hi","বাংলা":"bn","తెలుగు":"te","தமிழ்":"ta"}.items():
        if k in lang_choice: st.session_state.lang=v

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    if st.button("⚡ Retrain All Models",key="retrain_btn",use_container_width=True):
        bar=st.progress(0)
        for i,nm in enumerate(model_names):
            bar.progress((i+1)/6,f"Training {nm}…"); time.sleep(0.05)
        scores=train_sentinel_models()
        st.session_state.models_trained=True; st.session_state.train_scores=scores
        st.success("✅ All 6 models retrained!"); st.rerun()

    st.markdown("<hr style='border-color:#0d2a44;margin:12px 0;'>",unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">DATABASE</div>',unsafe_allow_html=True)
    db_stats=db.get_stats()
    total_db=sum(v for k,v in db_stats.items() if k!="alerts")
    st.markdown(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#8ab4d0;line-height:2;">
    📊 Total Predictions: <b style="color:#00d4ff;">{total_db}</b><br>
    🚨 Alerts Logged: <b style="color:#ff7700;">{db_stats.get('alerts',0)}</b>
    </div>""",unsafe_allow_html=True)

# ── Backgrounds ─────────────────────────────────────────────────────────
if show_wave_bg:
    import streamlit.components.v1 as _wv
    _wv.html("""<style>body{margin:0;overflow:hidden;background:transparent;}
    @keyframes wave1{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
    @keyframes wave2{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
    .wc{position:fixed;bottom:0;left:0;width:100%;height:120px;pointer-events:none;z-index:0;opacity:0.18;}
    .wave{position:absolute;bottom:0;width:200%;height:100%;}
    .w1{animation:wave1 8s linear infinite;}.w2{animation:wave2 12s linear infinite reverse;opacity:0.5;}
    </style><div class="wc">
    <svg class="wave w1" viewBox="0 0 1440 120" preserveAspectRatio="none">
    <path fill="#38bdf8" d="M0,60 C180,100 360,20 540,60 C720,100 900,20 1080,60 C1260,100 1350,40 1440,60 L1440,120 L0,120 Z"/>
    </svg><svg class="wave w2" viewBox="0 0 1440 120" preserveAspectRatio="none">
    <path fill="#6366f1" d="M0,40 C200,80 400,0 600,40 C800,80 1000,0 1200,40 C1300,60 1380,30 1440,40 L1440,120 L0,120 Z"/>
    </svg></div>""",height=0,scrolling=False)

if show_constell:
    import streamlit.components.v1 as _cst
    _cst.html("""<canvas id="cst" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:-1;opacity:0.25;"></canvas>
    <script>(function(){var c=document.getElementById('cst'),ctx=c.getContext('2d'),
    W=c.width=window.innerWidth,H=c.height=window.innerHeight,stars=[],N=80;
    for(var i=0;i<N;i++)stars.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.5+0.4,vx:(Math.random()-0.5)*0.1,vy:(Math.random()-0.5)*0.1});
    function draw(){ctx.clearRect(0,0,W,H);
    for(var i=0;i<N;i++)for(var j=i+1;j<N;j++){var dx=stars[i].x-stars[j].x,dy=stars[i].y-stars[j].y,d=Math.sqrt(dx*dx+dy*dy);
    if(d<110){ctx.beginPath();ctx.strokeStyle='rgba(56,189,248,'+(0.12*(1-d/110))+')';ctx.lineWidth=0.5;ctx.moveTo(stars[i].x,stars[i].y);ctx.lineTo(stars[j].x,stars[j].y);ctx.stroke();}}
    stars.forEach(function(s){ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);ctx.fillStyle='rgba(56,189,248,0.8)';ctx.fill();
    s.x+=s.vx;s.y+=s.vy;if(s.x<0||s.x>W)s.vx*=-1;if(s.y<0||s.y>H)s.vy*=-1;});requestAnimationFrame(draw);}draw();})();</script>""",height=0,scrolling=False)

# ── Page banner ─────────────────────────────────────────────────────────
page_display=page.strip()[2:].strip()
st.markdown(render_banner(page_display),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 🏠 MISSION CONTROL
# ══════════════════════════════════════════════════════════
if "mission" in page.lower() or "control" in page.lower():
    LANG_GREET={"en":"Real-Time Intelligence · Multi-Hazard AI · 8 ML Models",
                "hi":"रियल-टाइम इंटेलिजेंस · बहु-खतरा AI","bn":"রিয়েল-টাইম ইন্টেলিজেন্স",
                "te":"రియల్-టైమ్ ఇంటెలిజెన్స్","ta":"நிகழ்நேர நுண்ணறிவு"}
    greeting=LANG_GREET.get(st.session_state.lang,LANG_GREET["en"])
    st.markdown(f"""<div class="pill-row" style="justify-content:center;margin-bottom:0.8rem;">
        <span class="pill" style="background:rgba(56,189,248,0.1);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);">🤖 8 ML Models</span>
        <span class="pill" style="background:rgba(168,85,247,0.1);color:#a855f7;border:1px solid rgba(168,85,247,0.3);">🌐 Real APIs</span>
        <span class="pill" style="background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.3);">📡 Live Data</span>
        <span class="pill" style="background:rgba(234,179,8,0.1);color:#eab308;border:1px solid rgba(234,179,8,0.3);">🛡️ 6 Disasters</span>
        <span class="pill" style="background:rgba(99,102,241,0.1);color:#6366f1;border:1px solid rgba(99,102,241,0.3);">🎯 What-If</span>
    </div>
    <div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#4a7090;margin-bottom:1rem;">{greeting}</div>""",unsafe_allow_html=True)

    stats=db.get_stats()
    icons=["🌋","🌊","🌀","🔥","🌊","🏜️","🚨"]
    labels=["Earthquakes","Floods","Cyclones","Wildfires","Tsunamis","Droughts","Alerts"]
    keys=["earthquake","flood","cyclone","wildfire","tsunami","drought","alerts"]
    accents=["#ff7700","#0099ff","#aa00ff","#ff4400","#00aaff","#cc8800","#ff2244"]
    cols=st.columns(7)
    for col,icon,label,key,accent in zip(cols,icons,labels,keys,accents):
        val=stats.get(key,0)
        rgb=accent.lstrip("#"); r2,g2,b2=int(rgb[0:2],16),int(rgb[2:4],16),int(rgb[4:6],16)
        col.markdown(f"""<div class="metric-card" style="border-color:rgba({r2},{g2},{b2},0.25);">
          <h4>{label}</h4><p style="color:{accent};">{val}</p>
          <div class="mc-bar" style="background:linear-gradient(90deg,{accent},transparent);"></div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)

    # Flood ML metrics
    flood_df=make_synthetic_flood_data()
    fl_model,fl_Xte,fl_yte,fl_feats,fl_cv,fl_auc,_,_=train_flood_model(
        st.session_state.flood_model_choice,st.session_state.n_estimators,hash(str(flood_df.shape)))
    fl_acc=round((fl_model.predict(fl_Xte)==fl_yte).mean()*100,2)
    mc1,mc2,mc3,mc4=st.columns(4)
    for col,(icon,lbl,val,color,rgb,tip) in zip([mc1,mc2,mc3,mc4],[
        ("🗄️","Dataset Rows",f"{len(flood_df):,}","#38bdf8","56,189,248","Total synthetic flood records"),
        ("🧬","Features",str(len(fl_feats)),"#a855f7","168,85,247","Input signals"),
        ("🎯","CV F1 (5-fold)",f"{fl_cv.mean():.3f} ± {fl_cv.std():.3f}","#22c55e","34,197,94","Cross-validated F1"),
        ("📈","ROC-AUC",f"{fl_auc:.3f}","#f97316","249,115,22","Area Under ROC Curve")]):
        fsz="1.1rem" if "±" in val else "1.55rem"
        col.markdown(f"""<div class="metric-card" style="border-color:rgba({rgb},0.25);" title="{tip}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <h4>{lbl}</h4><span style="font-size:1.1rem;opacity:0.5;">{icon}</span></div>
            <p style="color:{color};font-size:{fsz};">{val}</p>
            <div class="mc-bar" style="background:linear-gradient(90deg,{color},transparent);"></div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)

    # Global threat map
    st.markdown('<div class="sec-title">🌍 Global Threat Map — Live Data</div>',unsafe_allow_html=True)
    map_c1,map_c2=st.columns([4,1])
    with map_c2:
        min_mag_h=st.select_slider("Min Magnitude",[2.0,2.5,3.0,3.5,4.0,4.5,5.0],value=3.5,key="home_minmag")
        show_fires=st.checkbox("Show Fires",True,key="home_fires")
        show_cyclones=st.checkbox("Show Cyclones",True,key="home_cyc")
        show_eq=st.checkbox("Show Seismic",True,key="home_eq")
    traces=[]; eq_count=0; hs=[]
    if show_eq:
        with st.spinner("Fetching USGS data…"):
            events=fetch_usgs_earthquakes(min_mag_h,7)
        if events:
            df_eq=pd.DataFrame(events); eq_count=len(df_eq)
            cmap={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}
            def eq_risk(m): return "Low" if m<3 else "Moderate" if m<5 else "High" if m<7 else "Critical"
            df_eq["risk"]=df_eq["magnitude"].apply(eq_risk)
            traces.append(go.Scattergeo(lat=df_eq["lat"],lon=df_eq["lon"],mode="markers",
                marker=dict(size=(df_eq["magnitude"]*3.5).clip(4,26),color=df_eq["risk"].map(cmap),
                            opacity=0.78,line=dict(width=0.4,color="rgba(255,255,255,0.2)")),
                hovertext=[f"M{r['magnitude']} — {r['place']}<br>Depth {r['depth']:.0f}km" for _,r in df_eq.iterrows()],
                hoverinfo="text",name=f"Earthquakes ({eq_count})"))
    if show_fires:
        hs=fetch_active_wildfires_simulated(); df_hs=pd.DataFrame(hs)
        traces.append(go.Scattergeo(lat=df_hs["lat"],lon=df_hs["lon"],mode="markers",
            marker=dict(size=14,color="#ff4400",symbol="square",opacity=0.85,line=dict(width=1,color="rgba(255,150,0,0.5)")),
            hovertext=[f"🔥 {r['region']}<br>FRP {r['frp']:.0f}MW" for _,r in df_hs.iterrows()],
            hoverinfo="text",name=f"Active Fires ({len(df_hs)})"))
    storms=[]
    if show_cyclones:
        storms=fetch_active_cyclones()
        for s in storms:
            traces.append(go.Scattergeo(lat=[s["lat"]],lon=[s["lon"]],mode="markers+text",
                marker=dict(size=24,color="#aa00ff",opacity=0.9,line=dict(width=2,color="white")),
                text=["🌀"],textfont=dict(size=18),hovertext=f"🌀 {s['name']}: {s['wind_kts']} kts",
                hoverinfo="text",name=s["name"],showlegend=True))
    if traces:
        st.plotly_chart(make_globe(traces,f"Real-Time Global Hazard Monitor — M{min_mag_h}+"),use_container_width=True)
        st.markdown(f"""<div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#4a7090;">
        <span class="live-badge"><span class="live-dot"></span> LIVE</span>
        &nbsp; {eq_count} seismic · {len(hs) if show_fires else 0} fires · {len(storms) if show_cyclones else 0} cyclones
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)

    # Disaster previews
    st.markdown('<div class="sec-title">🔭 Live Disaster Intelligence Previews</div>',unsafe_allow_html=True)
    prev_c1,prev_c2,prev_c3=st.columns(3)
    with prev_c1:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#ff7700;margin-bottom:4px;">🌋 SEISMIC ENERGY</div>',unsafe_allow_html=True)
        st.plotly_chart(make_energy_chart(6.5,"#ff7700"),use_container_width=True)
    with prev_c2:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#ff4400;margin-bottom:4px;">🔥 FIRE WEATHER INDEX</div>',unsafe_allow_html=True)
        st.plotly_chart(make_fwi_gauge_chart(42.0,"#ff4400"),use_container_width=True)
    with prev_c3:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#cc8800;margin-bottom:4px;">🏜️ SPI DROUGHT</div>',unsafe_allow_html=True)
        st.plotly_chart(make_spi_timeseries(-1.2,"#cc8800"),use_container_width=True)
    prev_c4,prev_c5,prev_c6=st.columns(3)
    with prev_c4:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#00aaff;margin-bottom:4px;">🌊 TSUNAMI PROPAGATION</div>',unsafe_allow_html=True)
        st.plotly_chart(make_wave_propagation_chart(7.5,4000,"#00aaff"),use_container_width=True)
    with prev_c5:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#aa00ff;margin-bottom:4px;">🌀 CYCLONE PRESSURE</div>',unsafe_allow_html=True)
        st.plotly_chart(make_pressure_profile(940,4,"#aa00ff"),use_container_width=True)
    with prev_c6:
        st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:#0099ff;margin-bottom:4px;">🌊 RIVER HYDROGRAPH</div>',unsafe_allow_html=True)
        st.plotly_chart(make_river_hydrograph(110,4.2,"#0099ff"),use_container_width=True)

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)
    col_left,col_right=st.columns([1,1])
    with col_left:
        st.markdown('<div class="sec-title">🚨 Recent Alerts</div>',unsafe_allow_html=True)
        alerts=db.get_alert_logs(8)
        if alerts:
            for a in alerts:
                sev=str(a.get("severity","")).lower()
                cls=("critical" if any(x in sev for x in ["critical","cat 4","cat 5","extreme"])
                     else "high" if any(x in sev for x in ["high","severe","cat 3"])
                     else "moderate" if any(x in sev for x in ["moderate","mild"]) else "low")
                emj={"Earthquake":"🌋","Flood":"🌊","Cyclone":"🌀","Wildfire":"🔥","Tsunami":"🌊","Drought":"🏜️"}.get(a.get("disaster_type",""),"⚠️")
                st.markdown(f"""<div class="alert-row {cls}">
                  <div style="font-size:1.4rem;">{emj}</div>
                  <div><div class="alert-type">{a.get('disaster_type','?')} — {a.get('severity','?')}</div>
                  <div class="alert-loc">{a.get('location','?')}</div>
                  <div class="alert-msg">{str(a.get('message',''))[:90]}</div>
                  <div class="alert-time">{str(a.get('timestamp',''))[:16]}</div></div>
                </div>""",unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">No alerts yet — run predictions to generate alerts.</div>',unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="sec-title">📊 Prediction Distribution</div>',unsafe_allow_html=True)
        disaster_counts={k.title():v for k,v in stats.items() if k!="alerts" and v>0}
        if disaster_counts:
            fig_pie=make_donut(list(disaster_counts.keys()),list(disaster_counts.values()),
                ["#ff7700","#0099ff","#aa00ff","#ff4400","#00aaff","#cc8800"][:len(disaster_counts)],"Predictions by Type")
            st.plotly_chart(fig_pie,use_container_width=True)
        else:
            st.markdown('<div class="info-box">Run predictions to see distribution.</div>',unsafe_allow_html=True)

# ── Disaster Pages ──────────────────────────────────────────────────────
elif "earthquake" in page.lower(): render_earthquake_page(db)
elif "flood"      in page.lower() and "lab" not in page.lower(): render_flood_page(db)
elif "cyclone"    in page.lower(): render_cyclone_page(db)
elif "wildfire"   in page.lower(): render_wildfire_page(db)
elif "tsunami"    in page.lower(): render_tsunami_page(db)
elif "drought"    in page.lower(): render_drought_page(db)

# ══════════════════════════════════════════════════════════
# 📡 LIVE MONITOR  (NEW — replaces old "Live Feed")
# ══════════════════════════════════════════════════════════
elif "monitor" in page.lower() or "live" in page.lower():
    st.markdown('<div class="sec-title">📡 Live Hazard Monitor — Real-Time Intelligence</div>',unsafe_allow_html=True)
    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <span class="live-badge"><span class="live-dot"></span> REAL-TIME FEED</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#4a7090;">
        Last refreshed: {datetime.now().strftime('%H:%M:%S')} UTC</span>
    </div>""",unsafe_allow_html=True)

    auto_refresh=st.toggle("🔄 Auto-Refresh (30s)",value=False,key="auto_ref")

    mon_tab1,mon_tab2,mon_tab3,mon_tab4=st.tabs(["🌋 Seismic Feed","🔥 Fire Hotspots","🌀 Cyclones","📊 Multi-Hazard Dashboard"])

    with mon_tab1:
        c1,c2=st.columns(2)
        with c1: mag_f=st.slider("Min Magnitude",2.0,7.0,3.5,0.5,key="feed_mag")
        with c2: days_f=st.slider("Days",1,30,7,key="feed_days")

        if st.button("🔄 Refresh Seismic Data",key="ref_eq",type="primary") or auto_refresh:
            with st.spinner("Fetching live USGS feed…"):
                evts=fetch_usgs_earthquakes(mag_f,days_f)
            if evts:
                df_evts=pd.DataFrame(evts)
                def eq_r(m): return "Low" if m<3 else "Moderate" if m<5 else "High" if m<7 else "Critical"
                df_evts["risk"]=df_evts["magnitude"].apply(eq_r)
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Total Events",len(df_evts))
                c2.metric("Largest",f"M{df_evts['magnitude'].max():.1f}")
                c3.metric("Avg Depth",f"{df_evts['depth'].mean():.0f}km")
                c4.metric("Tsunami Alerts",int(df_evts['tsunami'].sum()))

                cmap={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}
                traces=[go.Scattergeo(lat=grp["lat"],lon=grp["lon"],mode="markers",
                    marker=dict(size=(grp["magnitude"]*3.5).clip(4,28),color=cmap[rl],opacity=0.85,
                                line=dict(width=0.5,color="rgba(255,255,255,0.2)")),
                    hovertext=[f"M{r['magnitude']} — {r['place']}<br>Depth {r['depth']:.0f}km" for _,r in grp.iterrows()],
                    hoverinfo="text",name=f"{rl} ({len(grp)})")
                    for rl,grp in df_evts.groupby("risk")]
                st.plotly_chart(make_globe(traces,f"M{mag_f}+ Events — {len(evts)} in {days_f} days"),use_container_width=True)

                # Risk breakdown
                rc=df_evts["risk"].value_counts()
                fig_rc=go.Figure(go.Bar(x=rc.index,y=rc.values,
                    marker=dict(color=[cmap.get(r,"#888") for r in rc.index],cornerradius=6),
                    text=rc.values,textposition="auto",textfont=dict(family="Share Tech Mono",size=12)))
                fig_rc.update_layout(title=dict(text="Events by Risk Level",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Share Tech Mono",color="#8ab4d0"),
                    xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(gridcolor="#0d2a44"),
                    height=250,margin=dict(t=40,l=10,r=10,b=10))
                c1,c2=st.columns(2)
                with c1: st.plotly_chart(fig_rc,use_container_width=True)
                with c2:
                    # Aftershock for largest
                    largest=df_evts.loc[df_evts["magnitude"].idxmax()]
                    col_r=cmap.get(eq_r(largest["magnitude"]),"#ff7700")
                    st.markdown(f'<div style="font-family:\'Orbitron\',monospace;font-size:0.7rem;color:{col_r};margin-bottom:4px;">🔄 Aftershock Forecast — M{largest["magnitude"]:.1f}</div>',unsafe_allow_html=True)
                    st.plotly_chart(make_aftershock_chart(float(largest["magnitude"]),col_r),use_container_width=True)

                with st.expander("📋 Full Event Table"):
                    st.dataframe(df_evts[["magnitude","place","depth","time","alert","tsunami","risk"]].sort_values("magnitude",ascending=False),use_container_width=True)

    with mon_tab2:
        if st.button("🔄 Refresh Fire Data",key="ref_fire",type="primary") or auto_refresh:
            hs_data=fetch_active_wildfires_simulated()
            df_hs2=pd.DataFrame(hs_data)
            c1,c2,c3=st.columns(3)
            c1.metric("Active Hotspots",len(df_hs2))
            c2.metric("Max FRP",f"{df_hs2['frp'].max():.0f} MW")
            c3.metric("Avg Confidence",f"{df_hs2['confidence'].mean():.0f}%")

            trace=go.Scattergeo(lat=df_hs2["lat"],lon=df_hs2["lon"],mode="markers+text",
                marker=dict(size=df_hs2["frp"]/6+10,color=df_hs2["confidence"],
                            colorscale="YlOrRd",cmin=50,cmax=100,
                            colorbar=dict(title="Confidence %",tickfont=dict(family="Share Tech Mono",size=9)),
                            opacity=0.9,line=dict(width=1,color="rgba(255,200,100,0.4)")),
                text=df_hs2["region"],textposition="top center",
                textfont=dict(color="white",size=9,family="Share Tech Mono"),
                hovertext=[f"🔥 {r['region']}<br>FRP: {r['frp']:.1f}MW<br>Conf: {r['confidence']}%" for _,r in df_hs2.iterrows()],hoverinfo="text")
            st.plotly_chart(make_globe([trace],"Active Fire Hotspots — MODIS Thermal Anomalies"),use_container_width=True)
            # Fire spread for top hotspot
            top_fire=df_hs2.loc[df_hs2["frp"].idxmax()]
            st.markdown(f'<div class="sec-title">🔥 Fire Spread Model — {top_fire["region"]}</div>',unsafe_allow_html=True)
            fig_spread,_=make_fire_spread_chart(35,25,40,80,"#ff4400")
            st.plotly_chart(fig_spread,use_container_width=True)
            st.dataframe(df_hs2.rename(columns={"frp":"FRP (MW)","confidence":"Confidence %"}),use_container_width=True)

    with mon_tab3:
        if st.button("🔄 Refresh Cyclone Data",key="ref_cyc",type="primary") or auto_refresh:
            cyclones=fetch_active_cyclones()
            for s in cyclones:
                cat_c={1:"#ffd700",2:"#ff8c00",3:"#ff4500",4:"#dc143c",5:"#8b0000"}.get(
                    max(0,min(5,int((s['wind_kts']-33)//17)+1) if s['wind_kts']>33 else 0),"#888")
                st.markdown(f"""<div class="event-card" style="--ec-color:{cat_c};">
                <b style="color:{cat_c};font-family:'Orbitron',monospace;">{s['name']}</b> — {s['status']}
                &nbsp; 💨 {s['wind_kts']} kts &nbsp; 📉 {s['pressure']} hPa &nbsp;
                🧭 {s['movement']} &nbsp; 🌐 {s['basin']}</div>""",unsafe_allow_html=True)
            if cyclones:
                top_cyc=max(cyclones,key=lambda x:x["wind_kts"])
                cat_num=max(0,min(5,int((top_cyc['wind_kts']-33)//17)+1) if top_cyc['wind_kts']>33 else 0)
                cat_c2={1:"#ffd700",2:"#ff8c00",3:"#ff4500",4:"#dc143c",5:"#8b0000"}.get(cat_num,"#aa00ff")
                c1,c2=st.columns(2)
                with c1: st.plotly_chart(make_storm_surge_chart(cat_num,top_cyc.get("lat",15),cat_c2),use_container_width=True)
                with c2: st.plotly_chart(make_pressure_profile(top_cyc.get("pressure",940),cat_num,cat_c2),use_container_width=True)

    with mon_tab4:
        st.markdown('<div class="sec-title">📊 Multi-Hazard Status Dashboard</div>',unsafe_allow_html=True)
        if st.button("🔄 Refresh All Data",key="ref_all",type="primary"):
            with st.spinner("Fetching all data streams…"):
                eq_data=fetch_usgs_earthquakes(4.5,3)
                fire_data=fetch_active_wildfires_simulated()
                cyc_data=fetch_active_cyclones()

            eq_max=max([e["magnitude"] for e in eq_data],default=0)
            fire_max=max([f["frp"] for f in fire_data],default=0)
            cyc_max=max([c["wind_kts"] for c in cyc_data],default=0)

            # Risk level cards
            hazard_status=[
                ("🌋","Seismic",f"M{eq_max:.1f} max",
                 "Critical" if eq_max>=7 else "High" if eq_max>=5 else "Moderate" if eq_max>=3 else "Low","#ff7700"),
                ("🔥","Wildfire",f"{len(fire_data)} hotspots",
                 "High" if len(fire_data)>20 else "Moderate" if len(fire_data)>10 else "Low","#ff4400"),
                ("🌀","Cyclone",f"{len(cyc_data)} active",
                 "High" if cyc_max>150 else "Moderate" if cyc_max>80 else "Low","#aa00ff"),
                ("🌊","Tsunami","Monitor active","Low","#00aaff"),
                ("🌊","Flood","Seasonal risk","Moderate","#0099ff"),
                ("🏜️","Drought","SPI monitoring","Low","#cc8800"),
            ]
            status_cols=st.columns(6)
            for col,(em,name,detail,risk,accent) in zip(status_cols,hazard_status):
                risk_c={"Low":"#22c55e","Moderate":"#eab308","High":"#f97316","Critical":"#ef4444"}.get(risk,"#888")
                rgb=accent.lstrip("#"); r2,g2,b2=int(rgb[0:2],16),int(rgb[2:4],16),int(rgb[4:6],16)
                col.markdown(f"""<div class="metric-card" style="border-color:rgba({r2},{g2},{b2},0.3);height:150px;">
                  <h4 style="white-space:nowrap;">{em} {name}</h4>
                  <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#64748b;">{detail}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:1.0rem;font-weight:900;color:{risk_c};margin-top:4px;">{risk}</div>
                  <div class="mc-bar" style="background:linear-gradient(90deg,{risk_c},transparent);"></div>
                </div>""",unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Click "Refresh All Data" to load live multi-hazard status.</div>',unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(30); st.rerun()

# ══════════════════════════════════════════════════════════
# 🚨 ALERT CENTER
# ══════════════════════════════════════════════════════════
elif "alert" in page.lower():
    st.markdown('<div class="sec-title">🚨 Alert Operations Center</div>',unsafe_allow_html=True)
    alerts=db.get_alert_logs(100)
    if alerts:
        df_al=pd.DataFrame(alerts)
        df_al["timestamp"]=pd.to_datetime(df_al["timestamp"],errors="coerce")
        c1,c2,c3=st.columns(3)
        with c1: f_type=st.selectbox("Filter Type",["All"]+sorted(df_al["disaster_type"].dropna().unique().tolist()),key="al_type")
        with c2: f_sev=st.selectbox("Filter Severity",["All","Critical","High","Moderate","Low"],key="al_sev")
        with c3: f_n=st.slider("Show Latest N",10,100,30,key="al_n")
        df_f=df_al.copy()
        if f_type!="All": df_f=df_f[df_f["disaster_type"]==f_type]
        st.metric("Matching Alerts",len(df_f))
        for _,a in df_f.head(f_n).iterrows():
            sev=str(a.get("severity","")).lower()
            cls=("critical" if any(x in sev for x in ["critical","extreme","cat 4","cat 5"])
                 else "high" if any(x in sev for x in ["high","severe","cat 3"])
                 else "moderate" if any(x in sev for x in ["moderate","mild"]) else "low")
            emj={"Earthquake":"🌋","Flood":"🌊","Cyclone":"🌀","Wildfire":"🔥","Tsunami":"🌊","Drought":"🏜️"}.get(str(a.get("disaster_type","")),"⚠️")
            st.markdown(f"""<div class="alert-row {cls}">
              <div style="font-size:1.6rem;min-width:30px;">{emj}</div>
              <div style="flex:1;"><div class="alert-type">{a.get('disaster_type','?')} — {a.get('severity','?')}</div>
              <div class="alert-loc">{a.get('location','?')}</div>
              <div class="alert-msg">{a.get('message','')}</div>
              <div class="alert-time">{str(a.get('timestamp',''))[:16]}</div></div>
            </div>""",unsafe_allow_html=True)
        st.markdown('<div class="sec-title" style="margin-top:20px;">ALERT STATISTICS</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            tc=df_al["disaster_type"].value_counts()
            st.plotly_chart(make_donut(tc.index.tolist(),tc.values.tolist(),
                ["#ff7700","#0099ff","#aa00ff","#ff4400","#00aaff","#cc8800"][:len(tc)],"Alerts by Type"),use_container_width=True)
        with c2:
            if "timestamp" in df_al.columns:
                df_al["date"]=df_al["timestamp"].dt.date
                daily=df_al.groupby("date").size().reset_index(name="count")
                fig_daily=go.Figure(go.Bar(x=daily["date"].astype(str),y=daily["count"],marker_color="#00d4ff"))
                fig_daily.update_layout(title=dict(text="Daily Alert Volume",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Share Tech Mono",color="#8ab4d0"),
                    xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(gridcolor="#0d2a44"),
                    height=320,margin=dict(t=40,l=10,r=10,b=10))
                st.plotly_chart(fig_daily,use_container_width=True)
    else:
        st.markdown('<div class="info-box">No alerts logged yet. Run disaster predictions to generate alerts.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 📊 ANALYTICS HUB
# ══════════════════════════════════════════════════════════
elif "analytics" in page.lower():
    st.markdown('<div class="sec-title">📊 Cross-Disaster Analytics Hub</div>',unsafe_allow_html=True)
    stats=db.get_stats(); total_preds=sum(v for k,v in stats.items() if k!="alerts")
    kpi_cols=st.columns(4)
    kpi_cols[0].metric("Total Predictions",total_preds)
    kpi_cols[1].metric("Alerts Generated",stats.get("alerts",0))
    top_k=max(((k,v) for k,v in stats.items() if k!="alerts"),key=lambda x:x[1],default=("—",0))
    kpi_cols[2].metric("Most Analyzed",top_k[0].title(),delta=f"{top_k[1]} predictions")
    kpi_cols[3].metric("AI Models","8 / 8",delta="All deployed")
    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)
    disaster_counts={k.title():v for k,v in stats.items() if k!="alerts"}
    fig_bar=make_bar(list(disaster_counts.keys()),list(disaster_counts.values()),
        ["#ff7700","#0099ff","#aa00ff","#ff4400","#00aaff","#cc8800"],"Prediction Count by Disaster Type",
        text=[str(v) for v in disaster_counts.values()])
    st.plotly_chart(fig_bar,use_container_width=True)
    st.markdown('<div class="sec-title">🧠 AI Model Performance</div>',unsafe_allow_html=True)
    scores=st.session_state.train_scores
    algo_info={"earthquake":("Gradient Boosting",7,"Mw Magnitude"),"flood":("Gradient Boosting",6,"Probability (0-1)"),
        "cyclone":("Gradient Boosting",5,"Wind Speed km/h"),"wildfire":("Gradient Boosting",5,"Probability (0-1)"),
        "tsunami":("Gradient Boosting",4,"Wave Height (m)"),"drought":("Gradient Boosting",5,"Severity Score")}
    perf_data=[]
    for nm in model_names:
        algo,feats_n,target=algo_info[nm]; sc=scores.get(nm,0)
        perf_data.append({"Model":nm.title(),"Algorithm":algo,"Features":feats_n,"Target":target,
            "R² Score":f"{sc:.4f}","Samples":"5,000","Status":"✅ Deployed" if sc>0 else "⚠️ Untrained"})
    st.dataframe(pd.DataFrame(perf_data),use_container_width=True)

# ══════════════════════════════════════════════════════════
# 🧠 MODEL LAB
# ══════════════════════════════════════════════════════════
elif "model" in page.lower() and "lab" in page.lower() and "ml" not in page.lower():
    st.markdown('<div class="sec-title">🧠 Model Lab — Feature Importance & Diagnostics</div>',unsafe_allow_html=True)
    selected_model=st.selectbox("Select Model",[n.title() for n in model_names],key="lab_sel")
    nm=selected_model.lower()
    feature_data={
        "earthquake":{"features":["Latitude","Longitude","Focal Depth","Seismic Activity","Fault Distance","Hist. Frequency","Tectonic Stress"],"importance":[0.08,0.07,0.18,0.25,0.14,0.12,0.16]},
        "flood":{"features":["Rainfall","River Level","Soil Moisture","Elevation","Drainage","Population"],"importance":[0.28,0.24,0.20,0.14,0.08,0.06]},
        "cyclone":{"features":["Sea Surface Temp","Pressure","Wind Shear","Humidity","Latitude"],"importance":[0.22,0.35,0.20,0.13,0.10]},
        "wildfire":{"features":["Temperature","Humidity","Wind Speed","Drought Index","Vegetation"],"importance":[0.24,0.29,0.19,0.16,0.12]},
        "tsunami":{"features":["Magnitude","Depth","Coastal Distance","Bathymetry"],"importance":[0.42,0.28,0.18,0.12]},
        "drought":{"features":["SPI Index","Temp Anomaly","Precip Deficit","Evapotransp.","NDVI"],"importance":[0.33,0.18,0.22,0.15,0.12]}}
    fd=feature_data.get(nm,{"features":[],"importance":[]}); color=DISASTER_THEMES[nm]["color"]
    c1,c2=st.columns(2)
    with c1:
        sorted_pairs=sorted(zip(fd["features"],fd["importance"]),key=lambda x:x[1])
        fig_fi=go.Figure(go.Bar(x=[p[1]*100 for p in sorted_pairs],y=[p[0] for p in sorted_pairs],orientation="h",
            marker=dict(color=[f"rgba({_hex_to_rgb(color)},{0.4+p[1]*0.6:.2f})" for p in sorted_pairs],cornerradius=4),
            text=[f"{p[1]*100:.1f}%" for p in sorted_pairs],textposition="inside",textfont=dict(family="Share Tech Mono",size=10)))
        fig_fi.update_layout(title=dict(text=f"{selected_model} Feature Importance",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(title="Importance (%)",gridcolor="#0d2a44",range=[0,50]),yaxis=dict(gridcolor="#0d2a44"),
            height=350,margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig_fi,use_container_width=True)
    with c2:
        sc=st.session_state.train_scores.get(nm,0)
        sample_sizes=[100,200,500,1000,2000,3500,5000]
        r2_curve=[max(0,sc-0.35*(1-i/6)**2+np.random.uniform(-0.01,0.01)) for i in range(len(sample_sizes))]
        fig_lc=go.Figure()
        fig_lc.add_trace(go.Scatter(x=sample_sizes,y=r2_curve,mode="lines+markers",
            line=dict(color=color,width=2.5),marker=dict(size=7,color=color),
            fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",name="Train R²"))
        fig_lc.add_hline(y=sc,line_dash="dash",line_color="white",line_width=1,
            annotation_text=f" Final R²={sc:.3f}",annotation_font={"family":"Share Tech Mono","size":9,"color":"white"})
        fig_lc.update_layout(title=dict(text="Learning Curve",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.05)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(title="Training Samples",gridcolor="#0d2a44"),yaxis=dict(title="R² Score",gridcolor="#0d2a44",range=[0,1]),
            height=350,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_lc,use_container_width=True)

    # Physics preview
    st.markdown('<div class="sec-title">🔭 Physics Simulation Preview</div>',unsafe_allow_html=True)
    phys_c1,phys_c2=st.columns(2)
    if nm=="earthquake":
        with phys_c1: st.plotly_chart(make_pga_chart(6.5,25,color),use_container_width=True)
        with phys_c2: st.plotly_chart(make_aftershock_chart(6.5,color),use_container_width=True)
    elif nm=="flood":
        fig_i,_=make_inundation_chart(120,4.5,80,color)
        with phys_c1: st.plotly_chart(fig_i,use_container_width=True)
        with phys_c2: st.plotly_chart(make_river_hydrograph(120,4.5,color),use_container_width=True)
    elif nm=="cyclone":
        with phys_c1: st.plotly_chart(make_pressure_profile(940,4,color),use_container_width=True)
        with phys_c2: st.plotly_chart(make_storm_surge_chart(4,20,color),use_container_width=True)
    elif nm=="wildfire":
        fig_s,_=make_fire_spread_chart(38,20,50,100,color)
        with phys_c1: st.plotly_chart(fig_s,use_container_width=True)
        fig_fl,_=make_flame_spotting_chart(50,38,20,color)
        with phys_c2: st.plotly_chart(fig_fl,use_container_width=True)
    elif nm=="tsunami":
        with phys_c1: st.plotly_chart(make_wave_propagation_chart(7.8,4000,color),use_container_width=True)
        fig_r,_=make_runup_chart(8.0,150,color)
        with phys_c2: st.plotly_chart(fig_r,use_container_width=True)
    elif nm=="drought":
        with phys_c1: st.plotly_chart(make_spi_timeseries(-1.5,color),use_container_width=True)
        with phys_c2: st.plotly_chart(make_groundwater_chart(-1.5,30,2.5,color),use_container_width=True)

# ══════════════════════════════════════════════════════════
# 🔬 ML FLOOD LAB
# ══════════════════════════════════════════════════════════
elif "ml" in page.lower() and "flood" in page.lower() or "lab" in page.lower() and "ml" in page.lower():
    st.markdown('<div class="sec-title">🔬 ML Flood Lab — Full Diagnostic Suite</div>',unsafe_allow_html=True)
    flood_df3=make_synthetic_flood_data()
    model_choice=st.session_state.flood_model_choice; n_est3=st.session_state.n_estimators
    fl_mdl,fl_Xte3,fl_yte3,fl_feats3,fl_cv3,fl_auc3,fl_Xtr3,fl_ytr3=train_flood_model(
        model_choice,n_est3,hash(str(flood_df3.shape)))
    fl_acc3=round((fl_mdl.predict(fl_Xte3)==fl_yte3).mean()*100,2)
    icon_m3,_,col_m3=MODEL_META[model_choice]
    st.markdown(f"""<div class="accuracy-banner">
        <span style="font-size:2rem;">{icon_m3}</span>
        <div><div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:2px;">{model_choice} — Accuracy</div>
        <div style="font-size:1.8rem;font-weight:900;color:#22c55e;font-family:'Orbitron',monospace;">{fl_acc3}%</div></div>
        <div style="margin-left:auto;text-align:right;font-family:'Share Tech Mono',monospace;">
            <div style="font-size:0.65rem;color:#4a7090;">ROC-AUC</div>
            <div style="font-size:1.4rem;color:#38bdf8;font-family:'Orbitron',monospace;">{fl_auc3:.3f}</div>
        </div>
        <div style="margin-left:1rem;text-align:right;font-family:'Share Tech Mono',monospace;">
            <div style="font-size:0.65rem;color:#4a7090;">CV F1 (5-fold)</div>
            <div style="font-size:1rem;color:#a855f7;">{fl_cv3.mean():.3f} ± {fl_cv3.std():.3f}</div>
        </div>
    </div>""",unsafe_allow_html=True)

    lab_t1,lab_t2,lab_t3,lab_t4,lab_t5=st.tabs(["📈 ROC / PR Curves","🎯 Confusion Matrix","🏆 Model Comparison","🔍 Anomaly Detection","📋 Data Explorer"])

    with lab_t1:
        fl_proba=fl_mdl.predict_proba(fl_Xte3)[:,1]
        fpr3,tpr3,_=roc_curve(fl_yte3,fl_proba); prec3,rec3,_=precision_recall_curve(fl_yte3,fl_proba)
        c1,c2=st.columns(2)
        with c1:
            fig_roc3=go.Figure()
            fig_roc3.add_trace(go.Scatter(x=fpr3,y=tpr3,mode="lines",name=f"ROC (AUC={fl_auc3:.3f})",
                line=dict(color=col_m3,width=2.5),fill="tozeroy",fillcolor="rgba(56,189,248,0.06)"))
            fig_roc3.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(color="#334155",dash="dash",width=1))
            fig_roc3.update_layout(title=dict(text=f"ROC — {model_choice}",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.04)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                xaxis=dict(title="FPR",gridcolor="#0d2a44"),yaxis=dict(title="TPR",gridcolor="#0d2a44"),
                height=320,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_roc3,use_container_width=True)
        with c2:
            fig_pr3=go.Figure()
            fig_pr3.add_trace(go.Scatter(x=rec3,y=prec3,mode="lines",name="PR Curve",
                line=dict(color="#a855f7",width=2.5),fill="tozeroy",fillcolor="rgba(168,85,247,0.06)"))
            fig_pr3.update_layout(title=dict(text="Precision-Recall",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.04)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                xaxis=dict(title="Recall",gridcolor="#0d2a44"),yaxis=dict(title="Precision",gridcolor="#0d2a44"),
                height=320,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pr3,use_container_width=True)
        # Threshold sensitivity
        st.markdown('<div class="sec-title">⚙️ Threshold Sensitivity</div>',unsafe_allow_html=True)
        thresholds_range=np.linspace(0.1,0.9,50)
        precisions,recalls,f1s=[],[],[]
        for thr in thresholds_range:
            preds_thr=(fl_proba>=thr).astype(int)
            tp=((preds_thr==1)&(fl_yte3==1)).sum(); fp=((preds_thr==1)&(fl_yte3==0)).sum()
            fn=((preds_thr==0)&(fl_yte3==1)).sum()
            p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
            precisions.append(p); recalls.append(r); f1s.append(2*p*r/(p+r+1e-9))
        fig_thr=go.Figure()
        fig_thr.add_trace(go.Scatter(x=thresholds_range,y=precisions,name="Precision",line=dict(color="#38bdf8",width=2)))
        fig_thr.add_trace(go.Scatter(x=thresholds_range,y=recalls,name="Recall",line=dict(color="#a855f7",width=2)))
        fig_thr.add_trace(go.Scatter(x=thresholds_range,y=f1s,name="F1",line=dict(color="#22c55e",width=2)))
        fig_thr.add_vline(x=0.5,line_dash="dash",line_color="white",line_width=1,
            annotation_text=" Default 0.5",annotation_font={"family":"Share Tech Mono","size":9,"color":"white"})
        fig_thr.update_layout(title=dict(text="Threshold vs Precision/Recall/F1",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.04)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(title="Decision Threshold",gridcolor="#0d2a44"),yaxis=dict(title="Score",gridcolor="#0d2a44",range=[0,1]),
            height=300,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_thr,use_container_width=True)

    with lab_t2:
        fl_preds3=fl_mdl.predict(fl_Xte3)
        cm=confusion_matrix(fl_yte3,fl_preds3)
        fig_cm=go.Figure(go.Heatmap(z=cm,x=["Pred: No Flood","Pred: Flood"],y=["Actual: No Flood","Actual: Flood"],
            colorscale=[[0,"#020c1a"],[0.5,"#0d2a44"],[1,col_m3]],
            text=cm,texttemplate="%{text}",textfont=dict(size=18,family="Orbitron",color="white"),showscale=True))
        fig_cm.update_layout(title=dict(text=f"Confusion Matrix — {model_choice}",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            height=360,margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig_cm,use_container_width=True)
        cr=classification_report(fl_yte3,fl_preds3,target_names=["No Flood","Flood"],output_dict=True)
        st.dataframe(pd.DataFrame(cr).T.round(3),use_container_width=True)

    with lab_t3:
        comp_df2,roc_data2,_,_=train_all_flood_models(hash(str(flood_df3.shape)))
        best_auc=comp_df2["ROC-AUC"].max()
        bar_colors=[col_m3 if row["Model"]==model_choice else "#22c55e" if row["ROC-AUC"]==best_auc else "#1e3a5f" for _,row in comp_df2.iterrows()]
        fig_comp=go.Figure(go.Bar(x=comp_df2["Model"],y=comp_df2["ROC-AUC"],
            marker=dict(color=bar_colors,cornerradius=6),
            text=[f"{v:.3f}" for v in comp_df2["ROC-AUC"]],textposition="auto",textfont=dict(family="Share Tech Mono",size=10)))
        fig_comp.update_layout(title=dict(text="8-Model ROC-AUC Comparison",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(gridcolor="#0d2a44",tickangle=-25),yaxis=dict(title="ROC-AUC",gridcolor="#0d2a44",range=[0,1.05]),
            height=320,margin=dict(t=40,l=10,r=10,b=60))
        st.plotly_chart(fig_comp,use_container_width=True)
        st.dataframe(comp_df2.reset_index(drop=True),use_container_width=True)

    with lab_t4:
        st.markdown('<div class="sec-title">🔍 Anomaly Detection (Isolation Forest)</div>',unsafe_allow_html=True)
        numeric_cols=flood_df3.select_dtypes(include=np.number).columns.tolist()
        feat_cols=[c for c in numeric_cols if c!="Flood Occurred"]
        iso_data=flood_df3[feat_cols].copy()
        scaler3=StandardScaler(); iso_scaled=scaler3.fit_transform(iso_data)
        iso_forest=IsolationForest(contamination=0.05,random_state=42)
        iso_labels=iso_forest.fit_predict(iso_scaled); iso_scores=iso_forest.decision_function(iso_scaled)
        result3=iso_data.copy()
        result3["Anomaly"]=np.where(iso_labels==-1,"⚠️ Anomaly","✅ Normal")
        result3["Score"]=iso_scores
        n_anom=(iso_labels==-1).sum()
        c1,c2,c3=st.columns(3)
        c1.metric("Total Records",len(iso_labels))
        c2.metric("Anomalies",n_anom,delta=f"{n_anom/len(iso_labels)*100:.1f}%")
        c3.metric("Normal",(iso_labels==1).sum())
        pca3=PCA(n_components=2,random_state=42); coords=pca3.fit_transform(iso_scaled)
        fig_pca=go.Figure()
        for lbl,color_a in [("✅ Normal","#22c55e"),("⚠️ Anomaly","#ef4444")]:
            mask=result3["Anomaly"]==lbl
            fig_pca.add_trace(go.Scatter(x=coords[mask,0],y=coords[mask,1],mode="markers",name=lbl,
                marker=dict(size=5 if "Normal" in lbl else 10,color=color_a,opacity=0.6 if "Normal" in lbl else 0.9,
                            line=dict(width=1,color="white") if "Anomaly" in lbl else dict(width=0))))
        fig_pca.update_layout(title=dict(text="PCA 2D — Anomaly Clusters",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.04)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(title="PC1",gridcolor="#0d2a44"),yaxis=dict(title="PC2",gridcolor="#0d2a44"),
            height=350,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_pca,use_container_width=True)
        st.dataframe(result3[result3["Anomaly"]=="⚠️ Anomaly"].head(20),use_container_width=True)

    with lab_t5:
        st.markdown('<div class="sec-title">📋 Flood Dataset Explorer</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.metric("Total Rows",len(flood_df3))
            st.metric("Flood Events",flood_df3["Flood Occurred"].sum())
            st.metric("Flood Rate",f"{flood_df3['Flood Occurred'].mean()*100:.1f}%")
        with c2:
            fig_dist=go.Figure(go.Histogram(x=flood_df3["Rainfall (mm)"],nbinsx=40,marker_color="#38bdf8",opacity=0.75))
            fig_dist.update_layout(title=dict(text="Rainfall Distribution",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(gridcolor="#0d2a44"),height=260,margin=dict(t=40,l=10,r=10,b=10))
            st.plotly_chart(fig_dist,use_container_width=True)
        st.dataframe(flood_df3.describe().round(2),use_container_width=True)
        num_flood=flood_df3.select_dtypes(include=np.number); corr_mat=num_flood.corr()
        fig_corr=go.Figure(go.Heatmap(z=corr_mat.values,x=corr_mat.columns,y=corr_mat.columns,
            colorscale="RdBu_r",zmid=0,text=corr_mat.round(2).values,texttemplate="%{text}",
            textfont=dict(size=8),showscale=True))
        fig_corr.update_layout(title=dict(text="Feature Correlation Matrix",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            height=450,margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig_corr,use_container_width=True)

# ══════════════════════════════════════════════════════════
# 🎯 WHAT-IF SCENARIOS
# ══════════════════════════════════════════════════════════
elif "what" in page.lower() or "scenario" in page.lower():
    st.markdown('<div class="sec-title">🎯 What-If Scenario Engine — Multi-Hazard Simulation</div>',unsafe_allow_html=True)
    st.markdown("""<div class="insight-panel">
        <b>🔬 How it works:</b> Adjust sliders → all 6 hazard physics models update instantly.
        Charts reflect real inputs from the same equations used in the prediction pages.
    </div>""",unsafe_allow_html=True)
    sc_c1,sc_c2,sc_c3=st.columns(3)
    with sc_c1:
        wi_rainfall=st.slider("🌧️ Rainfall (mm/day)",0.0,300.0,st.session_state.whatif_rainfall,5.0,key="wi_rain")
        wi_wind=st.slider("💨 Wind Speed (km/h)",0.0,350.0,st.session_state.whatif_wind,5.0,key="wi_wind")
    with sc_c2:
        wi_mag=st.slider("🌋 Seismic Magnitude",1.0,9.9,st.session_state.whatif_mag,0.1,key="wi_mag")
        wi_temp=st.slider("🌡️ Temperature (°C)",0.0,55.0,35.0,1.0,key="wi_temp")
    with sc_c3:
        wi_spi=st.slider("🏜️ SPI Index",-3.0,2.0,st.session_state.whatif_spi,0.1,key="wi_spi")
        wi_humid=st.slider("💧 Humidity (%)",0.0,100.0,45.0,1.0,key="wi_humid")
    st.session_state.whatif_rainfall=wi_rainfall; st.session_state.whatif_wind=wi_wind
    st.session_state.whatif_mag=wi_mag; st.session_state.whatif_spi=wi_spi

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚡ Live Hazard Risk Assessment</div>',unsafe_allow_html=True)

    flood_score=min(1.0,(wi_rainfall/200)*0.4+(wi_humid/100)*0.2+0.1)
    flood_risk="Critical" if flood_score>0.75 else "High" if flood_score>0.5 else "Moderate" if flood_score>0.25 else "Low"
    eq_risk="Critical" if wi_mag>=7.0 else "High" if wi_mag>=5.0 else "Moderate" if wi_mag>=3.0 else "Low"
    ts_risk="Critical" if wi_mag>=7.5 else "High" if wi_mag>=6.5 else "Moderate" if wi_mag>=5.5 else "Low"
    cyc_cat=0 if wi_wind<63 else 1 if wi_wind<119 else 2 if wi_wind<154 else 3 if wi_wind<178 else 4 if wi_wind<209 else 5
    cyc_risk=["Low","Moderate","High","High","Critical","Critical"][cyc_cat]
    fwi_val=max(0,(wi_temp-20)*1.5+(100-wi_humid)*0.5+wi_wind*0.3-wi_rainfall*0.4)
    wf_risk="Critical" if fwi_val>80 else "High" if fwi_val>50 else "Moderate" if fwi_val>25 else "Low"
    dr_sev="Extreme" if wi_spi<-2 else "Severe" if wi_spi<-1.5 else "Moderate" if wi_spi<-1 else "Mild" if wi_spi<0 else "No Drought"
    dr_risk="Critical" if wi_spi<-2 else "High" if wi_spi<-1.5 else "Moderate" if wi_spi<-1 else "Low"

    hazards=[("🌋","Earthquake",f"Mw {wi_mag:.1f}",eq_risk,"#ff7700"),
             ("🌊","Flood",f"Rain {wi_rainfall:.0f}mm",flood_risk,"#0099ff"),
             ("🌀","Cyclone",f"Cat-{cyc_cat} ({wi_wind:.0f}km/h)",cyc_risk,"#aa00ff"),
             ("🔥","Wildfire",f"FWI {fwi_val:.0f}",wf_risk,"#ff4400"),
             ("🌊","Tsunami",f"Mw {wi_mag:.1f} trigger",ts_risk,"#00aaff"),
             ("🏜️","Drought",dr_sev,dr_risk,"#cc8800")]
    hz_cols=st.columns(6)
    for col,(em,name,detail,risk,accent) in zip(hz_cols,hazards):
        rgb=accent.lstrip("#"); r2,g2,b2=int(rgb[0:2],16),int(rgb[2:4],16),int(rgb[4:6],16)
        risk_c={"Low":"#22c55e","Moderate":"#eab308","High":"#f97316","Critical":"#ef4444"}.get(risk,"#888")
        col.markdown(f"""<div class="metric-card" style="border-color:rgba({r2},{g2},{b2},0.3);height:150px;">
          <h4 style="white-space:nowrap;">{em} {name}</h4>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#64748b;">{detail}</div>
          <div style="font-family:'Orbitron',monospace;font-size:1.0rem;font-weight:900;color:{risk_c};margin-top:4px;">{risk}</div>
          <div class="mc-bar" style="background:linear-gradient(90deg,{risk_c},transparent);"></div>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="ani-divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🔭 Physics Simulations — All 6 Hazards Updated Live</div>',unsafe_allow_html=True)
    wi_tab1,wi_tab2,wi_tab3,wi_tab4,wi_tab5,wi_tab6=st.tabs(["🌋 Earthquake","🌊 Flood","🌀 Cyclone","🔥 Wildfire","🌊 Tsunami","🏜️ Drought"])
    with wi_tab1:
        col_eq={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}.get(eq_risk,"#ff7700")
        wc1,wc2=st.columns(2)
        with wc1: st.plotly_chart(make_energy_chart(wi_mag,col_eq),use_container_width=True)
        with wc2: st.plotly_chart(make_pga_chart(wi_mag,25.0,col_eq),use_container_width=True)
        st.plotly_chart(make_aftershock_chart(wi_mag,col_eq),use_container_width=True)
    with wi_tab2:
        col_fl={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}.get(flood_risk,"#0099ff")
        wc1,wc2=st.columns(2)
        fig_i,_=make_inundation_chart(wi_rainfall,3.0+wi_rainfall/50,80,col_fl)
        with wc1: st.plotly_chart(fig_i,use_container_width=True)
        with wc2: st.plotly_chart(make_river_hydrograph(wi_rainfall,3.0+wi_rainfall/50,col_fl),use_container_width=True)
    with wi_tab3:
        col_cy="#aa00ff"
        wc1,wc2=st.columns(2)
        with wc1: st.plotly_chart(make_pressure_profile(max(880,1013-wi_wind*0.4),cyc_cat,col_cy),use_container_width=True)
        with wc2: st.plotly_chart(make_storm_surge_chart(cyc_cat,20.0,col_cy),use_container_width=True)
    with wi_tab4:
        col_wf={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}.get(wf_risk,"#ff4400")
        wc1,wc2=st.columns(2)
        fig_s=_fire_spread_fig(wi_temp,wi_humid,wi_wind,80.0,col_wf)
        with wc1: st.plotly_chart(fig_s,use_container_width=True)
        fig_fl=_flame_spotting_fig(wi_wind,wi_temp,wi_humid,col_wf)
        with wc2: st.plotly_chart(fig_fl,use_container_width=True)
        st.plotly_chart(make_fwi_gauge_chart(fwi_val,col_wf),use_container_width=True)
    with wi_tab5:
        col_ts={"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}.get(ts_risk,"#00aaff")
        wc1,wc2=st.columns(2)
        with wc1: st.plotly_chart(make_wave_propagation_chart(wi_mag,4000,col_ts),use_container_width=True)
        fig_r,_=make_runup_chart(max(0.1,0.001*10**(0.5*(wi_mag-7))),150,col_ts)
        with wc2: st.plotly_chart(fig_r,use_container_width=True)
    with wi_tab6:
        col_dr={"Low":"#22c55e","Moderate":"#eab308","High":"#f97316","Critical":"#ef4444"}.get(dr_risk,"#cc8800")
        wc1,wc2=st.columns(2)
        with wc1: st.plotly_chart(make_spi_timeseries(wi_spi,col_dr),use_container_width=True)
        with wc2: st.plotly_chart(make_groundwater_chart(wi_spi,30.0-wi_rainfall*0.1,wi_temp-25,col_dr),use_container_width=True)
        st.plotly_chart(make_crop_impact_chart(dr_sev,col_dr),use_container_width=True)

    # Cascade narrative
    all_risks={"Earthquake":eq_risk,"Flood":flood_risk,"Cyclone":cyc_risk,"Wildfire":wf_risk,"Tsunami":ts_risk,"Drought":dr_risk}
    critical_list=[k for k,v in all_risks.items() if v=="Critical"]
    high_list=[k for k,v in all_risks.items() if v=="High"]
    if critical_list: cascade_msg=f'<span class="ins-warn">⚠️ CRITICAL: <b>{", ".join(critical_list)}</b> — Immediate coordination required.</span>'
    elif high_list: cascade_msg=f'<span style="color:#f97316;">⚡ HIGH: <b>{", ".join(high_list)}</b> — Elevated monitoring recommended.</span>'
    else: cascade_msg='<span class="ins-ok">✅ All hazards MODERATE or below under current scenario.</span>'
    st.markdown(f"""<div class="insight-panel">
        <b>📋 Cascade Risk Summary</b> — Mw {wi_mag:.1f} · {wi_rainfall:.0f}mm rain · {wi_wind:.0f}km/h wind · {wi_temp:.0f}°C · SPI {wi_spi:.1f}<br><br>
        {cascade_msg}
    </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(f"""<style>
.footer-wrap{{margin-top:2rem;background:linear-gradient(135deg,#020c18,#030e1c);
    border:1px solid rgba(56,189,248,0.08);border-radius:16px;padding:1.4rem 2rem;position:relative;overflow:hidden;}}
.footer-wrap::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,#38bdf8,#6366f1,#a855f7,transparent);
    background-size:200% auto;animation:shimmerSlide 3s linear infinite;}}
.footer-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1rem;align-items:start;}}
.footer-title{{font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;letter-spacing:3px;color:#38bdf8;margin:0 0 0.3rem;}}
.footer-tagline{{font-size:0.68rem;color:#334155;letter-spacing:1px;margin:0;}}
.footer-col-title{{font-size:0.62rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#475569;margin:0 0 0.5rem;}}
.footer-item{{font-size:0.72rem;color:#334155;margin:0.2rem 0;display:flex;align-items:center;gap:0.4rem;}}
.footer-item b{{color:#64748b;}}
.footer-bottom{{border-top:1px solid rgba(255,255,255,0.04);margin-top:1rem;padding-top:0.8rem;
    display:flex;justify-content:space-between;align-items:center;font-size:0.65rem;color:#1e3a5f;letter-spacing:1px;}}
.footer-status{{display:inline-flex;align-items:center;gap:0.4rem;
    background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
    border-radius:999px;padding:0.15rem 0.6rem;font-size:0.62rem;font-weight:700;color:#22c55e;letter-spacing:1px;}}
.footer-dot{{width:5px;height:5px;border-radius:50%;background:#22c55e;}}
</style>
<div class="footer-wrap">
    <div class="footer-grid">
        <div><p class="footer-title">🛰️ SENTINEL</p><p class="footer-tagline">AI · v3.0.0 · Multi-Hazard Intelligence</p></div>
        <div><p class="footer-col-title">🤖 ML Stack</p>
            <div class="footer-item">🌲 <b>Random Forest</b></div>
            <div class="footer-item">🚀 <b>Gradient Boosting</b></div>
            <div class="footer-item">🌳 <b>Extra Trees</b> + 5 more</div></div>
        <div><p class="footer-col-title">📡 Data Sources</p>
            <div class="footer-item">🌍 <b>USGS</b> Earthquake Feed</div>
            <div class="footer-item">🌊 <b>Open-Meteo</b> Flood API</div>
            <div class="footer-item">🔥 <b>NASA FIRMS</b> Fires</div></div>
        <div><p class="footer-col-title">⚙️ System</p>
            <div class="footer-item">🐍 <b>Python</b> + Streamlit</div>
            <div class="footer-item">📊 <b>Plotly</b> Visualizations</div>
            <div class="footer-item">🗄️ <b>SQLite</b> + SQLAlchemy</div></div>
    </div>
    <div class="footer-bottom">
        <span>© 2026 SENTINEL AI · Built with Streamlit · Research & educational use</span>
        <span class="footer-status"><span class="footer-dot"></span> System Active · {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>""",unsafe_allow_html=True)
