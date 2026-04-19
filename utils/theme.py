"""
SENTINEL v2.0 — Global CSS, Animations & Theme System
Mind-blowing sci-fi UI with particle effects, holographic elements,
neural network backgrounds, and fluid animations.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Exo+2:ital,wght@0,200;0,400;0,600;1,300&display=swap');

/* ═══════════════════════════════════════════════
   ROOT VARIABLES
═══════════════════════════════════════════════ */
:root {
  --c-bg:        #020812;
  --c-bg2:       #040f1e;
  --c-bg3:       #071525;
  --c-card:      #061220;
  --c-blue:      #00d4ff;
  --c-teal:      #00ffcc;
  --c-purple:    #9d00ff;
  --c-red:       #ff2244;
  --c-orange:    #ff7700;
  --c-yellow:    #ffd700;
  --c-green:     #00ff88;
  --c-text:      #c8e6ff;
  --c-muted:     #4a7090;
  --c-border:    #0d2a44;
  --glow-b:      0 0 25px rgba(0,212,255,0.4);
  --glow-g:      0 0 25px rgba(0,255,136,0.4);
  --glow-r:      0 0 25px rgba(255,34,68,0.4);
  --glow-p:      0 0 25px rgba(157,0,255,0.4);
}

/* ═══════════════════════════════════════════════
   STREAMLIT OVERRIDES
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton, .stToolbar { display: none !important; }
.block-container { padding: 0 1rem 2rem !important; max-width: 100% !important; }
.stApp { background: var(--c-bg) !important; }

/* ═══════════════════════════════════════════════
   ANIMATED DEEP-SPACE BACKGROUND
═══════════════════════════════════════════════ */
.stApp {
  background:
    radial-gradient(ellipse 120% 80% at 10% 20%, rgba(0,60,120,0.15) 0%, transparent 60%),
    radial-gradient(ellipse 80% 120% at 90% 80%, rgba(80,0,160,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 60% 60% at 50% 50%, rgba(0,20,50,0.8) 0%, transparent 100%),
    #020812 !important;
  font-family: 'Rajdhani', sans-serif;
  color: var(--c-text);
  overflow-x: hidden;
}

/* Animated grid lines */
.stApp::before {
  content: "";
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
  background-size: 60px 60px;
  animation: gridScroll 20s linear infinite;
  pointer-events: none; z-index: 0;
}

@keyframes gridScroll {
  0%   { background-position: 0 0; }
  100% { background-position: 60px 60px; }
}

/* Scanline effect */
.stApp::after {
  content: "";
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,212,255,0.012) 3px, rgba(0,212,255,0.012) 4px
  );
  pointer-events: none; z-index: 1;
  animation: scanMove 12s linear infinite;
}
@keyframes scanMove {
  0%   { transform: translateY(0); }
  100% { transform: translateY(60px); }
}

/* ═══════════════════════════════════════════════
   TOP BANNER — HOLOGRAPHIC
═══════════════════════════════════════════════ */
.sentinel-banner {
  position: relative;
  background: linear-gradient(135deg, #020f1f 0%, #041525 50%, #020f1f 100%);
  border-bottom: 1px solid rgba(0,212,255,0.3);
  padding: 16px 28px;
  margin: 0 -1rem 1.5rem;
  display: flex; align-items: center; gap: 20px;
  overflow: hidden;
  box-shadow: 0 4px 40px rgba(0,212,255,0.15), inset 0 1px 0 rgba(0,212,255,0.1);
}

/* Animated sweep light */
.sentinel-banner::before {
  content: "";
  position: absolute; top: 0; left: -120%;
  width: 60%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.06), rgba(0,255,204,0.04), transparent);
  animation: bannerSweep 5s ease-in-out infinite;
}
@keyframes bannerSweep { 0%,100%{left:-120%} 50%{left:160%} }

/* Corner hex markers */
.sentinel-banner::after {
  content: "";
  position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
  width: 80px; height: 40px;
  background:
    linear-gradient(60deg, transparent 30%, rgba(0,212,255,0.15) 30%, rgba(0,212,255,0.15) 70%, transparent 70%),
    linear-gradient(-60deg, transparent 30%, rgba(0,212,255,0.1) 30%, rgba(0,212,255,0.1) 70%, transparent 70%);
}

.banner-logo {
  font-family: 'Orbitron', monospace;
  font-size: 2.2rem; font-weight: 900;
  letter-spacing: 6px;
  background: linear-gradient(90deg, #00d4ff 0%, #00ffcc 30%, #9d00ff 60%, #00d4ff 100%);
  background-size: 300% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: logoShimmer 4s linear infinite;
  text-shadow: none;
  filter: drop-shadow(0 0 20px rgba(0,212,255,0.5));
}
@keyframes logoShimmer { 0%{background-position:0%} 100%{background-position:300%} }

.banner-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem; letter-spacing: 3px;
  color: var(--c-muted); text-transform: uppercase;
  margin-top: 3px;
}

.banner-right {
  margin-left: auto;
  display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
}

.live-indicator {
  display: flex; align-items: center; gap: 6px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; color: var(--c-green);
  letter-spacing: 2px;
}
.live-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--c-green); box-shadow: 0 0 8px var(--c-green);
  animation: livePulse 1.4s ease-in-out infinite;
}
@keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.6)} }

.clock-display {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem; color: rgba(0,212,255,0.6);
  letter-spacing: 1px;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #030d1a 0%, #020812 100%) !important;
  border-right: 1px solid var(--c-border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.sidebar-logo-section {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--c-border);
  margin-bottom: 8px;
  text-align: center;
  background: linear-gradient(180deg, rgba(0,212,255,0.04), transparent);
}
.sidebar-logo {
  font-family: 'Orbitron', monospace;
  font-size: 1.1rem; font-weight: 700;
  color: var(--c-blue); letter-spacing: 4px;
}
.sidebar-version {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem; color: var(--c-muted);
  letter-spacing: 2px; margin-top: 2px;
}

.nav-section-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; color: var(--c-muted);
  letter-spacing: 3px; text-transform: uppercase;
  padding: 8px 16px 4px;
}

/* ═══════════════════════════════════════════════
   METRIC / STAT CARDS
═══════════════════════════════════════════════ */
.stat-card {
  position: relative;
  background: linear-gradient(135deg, var(--c-card), var(--c-bg2));
  border: 1px solid var(--c-border);
  border-radius: 12px;
  padding: 18px 16px;
  text-align: center;
  overflow: hidden;
  transition: all 0.35s cubic-bezier(0.23, 1, 0.32, 1);
  cursor: default;
}
.stat-card::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--card-accent, #00d4ff), transparent);
  opacity: 0.6;
  animation: topGlow 3s ease-in-out infinite;
  animation-delay: var(--card-delay, 0s);
}
@keyframes topGlow { 0%,100%{opacity:0.2} 50%{opacity:1} }

.stat-card:hover {
  border-color: var(--card-accent, #00d4ff);
  box-shadow: 0 0 30px rgba(0,212,255,0.2), 0 8px 32px rgba(0,0,0,0.4);
  transform: translateY(-4px) scale(1.02);
}

.stat-icon { font-size: 1.8rem; margin-bottom: 6px; display: block; }

.stat-value {
  font-family: 'Orbitron', monospace;
  font-size: 2.2rem; font-weight: 700;
  color: var(--card-accent, #00d4ff);
  text-shadow: 0 0 20px currentColor;
  line-height: 1;
}
.stat-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; letter-spacing: 2px;
  color: var(--c-muted); text-transform: uppercase;
  margin-top: 5px;
}
.stat-delta {
  font-size: 0.7rem; color: var(--c-green);
  margin-top: 3px; font-family: 'Share Tech Mono', monospace;
}

/* ═══════════════════════════════════════════════
   DISASTER SECTION HEADERS
═══════════════════════════════════════════════ */
.dis-header {
  position: relative;
  display: flex; align-items: center; gap: 20px;
  padding: 22px 28px;
  border-radius: 16px;
  margin-bottom: 24px;
  border: 1px solid var(--dh-color, #00d4ff);
  background: var(--dh-bg, linear-gradient(135deg, #041525, #020f1f));
  overflow: hidden;
  animation: headerIn 0.6s ease-out;
}
@keyframes headerIn { 0%{opacity:0;transform:translateY(-15px)} 100%{opacity:1;transform:translateY(0)} }

.dis-header::before {
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 100% at 80% 50%, rgba(255,255,255,0.03), transparent);
}
.dis-header::after {
  content: "";
  position: absolute; top: 0; left: -100%;
  width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
  animation: headerSweep 6s ease-in-out infinite;
}
@keyframes headerSweep { 0%,100%{left:-100%} 60%{left:200%} }

.dis-icon {
  font-size: 3.2rem;
  filter: drop-shadow(0 0 18px var(--dh-color, #00d4ff));
  animation: iconFloat 4s ease-in-out infinite;
}
@keyframes iconFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }

.dis-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.5rem; font-weight: 700;
  color: #fff; margin: 0; letter-spacing: 2px;
}
.dis-subtitle {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.9rem; color: rgba(200,230,255,0.6);
  margin: 4px 0 0; letter-spacing: 1px;
}

.dis-badges {
  margin-left: auto; display: flex; flex-direction: column;
  align-items: flex-end; gap: 5px;
}
.dis-badge {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; letter-spacing: 2px;
  padding: 3px 10px; border-radius: 3px;
  border: 1px solid var(--dh-color, #00d4ff);
  color: var(--dh-color, #00d4ff);
  background: rgba(0,212,255,0.07);
}

/* ═══════════════════════════════════════════════
   PREDICTION RESULT CARD — HOLOGRAPHIC
═══════════════════════════════════════════════ */
.pred-card {
  position: relative;
  background: linear-gradient(135deg, var(--c-card), #030e1c);
  border: 2px solid var(--pc-color, #00d4ff);
  border-radius: 20px;
  padding: 30px 24px;
  margin: 20px 0;
  text-align: center;
  overflow: hidden;
  animation: predCardIn 0.6s cubic-bezier(0.23, 1, 0.32, 1);
  box-shadow: 0 0 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.03) inset;
}
@keyframes predCardIn {
  0%  { opacity:0; transform:scale(0.92) translateY(20px); }
  100%{ opacity:1; transform:scale(1) translateY(0); }
}

/* Holographic shimmer */
.pred-card::before {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(
    135deg,
    rgba(255,255,255,0) 0%,
    rgba(255,255,255,0.02) 30%,
    rgba(255,255,255,0.04) 50%,
    rgba(255,255,255,0.02) 70%,
    rgba(255,255,255,0) 100%
  );
  animation: holoShimmer 4s ease-in-out infinite;
}
@keyframes holoShimmer {
  0%,100%{ background-position: -200% 0; opacity:0.5; }
  50%    { background-position: 200% 0; opacity:1; }
}

/* Corner accents */
.pred-card::after {
  content: "";
  position: absolute; bottom: 0; right: 0;
  width: 60px; height: 60px;
  background: radial-gradient(circle at 100% 100%, var(--pc-color, #00d4ff) 0%, transparent 70%);
  opacity: 0.15;
}

.pred-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; letter-spacing: 4px;
  color: var(--c-muted); text-transform: uppercase;
  display: block; margin-bottom: 12px;
}
.pred-value {
  font-family: 'Orbitron', monospace;
  font-size: 4rem; font-weight: 900; line-height: 1;
  color: var(--pc-color, #00d4ff);
  text-shadow: 0 0 40px currentColor, 0 0 80px currentColor;
  display: block; margin-bottom: 6px;
  animation: valueGlow 2s ease-in-out infinite;
}
@keyframes valueGlow { 0%,100%{filter:brightness(1)} 50%{filter:brightness(1.2)} }

.pred-unit {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem; color: var(--c-muted);
  display: block; margin-bottom: 20px;
}

.pred-tags {
  display: flex; flex-wrap: wrap;
  justify-content: center; gap: 10px;
}
.pred-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  padding: 5px 14px; border-radius: 20px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  color: var(--c-text);
}
.pred-tag.accent {
  border-color: var(--pc-color, #00d4ff);
  background: rgba(0,212,255,0.08);
  color: var(--pc-color, #00d4ff);
}

/* ═══════════════════════════════════════════════
   ALERT ITEMS
═══════════════════════════════════════════════ */
.alert-row {
  display: flex; align-items: flex-start; gap: 14px;
  padding: 12px 16px; border-radius: 10px;
  margin: 6px 0;
  border-left: 3px solid var(--al-color, #00d4ff);
  background: linear-gradient(90deg, rgba(0,212,255,0.06) 0%, transparent 100%);
  transition: all 0.25s;
  animation: alertIn 0.4s ease;
}
@keyframes alertIn { 0%{opacity:0;transform:translateX(-12px)} 100%{opacity:1;transform:translateX(0)} }
.alert-row:hover { background: linear-gradient(90deg, rgba(0,212,255,0.1) 0%, transparent 100%); transform:translateX(3px); }

.alert-row.critical { --al-color:#ff2244; background:linear-gradient(90deg,rgba(255,34,68,0.08) 0%,transparent); }
.alert-row.high     { --al-color:#ff7700; background:linear-gradient(90deg,rgba(255,119,0,0.08) 0%,transparent); }
.alert-row.moderate { --al-color:#ffd700; background:linear-gradient(90deg,rgba(255,215,0,0.07) 0%,transparent); }
.alert-row.low      { --al-color:#00ff88; background:linear-gradient(90deg,rgba(0,255,136,0.06) 0%,transparent); }

.alert-type {
  font-family: 'Orbitron', monospace;
  font-size: 0.7rem; letter-spacing: 1px;
  color: var(--al-color);
}
.alert-loc {
  font-size: 0.78rem; color: var(--c-muted);
  font-family: 'Share Tech Mono', monospace;
}
.alert-msg { font-size: 0.82rem; color: var(--c-text); margin: 2px 0; }
.alert-time { font-size: 0.62rem; color: #3a5a70; font-family: 'Share Tech Mono', monospace; }

/* ═══════════════════════════════════════════════
   SECTION TITLES
═══════════════════════════════════════════════ */
.sec-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.75rem; letter-spacing: 4px;
  color: var(--c-blue); text-transform: uppercase;
  padding: 14px 0 10px;
  border-bottom: 1px solid var(--c-border);
  margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}
.sec-title::before {
  content: "//";
  color: var(--c-muted); font-size: 0.7rem;
}

/* ═══════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--c-bg2) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 3px !important;
  border: 1px solid var(--c-border) !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.75rem !important; letter-spacing: 1.5px !important;
  color: var(--c-muted) !important;
  border-radius: 7px !important;
  padding: 8px 18px !important;
  background: transparent !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: var(--c-card) !important;
  color: var(--c-blue) !important;
  box-shadow: 0 0 15px rgba(0,212,255,0.15), inset 0 1px 0 rgba(0,212,255,0.1) !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.stButton > button {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.72rem !important; letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
  position: relative !important; overflow: hidden !important;
}
.stButton > button::after {
  content: "" !important;
  position: absolute !important; inset: 0 !important;
  background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent) !important;
  opacity: 0 !important; transition: opacity 0.3s !important;
}
.stButton > button:hover::after { opacity: 1 !important; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00d4ff, #0066aa) !important;
  color: #000 !important; border: none !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.35), 0 4px 16px rgba(0,0,0,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 0 35px rgba(0,212,255,0.6), 0 8px 24px rgba(0,0,0,0.5) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ═══════════════════════════════════════════════
   INPUTS & SLIDERS
═══════════════════════════════════════════════ */
.stSlider [data-baseweb="slider"] { padding: 5px 0; }
.stSlider [data-baseweb="thumb"] { background: var(--c-blue) !important; border: 2px solid #000 !important; }
.stSlider [data-baseweb="track-fill"] { background: linear-gradient(90deg, var(--c-purple), var(--c-blue)) !important; }

.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: var(--c-bg2) !important;
  border: 1px solid var(--c-border) !important;
  color: var(--c-text) !important;
  border-radius: 8px !important;
  font-family: 'Share Tech Mono', monospace !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--c-blue) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ═══════════════════════════════════════════════
   DATAFRAME / TABLE
═══════════════════════════════════════════════ */
.stDataFrame { border: 1px solid var(--c-border) !important; border-radius: 10px !important; }
.stDataFrame [data-testid="StyledDataFrameDataCell"] {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.78rem !important;
}

/* ═══════════════════════════════════════════════
   PROGRESS & SPINNERS
═══════════════════════════════════════════════ */
.stProgress > div > div { background: linear-gradient(90deg, var(--c-purple), var(--c-blue), var(--c-teal)) !important; border-radius: 4px !important; }

/* ═══════════════════════════════════════════════
   RISK LEVEL BADGES
═══════════════════════════════════════════════ */
.risk-badge {
  display: inline-block;
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem; letter-spacing: 2px;
  padding: 4px 12px; border-radius: 4px;
  text-transform: uppercase;
}
.risk-low      { background:rgba(0,255,136,0.12);  border:1px solid #00ff88; color:#00ff88; }
.risk-moderate { background:rgba(255,215,0,0.12);  border:1px solid #ffd700; color:#ffd700; }
.risk-high     { background:rgba(255,119,0,0.12);  border:1px solid #ff7700; color:#ff7700; }
.risk-critical { background:rgba(255,34,68,0.12);  border:1px solid #ff2244; color:#ff2244;
                 animation: criticalPulse 1.2s ease-in-out infinite; }
@keyframes criticalPulse { 0%,100%{box-shadow:0 0 0 0 rgba(255,34,68,0)} 50%{box-shadow:0 0 12px 2px rgba(255,34,68,0.5)} }

/* ═══════════════════════════════════════════════
   TERMINAL / CONSOLE BLOCK
═══════════════════════════════════════════════ */
.terminal-block {
  background: #01070f;
  border: 1px solid var(--c-border);
  border-radius: 10px;
  padding: 16px 20px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  color: var(--c-green);
  line-height: 1.8;
  position: relative;
}
.terminal-block::before {
  content: "● ● ●";
  position: absolute; top: 8px; left: 14px;
  font-size: 0.5rem; color: #334;
  letter-spacing: 4px;
}
.terminal-block pre { margin: 0; padding-top: 12px; white-space: pre-wrap; }

/* ═══════════════════════════════════════════════
   ANIMATED PARTICLE CANVAS PLACEHOLDER
═══════════════════════════════════════════════ */
.particle-bg {
  position: relative;
  height: 3px;
  background: linear-gradient(90deg,
    transparent 0%, var(--c-purple) 20%,
    var(--c-blue) 50%, var(--c-teal) 80%, transparent 100%);
  margin: 8px 0 20px;
  overflow: hidden;
  border-radius: 2px;
}
.particle-bg::after {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.6) 50%, transparent 100%);
  width: 30%; animation: particleScan 2.5s linear infinite;
}
@keyframes particleScan { 0%{left:-30%} 100%{left:130%} }

/* ═══════════════════════════════════════════════
   WORLD MAP SECTION
═══════════════════════════════════════════════ */
.map-container {
  border: 1px solid var(--c-border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 0 30px rgba(0,0,0,0.5), inset 0 0 60px rgba(0,20,40,0.3);
}

/* ═══════════════════════════════════════════════
   STORM / EVENT CARDS
═══════════════════════════════════════════════ */
.event-card {
  background: var(--c-card);
  border: 1px solid var(--c-border);
  border-left: 3px solid var(--ec-color, #00d4ff);
  border-radius: 8px;
  padding: 12px 16px; margin: 6px 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.8rem;
  transition: all 0.25s;
}
.event-card:hover {
  border-left-color: var(--ec-color, #00d4ff);
  box-shadow: 0 0 15px rgba(0,212,255,0.1);
  transform: translateX(3px);
}

/* ═══════════════════════════════════════════════
   TIPS / INFO BOXES
═══════════════════════════════════════════════ */
.info-box {
  background: rgba(0,212,255,0.06);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.85rem;
  margin: 12px 0;
}
.warn-box {
  background: rgba(255,215,0,0.06);
  border: 1px solid rgba(255,215,0,0.2);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.85rem;
  margin: 12px 0;
}
.danger-box {
  background: rgba(255,34,68,0.06);
  border: 1px solid rgba(255,34,68,0.2);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.85rem;
  margin: 12px 0;
  animation: dangerPulse 2s ease-in-out infinite;
}
@keyframes dangerPulse { 0%,100%{border-color:rgba(255,34,68,0.2)} 50%{border-color:rgba(255,34,68,0.5)} }

/* ═══════════════════════════════════════════════
   LOADING SKELETON
═══════════════════════════════════════════════ */
.skeleton {
  background: linear-gradient(90deg, var(--c-bg2) 25%, var(--c-bg3) 50%, var(--c-bg2) 75%);
  background-size: 400% 100%;
  animation: skeletonLoad 1.5s ease infinite;
  border-radius: 6px; height: 20px; margin: 6px 0;
}
@keyframes skeletonLoad { 0%{background-position:100%} 100%{background-position:-100%} }

/* ═══════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-blue); }

/* ═══════════════════════════════════════════════
   METRIC DELTA COLORS
═══════════════════════════════════════════════ */
[data-testid="stMetricDelta"] { font-family: 'Share Tech Mono', monospace !important; }

/* ═══════════════════════════════════════════════
   EXPANDER
═══════════════════════════════════════════════ */
.streamlit-expanderHeader {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 1px !important;
  background: var(--c-bg2) !important;
  border-radius: 8px !important;
}

/* ═══════════════════════════════════════════════
   DIVIDER LINE ANIMATED
═══════════════════════════════════════════════ */
.ani-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--c-blue), var(--c-purple), var(--c-blue), transparent);
  background-size: 300% 100%;
  animation: divMove 4s linear infinite;
  margin: 20px 0;
  border: none;
}
@keyframes divMove { 0%{background-position:0%} 100%{background-position:300%} }

/* ═══════════════════════════════════════════════
   FEATURE CARDS (home page)
═══════════════════════════════════════════════ */
.feature-card {
  background: linear-gradient(135deg, var(--c-card), var(--c-bg2));
  border: 1px solid var(--c-border);
  border-radius: 14px;
  padding: 20px;
  height: 100%;
  transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
  position: relative; overflow: hidden;
}
.feature-card:hover {
  border-color: var(--fc-color, #00d4ff);
  box-shadow: 0 0 25px rgba(0,212,255,0.12);
  transform: translateY(-5px);
}
.feature-card::before {
  content: "";
  position: absolute; top: 0; right: 0;
  width: 80px; height: 80px;
  background: radial-gradient(circle at 100% 0%, var(--fc-color, #00d4ff), transparent 70%);
  opacity: 0.08;
}
.fc-icon { font-size: 2.4rem; display: block; margin-bottom: 12px; }
.fc-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.85rem; letter-spacing: 1px; color: var(--fc-color, #00d4ff);
  margin-bottom: 8px;
}
.fc-body { font-size: 0.82rem; color: var(--c-muted); line-height: 1.6; }
</style>
"""


def render_banner(page_name: str = ""):
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
    st_html = f"""
    <div class="sentinel-banner">
      <div>
        <div class="banner-logo">🛰 SENTINEL</div>
        <div class="banner-sub">Multi-Hazard Disaster Prediction &amp; Early Warning System &nbsp;|&nbsp; v2.0</div>
      </div>
      <div class="banner-right">
        <div class="live-indicator"><div class="live-dot"></div>SYSTEM ONLINE</div>
        <div class="clock-display">{ts}</div>
        {"<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;color:#9d00ff;letter-spacing:2px;'>▶ " + page_name.upper() + "</div>" if page_name else ""}
      </div>
    </div>
    <div class="particle-bg"></div>
    """
    return st_html


RISK_COLORS = {
    "Low": "#00ff88", "Moderate": "#ffd700",
    "High": "#ff7700", "Critical": "#ff2244",
    "No Drought": "#00ff88", "Mild": "#aaff44",
    "Severe": "#ff7700", "Extreme": "#ff2244",
}

DISASTER_THEMES = {
    "earthquake": {"color": "#ff7700", "bg": "linear-gradient(135deg,#1e0a00,#100500)", "emoji": "🌋"},
    "flood":      {"color": "#0099ff", "bg": "linear-gradient(135deg,#001830,#000d1a)", "emoji": "🌊"},
    "cyclone":    {"color": "#aa00ff", "bg": "linear-gradient(135deg,#150028,#0a0014)", "emoji": "🌀"},
    "wildfire":   {"color": "#ff4400", "bg": "linear-gradient(135deg,#1e0800,#100400)", "emoji": "🔥"},
    "tsunami":    {"color": "#00aaff", "bg": "linear-gradient(135deg,#001a30,#000d20)", "emoji": "🌊"},
    "drought":    {"color": "#cc8800", "bg": "linear-gradient(135deg,#1a1200,#0d0900)", "emoji": "🏜️"},
}
