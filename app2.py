"""
SENTINEL v3 — Disaster Pages Patch
Adds: Predict buttons · History tabs · Live animated line graphs · Real-time feed
Replaces all render_*_page calls with full inline implementations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════

RISK_COLORS = {
    "Low":      "#22c55e",
    "Moderate": "#eab308",
    "High":     "#f97316",
    "Critical": "#ef4444",
    "Extreme":  "#dc143c",
}

def _risk_label(score, thresholds=(25, 50, 75)):
    if score < thresholds[0]: return "Low"
    if score < thresholds[1]: return "Moderate"
    if score < thresholds[2]: return "High"
    return "Critical"

def _risk_color(label):
    return RISK_COLORS.get(label, "#888")

def _base_layout(title="", height=280):
    return dict(
        title=dict(text=title, font={"family": "Orbitron", "size": 12, "color": "#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.03)",
        font=dict(family="Share Tech Mono", color="#8ab4d0"),
        xaxis=dict(gridcolor="#0d2a44", showgrid=True),
        yaxis=dict(gridcolor="#0d2a44", showgrid=True),
        height=height,
        margin=dict(t=45, l=10, r=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )

def _result_card(risk_label, score, detail_lines, accent):
    rc = _risk_color(risk_label)
    lines_html = "".join(f'<div style="margin:2px 0;font-size:0.72rem;color:#8ab4d0;">{l}</div>' for l in detail_lines)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#020c1a,#071528);
                border:2px solid {rc};border-radius:16px;padding:1.2rem 1.6rem;
                margin:0.6rem 0;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;
                  background:linear-gradient(90deg,{rc},{accent},{rc});"></div>
      <div style="display:flex;align-items:center;gap:1.2rem;">
        <div style="text-align:center;min-width:80px;">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;color:{rc};">{score:.0f}%</div>
          <div style="font-family:'Orbitron',monospace;font-size:0.65rem;color:{rc};letter-spacing:2px;">{risk_label.upper()}</div>
        </div>
        <div style="flex:1;border-left:1px solid rgba(255,255,255,0.08);padding-left:1rem;">
          {lines_html}
        </div>
      </div>
      <div style="margin-top:8px;height:4px;background:#0d2a44;border-radius:4px;">
        <div style="width:{score}%;height:100%;background:linear-gradient(90deg,{rc},transparent);border-radius:4px;
                    transition:width 1s ease;"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def _live_history_chart(history_key, ylabel, color, title, units=""):
    """Renders an animated line graph from session state history list."""
    data = st.session_state.get(history_key, [])
    if not data:
        st.markdown(f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#334155;padding:12px;">No predictions yet — run a prediction to see live feed.</div>', unsafe_allow_html=True)
        return
    times = [d["time"] for d in data]
    values = [d["value"] for d in data]
    risks = [d.get("risk", "Low") for d in data]
    risk_c = [_risk_color(r) for r in risks]

    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines",
        line=dict(color=color, width=0),
        fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
        showlegend=False, hoverinfo="skip"
    ))
    # Main line
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines+markers",
        line=dict(color=color, width=2.5, shape="spline"),
        marker=dict(size=[10 if r in ("Critical","High") else 6 for r in risks],
                    color=risk_c, line=dict(width=1.5, color="white")),
        hovertemplate=f"<b>%{{x}}</b><br>{ylabel}: %{{y:.1f}}{units}<extra></extra>",
        name=ylabel
    ))
    # Threshold lines
    if ylabel == "Risk Score":
        fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", line_width=1,
                      annotation_text=" Critical", annotation_font={"family":"Share Tech Mono","size":8,"color":"#ef4444"})
        fig.add_hline(y=50, line_dash="dash", line_color="#f97316", line_width=1,
                      annotation_text=" High", annotation_font={"family":"Share Tech Mono","size":8,"color":"#f97316"})

    layout = _base_layout(title, height=260)
    layout["yaxis"]["title"] = f"{ylabel} ({units})" if units else ylabel
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

def _append_history(key, value, risk, label=""):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "value": float(value),
        "risk": risk,
        "label": label,
    })
    # Keep only last 30
    if len(st.session_state[key]) > 30:
        st.session_state[key] = st.session_state[key][-30:]

def _history_table(key):
    data = st.session_state.get(key, [])
    if not data:
        return
    df = pd.DataFrame(data)
    df.index = range(1, len(df)+1)
    rc_map = {"Low":"🟢","Moderate":"🟡","High":"🟠","Critical":"🔴","Extreme":"🔴"}
    df["risk"] = df["risk"].apply(lambda r: f"{rc_map.get(r,'⚪')} {r}")
    st.dataframe(df.rename(columns={"time":"Time","value":"Score","risk":"Risk","label":"Details"}),
                 use_container_width=True)

def _predict_btn(key, label="⚡  RUN PREDICTION", color="#00d4ff"):
    return st.button(label, key=key, type="primary", use_container_width=True)

def _section(title):
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.75rem;font-weight:700;'
                f'color:#8ab4d0;letter-spacing:3px;text-transform:uppercase;margin:1.2rem 0 0.5rem;'
                f'border-bottom:1px solid #0d2a44;padding-bottom:6px;">{title}</div>', unsafe_allow_html=True)

def _live_pulse_bar(value, max_val, color):
    pct = min(100, value / max_val * 100)
    st.markdown(f"""
    <div style="background:#0d2a44;border-radius:6px;height:8px;margin:4px 0;overflow:hidden;">
      <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color},{color}88);
                  border-radius:6px;transition:width 0.8s ease;
                  box-shadow:0 0 8px {color}66;"></div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 🌋 EARTHQUAKE PAGE
# ═══════════════════════════════════════════════════════════

def render_earthquake_page_patched(db):
    from disasters.earthquake import make_energy_chart, make_pga_chart, make_aftershock_chart

    color = "#ff7700"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🌋 SEISMIC RISK ASSESSMENT ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.number_input("Latitude", -90.0, 90.0, 28.6, 0.1, key="eq_lat")
            lon = st.number_input("Longitude", -180.0, 180.0, 77.2, 0.1, key="eq_lon")
            depth = st.slider("Focal Depth (km)", 1, 700, 25, key="eq_depth")
        with c2:
            mag = st.slider("Expected Magnitude (Mw)", 1.0, 9.9, 6.5, 0.1, key="eq_mag")
            seismic_act = st.slider("Seismic Activity Index (0-10)", 0, 10, 7, key="eq_act")
            fault_dist = st.slider("Distance from Fault (km)", 0, 200, 15, key="eq_fault")
        with c3:
            hist_freq = st.slider("Historical Frequency (events/yr)", 0, 50, 12, key="eq_hist")
            tectonic = st.slider("Tectonic Stress (0-10)", 0, 10, 8, key="eq_tect")
            pop_density = st.number_input("Population Density (per km²)", 0, 50000, 8000, 100, key="eq_pop")

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="eq_clr", use_container_width=True):
                st.session_state["eq_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("eq_run_btn", "⚡  RUN EARTHQUAKE PREDICTION", color)

        if run:
            with st.spinner("Running seismic risk model…"):
                time.sleep(0.4)
                score = min(100, (
                    mag * 8 +
                    seismic_act * 2.5 +
                    tectonic * 2.0 +
                    hist_freq * 0.8 +
                    max(0, 10 - fault_dist / 10) * 1.5 +
                    max(0, 10 - depth / 70) * 1.5 +
                    np.random.normal(0, 3)
                ))
                score = max(0, score)
                risk = "Critical" if score > 75 else "High" if score > 50 else "Moderate" if score > 25 else "Low"
                energy = 10 ** (1.5 * mag + 4.8)
                pga = 10 ** (0.5 * mag - 1.5) * max(0.1, 1 - depth / 700)

                _append_history("eq_history", score, risk,
                                f"Mw{mag:.1f} d={depth}km lat={lat:.1f}")
                db.save_prediction("earthquake", {
                    "latitude": lat, "longitude": lon, "magnitude": mag,
                    "depth": depth, "risk_score": score
                }, risk, f"{lat:.2f}°N {lon:.2f}°E")

                _result_card(risk, score, [
                    f"📍 Location: {lat:.2f}°N, {lon:.2f}°E",
                    f"⚡ Magnitude: Mw {mag:.1f}  |  Depth: {depth} km",
                    f"🔥 Seismic Energy: {energy:.2e} J",
                    f"📈 Est. PGA: {pga:.3f} g  |  Tectonic Stress: {tectonic}/10",
                    f"🏙️ Population at risk: {pop_density:,}/km²",
                ], color)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2, pc3 = st.columns(3)
                with pc1: st.plotly_chart(make_energy_chart(mag, color), use_container_width=True)
                with pc2: st.plotly_chart(make_pga_chart(mag, float(depth), color), use_container_width=True)
                with pc3: st.plotly_chart(make_aftershock_chart(mag, color), use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — RISK SCORE OVER TIME")
        _live_history_chart("eq_history", "Risk Score", color, "🌋 Earthquake Risk Score Timeline", "%")
        _section("MAGNITUDE HISTORY")
        # magnitude line chart from history
        data = st.session_state.get("eq_history", [])
        if data:
            mags = [float(d["label"].split("Mw")[1].split(" ")[0]) if "Mw" in d.get("label","") else 0 for d in data]
            fig_m = go.Figure(go.Scatter(
                x=[d["time"] for d in data], y=mags,
                mode="lines+markers", line=dict(color="#ff4400", width=2, shape="spline"),
                marker=dict(size=8, color="#ff4400"), fill="tozeroy",
                fillcolor="rgba(255,68,0,0.07)",
            ))
            fig_m.update_layout(**_base_layout("Magnitude Over Predictions", 220))
            fig_m.update_layout(yaxis=dict(title="Mw", gridcolor="#0d2a44", range=[0,10]))
            st.plotly_chart(fig_m, use_container_width=True)
        _section("FULL LOG")
        _history_table("eq_history")

    with tab_live:
        _section("LIVE SEISMIC MONITOR")
        st.markdown("""<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#4a7090;margin-bottom:8px;">
        Auto-refreshing simulated seismic telemetry. Each data point = one sensor reading.</div>""", unsafe_allow_html=True)
        if st.button("▶  Start Live Seismic Stream", key="eq_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            for i in range(20):
                t = datetime.now().strftime("%H:%M:%S")
                val = 2.0 + np.random.exponential(1.2)
                live_data.append({"time": t, "mag": round(val, 2), "depth": np.random.randint(5, 150)})
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=[d["time"] for d in live_data],
                    y=[d["mag"] for d in live_data],
                    mode="lines+markers",
                    line=dict(color=color, width=2, shape="spline"),
                    marker=dict(size=7, color=[_risk_color("Critical" if m>5 else "High" if m>3.5 else "Moderate" if m>2.5 else "Low") for m in [d["mag"] for d in live_data]]),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.07)",
                    name="Magnitude"
                ))
                fig_live.update_layout(**_base_layout(f"🌋 Live Seismic Feed — {t}", 280))
                fig_live.update_layout(yaxis=dict(title="Magnitude (Mw)", gridcolor="#0d2a44", range=[0, 8]))
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.3)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        mag_c = st.slider("Magnitude for charts", 1.0, 9.9, 6.5, 0.1, key="eq_chart_mag")
        dep_c = st.slider("Depth for charts (km)", 1, 700, 30, key="eq_chart_dep")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.plotly_chart(make_energy_chart(mag_c, color), use_container_width=True)
        with cc2: st.plotly_chart(make_pga_chart(mag_c, float(dep_c), color), use_container_width=True)
        with cc3: st.plotly_chart(make_aftershock_chart(mag_c, color), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🌊 FLOOD PAGE
# ═══════════════════════════════════════════════════════════

def render_flood_page_patched(db):
    from disasters.flood import make_inundation_chart, make_river_hydrograph, make_soil_saturation_chart

    color = "#0099ff"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🌊 FLOOD RISK PREDICTION ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            rainfall = st.slider("Rainfall (mm/day)", 0.0, 300.0, 120.0, 5.0, key="fl_rain")
            river_level = st.slider("River Level (m)", 0.0, 20.0, 4.5, 0.1, key="fl_river")
            elevation = st.slider("Elevation (m)", 0.0, 3000.0, 80.0, 10.0, key="fl_elev")
        with c2:
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 75.0, 1.0, key="fl_humid")
            river_disc = st.slider("River Discharge (m³/s)", 0, 12000, 2500, 100, key="fl_disc")
            soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty"], key="fl_soil")
        with c3:
            land_cover = st.selectbox("Land Cover", ["Urban", "Rural", "Forest", "Agriculture"], key="fl_land")
            pop_density = st.number_input("Population Density (/km²)", 0, 25000, 5000, 100, key="fl_pop")
            hist_floods = st.slider("Historical Floods (count)", 0, 20, 5, key="fl_hist")

        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="fl_clr", use_container_width=True):
                st.session_state["fl_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("fl_run_btn", "⚡  RUN FLOOD PREDICTION", color)

        if run:
            with st.spinner("Running flood risk model…"):
                time.sleep(0.4)
                soil_boost = {"Clay": 10, "Silty": 5, "Loamy": 0, "Sandy": -8}.get(soil_type, 0)
                land_boost = {"Urban": 12, "Agriculture": 4, "Rural": 0, "Forest": -10}.get(land_cover, 0)
                score = min(100, max(0,
                    rainfall * 0.22 +
                    river_level * 3.0 +
                    humidity * 0.15 +
                    river_disc * 0.004 +
                    hist_floods * 1.2 +
                    max(0, 10 - elevation / 100) * 2 +
                    soil_boost + land_boost +
                    np.random.normal(0, 3)
                ))
                risk = _risk_label(score)
                flood_prob = score / 100

                _append_history("fl_history", score, risk,
                                f"Rain:{rainfall:.0f}mm River:{river_level:.1f}m")
                db.save_prediction("flood", {
                    "rainfall": rainfall, "river_level": river_level,
                    "elevation": elevation, "risk_score": score
                }, risk, f"Elev {elevation:.0f}m | {land_cover}")

                _result_card(risk, score, [
                    f"🌧️ Rainfall: {rainfall:.0f} mm/day  |  Humidity: {humidity:.0f}%",
                    f"🌊 River Level: {river_level:.1f} m  |  Discharge: {river_disc:,} m³/s",
                    f"🏔️ Elevation: {elevation:.0f} m  |  Soil: {soil_type}",
                    f"🏙️ Land Cover: {land_cover}  |  Historical Floods: {hist_floods}",
                    f"📊 Flood Probability: {flood_prob:.1%}",
                ], color)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2 = st.columns(2)
                with pc1:
                    res = make_inundation_chart(rainfall, river_level, elevation, color)
                    fig_i = res[0] if isinstance(res, tuple) else res
                    st.plotly_chart(fig_i, use_container_width=True)
                with pc2:
                    st.plotly_chart(make_river_hydrograph(rainfall, river_level, color), use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — FLOOD RISK SCORE")
        _live_history_chart("fl_history", "Risk Score", color, "🌊 Flood Risk Score Timeline", "%")
        _section("RAINFALL vs RIVER LEVEL")
        data = st.session_state.get("fl_history", [])
        if data:
            rains = []
            rivers = []
            for d in data:
                lbl = d.get("label", "")
                try:
                    rains.append(float(lbl.split("Rain:")[1].split("mm")[0]))
                    rivers.append(float(lbl.split("River:")[1].split("m")[0]))
                except:
                    rains.append(0); rivers.append(0)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[d["time"] for d in data], y=rains, mode="lines+markers",
                name="Rainfall (mm)", line=dict(color="#00d4ff", width=2, shape="spline"),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
            fig2.add_trace(go.Scatter(x=[d["time"] for d in data], y=rivers, mode="lines+markers",
                name="River Level (m)", line=dict(color="#0099ff", width=2, shape="spline"), yaxis="y2"))
            fig2.update_layout(**_base_layout("Rainfall & River Level History", 250))
            fig2.update_layout(yaxis2=dict(overlaying="y", side="right", gridcolor="#0d2a44", title="River Level (m)"))
            st.plotly_chart(fig2, use_container_width=True)
        _section("FULL LOG")
        _history_table("fl_history")

    with tab_live:
        _section("LIVE RIVER GAUGE MONITOR")
        if st.button("▶  Start Live River Stream", key="fl_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            base_level = 3.5
            for i in range(25):
                t = datetime.now().strftime("%H:%M:%S")
                base_level += np.random.normal(0.05, 0.2)
                base_level = max(0.5, min(15, base_level))
                live_data.append({"time": t, "level": round(base_level, 2),
                                  "rain": max(0, np.random.normal(80, 30))})
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=[d["time"] for d in live_data], y=[d["level"] for d in live_data],
                    mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.1)",
                    marker=dict(size=7, color=[_risk_color("Critical" if l>8 else "High" if l>5 else "Moderate" if l>3 else "Low") for l in [d["level"] for d in live_data]]),
                    name="River Level"
                ))
                fig_live.add_hline(y=8, line_dash="dash", line_color="#ef4444", line_width=1,
                                   annotation_text=" Flood threshold", annotation_font={"size":8,"color":"#ef4444","family":"Share Tech Mono"})
                fig_live.update_layout(**_base_layout(f"🌊 Live River Gauge — {t}", 300))
                fig_live.update_layout(yaxis=dict(title="Water Level (m)", gridcolor="#0d2a44", range=[0, 15]))
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.25)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        rc1, rc2 = st.columns(2)
        with rc1:
            rain_c = st.slider("Rainfall (mm)", 0.0, 300.0, 120.0, 5.0, key="fl_chart_rain")
            rl_c = st.slider("River Level (m)", 0.0, 20.0, 4.5, 0.1, key="fl_chart_rl")
        with rc2:
            elev_c = st.slider("Elevation (m)", 0.0, 3000.0, 80.0, 10.0, key="fl_chart_elev")
        cc1, cc2 = st.columns(2)
        with cc1:
            res = make_inundation_chart(rain_c, rl_c, elev_c, color)
            fig_i = res[0] if isinstance(res, tuple) else res
            st.plotly_chart(fig_i, use_container_width=True)
        with cc2:
            st.plotly_chart(make_river_hydrograph(rain_c, rl_c, color), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🌀 CYCLONE PAGE
# ═══════════════════════════════════════════════════════════

def render_cyclone_page_patched(db):
    from disasters.cyclone import make_pressure_profile, make_storm_surge_chart, make_track_forecast

    color = "#aa00ff"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🌀 CYCLONE INTENSITY PREDICTION ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            sst = st.slider("Sea Surface Temp (°C)", 20.0, 35.0, 29.0, 0.1, key="cy_sst")
            pressure = st.slider("Central Pressure (hPa)", 850, 1013, 950, 1, key="cy_pres")
            wind_kts = st.slider("Max Wind Speed (knots)", 20, 200, 100, 5, key="cy_wind")
        with c2:
            wind_shear = st.slider("Wind Shear (m/s)", 0, 30, 5, 1, key="cy_shear")
            humidity = st.slider("Mid-level Humidity (%)", 30, 100, 70, 1, key="cy_humid")
            lat = st.number_input("Latitude", -40.0, 40.0, 15.0, 0.5, key="cy_lat")
        with c3:
            lon = st.number_input("Longitude", -180.0, 180.0, 88.0, 0.5, key="cy_lon")
            basin = st.selectbox("Basin", ["Bay of Bengal", "Arabian Sea", "Pacific NW", "Atlantic", "Indian Ocean"], key="cy_basin")
            landfall_dist = st.slider("Landfall Distance (km)", 0, 2000, 300, 50, key="cy_lf")

        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="cy_clr", use_container_width=True):
                st.session_state["cy_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("cy_run_btn", "⚡  RUN CYCLONE PREDICTION", color)

        if run:
            with st.spinner("Running cyclone intensity model…"):
                time.sleep(0.4)
                cat_num = (0 if wind_kts < 63 else 1 if wind_kts < 83 else
                           2 if wind_kts < 96 else 3 if wind_kts < 113 else
                           4 if wind_kts < 137 else 5)
                score = min(100, max(0,
                    (sst - 26) * 5 +
                    (1013 - pressure) * 0.3 +
                    wind_kts * 0.35 +
                    (30 - wind_shear) * 1.2 +
                    humidity * 0.2 +
                    max(0, 10 - landfall_dist / 200) * 3 +
                    np.random.normal(0, 3)
                ))
                risk = "Critical" if cat_num >= 4 else "High" if cat_num >= 3 else "Moderate" if cat_num >= 1 else "Low"
                cat_c = {0:"#22c55e",1:"#eab308",2:"#f97316",3:"#ff4500",4:"#dc143c",5:"#8b0000"}.get(cat_num, color)

                _append_history("cy_history", score, risk, f"Cat{cat_num} {wind_kts}kts P:{pressure}hPa")
                db.save_prediction("cyclone", {
                    "wind_kts": wind_kts, "pressure": pressure, "sst": sst,
                    "category": cat_num, "risk_score": score
                }, risk, f"{basin} ({lat:.1f}°N)")

                _result_card(risk, score, [
                    f"🌀 Category: {cat_num}  |  Wind: {wind_kts} kts ({wind_kts*1.852:.0f} km/h)",
                    f"📉 Central Pressure: {pressure} hPa  |  SST: {sst}°C",
                    f"💨 Wind Shear: {wind_shear} m/s  |  Humidity: {humidity}%",
                    f"🌍 Basin: {basin}  |  Landfall in: {landfall_dist} km",
                    f"⚠️ Storm Category: {'SUPERTYPHOON' if cat_num==5 else f'Category-{cat_num}'}",
                ], cat_c)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2 = st.columns(2)
                with pc1: st.plotly_chart(make_pressure_profile(pressure, cat_num, cat_c), use_container_width=True)
                with pc2: st.plotly_chart(make_storm_surge_chart(cat_num, lat, cat_c), use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — CYCLONE RISK SCORE")
        _live_history_chart("cy_history", "Risk Score", color, "🌀 Cyclone Risk Score Timeline", "%")
        _section("WIND SPEED HISTORY (knots)")
        data = st.session_state.get("cy_history", [])
        if data:
            winds = []
            for d in data:
                try: winds.append(float(d["label"].split("kts")[0].split(" ")[-1]))
                except: winds.append(0)
            fig_w = go.Figure(go.Scatter(
                x=[d["time"] for d in data], y=winds,
                mode="lines+markers", line=dict(color="#aa00ff", width=2.5, shape="spline"),
                fill="tozeroy", fillcolor="rgba(170,0,255,0.07)",
                marker=dict(size=8, color="#aa00ff")
            ))
            fig_w.update_layout(**_base_layout("Wind Speed Over Predictions", 220))
            fig_w.update_layout(yaxis=dict(title="Wind (kts)", gridcolor="#0d2a44"))
            st.plotly_chart(fig_w, use_container_width=True)
        _section("FULL LOG")
        _history_table("cy_history")

    with tab_live:
        _section("LIVE CYCLONE WIND MONITOR")
        if st.button("▶  Start Live Cyclone Stream", key="cy_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            wind = 60.0
            for i in range(25):
                t = datetime.now().strftime("%H:%M:%S")
                wind += np.random.normal(1, 5)
                wind = max(30, min(200, wind))
                cat = int(max(0, min(5, (wind - 33) / 17)))
                live_data.append({"time": t, "wind": round(wind, 1), "cat": cat})
                cat_colors_live = [_risk_color("Critical" if d["cat"]>=4 else "High" if d["cat"]>=3 else "Moderate" if d["cat"]>=1 else "Low") for d in live_data]
                fig_live = go.Figure(go.Scatter(
                    x=[d["time"] for d in live_data], y=[d["wind"] for d in live_data],
                    mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                    marker=dict(size=8, color=cat_colors_live),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
                ))
                fig_live.update_layout(**_base_layout(f"🌀 Live Cyclone Monitor — {t}", 300))
                fig_live.update_layout(yaxis=dict(title="Wind Speed (kts)", gridcolor="#0d2a44", range=[0,220]))
                for thr, label, lc in [(64,"Cat 1","#eab308"),(96,"Cat 3","#f97316"),(137,"Cat 5","#ef4444")]:
                    fig_live.add_hline(y=thr, line_dash="dash", line_color=lc, line_width=1,
                        annotation_text=f" {label}", annotation_font={"size":8,"color":lc,"family":"Share Tech Mono"})
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.25)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        rc1, rc2 = st.columns(2)
        with rc1:
            p_c = st.slider("Central Pressure (hPa)", 850, 1013, 940, 1, key="cy_chart_p")
            cat_c_n = st.slider("Category (0-5)", 0, 5, 3, key="cy_chart_cat")
        with rc2:
            lat_c = st.slider("Latitude", -40.0, 40.0, 20.0, 0.5, key="cy_chart_lat")
        cat_c2 = {0:"#22c55e",1:"#eab308",2:"#f97316",3:"#ff4500",4:"#dc143c",5:"#8b0000"}.get(cat_c_n, color)
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(make_pressure_profile(p_c, cat_c_n, cat_c2), use_container_width=True)
        with cc2: st.plotly_chart(make_storm_surge_chart(cat_c_n, lat_c, cat_c2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🔥 WILDFIRE PAGE
# ═══════════════════════════════════════════════════════════

def render_wildfire_page_patched(db):
    from disasters.wildfire import make_fire_spread_chart, make_flame_spotting_chart, make_fwi_gauge_chart

    color = "#ff4400"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🔥 WILDFIRE RISK PREDICTION ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.slider("Temperature (°C)", 0.0, 55.0, 38.0, 0.5, key="wf_temp")
            humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 20.0, 1.0, key="wf_humid")
            wind = st.slider("Wind Speed (km/h)", 0.0, 150.0, 50.0, 1.0, key="wf_wind")
        with c2:
            drought_idx = st.slider("Drought Index (0-10)", 0.0, 10.0, 7.0, 0.1, key="wf_drought")
            veg_density = st.slider("Vegetation Density (%)", 0, 100, 75, 1, key="wf_veg")
            slope = st.slider("Terrain Slope (°)", 0, 60, 15, 1, key="wf_slope")
        with c3:
            rainfall_7d = st.slider("7-Day Rainfall (mm)", 0.0, 200.0, 5.0, 1.0, key="wf_rain7")
            veg_type = st.selectbox("Vegetation Type", ["Grassland", "Shrubland", "Forest", "Mixed"], key="wf_vegtype")
            lat = st.number_input("Latitude", -90.0, 90.0, 34.0, 0.5, key="wf_lat")

        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="wf_clr", use_container_width=True):
                st.session_state["wf_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("wf_run_btn", "⚡  RUN WILDFIRE PREDICTION", color)

        if run:
            with st.spinner("Running wildfire risk model…"):
                time.sleep(0.4)
                fwi = max(0, (temp - 20) * 1.5 + (100 - humidity) * 0.5 + wind * 0.3 - rainfall_7d * 0.4)
                veg_boost = {"Forest": 10, "Shrubland": 7, "Mixed": 5, "Grassland": 2}.get(veg_type, 5)
                score = min(100, max(0,
                    temp * 0.8 +
                    (100 - humidity) * 0.4 +
                    wind * 0.25 +
                    drought_idx * 3.5 +
                    veg_density * 0.15 +
                    slope * 0.5 +
                    veg_boost -
                    rainfall_7d * 0.3 +
                    np.random.normal(0, 3)
                ))
                risk = _risk_label(score)

                _append_history("wf_history", score, risk,
                                f"T:{temp:.0f}°C H:{humidity:.0f}% W:{wind:.0f}km/h")
                db.save_prediction("wildfire", {
                    "temperature": temp, "humidity": humidity, "wind": wind,
                    "fwi": fwi, "risk_score": score
                }, risk, f"Lat {lat:.1f}° | {veg_type}")

                _result_card(risk, score, [
                    f"🌡️ Temperature: {temp:.1f}°C  |  Humidity: {humidity:.0f}%",
                    f"💨 Wind: {wind:.0f} km/h  |  Drought Index: {drought_idx:.1f}/10",
                    f"🌿 Vegetation: {veg_type} ({veg_density}% density)  |  Slope: {slope}°",
                    f"🌧️ 7-day Rainfall: {rainfall_7d:.0f} mm",
                    f"🔥 Fire Weather Index (FWI): {fwi:.1f}",
                ], color)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    res = make_fire_spread_chart(temp, humidity, wind, veg_density, color)
                    st.plotly_chart(res[0] if isinstance(res, tuple) else res, use_container_width=True)
                with pc2:
                    res2 = make_flame_spotting_chart(wind, temp, humidity, color)
                    st.plotly_chart(res2[0] if isinstance(res2, tuple) else res2, use_container_width=True)
                with pc3:
                    st.plotly_chart(make_fwi_gauge_chart(fwi, color), use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — FIRE RISK SCORE")
        _live_history_chart("wf_history", "Risk Score", color, "🔥 Wildfire Risk Score Timeline", "%")
        _section("TEMPERATURE & HUMIDITY HISTORY")
        data = st.session_state.get("wf_history", [])
        if data:
            temps_h, humids_h = [], []
            for d in data:
                lbl = d.get("label", "")
                try:
                    temps_h.append(float(lbl.split("T:")[1].split("°")[0]))
                    humids_h.append(float(lbl.split("H:")[1].split("%")[0]))
                except:
                    temps_h.append(0); humids_h.append(0)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[d["time"] for d in data], y=temps_h, mode="lines+markers",
                name="Temp (°C)", line=dict(color="#ff7700", width=2, shape="spline"),
                fill="tozeroy", fillcolor="rgba(255,119,0,0.07)"))
            fig2.add_trace(go.Scatter(x=[d["time"] for d in data], y=humids_h, mode="lines+markers",
                name="Humidity (%)", line=dict(color="#38bdf8", width=2, shape="spline"), yaxis="y2"))
            fig2.update_layout(**_base_layout("Temperature & Humidity History", 250))
            fig2.update_layout(yaxis2=dict(overlaying="y", side="right", gridcolor="#0d2a44", title="Humidity (%)"))
            st.plotly_chart(fig2, use_container_width=True)
        _section("FULL LOG")
        _history_table("wf_history")

    with tab_live:
        _section("LIVE FIRE WEATHER MONITOR")
        if st.button("▶  Start Live Fire Weather Stream", key="wf_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            base_fwi = 30.0
            for i in range(25):
                t = datetime.now().strftime("%H:%M:%S")
                base_fwi += np.random.normal(0.5, 4)
                base_fwi = max(0, min(150, base_fwi))
                live_data.append({"time": t, "fwi": round(base_fwi, 1)})
                fwi_colors = [_risk_color("Critical" if d["fwi"]>80 else "High" if d["fwi"]>50 else "Moderate" if d["fwi"]>25 else "Low") for d in live_data]
                fig_live = go.Figure(go.Scatter(
                    x=[d["time"] for d in live_data], y=[d["fwi"] for d in live_data],
                    mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                    marker=dict(size=8, color=fwi_colors),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
                ))
                for thr, lbl_t, lc in [(25,"Moderate","#eab308"),(50,"High","#f97316"),(80,"Critical","#ef4444")]:
                    fig_live.add_hline(y=thr, line_dash="dash", line_color=lc, line_width=1,
                        annotation_text=f" {lbl_t}", annotation_font={"size":8,"color":lc,"family":"Share Tech Mono"})
                fig_live.update_layout(**_base_layout(f"🔥 Live FWI Monitor — {t}", 300))
                fig_live.update_layout(yaxis=dict(title="Fire Weather Index", gridcolor="#0d2a44", range=[0,150]))
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.25)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        rc1, rc2 = st.columns(2)
        with rc1:
            temp_c = st.slider("Temperature (°C)", 0.0, 55.0, 38.0, 0.5, key="wf_chart_t")
            hum_c = st.slider("Humidity (%)", 0.0, 100.0, 20.0, 1.0, key="wf_chart_h")
        with rc2:
            wind_c = st.slider("Wind (km/h)", 0.0, 150.0, 50.0, 1.0, key="wf_chart_w")
            veg_c = st.slider("Vegetation (%)", 0, 100, 75, 1, key="wf_chart_v")
        fwi_c = max(0, (temp_c-20)*1.5 + (100-hum_c)*0.5 + wind_c*0.3)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            res = make_fire_spread_chart(temp_c, hum_c, wind_c, veg_c, color)
            st.plotly_chart(res[0] if isinstance(res, tuple) else res, use_container_width=True)
        with cc2:
            res2 = make_flame_spotting_chart(wind_c, temp_c, hum_c, color)
            st.plotly_chart(res2[0] if isinstance(res2, tuple) else res2, use_container_width=True)
        with cc3:
            st.plotly_chart(make_fwi_gauge_chart(fwi_c, color), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🌊 TSUNAMI PAGE
# ═══════════════════════════════════════════════════════════

def render_tsunami_page_patched(db):
    from disasters.tsunami import make_wave_propagation_chart, make_travel_time_chart, make_runup_chart

    color = "#00aaff"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🌊 TSUNAMI RISK PREDICTION ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            mag = st.slider("Earthquake Magnitude (Mw)", 5.0, 9.5, 7.8, 0.1, key="ts_mag")
            depth = st.slider("Focal Depth (km)", 0, 100, 20, 1, key="ts_depth")
            lat = st.number_input("Epicenter Latitude", -60.0, 60.0, 8.5, 0.5, key="ts_lat")
        with c2:
            lon = st.number_input("Epicenter Longitude", -180.0, 180.0, 93.5, 0.5, key="ts_lon")
            coastal_dist = st.slider("Coastal Distance (km)", 1, 1000, 150, 10, key="ts_cdist")
            bathymetry = st.slider("Ocean Depth (m)", 500, 8000, 4000, 100, key="ts_bath")
        with c3:
            coast_type = st.selectbox("Coastline Type", ["V-shaped Bay", "Open Coast", "Shallow Shelf", "Delta"], key="ts_coast")
            pop_coast = st.number_input("Coastal Population (thousands)", 0, 10000, 500, 50, key="ts_pop")
            warning_time = st.slider("Warning System (minutes)", 0, 60, 10, 1, key="ts_warn")

        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="ts_clr", use_container_width=True):
                st.session_state["ts_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("ts_run_btn", "⚡  RUN TSUNAMI PREDICTION", color)

        if run:
            with st.spinner("Running tsunami propagation model…"):
                time.sleep(0.5)
                wave_height = max(0.1, 0.001 * 10 ** (0.5 * (mag - 7))) * max(0.1, 1 - depth / 100)
                coast_mult = {"V-shaped Bay": 2.5, "Shallow Shelf": 1.8, "Delta": 1.5, "Open Coast": 1.0}.get(coast_type, 1.0)
                runup = wave_height * coast_mult * max(1, 2 - coastal_dist / 500)
                travel_hrs = coastal_dist / (np.sqrt(9.81 * bathymetry) * 3.6)

                score = min(100, max(0,
                    (mag - 5) * 15 +
                    max(0, 30 - depth) * 1.0 +
                    wave_height * 5 +
                    runup * 3 -
                    coastal_dist * 0.03 -
                    warning_time * 0.5 +
                    np.random.normal(0, 3)
                ))
                risk = "Critical" if mag >= 7.5 else "High" if mag >= 6.5 else "Moderate" if mag >= 5.5 else "Low"

                _append_history("ts_history", score, risk, f"Mw{mag:.1f} H:{wave_height:.1f}m R:{runup:.1f}m")
                db.save_prediction("tsunami", {
                    "magnitude": mag, "depth": depth, "wave_height": wave_height,
                    "runup": runup, "risk_score": score
                }, risk, f"Lat{lat:.1f} Lon{lon:.1f}")

                _result_card(risk, score, [
                    f"⚡ Trigger: Mw {mag:.1f} at {depth} km depth",
                    f"🌊 Est. Wave Height: {wave_height:.2f} m  |  Runup: {runup:.2f} m",
                    f"⏱️ Travel Time to Coast: {travel_hrs*60:.0f} min ({travel_hrs:.2f} hrs)",
                    f"🏖️ Coastline: {coast_type}  |  Distance: {coastal_dist} km",
                    f"⚠️ Warning Lead Time: {warning_time} min  |  Pop. at risk: {pop_coast}k",
                ], color)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2 = st.columns(2)
                with pc1: st.plotly_chart(make_wave_propagation_chart(mag, bathymetry, color), use_container_width=True)
                with pc2:
                    res = make_runup_chart(wave_height, coastal_dist, color)
                    st.plotly_chart(res[0] if isinstance(res, tuple) else res, use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — TSUNAMI RISK SCORE")
        _live_history_chart("ts_history", "Risk Score", color, "🌊 Tsunami Risk Score Timeline", "%")
        _section("WAVE HEIGHT HISTORY")
        data = st.session_state.get("ts_history", [])
        if data:
            heights = []
            for d in data:
                try: heights.append(float(d["label"].split("H:")[1].split("m")[0]))
                except: heights.append(0)
            fig_h = go.Figure(go.Scatter(
                x=[d["time"] for d in data], y=heights,
                mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
                marker=dict(size=8, color=color)
            ))
            fig_h.update_layout(**_base_layout("Estimated Wave Height History", 220))
            fig_h.update_layout(yaxis=dict(title="Wave Height (m)", gridcolor="#0d2a44"))
            st.plotly_chart(fig_h, use_container_width=True)
        _section("FULL LOG")
        _history_table("ts_history")

    with tab_live:
        _section("LIVE OCEAN MONITOR")
        if st.button("▶  Start Live Ocean Stream", key="ts_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            base_wave = 0.1
            for i in range(25):
                t = datetime.now().strftime("%H:%M:%S")
                base_wave += np.random.normal(0, 0.05)
                base_wave = max(0.01, min(15, base_wave))
                live_data.append({"time": t, "wave": round(base_wave, 3)})
                fig_live = go.Figure(go.Scatter(
                    x=[d["time"] for d in live_data], y=[d["wave"] for d in live_data],
                    mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                    marker=dict(size=7, color=[_risk_color("Critical" if w>3 else "High" if w>1 else "Moderate" if w>0.3 else "Low") for w in [d["wave"] for d in live_data]]),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.1)",
                ))
                fig_live.update_layout(**_base_layout(f"🌊 Live Ocean Gauge — {t}", 300))
                fig_live.update_layout(yaxis=dict(title="Wave Height (m)", gridcolor="#0d2a44", range=[0,15]))
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.25)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        rc1, rc2 = st.columns(2)
        with rc1:
            mag_c = st.slider("Magnitude", 5.0, 9.5, 7.8, 0.1, key="ts_chart_m")
            bath_c = st.slider("Ocean Depth (m)", 500, 8000, 4000, 100, key="ts_chart_b")
        with rc2:
            cdist_c = st.slider("Coastal Distance (km)", 1, 1000, 150, 10, key="ts_chart_d")
        wh_c = max(0.01, 0.001 * 10 ** (0.5 * (mag_c - 7)))
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(make_wave_propagation_chart(mag_c, bath_c, color), use_container_width=True)
        with cc2:
            res = make_runup_chart(wh_c, cdist_c, color)
            st.plotly_chart(res[0] if isinstance(res, tuple) else res, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# 🏜️ DROUGHT PAGE
# ═══════════════════════════════════════════════════════════

def render_drought_page_patched(db):
    from disasters.drought import (make_spi_timeseries, make_water_stress_chart,
                                    make_groundwater_chart, make_crop_impact_chart)

    color = "#cc8800"
    st.markdown(f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:{color};letter-spacing:3px;margin-bottom:1rem;">🏜️ DROUGHT SEVERITY PREDICTION ENGINE</div>', unsafe_allow_html=True)

    tab_pred, tab_hist, tab_live, tab_charts = st.tabs(["⚡ Predict", "📋 History", "📡 Live Feed", "📊 Physics Charts"])

    with tab_pred:
        _section("INPUT PARAMETERS")
        c1, c2, c3 = st.columns(3)
        with c1:
            spi = st.slider("SPI Index", -3.0, 2.0, -1.5, 0.05, key="dr_spi")
            temp_anom = st.slider("Temp Anomaly (°C)", -3.0, 5.0, 1.8, 0.1, key="dr_tanom")
            precip_def = st.slider("Precip Deficit (%)", 0, 100, 45, 1, key="dr_pdef")
        with c2:
            evap = st.slider("Evapotranspiration (mm/day)", 0.0, 15.0, 6.5, 0.1, key="dr_evap")
            ndvi = st.slider("NDVI (vegetation health)", -1.0, 1.0, 0.25, 0.01, key="dr_ndvi")
            gw_level = st.slider("Groundwater Level (m below)", 0.0, 50.0, 15.0, 0.5, key="dr_gw")
        with c3:
            crop_area = st.slider("Affected Crop Area (%)", 0, 100, 40, 1, key="dr_crop")
            water_stress = st.slider("Water Stress Index (0-10)", 0.0, 10.0, 6.5, 0.1, key="dr_wstress")
            region = st.selectbox("Region Type", ["Arid", "Semi-arid", "Sub-humid", "Humid"], key="dr_region")

        col_btn, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear History", key="dr_clr", use_container_width=True):
                st.session_state["dr_history"] = []
                st.rerun()
        with col_btn:
            run = _predict_btn("dr_run_btn", "⚡  RUN DROUGHT PREDICTION", color)

        if run:
            with st.spinner("Running drought severity model…"):
                time.sleep(0.4)
                region_mult = {"Arid": 1.4, "Semi-arid": 1.2, "Sub-humid": 1.0, "Humid": 0.8}.get(region, 1.0)
                score = min(100, max(0,
                    abs(min(0, spi)) * 20 * region_mult +
                    max(0, temp_anom) * 5 +
                    precip_def * 0.35 +
                    evap * 2 +
                    max(0, 0.5 - ndvi) * 30 +
                    gw_level * 0.5 +
                    water_stress * 2.5 +
                    np.random.normal(0, 3)
                ))
                sev = ("Extreme" if spi < -2 else "Severe" if spi < -1.5 else
                       "Moderate" if spi < -1 else "Mild" if spi < 0 else "No Drought")
                risk = "Critical" if spi < -2 else "High" if spi < -1.5 else "Moderate" if spi < -1 else "Low"

                _append_history("dr_history", score, risk, f"SPI:{spi:.2f} {sev}")
                db.save_prediction("drought", {
                    "spi": spi, "temp_anomaly": temp_anom, "precip_deficit": precip_def,
                    "ndvi": ndvi, "risk_score": score
                }, risk, f"{region} | SPI {spi:.2f}")

                _result_card(risk, score, [
                    f"📊 SPI Index: {spi:.2f}  →  {sev}",
                    f"🌡️ Temp Anomaly: +{temp_anom:.1f}°C  |  Precip Deficit: {precip_def}%",
                    f"💧 Evapotranspiration: {evap:.1f} mm/day  |  GW Level: {gw_level:.1f} m below",
                    f"🌿 NDVI: {ndvi:.2f}  |  Crop Impact: {crop_area}%",
                    f"🚰 Water Stress: {water_stress:.1f}/10  |  Region: {region}",
                ], color)

                _section("PHYSICS SIMULATIONS")
                pc1, pc2 = st.columns(2)
                with pc1: st.plotly_chart(make_spi_timeseries(spi, color), use_container_width=True)
                with pc2: st.plotly_chart(make_groundwater_chart(spi, gw_level, temp_anom, color), use_container_width=True)
                pc3, pc4 = st.columns(2)
                with pc3: st.plotly_chart(make_water_stress_chart(water_stress, color), use_container_width=True)
                with pc4: st.plotly_chart(make_crop_impact_chart(sev, color), use_container_width=True)

    with tab_hist:
        _section("PREDICTION HISTORY — DROUGHT SEVERITY SCORE")
        _live_history_chart("dr_history", "Risk Score", color, "🏜️ Drought Severity Timeline", "%")
        _section("SPI INDEX HISTORY")
        data = st.session_state.get("dr_history", [])
        if data:
            spis = []
            for d in data:
                try: spis.append(float(d["label"].split("SPI:")[1].split(" ")[0]))
                except: spis.append(0)
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                x=[d["time"] for d in data], y=spis,
                marker=dict(color=[_risk_color("Critical" if s<-2 else "High" if s<-1.5 else "Moderate" if s<-1 else "Low") for s in spis]),
                name="SPI"
            ))
            fig_s.add_hline(y=-1, line_dash="dash", line_color="#eab308", line_width=1)
            fig_s.add_hline(y=-1.5, line_dash="dash", line_color="#f97316", line_width=1)
            fig_s.add_hline(y=-2, line_dash="dash", line_color="#ef4444", line_width=1)
            fig_s.update_layout(**_base_layout("SPI Index History", 240))
            fig_s.update_layout(yaxis=dict(title="SPI", gridcolor="#0d2a44", range=[-3.5,2.5]))
            st.plotly_chart(fig_s, use_container_width=True)
        _section("FULL LOG")
        _history_table("dr_history")

    with tab_live:
        _section("LIVE DROUGHT MONITOR")
        if st.button("▶  Start Live SPI Stream", key="dr_live_start", type="primary"):
            placeholder = st.empty()
            live_data = []
            base_spi = -0.5
            for i in range(25):
                t = datetime.now().strftime("%H:%M:%S")
                base_spi += np.random.normal(-0.03, 0.15)
                base_spi = max(-3.0, min(2.0, base_spi))
                live_data.append({"time": t, "spi": round(base_spi, 3)})
                spi_colors = [_risk_color("Critical" if s<-2 else "High" if s<-1.5 else "Moderate" if s<-1 else "Low") for s in [d["spi"] for d in live_data]]
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=[d["time"] for d in live_data], y=[d["spi"] for d in live_data],
                    mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                    marker=dict(size=8, color=spi_colors),
                    fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
                ))
                for thr, lbl_t, lc in [(-1,"Mild","#eab308"),(-1.5,"Severe","#f97316"),(-2,"Extreme","#ef4444")]:
                    fig_live.add_hline(y=thr, line_dash="dash", line_color=lc, line_width=1,
                        annotation_text=f" {lbl_t}", annotation_font={"size":8,"color":lc,"family":"Share Tech Mono"})
                fig_live.update_layout(**_base_layout(f"🏜️ Live SPI Monitor — {t}", 300))
                fig_live.update_layout(yaxis=dict(title="SPI Index", gridcolor="#0d2a44", range=[-3.5, 2.5]))
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.25)

    with tab_charts:
        _section("PHYSICS CHART EXPLORER")
        rc1, rc2 = st.columns(2)
        with rc1:
            spi_c = st.slider("SPI Index", -3.0, 2.0, -1.5, 0.05, key="dr_chart_spi")
            gw_c = st.slider("GW Level (m below)", 0.0, 50.0, 15.0, 0.5, key="dr_chart_gw")
        with rc2:
            ws_c = st.slider("Water Stress (0-10)", 0.0, 10.0, 6.5, 0.1, key="dr_chart_ws")
            ta_c = st.slider("Temp Anomaly (°C)", -3.0, 5.0, 1.8, 0.1, key="dr_chart_ta")
        sev_c = "Extreme" if spi_c < -2 else "Severe" if spi_c < -1.5 else "Moderate" if spi_c < -1 else "Mild" if spi_c < 0 else "No Drought"
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(make_spi_timeseries(spi_c, color), use_container_width=True)
        with cc2: st.plotly_chart(make_groundwater_chart(spi_c, gw_c, ta_c, color), use_container_width=True)
        cc3, cc4 = st.columns(2)
        with cc3: st.plotly_chart(make_water_stress_chart(ws_c, color), use_container_width=True)
        with cc4: st.plotly_chart(make_crop_impact_chart(sev_c, color), use_container_width=True)
