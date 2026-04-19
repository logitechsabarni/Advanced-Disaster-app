"""SENTINEL v3 — Wildfire Module (Enhanced)
ENHANCED: Live-reactive FWI · Fire spread simulation · Spotting distance · Flame length · Resource deployment
"""
import numpy as np, streamlit as st, plotly.graph_objects as go, pandas as pd
import sys, os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from models.trainer import load_model
from utils.api_utils import fetch_weather_data, fetch_active_wildfires_simulated
from utils.charts import make_gauge, make_radar, make_globe, make_timeseries, make_donut, _hex_to_rgb
from utils.theme import RISK_COLORS, DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["wildfire"]; COLOR = THEME["color"]

def get_risk(p):
    return "Low" if p<0.25 else "Moderate" if p<0.5 else "High" if p<0.75 else "Critical"

def make_fire_spread_chart(temp, humidity, wind, veg, color):
    """Fire spread rate and area over time (Rothermel model approximation)"""
    hours = np.arange(0, 48, 0.5)
    # Rothermel-inspired spread rate (m/min)
    spread_rate = (0.3 + temp/100 + wind/50 + veg/200 - humidity/200) * max(0.1, 1-humidity/200)
    spread_rate = max(0.01, spread_rate)
    # Elliptical fire perimeter
    area_ha = np.pi * (spread_rate * hours * 60 / 100)**2  # in hectares
    perimeter_km = 2 * np.pi * (spread_rate * hours * 60 / 1000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=area_ha, mode="lines", name="Burned Area (ha)",
        line=dict(color=color, width=2.5), fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.12)"))
    fig.add_trace(go.Scatter(x=hours, y=perimeter_km*100, mode="lines", name="Perimeter (×100m)",
        line=dict(color="#ffd700", width=1.5, dash="dash")))
    for milestone, label in [(100,"Town evacuation zone"),(1000,"Major incident"),(10000,"Mega-fire")]:
        fig.add_hline(y=milestone, line_dash="dot", line_color="#4a7090", line_width=1,
            annotation_text=f" {label}",annotation_font={"family":"Share Tech Mono","size":8,"color":"#4a7090"})
    fig.update_layout(title=dict(text=f"🔥 Fire Spread Simulation ({spread_rate:.2f}m/min rate)",
        font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Hours since ignition",gridcolor="#0d2a44"),
        yaxis=dict(title="Area (ha) / Perimeter (×100m)",gridcolor="#0d2a44",type="log"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=300,margin=dict(t=45,l=10,r=10,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig, spread_rate

def make_flame_spotting_chart(wind, temp, humidity, color):
    """Spotting distance and flame length with wind speed"""
    wind_speeds = np.linspace(0, 100, 200)
    # Flame length (Nelson eq.)
    flame_len = 0.0775 * (max(1,wind)**0.46) * (temp/20) * max(0.1, (100-humidity)/100)
    # Spotting distance (simplified Albini)
    spot_dist = 0.1 * wind_speeds**1.5 * max(0.1,(100-humidity)/100) * (temp/30)
    actual_flame = 0.0775 * (max(1,wind)**0.46) * (temp/20) * max(0.1,(100-humidity)/100)
    actual_spot = 0.1 * wind**1.5 * max(0.1,(100-humidity)/100) * (temp/30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wind_speeds, y=spot_dist, mode="lines", name="Spotting Distance (m)",
        line=dict(color=color, width=2.5), fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.08)"))
    fig.add_trace(go.Scatter(x=[wind], y=[actual_spot], mode="markers", name=f"Current ({actual_spot:.0f}m)",
        marker=dict(size=14, color="white", symbol="star",
                    line=dict(color=color, width=2))))
    fig.update_layout(title=dict(text="🌬️ Spotting Distance vs Wind Speed",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Wind Speed (km/h)",gridcolor="#0d2a44"),
        yaxis=dict(title="Max Spotting Distance (m)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=280,margin=dict(t=45,l=10,r=10,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig, actual_flame

def make_fwi_gauge_chart(fwi, color):
    """FWI component breakdown"""
    # Sub-indices (approximate)
    ffmc = min(100, fwi * 1.1)
    dmc  = min(100, fwi * 0.9)
    dc   = min(100, fwi * 0.95)
    isi  = min(100, fwi * 1.05)
    bui  = min(100, fwi * 0.85)
    categories = ["FFMC", "DMC", "DC", "ISI", "BUI"]
    values     = [ffmc, dmc, dc, isi, bui]
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker=dict(color=[f"rgba({_hex_to_rgb(color)},{0.4+v/100*0.6:.2f})" for v in values],
                    cornerradius=6, line=dict(color=color, width=0.5)),
        text=[f"{v:.0f}" for v in values], textposition="auto",
        textfont=dict(family="Share Tech Mono", size=11)
    ))
    fig.update_layout(title=dict(text="📊 FWI Sub-Index Breakdown",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(title="Index Value (0-100)",gridcolor="#0d2a44",range=[0,105]),
        height=270,margin=dict(t=45,l=10,r=10,b=10))
    return fig

def render_wildfire_page(db: DBManager):
    st.markdown(f"""<div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Wildfire Risk Predictor</div>
      <div class="dis-subtitle">Fire Weather Index · Spread simulation · Spotting distance · NASA FIRMS hotspots</div></div>
      <div class="dis-badges"><div class="dis-badge">FWI MODEL</div>
      <div class="dis-badge">NASA FIRMS</div><div class="dis-badge">LIVE REACTIVE</div></div>
    </div>""", unsafe_allow_html=True)

    model = load_model("wildfire")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict","🛰️ Hotspots","🔥 Fire Physics","📊 History"])

    with tab1:
        col1,col2=st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">🌡️ Meteorological</div>',unsafe_allow_html=True)
            region  =st.text_input("Region","California, USA",key="wf_reg2")
            lat     =st.number_input("Lat",-90.0,90.0,37.5,key="wf_lat2")
            lon     =st.number_input("Lon",-180.0,180.0,-119.5,key="wf_lon2")
            temp    =st.slider("Temperature (°C)",-10.0,50.0,38.0,key="wf_t2")
            humidity=st.slider("Relative Humidity (%)",0.0,100.0,14.0,key="wf_h2")
            if st.button("🌐 Fetch Live Weather",key="wf_live2"):
                with st.spinner("Fetching…"):
                    data=fetch_weather_data(lat,lon)
                if data and "current" in data:
                    c=data["current"]
                    temp=c.get("temperature_2m",temp); humidity=c.get("relative_humidity_2m",humidity)
                    st.success(f"Live: {temp}°C | {humidity}% RH | Wind: {c.get('wind_speed_10m',0)}km/h")
        with col2:
            st.markdown('<div class="sec-title">🌿 Environmental</div>',unsafe_allow_html=True)
            wind_speed  =st.slider("Wind Speed (km/h)",0.0,100.0,45.0,key="wf_w2")
            drought_idx =st.slider("Drought Index (KBDI)",0.0,10.0,7.5,key="wf_d2")
            veg_density =st.slider("Vegetation Density (%)",0.0,100.0,70.0,key="wf_v2")
            fwi = temp/50*30 + (100-humidity)/100*30 + wind_speed/100*20 + drought_idx/10*20
            fwi_risk="🟢 Low" if fwi<30 else "🟡 Moderate" if fwi<50 else "🟠 High" if fwi<70 else "🔴 Critical"
            st.markdown(f"""<div class="info-box">
            🔥 <b>Fire Weather Index: {fwi:.1f}/100</b><br>Classification: <b>{fwi_risk}</b>
            </div>""", unsafe_allow_html=True)

        # LIVE REACTIVE
        if model:
            prob_live=float(np.clip(model.predict(np.array([[temp,humidity,wind_speed,drought_idx,veg_density]]))[0],0,1))
            risk_live=get_risk(prob_live); col_live=RISK_COLORS[risk_live]; pct_live=round(prob_live*100,1)
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">{pct_live}%</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{risk_live}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">FWI: {fwi:.1f}</div>
            </div>""", unsafe_allow_html=True)
            lc1,lc2=st.columns(2)
            with lc1:
                fig_spread,sr=make_fire_spread_chart(temp,humidity,wind_speed,veg_density,col_live)
                st.plotly_chart(fig_spread,use_container_width=True)
            with lc2: st.plotly_chart(make_fwi_gauge_chart(fwi,col_live),use_container_width=True)

        if st.button("🔥 Save & Full Wildfire Prediction",type="primary",key="wf_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            prob=float(np.clip(model.predict(np.array([[temp,humidity,wind_speed,drought_idx,veg_density]]))[0],0,1))
            risk=get_risk(prob); col_r=RISK_COLORS[risk]; pct=round(prob*100,1)
            db.save_wildfire({"region":region,"temperature":temp,"humidity":humidity,
                              "wind_speed":wind_speed,"drought_index":drought_idx,
                              "fire_probability":prob,"risk_level":risk})
            db.log_alert({"disaster_type":"Wildfire","severity":risk,"location":region,
                          "message":f"Fire prob {pct}% Temp {temp}°C Humidity {humidity}% FWI {fwi:.1f}"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Wildfire Ignition Probability</span><span class="pred-value">{pct}%</span>
              <span class="pred-unit">Fire Weather Index: {fwi:.1f} / 100</span>
              <div class="pred-tags"><span class="pred-tag accent">Risk: {risk}</span>
              <span class="pred-tag">Temp: {temp}°C</span><span class="pred-tag">Humidity: {humidity}%</span>
              <span class="pred-tag">Wind: {wind_speed:.0f}km/h</span><span class="pred-tag">Drought: {drought_idx}/10</span>
              </div></div>""", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_gauge(pct,100,"Fire Probability %",col_r,"%"),use_container_width=True)
            with c2:
                vals=[temp/50*100,100-humidity,wind_speed,drought_idx*10,veg_density]
                st.plotly_chart(make_radar(["Temperature","Low Humidity","Wind","Drought","Vegetation"],vals,col_r,"Risk Factors"),use_container_width=True)
            recs={"Critical":["🔴 Immediate evacuation","✈️ Request aerial resources","🚒 Position crews","📻 Activate broadcast"],
                  "High":["🟠 Issue Red Flag Warning","🚒 Increase readiness","📍 Restrict access"],
                  "Moderate":["🟡 Monitor hourly","💧 Fill reservoirs"],"Low":["🟢 Normal protocols"]}
            st.markdown('<div class="sec-title">🚨 Recommended Actions</div>',unsafe_allow_html=True)
            for rec in recs.get(risk,[]): st.markdown(f"- {rec}")

    with tab2:
        st.markdown('<div class="sec-title">🛰️ NASA FIRMS Active Fire Hotspots</div>',unsafe_allow_html=True)
        hotspots=fetch_active_wildfires_simulated(); df=pd.DataFrame(hotspots)
        trace=go.Scattergeo(lat=df["lat"],lon=df["lon"],mode="markers+text",
            marker=dict(size=df["frp"]/6+10,color=df["confidence"],colorscale="YlOrRd",cmin=50,cmax=100,
                colorbar=dict(title="Confidence %",tickfont=dict(family="Share Tech Mono",size=9)),
                opacity=0.9,line=dict(width=1,color="rgba(255,200,100,0.4)")),
            text=df["region"],textposition="top center",textfont=dict(color="white",size=9,family="Share Tech Mono"),
            hovertext=[f"🔥 {r['region']}<br>FRP: {r['frp']:.1f}MW<br>Conf: {r['confidence']}%" for _,r in df.iterrows()],hoverinfo="text")
        st.plotly_chart(make_globe([trace],"Active Fire Hotspots (MODIS Thermal Anomalies)"),use_container_width=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Active Hotspots",len(df)); c2.metric("Max FRP",f"{df['frp'].max():.0f} MW"); c3.metric("Avg Confidence",f"{df['confidence'].mean():.0f}%")

        # FRP bar chart
        fig_frp=go.Figure(go.Bar(x=df["region"],y=df["frp"],
            marker=dict(color=df["confidence"],colorscale="YlOrRd",cornerradius=5),
            text=[f"{f:.0f}MW" for f in df["frp"]],textposition="auto",textfont=dict(family="Share Tech Mono",size=9)))
        fig_frp.update_layout(title=dict(text="Fire Radiative Power by Region",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(gridcolor="#0d2a44",tickangle=-30),yaxis=dict(title="FRP (MW)",gridcolor="#0d2a44"),
            height=280,margin=dict(t=45,l=10,r=10,b=80))
        st.plotly_chart(fig_frp,use_container_width=True)
        st.dataframe(df.rename(columns={"frp":"FRP (MW)","confidence":"Confidence %"}),use_container_width=True)

    with tab3:
        st.markdown('<div class="sec-title">🔥 Fire Behavior Physics</div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: fp_temp=st.slider("Temp (°C)",-10.0,50.0,38.0,key="fp_t"); fp_hum=st.slider("Humidity (%)",0.0,100.0,14.0,key="fp_h")
        with col2: fp_wind=st.slider("Wind (km/h)",0.0,100.0,45.0,key="fp_w"); fp_veg=st.slider("Vegetation (%)",0.0,100.0,70.0,key="fp_v")
        fp_fwi=fp_temp/50*30+(100-fp_hum)/100*30+fp_wind/100*20
        fp_color=RISK_COLORS[get_risk(fp_fwi/100)]
        fig_spr,sp_rate=make_fire_spread_chart(fp_temp,fp_hum,fp_wind,fp_veg,fp_color)
        st.plotly_chart(fig_spr,use_container_width=True)
        fig_spot,flame_len=make_flame_spotting_chart(fp_wind,fp_temp,fp_hum,fp_color)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(fig_spot,use_container_width=True)
        with c2: st.plotly_chart(make_fwi_gauge_chart(fp_fwi,fp_color),use_container_width=True)
        st.markdown(f"""<div class="terminal-block"><pre>
[FIRE BEHAVIOR ANALYSIS]
Spread Rate      :  {sp_rate:.2f} m/min  ({sp_rate*60:.0f} m/hr)
Flame Length     :  ~{flame_len:.1f} m
Spotting Dist    :  ~{0.1*fp_wind**1.5*max(0.1,(100-fp_hum)/100)*(fp_temp/30):.0f} m
FWI              :  {fp_fwi:.1f} / 100
Fire Intensity   :  {"EXTREME" if fp_fwi>70 else "HIGH" if fp_fwi>50 else "MODERATE" if fp_fwi>30 else "LOW"}
Area at 24h (est) :  {np.pi*(sp_rate*24*60/100)**2:.0f} ha
</pre></div>""", unsafe_allow_html=True)

    with tab4:
        history=db.get_recent_predictions("wildfire")
        if history:
            df_h=pd.DataFrame(history); df_h["timestamp"]=pd.to_datetime(df_h["timestamp"],errors="coerce")
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_timeseries(df_h.head(25),"timestamp","fire_probability",COLOR,"Fire Probability History"),use_container_width=True)
            with c2:
                rc=df_h["risk_level"].value_counts()
                st.plotly_chart(make_donut(rc.index.tolist(),rc.values.tolist(),[RISK_COLORS.get(r,"#aaa") for r in rc.index],"Risk Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No wildfire history yet.</div>',unsafe_allow_html=True)
