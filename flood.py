"""SENTINEL v3 — Flood Module (Enhanced)
ENHANCED: Live-reactive risk estimate · Inundation depth model · River flow simulation · Evacuation zones
"""
import numpy as np, streamlit as st, plotly.graph_objects as go, pandas as pd
import sys, os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from models.trainer import load_model
from utils.api_utils import fetch_precipitation_forecast, geocode_location
from utils.charts import make_gauge, make_radar, make_globe, make_timeseries, make_bar, make_donut, _hex_to_rgb
from utils.theme import RISK_COLORS, DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["flood"]; COLOR = THEME["color"]

def get_risk(p):
    return "Low" if p<0.25 else "Moderate" if p<0.5 else "High" if p<0.75 else "Critical"

def make_inundation_chart(rainfall, river_level, elevation, color):
    """Simulated cross-section inundation depth chart"""
    distances = np.linspace(-2000, 2000, 400)
    # Terrain cross-section (v-shaped valley)
    terrain = elevation + (np.abs(distances)/100)**1.5
    # Water level based on river and rainfall
    water_level = elevation - 2 + river_level*1.5 + rainfall/80
    inundation = np.where(terrain < water_level, water_level - terrain, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances, y=terrain, mode="lines", name="Terrain",
        line=dict(color="#8b6914", width=2), fill="tozeroy",
        fillcolor="rgba(139,105,20,0.25)"))
    fig.add_trace(go.Scatter(x=distances, y=np.where(terrain<water_level, water_level, terrain),
        mode="lines", name="Water Surface", line=dict(color=color, width=2, dash="dash"),
        fill="tonexty", fillcolor=f"rgba({_hex_to_rgb(color)},0.2)"))
    fig.add_hline(y=water_level, line_dash="dot", line_color=color, line_width=1,
        annotation_text=f" Flood Level {water_level:.1f}m",
        annotation_font={"family":"Share Tech Mono","size":9,"color":color})
    fig.update_layout(title=dict(text="🏞️ Cross-Section Inundation Model",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Distance from River (m)",gridcolor="#0d2a44"),
        yaxis=dict(title="Elevation (m)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=300,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig, max(0, water_level-elevation)

def make_river_hydrograph(rainfall, river_level, color):
    """Simulated river hydrograph with flood thresholds"""
    hours = np.arange(0, 120, 1)
    # Rising limb then falling limb
    peak_h = 24 + rainfall/20
    base_flow = river_level * 50
    peak_flow = base_flow + rainfall * 15 * np.exp(-((hours-peak_h)**2)/(2*12**2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=peak_flow, mode="lines", name="Discharge",
        line=dict(color=color, width=2.5), fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.1)"))
    for label, threshold in [("Action Level", base_flow*1.3), ("Flood Alert", base_flow*1.8), ("Severe Flood", base_flow*2.5)]:
        fig.add_hline(y=threshold, line_dash="dot", line_color="#ffd700", line_width=1,
            annotation_text=f" {label}", annotation_font={"family":"Share Tech Mono","size":8,"color":"#ffd700"})
    fig.update_layout(title=dict(text="💧 River Hydrograph Simulation",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Time (hours)",gridcolor="#0d2a44"),
        yaxis=dict(title="Discharge (m³/s)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=280,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_soil_saturation_chart(soil_moisture, rainfall, color):
    """Soil moisture dynamics over time"""
    hours = np.arange(0, 72, 1)
    capacity = 100
    # Moisture dynamics
    moisture = soil_moisture + (rainfall/10) * np.exp(-hours/12)
    moisture = np.clip(moisture, 0, capacity)
    runoff_fraction = np.where(moisture > 80, (moisture-80)/20, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=moisture, mode="lines", name="Soil Moisture %",
        line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.1)"))
    fig.add_trace(go.Scatter(x=hours, y=runoff_fraction*100, mode="lines", name="Surface Runoff %",
        line=dict(color="#ff7700", width=1.5, dash="dash")))
    fig.add_hline(y=80, line_dash="dot", line_color="#ff2244", line_width=1,
        annotation_text=" Field Capacity Exceeded",
        annotation_font={"family":"Share Tech Mono","size":9,"color":"#ff2244"})
    fig.update_layout(title=dict(text="🌱 Soil Saturation Dynamics",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Hours",gridcolor="#0d2a44"),yaxis=dict(title="%",gridcolor="#0d2a44",range=[0,105]),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=260,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def render_flood_page(db: DBManager):
    st.markdown(f"""<div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Flood Risk Analyzer</div>
      <div class="dis-subtitle">Hydrological AI · Live precipitation · Inundation modeling · Soil saturation</div></div>
      <div class="dis-badges"><div class="dis-badge">GRADIENT BOOSTING</div>
      <div class="dis-badge">OPEN-METEO API</div><div class="dis-badge">LIVE REACTIVE</div></div>
    </div>""", unsafe_allow_html=True)

    model = load_model("flood")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict","🌧️ Precip Forecast","💧 Hydro Analysis","📊 History"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">📍 Region</div>',unsafe_allow_html=True)
            region = st.text_input("Region Name","Bangladesh Delta",key="fl_reg")
            c1a,c1b = st.columns(2)
            with c1a: lat=st.number_input("Lat",-90.0,90.0,23.68,key="fl_lat2")
            with c1b: lon=st.number_input("Lon",-180.0,180.0,90.35,key="fl_lon2")
            st.markdown('<div class="sec-title">💧 Hydrological</div>',unsafe_allow_html=True)
            rainfall    = st.slider("24h Rainfall (mm)",0.0,500.0,120.0,key="fl_r2")
            river_level = st.slider("River Level above normal (m)",0.0,20.0,5.0,key="fl_rv2")
            soil_moisture=st.slider("Soil Moisture (%)",0.0,100.0,75.0,key="fl_sm2")
        with col2:
            st.markdown('<div class="sec-title">🏔️ Terrain</div>',unsafe_allow_html=True)
            elevation  = st.slider("Elevation (m)",0.0,500.0,10.0,key="fl_el2")
            drainage   = st.slider("Drainage Capacity (m³/s)",0.0,10.0,2.0,key="fl_dr2")
            pop_density= st.slider("Population (per km²)",0.0,5000.0,800.0,key="fl_pd2")
            if st.button("🌐 Fetch Live Precipitation",key="fl_live2"):
                with st.spinner("Fetching Open-Meteo…"):
                    fdata = fetch_precipitation_forecast(lat,lon)
                if fdata.get("precipitation_7day_total",0)>0:
                    rainfall = min(500,fdata["precipitation_7day_total"]/7*24)
                    st.success(f"7-day: {fdata['precipitation_7day_total']:.1f}mm | Peak: {fdata['max_daily_rain']:.1f}mm")

        # LIVE REACTIVE ESTIMATE
        if model:
            prob_live = float(np.clip(model.predict(np.array([[rainfall,river_level,soil_moisture,elevation,drainage,pop_density]]))[0],0,1))
            risk_live = get_risk(prob_live); col_live = RISK_COLORS[risk_live]; pct_live = round(prob_live*100,1)
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">{pct_live}%</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{risk_live}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">
                Flood Probability Index
              </div>
            </div>""", unsafe_allow_html=True)

            # Live charts
            lc1,lc2 = st.columns(2)
            with lc1:
                fig_inund, max_depth = make_inundation_chart(rainfall, river_level, elevation, col_live)
                st.plotly_chart(fig_inund, use_container_width=True)
                if max_depth > 0:
                    st.markdown(f'<div class="warn-box">⚠️ Max inundation depth: <b>{max_depth:.1f}m</b></div>', unsafe_allow_html=True)
            with lc2:
                st.plotly_chart(make_river_hydrograph(rainfall, river_level, col_live), use_container_width=True)
            st.plotly_chart(make_soil_saturation_chart(soil_moisture, rainfall, col_live), use_container_width=True)

        if st.button("🌊 Save & Full Flood Prediction",type="primary",key="fl_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            prob=float(np.clip(model.predict(np.array([[rainfall,river_level,soil_moisture,elevation,drainage,pop_density]]))[0],0,1))
            risk=get_risk(prob); col_r=RISK_COLORS[risk]; pct=round(prob*100,1)
            db.save_flood({"region":region,"rainfall_mm":rainfall,"river_level":river_level,
                           "soil_moisture":soil_moisture,"flood_probability":prob,"risk_level":risk})
            db.log_alert({"disaster_type":"Flood","severity":risk,"location":region,
                          "message":f"Flood {pct}% rain {rainfall:.0f}mm river +{river_level:.1f}m"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Flood Probability Index</span><span class="pred-value">{pct}%</span>
              <span class="pred-unit">Hydrological risk estimate</span>
              <div class="pred-tags"><span class="pred-tag accent">Risk: {risk}</span>
              <span class="pred-tag">Rain: {rainfall:.0f}mm</span><span class="pred-tag">River: +{river_level:.1f}m</span>
              <span class="pred-tag">Soil: {soil_moisture:.0f}%</span></div></div>""",unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(make_gauge(pct,100,"Flood Risk %",col_r,"%"),use_container_width=True)
            with c2:
                vals=[min(100,rainfall/2),river_level/20*100,soil_moisture,max(0,100-elevation/5),max(0,100-drainage*10)]
                st.plotly_chart(make_radar(["Rainfall","River Level","Soil Moisture","Low Elevation","Poor Drainage"],vals,col_r,"Risk Factors"),use_container_width=True)
            # Scenario analysis
            st.markdown('<div class="sec-title">🔬 Scenario Analysis</div>',unsafe_allow_html=True)
            scenarios=["Current","Rain +50%","Rain +100%","River +3m","Drainage +5","Rain -50%"]
            feats_list=[[rainfall,river_level,soil_moisture,elevation,drainage,pop_density],
                [rainfall*1.5,river_level,soil_moisture,elevation,drainage,pop_density],
                [rainfall*2,river_level,soil_moisture,elevation,drainage,pop_density],
                [rainfall,river_level+3,soil_moisture,elevation,drainage,pop_density],
                [rainfall,river_level,soil_moisture,elevation,min(10,drainage+5),pop_density],
                [rainfall*0.5,river_level,soil_moisture,elevation,drainage,pop_density]]
            sc_probs=[round(float(np.clip(model.predict(np.array([f]))[0],0,1))*100,1) for f in feats_list]
            sc_colors=[RISK_COLORS[get_risk(p/100)] for p in sc_probs]
            fig_sc=go.Figure(go.Bar(x=scenarios,y=sc_probs,
                marker=dict(color=sc_colors,cornerradius=6,line=dict(width=0.5,color="rgba(255,255,255,0.1)")),
                text=[f"{p}%" for p in sc_probs],textposition="auto",textfont=dict(family="Share Tech Mono",size=11)))
            fig_sc.update_layout(title=dict(text="What-If Scenarios",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(title="%",gridcolor="#0d2a44",range=[0,105]),
                height=280,margin=dict(t=40,l=10,r=10,b=10))
            st.plotly_chart(fig_sc,use_container_width=True)

    with tab2:
        st.markdown('<div class="sec-title">🌍 Global Flood Hotspots</div>',unsafe_allow_html=True)
        flood_zones=[{"Region":"Bangladesh","lat":23.7,"lon":90.4,"risk":95},
            {"Region":"Pakistan (Indus)","lat":25.9,"lon":68.4,"risk":88},
            {"Region":"India (Assam)","lat":26.1,"lon":91.7,"risk":85},
            {"Region":"Nigeria (Niger Delta)","lat":4.8,"lon":7.0,"risk":82},
            {"Region":"Thailand","lat":13.7,"lon":100.5,"risk":78},
            {"Region":"Mozambique","lat":-18.6,"lon":35.5,"risk":76},
            {"Region":"Netherlands","lat":52.4,"lon":5.3,"risk":60},
            {"Region":"Philippines","lat":14.6,"lon":121.0,"risk":80},
            {"Region":"Vietnam (Mekong)","lat":10.8,"lon":106.6,"risk":83}]
        df_fz=pd.DataFrame(flood_zones)
        trace=go.Scattergeo(lat=df_fz["lat"],lon=df_fz["lon"],mode="markers+text",
            marker=dict(size=df_fz["risk"]/5+8,color=df_fz["risk"],colorscale="RdYlBu_r",cmin=50,cmax=100,
                colorbar=dict(title="Risk %",tickfont=dict(family="Share Tech Mono",size=9)),
                opacity=0.85,line=dict(width=1,color="rgba(255,255,255,0.3)")),
            text=df_fz["Region"],textposition="top center",
            textfont=dict(color="white",size=9,family="Share Tech Mono"),
            hovertext=[f"{r['Region']}<br>Flood Risk: {r['risk']}%" for _,r in df_fz.iterrows()],hoverinfo="text")
        st.plotly_chart(make_globe([trace],"Global Flood Risk Hotspots"),use_container_width=True)

        # Risk bar chart for hotspots
        fig_fz=go.Figure(go.Bar(x=df_fz["Region"],y=df_fz["risk"],
            marker=dict(color=df_fz["risk"],colorscale="RdYlGn_r",cmin=50,cmax=100,cornerradius=5),
            text=[f"{r}%" for r in df_fz["risk"]],textposition="auto",textfont=dict(family="Share Tech Mono",size=10)))
        fig_fz.update_layout(title=dict(text="Flood Risk by Region",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            xaxis=dict(gridcolor="#0d2a44",tickangle=-30),yaxis=dict(title="Risk %",gridcolor="#0d2a44",range=[0,105]),
            height=300,margin=dict(t=45,l=10,r=10,b=80))
        st.plotly_chart(fig_fz,use_container_width=True)

    with tab3:
        st.markdown('<div class="sec-title">💧 Hydrological Deep Analysis</div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: h_rain=st.slider("Rainfall (mm)",0.0,500.0,120.0,key="h_rain"); h_river=st.slider("River Level (m)",0.0,20.0,5.0,key="h_river")
        with col2: h_soil=st.slider("Soil Moisture (%)",0.0,100.0,75.0,key="h_soil"); h_elev=st.slider("Elevation (m)",0.0,500.0,10.0,key="h_elev")
        h_color=RISK_COLORS[get_risk(min(1,h_rain/500*0.6+h_river/20*0.4))]
        fig_i,depth_val=make_inundation_chart(h_rain,h_river,h_elev,h_color)
        st.plotly_chart(fig_i,use_container_width=True)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(make_river_hydrograph(h_rain,h_river,h_color),use_container_width=True)
        with c2: st.plotly_chart(make_soil_saturation_chart(h_soil,h_rain,h_color),use_container_width=True)
        if depth_val>0:
            pop_affected=int(800*((depth_val/5)**0.7)*np.pi*(2)**2)
            st.markdown(f"""<div class="terminal-block"><pre>
[INUNDATION ASSESSMENT]
Max Flood Depth   :  {depth_val:.2f} m
Area at Risk      :  ~{max(0,(depth_val*200)):.0f} hectares (est.)
Pop. Affected     :  ~{pop_affected:,} people (est.)
Evacuation Zones  :  {"3 zones active" if depth_val>3 else "1 zone active" if depth_val>1 else "Monitor only"}
Response Level    :  {"EMERGENCY" if depth_val>5 else "HIGH ALERT" if depth_val>2 else "WATCH"}
</pre></div>""", unsafe_allow_html=True)

    with tab4:
        history=db.get_recent_predictions("flood")
        if history:
            df_h=pd.DataFrame(history); df_h["timestamp"]=pd.to_datetime(df_h["timestamp"],errors="coerce")
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_timeseries(df_h.head(25),"timestamp","flood_probability",COLOR,"Flood Probability History","%"),use_container_width=True)
            with c2:
                rc=df_h["risk_level"].value_counts()
                st.plotly_chart(make_donut(rc.index.tolist(),rc.values.tolist(),[RISK_COLORS.get(r,"#aaa") for r in rc.index],"Risk Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No flood prediction history yet.</div>',unsafe_allow_html=True)
