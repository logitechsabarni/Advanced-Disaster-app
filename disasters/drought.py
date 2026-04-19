"""SENTINEL v3 — Drought Module (Fixed + Enhanced)
Fixed: DB field names (spi_index, severity, duration_weeks) · History tab
"""
import numpy as np, streamlit as st, plotly.graph_objects as go, pandas as pd
import sys, os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from models.trainer import load_model
from utils.api_utils import fetch_weather_data
from utils.charts import make_gauge, make_radar, make_drought_timeline, make_globe, make_donut, make_timeseries, _hex_to_rgb
from utils.theme import RISK_COLORS, DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["drought"]; COLOR = THEME["color"]
SEV_COLORS={"No Drought":"#00ff88","Mild":"#aaff44","Moderate":"#ffd700","Severe":"#ff7700","Extreme":"#ff2244"}

def score_to_sev(s):
    return "No Drought" if s<0 else "Mild" if s<0.5 else "Moderate" if s<1 else "Severe" if s<1.5 else "Extreme"

def make_spi_timeseries(spi, color):
    months = np.arange(0,60)
    np.random.seed(42)
    noise = np.cumsum(np.random.normal(0,0.3,60))*0.5
    noise = noise-noise.mean()
    spi_series = noise+np.sin(months/12*2*np.pi)*0.5+spi*0.3
    spi_series[-1] = spi
    fig = go.Figure()
    for y_min,y_max,label,band_color in [(-4,-2,"Extreme Drought","rgba(255,34,68,0.15)"),
                                          (-2,-1.5,"Severe Drought","rgba(255,119,0,0.12)"),
                                          (-1.5,-1,"Moderate Drought","rgba(255,215,0,0.10)"),
                                          (-1,0,"Mild Drought","rgba(170,255,68,0.07)"),
                                          (0,3,"Normal / Wet","rgba(0,255,136,0.05)")]:
        fig.add_hrect(y0=y_min,y1=y_max,fillcolor=band_color,layer="below",line_width=0,
            annotation_text=label,annotation_font={"family":"Share Tech Mono","size":8,"color":"#4a7090"},
            annotation_position="right")
    fig.add_trace(go.Scatter(x=months,y=spi_series,mode="lines",name="SPI",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.06)"))
    fig.add_trace(go.Scatter(x=[months[-1]],y=[spi],mode="markers",name=f"Current SPI: {spi:.1f}",
        marker=dict(size=12,color="white",symbol="star",line=dict(color=color,width=2))))
    fig.add_hline(y=0,line_color="rgba(255,255,255,0.2)",line_width=1)
    fig.update_layout(title=dict(text="📉 Standardized Precipitation Index (60-month)",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Month",gridcolor="#0d2a44"),yaxis=dict(title="SPI",gridcolor="#0d2a44",range=[-4,3]),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=300,margin=dict(t=45,l=10,r=80,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_water_stress_chart(spi, evapotrans, precip_deficit, ndvi, color):
    categories = ["Precipitation\nDeficit","Evapotranspiration","Vegetation\nStress","SPI\nSeverity","Soil\nMoisture"]
    values = [precip_deficit, min(100,evapotrans*10), max(0,(0.8-ndvi)/0.8*100),
              min(100,max(0,-spi/3*100)), max(0,100-precip_deficit*0.5)]
    fig = go.Figure()
    bar_colors = [f"rgba({_hex_to_rgb(color)},{0.4+v/100*0.6:.2f})" for v in values]
    fig.add_trace(go.Bar(x=categories,y=values,marker=dict(color=bar_colors,cornerradius=6,line=dict(color=color,width=0.5)),
        text=[f"{v:.0f}%" for v in values],textposition="auto",textfont=dict(family="Share Tech Mono",size=11)))
    fig.update_layout(title=dict(text="💧 Water Stress Indicators",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(title="Stress Level (%)",gridcolor="#0d2a44",range=[0,105]),
        height=280,margin=dict(t=45,l=10,r=10,b=10))
    return fig

def make_groundwater_chart(spi, precip_deficit, temp_anomaly, color):
    months = np.arange(0,24)
    recharge_rate = max(0,1+spi*0.3-precip_deficit/100)
    depletion = max(0.2,0.5+temp_anomaly*0.1+precip_deficit/100)
    gw_level = 100-months*depletion*1.5+recharge_rate*np.sin(months/6*np.pi)*10
    gw_level = np.clip(gw_level,0,100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months,y=gw_level,mode="lines+markers",name="Groundwater Level %",
        line=dict(color=color,width=2.5),marker=dict(size=4,color=color),
        fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.08)"))
    fig.add_hline(y=30,line_dash="dot",line_color="#ff7700",line_width=1,
        annotation_text=" Critical Level",annotation_font={"family":"Share Tech Mono","size":9,"color":"#ff7700"})
    fig.update_layout(title=dict(text="🌊 Groundwater Depletion Model",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Months",gridcolor="#0d2a44"),yaxis=dict(title="Groundwater Level (%)",gridcolor="#0d2a44",range=[0,105]),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=270,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_crop_impact_chart(severity, color):
    crops = ["Wheat","Rice","Corn","Cotton","Soybeans","Vegetables"]
    sensitivity = [0.7,0.85,0.75,0.6,0.8,0.9]
    sev_map = {"No Drought":0,"Mild":0.3,"Moderate":0.55,"Severe":0.78,"Extreme":0.95}
    sev_val = sev_map.get(severity,0.5)
    losses = [round(s*sev_val*100,1) for s in sensitivity]
    loss_colors = [f"rgba({_hex_to_rgb(color)},{0.3+l/100*0.7:.2f})" for l in losses]
    fig = go.Figure(go.Bar(x=crops,y=losses,marker=dict(color=loss_colors,cornerradius=6,line=dict(color=color,width=0.5)),
        text=[f"{l:.0f}%" for l in losses],textposition="auto",textfont=dict(family="Share Tech Mono",size=11)))
    fig.update_layout(title=dict(text=f"🌾 Estimated Crop Yield Loss — {severity}",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        xaxis=dict(gridcolor="#0d2a44"),yaxis=dict(title="Yield Loss (%)",gridcolor="#0d2a44",range=[0,100]),
        height=270,margin=dict(t=45,l=10,r=10,b=10))
    return fig

def render_drought_page(db: DBManager):
    st.markdown(f"""<div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Drought Severity Monitor</div>
      <div class="dis-subtitle">SPI analysis · Water stress model · Crop impact · Groundwater depletion</div></div>
      <div class="dis-badges"><div class="dis-badge">GRADIENT BOOSTING</div>
      <div class="dis-badge">SPI INDEX</div><div class="dis-badge">LIVE REACTIVE</div></div>
    </div>""",unsafe_allow_html=True)

    model = load_model("drought")
    tab1,tab2,tab3,tab4 = st.tabs(["🔮 Predict","🌍 Global Hotspots","💧 Water Analysis","📊 History"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">💧 Climate Indices</div>',unsafe_allow_html=True)
            region       = st.text_input("Region","Sahel Region, Africa",key="dr_reg2")
            lat          = st.number_input("Lat",-90.0,90.0,12.0,key="dr_lat2")
            lon          = st.number_input("Lon",-180.0,180.0,15.0,key="dr_lon2")
            spi          = st.slider("SPI (Standardized Precip Index)",-3.0,3.0,-1.8,0.1,key="dr_spi2")
            temp_anomaly = st.slider("Temperature Anomaly (°C)",-3.0,5.0,2.5,0.1,key="dr_ta2")
            st.caption("SPI < -2: Extreme | -2 to -1.5: Severe | -1.5 to -1: Moderate | > 0: Normal")
        with col2:
            st.markdown('<div class="sec-title">🌱 Soil & Vegetation</div>',unsafe_allow_html=True)
            precip_deficit = st.slider("Precipitation Deficit (%)",0.0,100.0,65.0,key="dr_pd2")
            evapotrans     = st.slider("Evapotranspiration (mm/d)",0.0,10.0,7.5,key="dr_et2")
            ndvi           = st.slider("NDVI Vegetation Index",-0.2,0.9,0.15,0.01,key="dr_nd2")
            if st.button("🌐 Fetch Live Weather",key="dr_live2"):
                with st.spinner("Fetching…"):
                    data = fetch_weather_data(lat,lon)
                if data and "current" in data:
                    c = data["current"]
                    temp_live = c.get("temperature_2m",30)
                    st.success(f"Live: {temp_live}°C | anomaly: {round(temp_live-25,1):+.1f}°C")

        if model:
            score_live = float(model.predict(np.array([[spi,temp_anomaly,precip_deficit,evapotrans,ndvi]]))[0])
            sev_live = score_to_sev(score_live); col_live = SEV_COLORS.get(sev_live,COLOR)
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">{score_live:.2f}</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{sev_live}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">SPI: {spi:.1f}</div>
            </div>""",unsafe_allow_html=True)
            lc1,lc2 = st.columns(2)
            with lc1: st.plotly_chart(make_spi_timeseries(spi,col_live),use_container_width=True)
            with lc2: st.plotly_chart(make_water_stress_chart(spi,evapotrans,precip_deficit,ndvi,col_live),use_container_width=True)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(make_groundwater_chart(spi,precip_deficit,temp_anomaly,col_live),use_container_width=True)
            with c2: st.plotly_chart(make_crop_impact_chart(sev_live,col_live),use_container_width=True)

        if st.button("🏜️ Save & Full Drought Assessment",type="primary",key="dr_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            score = float(model.predict(np.array([[spi,temp_anomaly,precip_deficit,evapotrans,ndvi]]))[0])
            sev = score_to_sev(score); col_r = SEV_COLORS.get(sev,COLOR)
            duration_est = max(4,int(-spi*8+precip_deficit/10))
            # ✅ Correct DB field names matching DroughtPrediction model
            db.save_drought({"region":region,"spi_index":spi,"temperature_anomaly":temp_anomaly,
                "precipitation_deficit":precip_deficit,"severity":sev,"duration_weeks":duration_est})
            db.log_alert({"disaster_type":"Drought","severity":sev,"location":region,
                          "message":f"Score {score:.2f} SPI {spi:.1f} deficit {precip_deficit:.0f}%"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Drought Severity Score</span><span class="pred-value">{score:.2f}</span>
              <span class="pred-unit">Palmer Drought Severity Index analog · {sev}</span>
              <div class="pred-tags"><span class="pred-tag accent">{sev}</span>
              <span class="pred-tag">SPI: {spi:.1f}</span><span class="pred-tag">Deficit: {precip_deficit:.0f}%</span>
              <span class="pred-tag">Duration est: {duration_est}wks</span></div></div>""",unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(make_gauge(max(0,score),2,"Drought Severity",col_r),use_container_width=True)
            with c2:
                vals=[max(0,-spi/3*100),temp_anomaly/5*100,precip_deficit,evapotrans/10*100,max(0,(0.6-ndvi)/0.6*100)]
                st.plotly_chart(make_radar(["SPI","Temp Anomaly","Precip Deficit","Evapotransp.","Veg Stress"],vals,col_r,"Drought Indicators"),use_container_width=True)

    with tab2:
        st.markdown('<div class="sec-title">🌍 Global Drought Hotspots</div>',unsafe_allow_html=True)
        drought_zones=[{"Region":"Sahel, Africa","lat":14.0,"lon":10.0,"severity":2.1},
            {"Region":"Horn of Africa","lat":5.0,"lon":42.0,"severity":1.9},
            {"Region":"California, USA","lat":37.0,"lon":-119.0,"severity":1.4},
            {"Region":"Australia (SE)","lat":-33.0,"lon":146.0,"severity":1.2},
            {"Region":"Middle East","lat":27.0,"lon":42.0,"severity":1.7},
            {"Region":"N. China Plain","lat":36.0,"lon":115.0,"severity":1.1},
            {"Region":"Patagonia","lat":-43.0,"lon":-70.0,"severity":0.9},
            {"Region":"Mediterranean","lat":38.0,"lon":20.0,"severity":1.3},
            {"Region":"N.E. Brazil","lat":-7.0,"lon":-39.0,"severity":1.5}]
        df_dz = pd.DataFrame(drought_zones)
        df_dz["sev_label"] = [score_to_sev(s) for s in df_dz["severity"]]
        trace = go.Scattergeo(lat=df_dz["lat"],lon=df_dz["lon"],mode="markers+text",
            marker=dict(size=df_dz["severity"]*15+8,color=df_dz["severity"],colorscale="OrRd",cmin=0,cmax=2.5,
                colorbar=dict(title="Severity",tickfont=dict(family="Share Tech Mono",size=9)),
                opacity=0.85,line=dict(width=1,color="rgba(255,200,100,0.3)")),
            text=df_dz["Region"],textposition="top center",
            textfont=dict(color="white",size=9,family="Share Tech Mono"),
            hovertext=[f"{r['Region']}<br>{r['sev_label']} — Score: {r['severity']:.1f}" for _,r in df_dz.iterrows()],hoverinfo="text")
        st.plotly_chart(make_globe([trace],"Global Drought Severity Map"),use_container_width=True)

    with tab3:
        st.markdown('<div class="sec-title">💧 Water Stress Deep Analysis</div>',unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1: wa_spi=st.slider("SPI",-3.0,3.0,-1.5,0.1,key="wa_spi"); wa_et=st.slider("Evapotrans (mm/d)",0.0,10.0,7.0,key="wa_et")
        with col2: wa_def=st.slider("Precip Deficit (%)",0.0,100.0,60.0,key="wa_def"); wa_ndvi=st.slider("NDVI",-.2,.9,.2,.01,key="wa_ndvi")
        wa_score = max(0,(-wa_spi-1)*0.5+wa_def/200+wa_et/20)
        wa_sev = score_to_sev(wa_score); wa_color = SEV_COLORS.get(wa_sev,COLOR)
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(make_spi_timeseries(wa_spi,wa_color),use_container_width=True)
        with c2: st.plotly_chart(make_water_stress_chart(wa_spi,wa_et,wa_def,wa_ndvi,wa_color),use_container_width=True)
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(make_groundwater_chart(wa_spi,wa_def,2.0,wa_color),use_container_width=True)
        with c2: st.plotly_chart(make_crop_impact_chart(wa_sev,wa_color),use_container_width=True)

    with tab4:
        st.markdown('<div class="sec-title">📊 Drought Prediction History</div>',unsafe_allow_html=True)
        history = db.get_recent_predictions("drought")
        if history:
            df_h = pd.DataFrame(history)
            df_h["timestamp"] = pd.to_datetime(df_h["timestamp"],errors="coerce")
            display_cols = [c for c in ["timestamp","region","spi_index","temperature_anomaly","precipitation_deficit","severity","duration_weeks"] if c in df_h.columns]
            st.dataframe(df_h[display_cols].head(20),use_container_width=True)
            if "spi_index" in df_h.columns and df_h["spi_index"].notna().any():
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(make_timeseries(df_h.head(25),"timestamp","spi_index",COLOR,"SPI History"),use_container_width=True)
                with c2:
                    if "severity" in df_h.columns:
                        rc = df_h["severity"].value_counts()
                        st.plotly_chart(make_donut(rc.index.tolist(),rc.values.tolist(),[SEV_COLORS.get(r,"#aaa") for r in rc.index],"Severity Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No drought predictions yet — run a prediction to populate history.</div>',unsafe_allow_html=True)
