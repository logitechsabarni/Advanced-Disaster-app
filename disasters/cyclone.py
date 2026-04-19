"""SENTINEL v3 — Cyclone Module (Enhanced)
ENHANCED: Live-reactive intensity · Wind field spiral · Pressure profile · Track forecast · Storm surge
"""
import numpy as np, streamlit as st, plotly.graph_objects as go, pandas as pd
import sys, os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from models.trainer import load_model
from utils.api_utils import fetch_weather_data, fetch_active_cyclones
from utils.charts import make_wind_polar, make_gauge, make_globe, make_timeseries, make_donut, _hex_to_rgb
from utils.theme import DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["cyclone"]; COLOR = THEME["color"]
CAT_COLORS = {0:"#8ab4d0",1:"#ffd700",2:"#ff8c00",3:"#ff4500",4:"#dc143c",5:"#8b0000"}
CAT_NAMES  = {0:"TD/TS",1:"Cat-1 Minimal",2:"Cat-2 Moderate",3:"Cat-3 Extensive",4:"Cat-4 Extreme",5:"Cat-5 Catastrophic"}

def wind_to_cat(kph):
    kts=kph/1.852
    return 0 if kts<34 else 1 if kts<64 else 2 if kts<83 else 3 if kts<96 else 4 if kts<113 else 5

def make_pressure_profile(pressure, cat, color):
    """Pressure cross-section through cyclone eye"""
    r=np.linspace(-500,500,400); r0=35  # RMW in km
    # Holland B parameter
    B=1.5+min(1,(36-r0)/17)
    p_env=1013; p_c=pressure
    profile=p_c+(p_env-p_c)*np.exp(-(r0/np.abs(r+0.001))**B)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=r,y=profile,mode="lines",name="Pressure (hPa)",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.07)"))
    fig.add_vline(x=-r0,line_dash="dot",line_color="#ffd700",line_width=1,annotation_text=" RMW",annotation_font={"family":"Share Tech Mono","size":9,"color":"#ffd700"})
    fig.add_vline(x=r0,line_dash="dot",line_color="#ffd700",line_width=1)
    fig.add_vrect(x0=-r0,x1=r0,fillcolor="rgba(255,215,0,0.04)",layer="below",line_width=0,annotation_text="Eye Wall",annotation_font={"family":"Share Tech Mono","size":9,"color":"#4a7090"})
    fig.update_layout(title=dict(text="🌀 Pressure Cross-Section (Holland Model)",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Distance from Center (km)",gridcolor="#0d2a44"),
        yaxis=dict(title="Pressure (hPa)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=300,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_storm_surge_chart(cat, lat, color):
    """Storm surge height vs coastal distance"""
    coastal_dist=np.linspace(0,200,200)
    base_surge=[0,1.5,2.5,3.7,5.5,8.0][cat]
    surge=base_surge*np.exp(-coastal_dist/30)*np.cos(coastal_dist/15)**2
    surge=np.clip(surge,0,None)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=coastal_dist,y=surge,mode="lines",name="Surge Height (m)",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.15)"))
    for hgt,label in [(1,"Dangerous"),(2,"Life-threatening"),(4,"Catastrophic")]:
        fig.add_hline(y=hgt,line_dash="dot",line_color="#4a7090",line_width=1,
            annotation_text=f" {label}",annotation_font={"family":"Share Tech Mono","size":8,"color":"#4a7090"})
    fig.update_layout(title=dict(text=f"🌊 Storm Surge Profile — Cat {cat}",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Distance from Coast (km)",gridcolor="#0d2a44"),
        yaxis=dict(title="Surge Height (m)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=280,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_track_forecast(lat, lon, color):
    """Synthetic 5-day track forecast cone"""
    # Simulate NHC-style cone of uncertainty
    days=np.arange(0,6)
    drift_lat=lat+days*1.2+np.random.uniform(-0.3,0.3,6)
    drift_lon=lon+days*2.5+np.random.uniform(-0.5,0.5,6)
    cone_radius=[0,40,80,130,200,280]  # km uncertainty
    fig=go.Figure()
    # Cone (simplified as error bars)
    fig.add_trace(go.Scattergeo(lat=drift_lat,lon=drift_lon,mode="lines+markers",
        line=dict(color=color,width=3),
        marker=dict(size=[12]+[8]*5,color=[color]+["#ffd700"]*5,symbol="circle",
                    line=dict(color="white",width=1.5)),
        name="Forecast Track"))
    # Add uncertainty circles (simplified as scatter)
    for i,d in enumerate(days[1:],1):
        theta=np.linspace(0,2*np.pi,30)
        r_deg=cone_radius[i]/111
        cir_lat=drift_lat[i]+r_deg*np.cos(theta)
        cir_lon=drift_lon[i]+r_deg*np.sin(theta)/np.cos(np.radians(drift_lat[i]))
        fig.add_trace(go.Scattergeo(lat=cir_lat,lon=cir_lon,mode="lines",
            line=dict(color=f"rgba({_hex_to_rgb(color)},0.25)",width=1,dash="dot"),
            showlegend=False))
    fig.update_layout(
        geo=dict(showland=True,landcolor="rgba(20,40,60,0.9)",showocean=True,oceancolor="rgba(4,15,30,0.95)",
                 showcoastlines=True,coastlinecolor="#0d2a44",showframe=False,bgcolor="rgba(0,0,0,0)",
                 projection_type="natural earth",
                 center=dict(lat=float(np.mean(drift_lat)),lon=float(np.mean(drift_lon))),
                 lataxis=dict(range=[lat-10,lat+15]),lonaxis=dict(range=[lon-5,lon+20])),
        title=dict(text="📍 5-Day Track Forecast Cone",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=350,margin=dict(t=40,l=0,r=0,b=0),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def render_cyclone_page(db: DBManager):
    st.markdown(f"""<div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Cyclone Intensity Tracker</div>
      <div class="dis-subtitle">Saffir-Simpson AI · Wind field · Pressure profile · Storm surge · Track forecast</div></div>
      <div class="dis-badges"><div class="dis-badge">SAFFIR-SIMPSON</div>
      <div class="dis-badge">LIVE REACTIVE</div><div class="dis-badge">5 FEATURES</div></div>
    </div>""", unsafe_allow_html=True)

    model = load_model("cyclone")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict","🌀 Active Storms","📊 History","🌊 Surge & Track"])

    with tab1:
        col1,col2=st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">🌡️ Atmospheric</div>',unsafe_allow_html=True)
            sst       =st.slider("Sea Surface Temp (°C)",20.0,35.0,29.5,key="cy_sst2")
            pressure  =st.slider("Central Pressure (hPa)",870.0,1010.0,955.0,key="cy_p2")
            wind_shear=st.slider("Vertical Wind Shear (m/s)",0.0,30.0,7.0,key="cy_ws2")
            humidity  =st.slider("Mid-level Humidity (%)",40.0,100.0,78.0,key="cy_hm2")
            lat       =st.slider("Latitude",0.0,50.0,18.0,key="cy_lt2")
        with col2:
            st.markdown('<div class="sec-title">📡 Position & Metadata</div>',unsafe_allow_html=True)
            lon=st.slider("Longitude",-180.0,180.0,125.0,key="cy_ln2")
            warm_core=sst-26; ocean_heat=max(0,(sst-26)*15)
            st.markdown(f"""<div class="info-box">
            🌡️ SST Anomaly: <b>{warm_core:+.1f}°C</b><br>
            🌊 Ocean Heat: <b>~{ocean_heat:.0f} kJ/cm²</b><br>
            {"✅ Rapid intensification favored" if warm_core>1 and wind_shear<10 else "⚠️ Unfavorable conditions"}
            </div>""", unsafe_allow_html=True)
            if st.button("🌐 Fetch Live Conditions",key="cy_live2"):
                with st.spinner("Fetching atmospheric data…"):
                    data=fetch_weather_data(lat,lon)
                if data and "current" in data:
                    c=data["current"]
                    st.success(f"Wind: {c.get('wind_speed_10m',0)} km/h | Precip: {c.get('precipitation',0)}mm")

        # LIVE REACTIVE
        if model:
            wind_live=float(np.clip(model.predict(np.array([[sst,pressure,wind_shear,humidity,lat]]))[0],20,300))
            cat_live=wind_to_cat(wind_live); col_live=CAT_COLORS[cat_live]
            surge_live=round(0.5+cat_live*1.2,1)
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">{wind_live:.0f} km/h</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{CAT_NAMES[cat_live]}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">Surge: ~{surge_live}m</div>
            </div>""", unsafe_allow_html=True)
            lc1,lc2=st.columns(2)
            with lc1: st.plotly_chart(make_pressure_profile(pressure,cat_live,col_live),use_container_width=True)
            with lc2: st.plotly_chart(make_storm_surge_chart(cat_live,lat,col_live),use_container_width=True)

        if st.button("🌀 Save & Full Cyclone Prediction",type="primary",key="cy_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            wind_pred=float(np.clip(model.predict(np.array([[sst,pressure,wind_shear,humidity,lat]]))[0],20,300))
            cat=wind_to_cat(wind_pred); col_r=CAT_COLORS[cat]; cat_name=CAT_NAMES[cat]
            landfall=round(max(0,min(100,55-lat*0.9+np.random.uniform(-8,8))),1)
            surge_m=round(0.5+cat*1.2+np.random.uniform(-0.3,0.5),1)
            db.save_cyclone({"lat":lat,"lon":lon,"wind_speed":wind_pred,"pressure":pressure,
                             "category":cat,"intensity_pred":cat_name,"landfall_risk":landfall})
            db.log_alert({"disaster_type":"Cyclone","severity":cat_name,"location":f"{lat:.1f}°N {lon:.1f}°",
                          "message":f"Winds {wind_pred:.0f}km/h pressure {pressure}hPa cat{cat}"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Max Sustained Wind Speed</span>
              <span class="pred-value">{wind_pred:.0f}</span>
              <span class="pred-unit">km/h  ·  {wind_pred/1.852:.0f} knots  ·  {cat_name}</span>
              <div class="pred-tags"><span class="pred-tag accent">Category {cat}</span>
              <span class="pred-tag">Pressure: {pressure:.0f}hPa</span>
              <span class="pred-tag">Landfall Risk: {landfall}%</span>
              <span class="pred-tag">Storm Surge: ~{surge_m}m</span></div></div>""",unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_wind_polar(wind_pred,col_r,f"Wind Field — Category {cat}"),use_container_width=True)
            with c2: st.plotly_chart(make_gauge(wind_pred,300,"Wind Speed (km/h)",col_r," km/h"),use_container_width=True)
            # Category scale
            cat_winds=[119,154,178,209,252]; cat_labels=[f"Cat {i+1}" for i in range(5)]
            marker_colors=["rgba(255,255,255,0.4)" if i+1!=cat else CAT_COLORS[cat] for i in range(5)]
            fig_cat=go.Figure(go.Bar(x=cat_labels,y=cat_winds,
                marker=dict(color=marker_colors,cornerradius=6,line=dict(color=[CAT_COLORS[i+1] for i in range(5)],width=2)),
                text=[f"{w}km/h" for w in cat_winds],textposition="auto",textfont=dict(family="Share Tech Mono",size=10)))
            fig_cat.add_hline(y=wind_pred,line_color="white",line_dash="dash",line_width=2,
                annotation_text=f" Current: {wind_pred:.0f}km/h",annotation_font={"family":"Share Tech Mono","size":10,"color":"white"})
            fig_cat.update_layout(title=dict(text="📊 Saffir-Simpson Scale Position",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                yaxis=dict(title="Wind Speed (km/h)",gridcolor="#0d2a44"),xaxis=dict(gridcolor="#0d2a44"),
                height=280,margin=dict(t=40,l=10,r=10,b=10))
            st.plotly_chart(fig_cat,use_container_width=True)

    with tab2:
        st.markdown('<div class="sec-title">🌀 Active Tropical Cyclones</div>',unsafe_allow_html=True)
        storms=fetch_active_cyclones()
        storm_traces=[]
        for s in storms:
            cat_c={1:"#ffd700",2:"#ff8c00",3:"#ff4500",4:"#dc143c",5:"#8b0000"}.get(
                max(0,min(5,int((s['wind_kts']-33)//17)+1) if s['wind_kts']>33 else 0),"#888")
            st.markdown(f"""<div class="event-card" style="--ec-color:{cat_c};">
            <b style="color:{cat_c};font-family:'Orbitron',monospace;">{s['name']}</b> — {s['status']}
            &nbsp; 💨 {s['wind_kts']} kts &nbsp; 📉 {s['pressure']} hPa &nbsp; 🧭 {s['movement']}</div>""",unsafe_allow_html=True)
            storm_traces.append(go.Scattergeo(lat=[s["lat"]],lon=[s["lon"]],mode="markers+text",
                marker=dict(size=24,color=cat_c,opacity=0.9,line=dict(width=2,color="white")),
                text=["🌀"],textfont=dict(size=18),hovertext=f"🌀 {s['name']}: {s['wind_kts']} kts",
                hoverinfo="text",name=s["name"],showlegend=True))
        if storm_traces:
            st.plotly_chart(make_globe(storm_traces,"Active Tropical Cyclones — Global Overview"),use_container_width=True)
        else:
            st.info("No active cyclone data available at this time.")

    with tab3:
        history=db.get_recent_predictions("cyclone")
        if history:
            df_h=pd.DataFrame(history); df_h["timestamp"]=pd.to_datetime(df_h["timestamp"],errors="coerce")
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_timeseries(df_h.head(25),"timestamp","wind_speed",COLOR,"Wind Speed History","km/h"),use_container_width=True)
            with c2:
                rc=df_h["category"].value_counts()
                cat_names_map={str(k):v for k,v in CAT_NAMES.items()}
                st.plotly_chart(make_donut([str(k) for k in rc.index],rc.values.tolist(),
                    [CAT_COLORS.get(int(k),COLOR) for k in rc.index],"Category Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No cyclone prediction history yet.</div>',unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec-title">🌊 Storm Surge & Track Analysis</div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1:
            sg_cat=st.selectbox("Cyclone Category",[0,1,2,3,4,5],index=3,key="sg_cat",format_func=lambda x: f"Cat {x}: {CAT_NAMES[x]}")
            sg_lat=st.slider("Latitude",0.0,50.0,18.0,key="sg_lat")
        with col2:
            sg_lon=st.slider("Longitude",-180.0,180.0,125.0,key="sg_lon")
        sg_color=CAT_COLORS[sg_cat]
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(make_storm_surge_chart(sg_cat,sg_lat,sg_color),use_container_width=True)
        with c2: st.plotly_chart(make_pressure_profile(1013-sg_cat*20,sg_cat,sg_color),use_container_width=True)
        st.plotly_chart(make_track_forecast(sg_lat,sg_lon,sg_color),use_container_width=True)
