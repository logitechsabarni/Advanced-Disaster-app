"""SENTINEL v3 — Tsunami Module (Fixed + Enhanced)
Fixed: DB field names · History tab · Live-reactive wave height
"""
import numpy as np, streamlit as st, plotly.graph_objects as go, pandas as pd
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.trainer import load_model
from utils.api_utils import fetch_usgs_earthquakes
from utils.charts import make_wave_chart, make_gauge, make_globe, make_timeseries, make_donut, _hex_to_rgb
from utils.theme import DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["tsunami"]; COLOR = THEME["color"]

def get_risk(wh):
    return ("Low","#00ff88") if wh<0.5 else ("Moderate","#ffd700") if wh<2 else ("High","#ff7700") if wh<5 else ("Critical","#ff2244")

def make_wave_propagation_chart(magnitude, bathymetry, color):
    distances = np.linspace(10,5000,500)
    initial_height = 0.001*10**(0.5*(magnitude-7))
    deep_height = initial_height/np.sqrt(distances/10)
    shallow_factor = np.where(distances>3000,1.0,1.0+(3000-distances)/3000*4)
    wave_h = deep_height*shallow_factor
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances,y=wave_h,mode="lines",name="Wave Height (m)",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.1)"))
    fig.add_hline(y=0.5,line_dash="dot",line_color="#ffd700",line_width=1,
        annotation_text=" Dangerous (0.5m)",annotation_font={"family":"Share Tech Mono","size":9,"color":"#ffd700"})
    fig.add_hline(y=2.0,line_dash="dot",line_color="#ff7700",line_width=1,
        annotation_text=" Life-threatening (2m)",annotation_font={"family":"Share Tech Mono","size":9,"color":"#ff7700"})
    fig.update_layout(title=dict(text=f"🌊 Wave Propagation — Mw{magnitude:.1f}",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Distance from Source (km)",gridcolor="#0d2a44"),
        yaxis=dict(title="Wave Height (m)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=300,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_travel_time_chart(bathymetry, origin_lat, origin_lon, color):
    wave_speed_ms = np.sqrt(9.81*bathymetry)
    wave_speed_kmh = wave_speed_ms*3.6
    cities = [("Tokyo",35.7,139.7),("Manila",14.6,121.0),("Honolulu",21.3,-157.8),
              ("Sydney",-33.9,151.2),("Los Angeles",34.0,-118.2),("Jakarta",-6.2,106.8),
              ("Bangkok",13.7,100.5),("Mumbai",19.1,72.9),("Colombo",6.9,79.9)]
    results = []
    for name,clat,clon in cities:
        dist_km = np.sqrt((clat-origin_lat)**2+(clon-origin_lon)**2)*111
        eta_min = (dist_km/wave_speed_kmh)*60
        results.append({"City":name,"Distance (km)":round(dist_km),"ETA (min)":round(eta_min)})
    df = pd.DataFrame(results).sort_values("ETA (min)")
    fig = go.Figure(go.Bar(x=df["City"],y=df["ETA (min)"],
        marker=dict(color=df["ETA (min)"],colorscale="RdYlGn",cmin=0,cmax=600,cornerradius=5,line=dict(color=color,width=0.5)),
        text=[f"{v}min" for v in df["ETA (min)"]],textposition="auto",textfont=dict(family="Share Tech Mono",size=9)))
    fig.update_layout(title=dict(text="⏱️ ETA to Major Cities",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        xaxis=dict(gridcolor="#0d2a44",tickangle=-30),
        yaxis=dict(title="ETA (minutes)",gridcolor="#0d2a44"),
        height=300,margin=dict(t=45,l=10,r=10,b=80))
    return fig, df

def make_runup_chart(wave_height, coastal_dist, color):
    slopes = np.linspace(0.001,0.1,200)
    runup = wave_height*(slopes**(-0.25))*(coastal_dist/100)**(-0.1)
    runup = np.clip(runup,0,50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=slopes*100,y=runup,mode="lines",name="Max Runup (m)",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.1)"))
    fig.update_layout(title=dict(text="🏔️ Coastal Runup vs Beach Slope",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Beach Slope (%)",gridcolor="#0d2a44"),
        yaxis=dict(title="Max Runup (m)",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),
        height=280,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig, float(np.mean(runup))

def render_tsunami_page(db: DBManager):
    st.markdown(f"""<div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Tsunami Warning System</div>
      <div class="dis-subtitle">Wave height AI · Propagation model · ETA calculator · Runup estimation</div></div>
      <div class="dis-badges"><div class="dis-badge">GRADIENT BOOSTING</div>
      <div class="dis-badge">USGS REALTIME</div><div class="dis-badge">LIVE REACTIVE</div></div>
    </div>""", unsafe_allow_html=True)

    model = load_model("tsunami")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict","⚡ Seismic Triggers","🌊 Wave Analysis","📊 History"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">🌋 Seismic Source</div>',unsafe_allow_html=True)
            magnitude   = st.slider("Earthquake Magnitude (Mw)",5.0,9.5,8.2,0.1,key="ts_m2")
            depth       = st.slider("Focal Depth (km)",0.0,100.0,15.0,key="ts_d2")
            origin_lat  = st.slider("Origin Latitude",-60.0,60.0,-3.3,key="ts_ol2")
            origin_lon  = st.slider("Origin Longitude",-180.0,180.0,140.0,key="ts_on2")
        with col2:
            st.markdown('<div class="sec-title">🌊 Ocean Parameters</div>',unsafe_allow_html=True)
            coastal_dist = st.slider("Distance to Coast (km)",0.0,1000.0,150.0,key="ts_cd2")
            bathymetry   = st.slider("Avg Ocean Depth (m)",100.0,8000.0,4000.0,key="ts_bm2")
            potential = magnitude>=7.5 and depth<70
            if potential:
                st.markdown("""<div class="danger-box">⚠️ <b>TSUNAMI POTENTIAL DETECTED</b><br>
                Mw ≥ 7.5 with shallow focal depth</div>""",unsafe_allow_html=True)
            else:
                st.markdown("""<div class="info-box">✅ Below tsunami generation threshold</div>""",unsafe_allow_html=True)

        if model:
            wh_live = float(np.clip(model.predict(np.array([[magnitude,depth,coastal_dist,bathymetry]]))[0],0,35))
            risk_live,col_live = get_risk(wh_live)
            wave_speed_kmh = np.sqrt(9.81*bathymetry)*3.6
            eta_live = coastal_dist/wave_speed_kmh*60
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">{wh_live:.1f}m</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{risk_live}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">
                ETA: {eta_live:.0f}min | Speed: {wave_speed_kmh:.0f}km/h
              </div>
            </div>""",unsafe_allow_html=True)
            lc1,lc2 = st.columns(2)
            with lc1: st.plotly_chart(make_wave_propagation_chart(magnitude,bathymetry,col_live),use_container_width=True)
            with lc2:
                fig_r,_ = make_runup_chart(wh_live,coastal_dist,col_live)
                st.plotly_chart(fig_r,use_container_width=True)

        if st.button("🌊 Save & Full Tsunami Analysis",type="primary",key="ts_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            wave_height = float(np.clip(model.predict(np.array([[magnitude,depth,coastal_dist,bathymetry]]))[0],0,35))
            risk,col_r = get_risk(wave_height)
            wave_speed_kmh = np.sqrt(9.81*bathymetry)*3.6
            eta_min = round(coastal_dist/wave_speed_kmh*60,0)
            max_runup = round(wave_height*(2.5+magnitude/9*4),1)
            db.save_tsunami({"origin_lat":origin_lat,"origin_lon":origin_lon,
                "earthquake_mag":magnitude,"depth":depth,
                "wave_height_pred":wave_height,"eta_minutes":eta_min,
                "affected_coasts":f"dist={coastal_dist:.0f}km"})
            db.log_alert({"disaster_type":"Tsunami","severity":risk,
                          "location":f"{origin_lat:.1f}°,{origin_lon:.1f}°",
                          "message":f"Wave {wave_height:.1f}m ETA {eta_min:.0f}min from M{magnitude}"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Predicted Wave Height</span><span class="pred-value">{wave_height:.1f}</span>
              <span class="pred-unit">meters at coast · {risk} Risk</span>
              <div class="pred-tags"><span class="pred-tag accent">Risk: {risk}</span>
              <span class="pred-tag">ETA: {eta_min:.0f} min</span><span class="pred-tag">Max Runup: ~{max_runup:.1f}m</span>
              <span class="pred-tag">Speed: {wave_speed_kmh:.0f} km/h</span></div></div>""",unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(make_gauge(wave_height,35,"Wave Height (m)",col_r," m"),use_container_width=True)
            with c2: st.plotly_chart(make_wave_propagation_chart(magnitude,bathymetry,col_r),use_container_width=True)
            fig_ttm,df_cities = make_travel_time_chart(bathymetry,origin_lat,origin_lon,col_r)
            st.plotly_chart(fig_ttm,use_container_width=True)
            with st.expander("📋 ETA by City"): st.dataframe(df_cities,use_container_width=True)

    with tab2:
        st.markdown('<div class="sec-title">⚡ Real-Time Seismic Triggers</div>',unsafe_allow_html=True)
        min_ts_mag = st.slider("Min Magnitude",5.0,8.0,6.5,0.5,key="ts_feed_mag")
        with st.spinner("Fetching USGS data…"):
            events = fetch_usgs_earthquakes(min_ts_mag,14)
        if events:
            df = pd.DataFrame(events)
            df["tsunami_risk"] = df.apply(lambda r:"⚠️ HIGH" if r["magnitude"]>=7.5 and r["depth"]<70
                                           else "🟡 WATCH" if r["magnitude"]>=7.0 else "🟢 LOW",axis=1)
            st.metric("Potential Events",len(df[df["magnitude"]>=7.5]),delta=f"of {len(df)} monitored")
            traces = [go.Scattergeo(lat=df["lat"],lon=df["lon"],mode="markers",
                marker=dict(size=(df["magnitude"]*4).clip(5,30),
                            color=["#ff2244" if r.get("tsunami")==1 else "#ffd700" if r["magnitude"]>=7.0 else "#00ff88" for _,r in df.iterrows()],
                            opacity=0.85,line=dict(width=0.5,color="rgba(255,255,255,0.2)")),
                hovertext=[f"M{r['magnitude']} — {r['place']}<br>Depth {r['depth']:.0f}km" for _,r in df.iterrows()],
                hoverinfo="text",name="Seismic Events")]
            st.plotly_chart(make_globe(traces,f"M{min_ts_mag}+ Events — Tsunami Risk"),use_container_width=True)
            st.dataframe(df[["magnitude","place","depth","tsunami","tsunami_risk"]].sort_values("magnitude",ascending=False).head(20),use_container_width=True)

    with tab3:
        st.markdown('<div class="sec-title">🌊 Wave Behavior Analysis</div>',unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1:
            wa_mag  = st.slider("Source Magnitude",6.0,9.5,8.0,0.1,key="wa_mag")
            wa_bath = st.slider("Ocean Depth (m)",100.0,8000.0,4000.0,key="wa_bath")
        with col2:
            wa_wh   = st.slider("Offshore Wave Height (m)",0.1,20.0,3.0,0.1,key="wa_wh")
            wa_dist = st.slider("Coastal Distance (km)",10.0,500.0,150.0,key="wa_dist")
        wa_color = get_risk(wa_wh)[1]
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(make_wave_propagation_chart(wa_mag,wa_bath,wa_color),use_container_width=True)
        with c2:
            fig_r,_ = make_runup_chart(wa_wh,wa_dist,wa_color)
            st.plotly_chart(fig_r,use_container_width=True)
        fig_tt,_ = make_travel_time_chart(wa_bath,0,140,wa_color)
        st.plotly_chart(fig_tt,use_container_width=True)

    with tab4:
        st.markdown('<div class="sec-title">📊 Tsunami Prediction History</div>',unsafe_allow_html=True)
        history = db.get_recent_predictions("tsunami")
        if history:
            df_h = pd.DataFrame(history)
            df_h["timestamp"] = pd.to_datetime(df_h["timestamp"],errors="coerce")
            display_cols = [c for c in ["timestamp","earthquake_mag","depth","wave_height_pred","eta_minutes","origin_lat","origin_lon"] if c in df_h.columns]
            st.dataframe(df_h[display_cols].head(20),use_container_width=True)
            if "wave_height_pred" in df_h.columns and df_h["wave_height_pred"].notna().any():
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(make_timeseries(df_h.head(25),"timestamp","wave_height_pred",COLOR,"Wave Height History","m"),use_container_width=True)
                with c2:
                    df_h["risk_level"] = df_h["wave_height_pred"].apply(lambda x:get_risk(x)[0] if pd.notna(x) else "Unknown")
                    rc = df_h["risk_level"].value_counts()
                    colors_map = {"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}
                    st.plotly_chart(make_donut(rc.index.tolist(),rc.values.tolist(),[colors_map.get(r,"#aaa") for r in rc.index],"Risk Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No tsunami predictions yet — run a prediction to populate history.</div>',unsafe_allow_html=True)
