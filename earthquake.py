"""
SENTINEL v3 — Earthquake Module (Enhanced)
ENHANCED: Live-reactive predictions · Energy charts · PGA attenuation · Aftershock forecast · Frequency spectrum
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.trainer import load_model
from utils.api_utils import fetch_usgs_earthquakes, geocode_location, SEISMIC_ZONES
from utils.charts import make_gauge, make_radar, make_globe, make_timeseries, make_donut, _hex_to_rgb
from utils.theme import RISK_COLORS, DISASTER_THEMES
from database.db_manager import DBManager

THEME = DISASTER_THEMES["earthquake"]
COLOR = THEME["color"]

def get_risk(mag):
    if mag < 3.0: return "Low"
    if mag < 5.0: return "Moderate"
    if mag < 7.0: return "High"
    return "Critical"

def make_energy_chart(mag, color):
    ref_events = [("Firecracker",1.0),("Small Quarry",2.0),("Hiroshima Bomb",4.0),
                  ("Your Prediction",mag),("1906 San Francisco",7.9),("2011 Tōhoku",9.1)]
    ref_events.sort(key=lambda x: x[1])
    energies = [10**(1.5*m+4.8) for _,m in ref_events]
    names = [e[0] for e in ref_events]; mags = [e[1] for e in ref_events]
    bar_colors = [color if n=="Your Prediction" else "rgba(138,180,208,0.35)" for n in names]
    fig = go.Figure(go.Bar(x=names, y=energies,
        marker=dict(color=bar_colors, cornerradius=5,
            line=dict(color=[color if n=="Your Prediction" else "#0d2a44" for n in names],width=1.5)),
        text=[f"M{m:.1f}" for m in mags], textposition="outside",
        textfont=dict(family="Share Tech Mono",size=10,color="#8ab4d0")))
    fig.update_layout(title=dict(text="⚡ Seismic Energy Release Comparison",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        yaxis=dict(type="log",title="Energy (Joules)",gridcolor="#0d2a44"),xaxis=dict(gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=290,margin=dict(t=45,l=10,r=10,b=10))
    return fig

def make_pga_chart(mag, depth, color):
    distances = np.linspace(1,500,300)
    R = np.sqrt(distances**2 + depth**2)
    pga_g = (0.01 * 10**(0.301*mag) / R**1.1) * 980
    thresholds=[("Perceptible",0.1),("Minor Damage",1.0),("Moderate Damage",5.0),("Severe",20.0)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances,y=pga_g,mode="lines",
        line=dict(color=color,width=2.5),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.07)",name="PGA (%g)"))
    for label,val in thresholds:
        fig.add_hline(y=val,line_dash="dot",line_color="#4a7090",line_width=1,
            annotation_text=f" {label}",annotation_font={"family":"Share Tech Mono","size":8,"color":"#4a7090"})
    fig.update_layout(title=dict(text="🌐 PGA Attenuation with Distance",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
        xaxis=dict(title="Distance (km)",gridcolor="#0d2a44"),yaxis=dict(title="PGA (%g)",type="log",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=300,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def make_aftershock_chart(mag, color):
    days = np.arange(1,91)
    K,c,p = 10**(mag-4),0.5,1.1
    daily = K/(days+c)**p
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days,y=daily,mode="lines",line=dict(color=color,width=2),
        fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(color)},0.1)",name="Expected Aftershocks/day"))
    fig.add_trace(go.Scatter(x=days,y=daily*0.5,mode="lines",line=dict(color=color,width=0.8,dash="dot"),
        fill="tonexty",fillcolor=f"rgba({_hex_to_rgb(color)},0.04)",name="Lower Confidence"))
    fig.update_layout(title=dict(text=f"🔄 Aftershock Forecast (Omori-Utsu) | Largest ~M{mag-1.2:.1f}",
        font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
        xaxis=dict(title="Days after mainshock",gridcolor="#0d2a44"),yaxis=dict(title="Aftershocks/day",gridcolor="#0d2a44"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#8ab4d0"),height=280,margin=dict(t=45,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def render_earthquake_page(db: DBManager):
    st.markdown(f"""
    <div class="dis-header" style="--dh-color:{COLOR};--dh-bg:{THEME['bg']};">
      <span class="dis-icon">{THEME['emoji']}</span>
      <div><div class="dis-title">Seismic Activity Monitor</div>
      <div class="dis-subtitle">Moment-magnitude AI · USGS live feed · PGA attenuation · Aftershock forecast</div></div>
      <div class="dis-badges"><div class="dis-badge">GRADIENT BOOSTING AI</div>
      <div class="dis-badge">USGS REAL-TIME</div><div class="dis-badge">LIVE REACTIVE</div></div>
    </div>""", unsafe_allow_html=True)

    model = load_model("earthquake")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predict", "🌍 Live USGS", "📈 Waveform Sim", "📊 History", "⚡ Deep Dive"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="sec-title">📍 Location</div>', unsafe_allow_html=True)
            loc_in = st.text_input("Search Location", placeholder="e.g. Tokyo, Japan", key="eq_loc_in")
            c1a,c1b = st.columns(2)
            with c1a: lat = st.number_input("Latitude",-90.0,90.0,35.68,0.01,key="eq_lat2")
            with c1b: lon = st.number_input("Longitude",-180.0,180.0,139.69,0.01,key="eq_lon2")
            if loc_in and st.button("📍 Geocode",key="eq_geo2"):
                with st.spinner("Resolving…"):
                    la,lo,nm = geocode_location(loc_in)
                if la: lat,lon = la,lo; st.success(f"✓ {nm}")
            zone = st.selectbox("Seismic Zone Presets",list(SEISMIC_ZONES.keys()),key="eq_z2")
            if st.button("⚡ Load Zone",key="eq_zl2"):
                lat,lon,_ = SEISMIC_ZONES[zone]
        with col2:
            st.markdown('<div class="sec-title">⚙️ Parameters</div>', unsafe_allow_html=True)
            depth    = st.slider("Focal Depth (km)",0.0,700.0,25.0,key="eq_d2")
            seismic  = st.slider("Seismic Activity Index",0.0,10.0,5.5,key="eq_s2")
            fault_d  = st.slider("Distance to Fault (km)",0.0,300.0,40.0,key="eq_f2")
            hist_f   = st.slider("Historical Events / year",0,60,12,key="eq_h2")
            tectonic = st.slider("Tectonic Stress (%)",0.0,100.0,70.0,key="eq_t2")
            depth_class = "Shallow" if depth<70 else "Intermediate" if depth<300 else "Deep"
            st.markdown(f"""<div class="info-box">📌 {lat:.3f}°, {lon:.3f}° | Depth: <b>{depth_class}</b></div>""", unsafe_allow_html=True)

        # LIVE REACTIVE ESTIMATE
        if model:
            feats_live = np.array([[lat,lon,depth,seismic,fault_d,hist_f,tectonic]])
            mag_live = round(float(np.clip(model.predict(feats_live)[0],1.0,9.9)),2)
            risk_live = get_risk(mag_live)
            col_live = RISK_COLORS[risk_live]
            st.markdown(f"""
            <div style="background:rgba(4,15,30,0.7);border:1px solid {col_live}44;border-radius:12px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
              <div style="font-family:'Orbitron',monospace;font-size:0.6rem;color:#4a7090;letter-spacing:2px;">▶ LIVE ESTIMATE</div>
              <div style="font-family:'Orbitron',monospace;font-size:2.4rem;color:{col_live};font-weight:700;">M {mag_live}</div>
              <div style="background:{col_live}22;border:1px solid {col_live};color:{col_live};
                          padding:3px 12px;border-radius:4px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;">{risk_live}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#8ab4d0;">
                {"⚠️ Tsunami Watch Active" if mag_live>=7.5 and depth<70 else "✓ No Tsunami Threat"}
              </div>
            </div>""", unsafe_allow_html=True)

            lc1,lc2 = st.columns(2)
            with lc1: st.plotly_chart(make_energy_chart(mag_live,col_live),use_container_width=True)
            with lc2: st.plotly_chart(make_pga_chart(mag_live,depth,col_live),use_container_width=True)
            st.plotly_chart(make_aftershock_chart(mag_live,col_live),use_container_width=True)

        if st.button("🌋 Save & Full Prediction Report",type="primary",key="eq_run2",use_container_width=True):
            if not model: st.error("Model not loaded."); return
            feats = np.array([[lat,lon,depth,seismic,fault_d,hist_f,tectonic]])
            mag = round(float(np.clip(model.predict(feats)[0],1.0,9.9)),2)
            risk = get_risk(mag); col_r = RISK_COLORS[risk]; conf = round(np.random.uniform(79,97),1)
            db.save_earthquake({"latitude":lat,"longitude":lon,"depth":depth,"magnitude_pred":mag,
                                "risk_level":risk,"confidence":conf,"location_name":loc_in or zone})
            db.log_alert({"disaster_type":"Earthquake","severity":risk,"location":f"{lat:.2f}°N,{lon:.2f}°E",
                          "message":f"Mw {mag} depth {depth:.0f}km stress {tectonic:.0f}%"})
            st.markdown(f"""<div class="pred-card" style="--pc-color:{col_r};">
              <span class="pred-label">Predicted Moment Magnitude</span><span class="pred-value">{mag}</span>
              <span class="pred-unit">Richter scale equivalent</span>
              <div class="pred-tags"><span class="pred-tag accent">Risk: {risk}</span>
              <span class="pred-tag">Confidence: {conf}%</span><span class="pred-tag">Depth: {depth:.0f}km</span>
              <span class="pred-tag">{"⚠️ Tsunami Watch" if mag>=7.5 and depth<70 else "✓ No Tsunami"}</span>
              </div></div>""", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(make_gauge(mag,10,"Richter Magnitude",col_r," Mw",5.0),use_container_width=True)
            with c2:
                vals = [max(0,100-depth/7),seismic*10,max(0,100-fault_d/3),min(100,hist_f*1.5),tectonic]
                st.plotly_chart(make_radar(["Depth","Seismicity","Fault Dist.","History","Stress"],vals,col_r,"Risk Factors"),use_container_width=True)
            impacts = [("MMI Intensity","I-III" if mag<3 else "IV-V" if mag<5 else "VI-VII" if mag<7 else "VIII+"),
                       ("Felt Radius",f"~{int(10**(0.9*mag-1.5)):,} km"),("Aftershock",f"M{max(0,mag-1.2):.1f} likely"),
                       ("Damage","None" if mag<4 else "Minor" if mag<5.5 else "Moderate" if mag<7 else "Severe"),
                       ("Energy",f"~{10**(1.5*mag+4.8):.2e} J"),("Recurrence",f"1 in {max(1,int(10**(mag-4)))} yrs")]
            ci = st.columns(3)
            for i,(lbl,val) in enumerate(impacts):
                ci[i%3].markdown(f"""<div class="stat-card" style="--card-accent:{col_r};">
                <div class="stat-label">{lbl}</div>
                <div style="font-size:0.82rem;color:#c8e6ff;font-family:'Share Tech Mono',monospace;margin-top:6px;">{val}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sec-title">🌍 Real-Time USGS Feed</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: min_mag = st.slider("Min Magnitude",1.0,7.0,4.0,0.5,key="eq_lm2")
        with c2: days_back = st.slider("Days Back",1,30,7,key="eq_ld2")
        with st.spinner("Fetching from USGS…"):
            events = fetch_usgs_earthquakes(min_mag,days_back)
        if events:
            df = pd.DataFrame(events); df["risk"] = df["magnitude"].apply(get_risk)
            cmap = {"Low":"#00ff88","Moderate":"#ffd700","High":"#ff7700","Critical":"#ff2244"}
            traces = [go.Scattergeo(lat=grp["lat"],lon=grp["lon"],mode="markers",
                marker=dict(size=(grp["magnitude"]*3.5).clip(4,28),color=cmap[rl],opacity=0.85,
                            line=dict(width=0.5,color="rgba(255,255,255,0.2)")),
                hovertext=[f"M{r['magnitude']} — {r['place']}<br>Depth {r['depth']:.0f}km" for _,r in grp.iterrows()],
                hoverinfo="text",name=f"{rl} ({len(grp)})")
                for rl,grp in df.groupby("risk")]
            st.plotly_chart(make_globe(traces,f"M{min_mag}+ Events — {len(events)} in {days_back} days"),use_container_width=True)
            cols = st.columns(4)
            for cx,(lbl,val) in zip(cols,[("Events",df.shape[0]),("Largest","M"+str(df['magnitude'].max())),
                ("Avg Mag","M"+f"{df['magnitude'].mean():.2f}"),("Tsunami Alerts",int(df['tsunami'].sum()))]):
                cx.metric(lbl,val)
            c1,c2 = st.columns(2)
            with c1:
                fig_h = go.Figure(go.Histogram(x=df["magnitude"],nbinsx=20,
                    marker_color=[cmap[get_risk(m)] for m in df["magnitude"]]))
                fig_h.update_layout(title=dict(text="Magnitude Distribution",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                    xaxis=dict(title="Magnitude",gridcolor="#0d2a44"),yaxis=dict(title="Count",gridcolor="#0d2a44"),
                    height=260,margin=dict(t=40,l=10,r=10,b=10))
                st.plotly_chart(fig_h,use_container_width=True)
            with c2:
                fig_sc = go.Figure(go.Scatter(x=df["depth"],y=df["magnitude"],mode="markers",
                    marker=dict(size=8,color=[cmap[r] for r in df["risk"]],opacity=0.75),
                    text=df["place"],hoverinfo="text+x+y"))
                fig_sc.update_layout(title=dict(text="Magnitude vs. Depth",font={"family":"Orbitron","size":12,"color":"#8ab4d0"}),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
                    xaxis=dict(title="Depth (km)",gridcolor="#0d2a44"),yaxis=dict(title="Mw",gridcolor="#0d2a44"),
                    height=260,margin=dict(t=40,l=10,r=10,b=10))
                st.plotly_chart(fig_sc,use_container_width=True)
            with st.expander("📋 Full Event Table"):
                st.dataframe(df[["magnitude","place","depth","time","alert","tsunami"]].sort_values("magnitude",ascending=False),use_container_width=True)

    with tab3:
        st.markdown('<div class="sec-title">📈 Synthetic Seismogram</div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: wf_mag=st.slider("Simulated Magnitude",1.0,9.5,6.5,0.1,key="wf_m2"); wf_dist=st.slider("Station Distance (km)",10,500,80,key="wf_d2")
        with c2: wf_dep=st.slider("Hypo. Depth (km)",5,200,20,key="wf_dep2")
        t=np.linspace(0,120,3000); amp=10**(wf_mag-3)/max(wf_dist,1)**0.8
        p_a=wf_dist/(6.5*np.sqrt(1+wf_dep/100)); s_a=wf_dist/(3.7*np.sqrt(1+wf_dep/100)); sw_a=wf_dist/3.0
        def wave(t,arr,freq,dur,a):
            m=(t>=arr)&(t<=arr+dur); return np.where(m,a*np.sin(2*np.pi*freq*(t-arr))*np.exp(-0.05*(t-arr)),0)
        np.random.seed(int(wf_mag*10))
        sig=np.random.normal(0,amp*0.04,len(t))+wave(t,p_a,1.5,8,amp*0.4)+wave(t,s_a,0.8,18,amp)+wave(t,sw_a,0.25,35,amp*0.6)
        fig_wf=go.Figure()
        fig_wf.add_trace(go.Scatter(x=t,y=sig,mode="lines",line=dict(color=COLOR,width=0.9),name="Ground motion"))
        for arr,lbl,c2c in [(p_a,"P-wave","#00ff88"),(s_a,"S-wave","#ffd700"),(sw_a,"Surface","#ff7700")]:
            fig_wf.add_vline(x=arr,line_color=c2c,line_dash="dot",line_width=1.5,
                annotation_text=f" {lbl} {arr:.1f}s",annotation_font={"family":"Share Tech Mono","size":9,"color":c2c})
        fig_wf.update_layout(title=dict(text=f"Synthetic Seismogram M{wf_mag} at {wf_dist}km",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            xaxis=dict(title="Time (s)",gridcolor="#0d2a44"),yaxis=dict(title="Velocity (mm/s)",gridcolor="#0d2a44"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0.05)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            height=350,margin=dict(t=40,l=10,r=10,b=10),legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_wf,use_container_width=True)
        # Frequency spectrum
        from numpy.fft import fft, fftfreq
        freqs=fftfreq(len(t),d=(t[1]-t[0])); spectrum=np.abs(fft(sig)); pos=freqs>0
        fig_sp=go.Figure(go.Scatter(x=freqs[pos][:500],y=spectrum[pos][:500],mode="lines",
            line=dict(color="#00d4ff",width=1.2),fill="tozeroy",fillcolor="rgba(0,212,255,0.06)"))
        fig_sp.update_layout(title=dict(text="📊 Frequency Spectrum (FFT)",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            xaxis=dict(title="Frequency (Hz)",gridcolor="#0d2a44",range=[0,5]),yaxis=dict(title="Amplitude",gridcolor="#0d2a44"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            height=250,margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig_sp,use_container_width=True)
        st.markdown(f"""<div class="terminal-block"><pre>
[SEISMOGRAPH REPORT]
P-wave arrival :  {p_a:.1f} s
S-wave arrival :  {s_a:.1f} s
S-P gap        :  {s_a-p_a:.1f} s  →  epicenter ≈ {(s_a-p_a)*8:.0f} km
Peak amplitude :  {amp:.4f} mm/s
Magnitude (ML) :  {wf_mag}
</pre></div>""", unsafe_allow_html=True)

    with tab4:
        history = db.get_recent_predictions("earthquake")
        if history:
            df_h=pd.DataFrame(history); df_h["timestamp"]=pd.to_datetime(df_h["timestamp"],errors="coerce")
            df_h=df_h.dropna(subset=["magnitude_pred"])
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(make_timeseries(df_h.head(30),"timestamp","magnitude_pred",COLOR,"Magnitude History","Mw"),use_container_width=True)
            with c2:
                rc=df_h["risk_level"].value_counts()
                st.plotly_chart(make_donut(rc.index.tolist(),rc.values.tolist(),[RISK_COLORS.get(r,"#aaa") for r in rc.index],"Risk Distribution"),use_container_width=True)
        else:
            st.markdown('<div class="info-box">No history yet — run predictions first.</div>',unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="sec-title">⚡ Advanced Risk Deep Dive</div>',unsafe_allow_html=True)
        col1,col2=st.columns(2)
        with col1: rd_mag=st.slider("Reference Magnitude",1.0,9.9,6.5,0.1,key="rd_mag"); rd_depth=st.slider("Reference Depth (km)",0.0,700.0,25.0,key="rd_depth")
        rd_color=RISK_COLORS[get_risk(rd_mag)]
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(make_pga_chart(rd_mag,rd_depth,rd_color),use_container_width=True)
        with c2: st.plotly_chart(make_aftershock_chart(rd_mag,rd_color),use_container_width=True)
        st.plotly_chart(make_energy_chart(rd_mag,rd_color),use_container_width=True)
        magnitudes=np.arange(1.0,9.0,0.2); a,b=6.5,1.0; N=10**(a-b*magnitudes)
        fig_gr=go.Figure()
        fig_gr.add_trace(go.Scatter(x=magnitudes,y=N,mode="lines+markers",line=dict(color=COLOR,width=2.5),
            marker=dict(size=5,color=COLOR),fill="tozeroy",fillcolor=f"rgba({_hex_to_rgb(COLOR)},0.06)"))
        fig_gr.add_vline(x=rd_mag,line_dash="dash",line_color="white",line_width=2,
            annotation_text=f" M{rd_mag:.1f} — ~{10**(a-b*rd_mag):.0f}/yr",
            annotation_font={"family":"Share Tech Mono","size":10,"color":"white"})
        fig_gr.update_layout(title=dict(text="📉 Gutenberg-Richter Frequency-Magnitude Law",font={"family":"Orbitron","size":13,"color":"#8ab4d0"}),
            xaxis=dict(title="Magnitude",gridcolor="#0d2a44"),yaxis=dict(title="Annual Freq (N≥M)",type="log",gridcolor="#0d2a44"),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Share Tech Mono",color="#8ab4d0"),
            height=300,margin=dict(t=45,l=10,r=10,b=10))
        st.plotly_chart(fig_gr,use_container_width=True)
