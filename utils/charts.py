"""
Advanced Chart Helpers — All charts use the SENTINEL dark sci-fi theme
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ── Shared layout base ────────────────────────────────────────
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Share Tech Mono, monospace", color="#8ab4d0", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=["#00d4ff", "#9d00ff", "#ff7700", "#00ff88", "#ff2244", "#ffd700"],
)

AXIS_STYLE = dict(
    gridcolor="rgba(13,42,68,0.8)",
    zerolinecolor="rgba(0,212,255,0.2)",
    showgrid=True,
    tickfont=dict(family="Share Tech Mono", color="#4a7090", size=10),
)


def _layout(**kwargs):
    d = dict(**LAYOUT_BASE)
    d.update(kwargs)
    return d


# ── Gauge chart ───────────────────────────────────────────────
def make_gauge(value, max_val, title, color, unit="", reference=None):
    steps = [
        {"range": [0, max_val * 0.25], "color": "rgba(0,255,136,0.08)"},
        {"range": [max_val * 0.25, max_val * 0.5], "color": "rgba(255,215,0,0.08)"},
        {"range": [max_val * 0.5, max_val * 0.75], "color": "rgba(255,119,0,0.08)"},
        {"range": [max_val * 0.75, max_val], "color": "rgba(255,34,68,0.08)"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if reference else "gauge+number",
        value=value,
        number={"suffix": unit, "font": {"family": "Orbitron", "size": 36, "color": color}},
        delta={"reference": reference, "increasing": {"color": "#ff2244"}, "decreasing": {"color": "#00ff88"}} if reference else None,
        title={"text": title, "font": {"family": "Orbitron", "size": 14, "color": "#8ab4d0"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#2a4060",
                     "tickfont": {"family": "Share Tech Mono", "size": 9, "color": "#4a7090"}},
            "bar": {"color": color, "thickness": 0.25,
                    "line": {"color": color, "width": 2}},
            "bgcolor": "rgba(4,15,30,0.8)",
            "borderwidth": 1, "bordercolor": "#0d2a44",
            "steps": steps,
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": value},
        }
    ))
    fig.update_layout(**_layout(height=280))
    return fig


# ── Radar chart ───────────────────────────────────────────────
def make_radar(categories, values, color, title):
    cats_closed = categories + [categories[0]]
    vals_closed = values + [values[0]]
    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.12)",
        line=dict(color=color, width=2),
        name=title
    ))
    # Glow line
    fig.add_trace(go.Scatterpolar(
        r=[v * 0.9 for v in vals_closed], theta=cats_closed,
        fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.06)",
        line=dict(color=color, width=0.5, dash="dot"),
        showlegend=False
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], visible=True,
                            tickfont={"family": "Share Tech Mono", "color": "#3a5a70", "size": 8},
                            gridcolor="rgba(13,42,68,0.8)", linecolor="#0d2a44"),
            angularaxis=dict(tickfont={"family": "Share Tech Mono", "color": "#8ab4d0", "size": 10},
                             gridcolor="rgba(13,42,68,0.8)", linecolor="#0d2a44"),
            bgcolor="rgba(4,15,30,0.9)"
        ),
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        **_layout(height=380, showlegend=False)
    )
    return fig


# ── Globe scatter map ─────────────────────────────────────────
def make_globe(traces_list, title="Global Threat Map"):
    fig = go.Figure(data=traces_list)
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True, coastlinecolor="#1a3a5c", coastlinewidth=0.8,
            showland=True, landcolor="#061220",
            showocean=True, oceancolor="#020d1a",
            showlakes=True, lakecolor="#030f1e",
            showrivers=True, rivercolor="#041525",
            showcountries=True, countrycolor="#0d2a44", countrywidth=0.5,
            bgcolor="rgba(0,0,0,0)",
            projection_type="natural earth",
        ),
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        **_layout(height=480),
        legend=dict(
            bgcolor="rgba(4,12,25,0.9)", bordercolor="#0d2a44", borderwidth=1,
            font={"family": "Share Tech Mono", "size": 10, "color": "#8ab4d0"}
        )
    )
    return fig


# ── Time-series line ──────────────────────────────────────────
def make_timeseries(df, x_col, y_col, color, title, y_label=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color, line=dict(width=1, color="white")),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
        name=y_label or y_col
    ))
    fig.update_layout(
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE, title=y_label),
        **_layout(height=300)
    )
    return fig


# ── Bar chart ─────────────────────────────────────────────────
def make_bar(x, y, colors, title, x_label="", y_label="", text=None):
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker=dict(
            color=colors,
            line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
            cornerradius=4,
        ),
        text=text, textposition="auto",
        textfont=dict(family="Share Tech Mono", size=10),
    ))
    fig.update_layout(
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        xaxis=dict(**AXIS_STYLE, title=x_label),
        yaxis=dict(**AXIS_STYLE, title=y_label),
        **_layout(height=320)
    )
    return fig


# ── Donut chart ───────────────────────────────────────────────
def make_donut(labels, values, colors, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#020812", width=2)),
        textfont=dict(family="Share Tech Mono", size=10),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        **_layout(height=320),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"family": "Share Tech Mono", "size": 10})
    )
    return fig


# ── Scatter 2D ────────────────────────────────────────────────
def make_scatter(df, x, y, color_col, color_map, title, size_col=None):
    fig = px.scatter(
        df, x=x, y=y, color=color_col,
        color_discrete_map=color_map,
        size=size_col, size_max=18,
        title=title
    )
    fig.update_layout(
        **_layout(height=320),
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"family": "Share Tech Mono", "size": 10})
    )
    return fig


# ── Wind polar chart ──────────────────────────────────────────
def make_wind_polar(wind_speed, color, title):
    theta = list(range(0, 361, 15))
    r_outer = [wind_speed * (0.82 + 0.18 * np.cos(np.radians(t * 2.5))) for t in theta]
    r_mid   = [wind_speed * (0.5 + 0.1 * np.sin(np.radians(t * 3))) for t in theta]
    r_inner = [wind_speed * 0.18 for _ in theta]

    fig = go.Figure()
    for r, op, name in [(r_outer, 0.75, "Outer Wind Band"),
                         (r_mid,   0.45, "Inner Spiral"),
                         (r_inner, 0.25, "Eye Wall")]:
        fig.add_trace(go.Barpolar(
            r=r, theta=theta, width=[15] * len(theta),
            marker_color=[color] * len(theta),
            marker_line_color="rgba(0,0,0,0.3)",
            marker_line_width=0.5,
            opacity=op, name=name
        ))
    fig.update_layout(
        title=dict(text=title, font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        polar=dict(
            radialaxis=dict(range=[0, wind_speed * 1.15], showticklabels=True,
                            tickfont={"family": "Share Tech Mono", "size": 8, "color": "#3a5a70"},
                            gridcolor="rgba(13,42,68,0.7)", linecolor="#0d2a44"),
            angularaxis=dict(tickfont={"family": "Share Tech Mono", "size": 9, "color": "#4a7090"},
                             gridcolor="rgba(13,42,68,0.7)"),
            bgcolor="rgba(4,15,30,0.95)"
        ),
        **_layout(height=420, showlegend=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"family": "Share Tech Mono", "size": 10})
    )
    return fig


# ── Wave propagation chart ────────────────────────────────────
def make_wave_chart(wave_height, coastal_dist, wave_speed_kmh, color):
    dist = np.linspace(1, max(coastal_dist * 1.5, 500), 300)
    height_decay = wave_height * np.sqrt(np.maximum(coastal_dist, 1) / dist)
    height_decay = np.clip(height_decay, 0, wave_height * 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist, y=height_decay,
        mode="lines", fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)},0.12)",
        line=dict(color=color, width=2.5),
        name="Wave Height"
    ))
    # Danger zones
    for h, label, c in [(5, "Catastrophic >5m", "#ff2244"),
                         (2, "Dangerous >2m",   "#ff7700"),
                         (0.5, "Advisory >0.5m",  "#ffd700")]:
        fig.add_hline(y=h, line_dash="dash", line_color=c, line_width=1,
                      annotation_text=label,
                      annotation_font={"family": "Share Tech Mono", "size": 9, "color": c},
                      annotation_position="top right")
    fig.add_vline(x=coastal_dist, line_color="white", line_width=1.5, line_dash="dot",
                  annotation_text=f"  Coast  {coastal_dist:.0f}km",
                  annotation_font={"family": "Share Tech Mono", "size": 9, "color": "white"})
    fig.update_layout(
        title=dict(text="Tsunami Wave Propagation Model", font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        xaxis=dict(**AXIS_STYLE, title="Distance from Epicenter (km)"),
        yaxis=dict(**AXIS_STYLE, title="Wave Height (m)"),
        **_layout(height=320)
    )
    return fig


# ── Drought timeline ──────────────────────────────────────────
def make_drought_timeline(severity_score, duration, color):
    weeks = np.arange(0, duration + 8)
    decay_rate = 0.04 + max(0, severity_score - 1) * 0.02
    values = [max(0, severity_score - w * decay_rate * (1 + np.random.uniform(-0.1, 0.1))) for w in weeks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=values,
        mode="lines+markers",
        fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(color)},0.1)",
        line=dict(color=color, width=2.5),
        marker=dict(size=6, color=color)
    ))
    for thresh, label, c in [(1.5, "Extreme", "#ff2244"),
                               (1.0, "Severe",  "#ff7700"),
                               (0.5, "Moderate","#ffd700")]:
        fig.add_hline(y=thresh, line_dash="dash", line_color=c, line_width=1,
                      annotation_text=label,
                      annotation_font={"family": "Share Tech Mono", "size": 9, "color": c})
    fig.update_layout(
        title=dict(text="Projected Drought Severity Timeline", font={"family": "Orbitron", "size": 13, "color": "#8ab4d0"}),
        xaxis=dict(**AXIS_STYLE, title="Week"),
        yaxis=dict(**AXIS_STYLE, title="Severity Index"),
        **_layout(height=300)
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
