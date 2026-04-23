"""
MLB Free Agent Value Calculator
================================
A data-driven contract valuation tool for MLB front offices.

Run locally:
    python -m streamlit run app.py

Author: [Your Name]
Course: ACC102 — Mini Assignment, Track 4
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="MLB FA Value Calculator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME — Editorial sports-analytics aesthetic (light mode)
# =============================================================================
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;600;800&display=swap');

  :root {
    --bg-primary:   #FAF8F4;
    --bg-card:      #FFFFFF;
    --bg-elevated:  #F1EEE8;
    --accent-green: #1B5E20;
    --accent-red:   #C62828;
    --accent-amber: #E65100;
    --accent-blue:  #0277BD;
    --text-primary: #1A1A1A;
    --text-muted:   #6B6B6B;
    --border:       #E0DCD2;
  }

  .stApp { background-color: var(--bg-primary); font-family: 'Manrope', sans-serif; color: var(--text-primary); }

  h1, h2, h3, h4, h5 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
  }
  p, label, span, div { color: var(--text-primary); }

  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted) !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
  }

  .brand-bar {
    border-left: 4px solid var(--accent-green);
    padding: 6px 0 6px 16px;
    margin-bottom: 16px;
  }
  .brand-bar .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--accent-green);
    font-weight: 600;
  }
  .brand-bar h1 {
    margin: 0 !important;
    font-size: 34px !important;
    line-height: 1.1;
  }
  .brand-bar p {
    color: var(--text-muted);
    font-size: 14px;
    margin: 4px 0 0 0;
  }

  .verdict {
    padding: 22px 26px;
    border-radius: 6px;
    margin: 16px 0;
    border-left: 5px solid;
    background: var(--bg-card);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .verdict-bargain  { background: #E8F5E9; border-color: var(--accent-green); }
  .verdict-overpaid { background: #FFEBEE; border-color: var(--accent-red);   }
  .verdict-fair     { background: #FFF3E0; border-color: var(--accent-amber); }
  .verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 600;
  }
  .verdict-amount {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.3;
  }
  .verdict-detail {
    font-size: 14px;
    color: var(--text-muted);
    margin-top: 8px;
  }

  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin: 28px 0 10px 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    font-weight: 600;
  }

  [data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] h2 { font-size: 18px !important; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 14px 22px;
    color: var(--text-muted);
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom: 2px solid var(--accent-green) !important;
  }

  .stDataFrame { font-family: 'IBM Plex Mono', monospace; }

  .footer {
    margin-top: 56px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.6;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def base_layout(**overrides):
    """Base layout for plotly. Pass overrides for axis configs."""
    layout = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#1A1A1A", size=12),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    layout.update(overrides)
    return layout

GRID = "#E0DCD2"
ACCENT_GREEN = "#1B5E20"
ACCENT_RED   = "#C62828"
ACCENT_AMBER = "#E65100"
ACCENT_BLUE  = "#0277BD"


# =============================================================================
# DATA LOADING
# =============================================================================
# Use a relative path: start at app.py's location, then go into data/processed/
DATA_DIR = Path(__file__).parent / "data" / "processed"

@st.cache_data
def load_data():
    try:
        master = pd.read_parquet(DATA_DIR / "master_dataset.parquet")
        aging  = pd.read_parquet(DATA_DIR / "aging_curve.parquet")
        dpw    = pd.read_parquet(DATA_DIR / "dollar_per_war.parquet")
        return {"master": master, "aging": aging, "dpw": dpw}
    except FileNotFoundError as e:
        st.error(
            f"❌ Data file missing: {e.filename}\n\n"
            f"Please run `analysis.ipynb` end-to-end first to generate "
            f"the processed Parquet files in `{DATA_DIR}/`."
        )
        st.stop()

data   = load_data()
master = data["master"]
aging  = data["aging"]
dpw    = data["dpw"]

trend_model = LinearRegression().fit(dpw[["yearID"]].values,
                                     dpw["dollar_per_war"].values)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def get_multiplier(age: int) -> float:
    if age < aging["age"].min():
        return float(aging.iloc[0]["multiplier"])
    if age > aging["age"].max():
        excess = age - aging["age"].max()
        return float(aging.iloc[-1]["multiplier"]) * (0.95 ** excess)
    return float(aging.loc[aging["age"] == age, "multiplier"].iloc[0])


def projected_dpw(year: int) -> float:
    return float(trend_model.predict([[year]])[0])


def estimate_contract(recent_war, current_age, contract_years, start_year):
    current_mult = get_multiplier(current_age)
    peak_war = recent_war / current_mult if current_mult > 0 else recent_war
    rows = []
    for offset in range(contract_years):
        yr   = start_year + offset
        age  = current_age + offset
        mult = get_multiplier(age)
        proj_war = peak_war * mult
        dpw_proj = projected_dpw(yr)
        rows.append({
            "Year": yr, "Age": age,
            "Multiplier": mult,
            "Projected WAR": proj_war,
            "$/WAR ($M)": dpw_proj / 1e6,
            "Value ($M)": proj_war * dpw_proj / 1e6,
        })
    df = pd.DataFrame(rows)
    return {"breakdown": df, "fair_value_M": df["Value ($M)"].sum()}


def monte_carlo_contract(recent_war, war_std, current_age, contract_years,
                          start_year, n_sims=1000, seed=42):
    rng = np.random.default_rng(seed)
    current_mult = get_multiplier(current_age)
    peak_war = recent_war / current_mult if current_mult > 0 else recent_war
    totals = np.zeros(n_sims)
    for offset in range(contract_years):
        yr   = start_year + offset
        age  = current_age + offset
        mult = get_multiplier(age)
        sampled = rng.normal(peak_war * mult, war_std, size=n_sims)
        sampled = np.maximum(sampled, -1.0)
        totals += sampled * projected_dpw(yr) / 1e6
    return {
        "mean":   float(np.mean(totals)),
        "median": float(np.median(totals)),
        "p05":    float(np.percentile(totals,  5)),
        "p95":    float(np.percentile(totals, 95)),
        "all_totals": totals,
    }


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
    <div class="brand-bar">
      <div class="eyebrow">⚾ FRONT OFFICE ANALYTICS · 2000–2016</div>
      <h1>MLB Free Agent Value Calculator</h1>
      <p>A data-driven contract valuation tool · Built with Lahman Database open data</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎛️ Market Context")
    latest = dpw.iloc[-1]
    earliest = dpw.iloc[0]
    growth = (latest["dollar_per_war"] / earliest["dollar_per_war"] - 1) * 100

    st.metric(
        f"$/WAR ({int(latest['yearID'])})",
        f"${latest['dollar_per_war']/1e6:.2f}M",
        delta=f"{growth:+.0f}% vs {int(earliest['yearID'])}",
    )
    st.metric("Players in dataset", f"{master['playerID'].nunique():,}")
    st.metric("Player-seasons analysed", f"{len(master):,}")

    st.markdown("---")
    st.markdown("### About this tool")
    st.caption(
        "Uses simplified WAR (wOBA / FIP-based), an empirically-derived "
        "aging curve, and league-wide $/WAR market price to estimate the "
        "fair value of MLB player contracts. "
        "Data: Lahman Baseball Database (1985–2016 salaries)."
    )


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "👤  Player Explorer",
    "💰  Contract Calculator",
    "🏆  Market Leaderboard",
])


# -----------------------------------------------------------------------------
# TAB 1
# -----------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-label">SELECT A PLAYER</div>', unsafe_allow_html=True)

    career = (master.groupby(["playerID", "name"])
              .agg(career_war=("WAR", "sum"),
                   seasons=("yearID", "count"))
              .reset_index()
              .sort_values("career_war", ascending=False))

    col_a, col_b = st.columns([2, 1])
    with col_a:
        names_list = career["name"].tolist()
        default_name = "Albert Pujols" if "Albert Pujols" in names_list else names_list[0]
        choice = st.selectbox(
            "Player",
            options=names_list,
            index=names_list.index(default_name),
            help="Sorted by career WAR within our 2000–2016 window.",
        )
    with col_b:
        st.markdown("&nbsp;")
        st.caption(f"Showing top **{len(career):,}** players ranked by career WAR")

    pdata = master[master["name"] == choice].sort_values("yearID").copy()
    pinfo = pdata.iloc[0]

    st.markdown('<div class="section-label">PLAYER PROFILE</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Position",      pinfo["player_type"])
    c2.metric("First Season",  int(pdata["yearID"].min()))
    c3.metric("Last Season",   int(pdata["yearID"].max()))
    c4.metric("Career WAR",    f"{pdata['WAR'].sum():.1f}")
    if pdata["salary"].notna().any():
        c5.metric("Career Salary", f"${pdata['salary'].sum()/1e6:.0f}M")
    else:
        c5.metric("Career Salary", "—")

    st.markdown('<div class="section-label">CAREER TIMELINE</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdata["yearID"], y=pdata["WAR"],
        name="WAR",
        marker_color=ACCENT_GREEN,
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>WAR: %{y:.2f}<extra></extra>",
    ))
    layout_extras = dict(
        xaxis=dict(title="Season", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="WAR", gridcolor=GRID, zeroline=False),
    )
    if pdata["salary"].notna().any():
        fig.add_trace(go.Scatter(
            x=pdata["yearID"], y=pdata["salary"]/1e6,
            name="Salary ($M)", yaxis="y2", mode="lines+markers",
            line=dict(color=ACCENT_AMBER, width=2.5),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Salary: $%{y:.1f}M<extra></extra>",
        ))
        layout_extras["yaxis2"] = dict(
            title="Salary ($M)", overlaying="y", side="right",
            showgrid=False, color=ACCENT_AMBER,
        )

    fig.update_layout(**base_layout(
        title=dict(text=f"{choice} — WAR vs Salary",
                   font=dict(size=16, family="Manrope")),
        legend=dict(orientation="h", y=1.12, x=0),
        height=400,
        **layout_extras,
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">SEASON-BY-SEASON</div>', unsafe_allow_html=True)
    show_cols = ["yearID", "age", "bat_PA", "IP", "WAR", "salary"]
    table = pdata[show_cols].rename(columns={
        "yearID": "Year", "age": "Age", "bat_PA": "PA", "IP": "IP",
        "WAR": "WAR", "salary": "Salary ($)",
    })
    st.dataframe(
        table.style.format({
            "PA": "{:.0f}", "IP": "{:.1f}", "WAR": "{:.2f}",
            "Salary ($)": lambda x: f"${x:,.0f}" if pd.notna(x) else "—",
        }),
        use_container_width=True, height=320,
    )


# -----------------------------------------------------------------------------
# TAB 2
# -----------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-label">CONTRACT INPUTS</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        recent_war = st.number_input(
            "Recent WAR", min_value=-2.0, max_value=15.0, value=4.0, step=0.5,
            help="The player's WAR over his most recent full season.",
        )
    with c2:
        current_age = st.number_input(
            "Current Age", min_value=21, max_value=42, value=29, step=1,
            help="Player's age at the START of the contract.",
        )
    with c3:
        contract_years = st.slider("Contract Length (yrs)", 1, 10, 5)
    with c4:
        start_year = st.number_input(
            "Start Year", min_value=2010, max_value=2030, value=2017, step=1,
        )

    actual_contract = st.number_input(
        "💵 Actual Contract Total ($M) — optional, leave 0 to skip comparison",
        min_value=0.0, max_value=600.0, value=0.0, step=5.0,
    )

    result = estimate_contract(recent_war, int(current_age),
                               int(contract_years), int(start_year))
    fair_M = result["fair_value_M"]

    st.markdown('<div class="section-label">VERDICT</div>', unsafe_allow_html=True)
    if actual_contract > 0:
        diff = actual_contract - fair_M
        diff_pct = (diff / fair_M * 100) if fair_M > 0 else 0
        if diff > fair_M * 0.15:
            cls, label, msg = "verdict-overpaid", "❌ OVERPAID", \
                f"Team is paying ${diff:,.1f}M above fair value ({diff_pct:+.0f}%)."
        elif diff < -fair_M * 0.15:
            cls, label, msg = "verdict-bargain", "✅ BARGAIN", \
                f"Team is saving ${-diff:,.1f}M vs fair value ({diff_pct:+.0f}%)."
        else:
            cls, label, msg = "verdict-fair", "⚖️ FAIR DEAL", \
                f"Within 15% of fair value (${diff:+,.1f}M)."
        st.markdown(f"""
            <div class="verdict {cls}">
              <div class="verdict-label">{label}</div>
              <div class="verdict-amount">Fair Value: ${fair_M:,.1f}M &nbsp;·&nbsp; Actual: ${actual_contract:,.1f}M</div>
              <div class="verdict-detail">{msg}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="verdict verdict-fair">
              <div class="verdict-label">ESTIMATED FAIR VALUE</div>
              <div class="verdict-amount">${fair_M:,.1f}M total over {contract_years} year{'s' if contract_years>1 else ''}</div>
              <div class="verdict-detail">Average annual value: ${fair_M/contract_years:,.1f}M · Enter an actual contract above to compare.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">YEAR-BY-YEAR PROJECTION</div>',
                unsafe_allow_html=True)

    bd = result["breakdown"].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bd["Year"], y=bd["Projected WAR"],
        marker_color=ACCENT_GREEN, opacity=0.7,
        name="Projected WAR",
        hovertemplate="<b>%{x}</b> (age %{customdata})<br>WAR: %{y:.2f}<extra></extra>",
        customdata=bd["Age"],
    ))
    fig.add_trace(go.Scatter(
        x=bd["Year"], y=bd["Value ($M)"],
        yaxis="y2", mode="lines+markers", name="Annual Value ($M)",
        line=dict(color=ACCENT_AMBER, width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Value: $%{y:.1f}M<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        xaxis=dict(title="Year", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Projected WAR", gridcolor=GRID, zeroline=False),
        yaxis2=dict(title="Annual Value ($M)", overlaying="y", side="right",
                    showgrid=False, color=ACCENT_AMBER),
        legend=dict(orientation="h", y=1.12, x=0),
        height=380,
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        bd.style.format({
            "Multiplier":   "{:.3f}",
            "Projected WAR":"{:.2f}",
            "$/WAR ($M)":   "{:.2f}",
            "Value ($M)":   "${:.2f}M",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="section-label">MONTE CARLO UNCERTAINTY ANALYSIS</div>',
                unsafe_allow_html=True)

    with st.expander("ℹ️  What is this?", expanded=False):
        st.markdown(
            "The point estimate above assumes the aging curve plays out exactly. "
            "In reality, year-to-year WAR varies considerably (injuries, regression, "
            "luck). Monte Carlo simulation runs the contract **1,000+ times** with "
            "random variation around the projected path, giving us a *distribution* "
            "of plausible contract outcomes — not just one number."
        )

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        war_std = st.slider(
            "Year-to-year WAR volatility (σ)",
            min_value=0.5, max_value=3.0, value=1.5, step=0.1,
            help="Standard deviation of yearly WAR around the projected curve. "
                 "1.0–2.0 is typical for established players.",
        )
    with mc_col2:
        n_sims = st.select_slider(
            "Number of simulations",
            options=[500, 1000, 2000, 5000], value=1000,
        )

    mc = monte_carlo_contract(recent_war, war_std, int(current_age),
                              int(contract_years), int(start_year), n_sims=n_sims)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median",          f"${mc['median']:.1f}M")
    m2.metric("Mean",            f"${mc['mean']:.1f}M")
    m3.metric("5th percentile",  f"${mc['p05']:.1f}M",
              help="In the worst 5% of scenarios, the contract is worth less than this.")
    m4.metric("95th percentile", f"${mc['p95']:.1f}M",
              help="In the best 5% of scenarios, the contract is worth more than this.")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc["all_totals"], nbinsx=50,
        marker_color=ACCENT_BLUE, opacity=0.75,
        hovertemplate="$%{x:.1f}M: %{y} simulations<extra></extra>",
    ))
    fig.add_vline(x=mc["median"], line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
                  annotation_text=f"Median ${mc['median']:.1f}M",
                  annotation_position="top")
    if actual_contract > 0:
        fig.add_vline(x=actual_contract, line=dict(color=ACCENT_RED, width=2),
                      annotation_text=f"Actual ${actual_contract:.0f}M",
                      annotation_position="top")
    fig.update_layout(**base_layout(
        title=dict(text=f"Distribution of {n_sims:,} simulated fair-value outcomes",
                   font=dict(size=14, family="Manrope")),
        xaxis=dict(title="Fair Contract Value ($M)", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Number of Simulations", gridcolor=GRID, zeroline=False),
        height=350, showlegend=False,
    ))
    st.plotly_chart(fig, use_container_width=True)

    if actual_contract > 0:
        prob_overpaid = float((mc["all_totals"] < actual_contract).mean()) * 100
        st.info(
            f"📊 In **{prob_overpaid:.0f}%** of simulations, the team's actual "
            f"${actual_contract:.0f}M offer exceeds the player's fair value — "
            f"that is the empirical probability of being overpaid."
        )


# -----------------------------------------------------------------------------
# TAB 3
# -----------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-label">FILTERS</div>', unsafe_allow_html=True)

    valued = master.dropna(subset=["salary", "WAR"]).copy()
    valued = valued.merge(dpw[["yearID", "dollar_per_war"]], on="yearID")
    valued["expected_value"] = valued["WAR"] * valued["dollar_per_war"]
    valued["surplus"]   = valued["expected_value"] - valued["salary"]
    valued["surplus_M"] = valued["surplus"] / 1e6
    valued["salary_M"]  = valued["salary"] / 1e6

    f1, f2, f3 = st.columns(3)
    with f1:
        yr_range = st.slider(
            "Year range",
            int(valued["yearID"].min()), int(valued["yearID"].max()),
            (2010, int(valued["yearID"].max())),
        )
    with f2:
        min_salary = st.slider(
            "Minimum salary ($M)", 0.5, 20.0, 3.0, 0.5,
            help="Filters out pre-arbitration players, who are structurally underpaid.",
        )
    with f3:
        ptype = st.selectbox("Player type", ["All", "Batter", "Pitcher"])

    filtered = valued[
        (valued["yearID"].between(*yr_range))
        & (valued["salary_M"] >= min_salary)
    ]
    if ptype != "All":
        filtered = filtered[filtered["player_type"] == ptype]

    st.caption(f"**{len(filtered):,}** player-seasons match these filters.")

    st.markdown('<div class="section-label">CONTRACT VALUE MAP</div>',
                unsafe_allow_html=True)

    fig = px.scatter(
        filtered, x="salary_M", y="WAR",
        color="surplus_M", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data={"name": True, "yearID": True, "age": True,
                    "salary_M": ":.1f", "WAR": ":.2f", "surplus_M": ":.1f"},
        labels={"salary_M": "Salary ($M)", "WAR": "WAR (simplified)",
                "surplus_M": "Surplus ($M)"},
    )
    avg_dpw = dpw[dpw["yearID"].between(*yr_range)]["dollar_per_war"].mean()
    if len(filtered):
        xs = np.linspace(min_salary, filtered["salary_M"].max(), 100)
        fig.add_trace(go.Scatter(
            x=xs, y=xs * 1e6 / avg_dpw, mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", dash="dash", width=1.5),
            name=f"Break-even (avg $/WAR ≈ ${avg_dpw/1e6:.1f}M)",
            hoverinfo="skip",
        ))
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")),
                      selector=dict(mode="markers"))
    fig.update_layout(**base_layout(
        xaxis=dict(title="Salary ($M)", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="WAR (simplified)", gridcolor=GRID, zeroline=False),
        height=480,
        legend=dict(orientation="h", y=1.08, x=0),
    ))
    st.plotly_chart(fig, use_container_width=True)

    L, R = st.columns(2)
    with L:
        st.markdown(
            '<div class="section-label" style="color:var(--accent-green);">'
            '🏆 TOP BARGAINS (Surplus +)</div>', unsafe_allow_html=True)
        bargains = filtered.nlargest(15, "surplus_M")[
            ["name", "yearID", "age", "WAR", "salary_M", "surplus_M"]
        ].rename(columns={"name": "Player", "yearID": "Year", "age": "Age",
                          "salary_M": "Salary ($M)", "surplus_M": "Surplus ($M)"})
        st.dataframe(
            bargains.style.format({"WAR": "{:.2f}",
                                    "Salary ($M)": "${:.1f}M",
                                    "Surplus ($M)": "${:+.1f}M"}),
            use_container_width=True, hide_index=True, height=560,
        )
    with R:
        st.markdown(
            '<div class="section-label" style="color:var(--accent-red);">'
            '💸 BIGGEST OVERPAYS (Surplus −)</div>', unsafe_allow_html=True)
        busts = filtered.nsmallest(15, "surplus_M")[
            ["name", "yearID", "age", "WAR", "salary_M", "surplus_M"]
        ].rename(columns={"name": "Player", "yearID": "Year", "age": "Age",
                          "salary_M": "Salary ($M)", "surplus_M": "Surplus ($M)"})
        st.dataframe(
            busts.style.format({"WAR": "{:.2f}",
                                "Salary ($M)": "${:.1f}M",
                                "Surplus ($M)": "${:+.1f}M"}),
            use_container_width=True, hide_index=True, height=560,
        )

    st.markdown('<div class="section-label">MARKET TREND · $/WAR OVER TIME</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dpw["yearID"], y=dpw["dollar_per_war"]/1e6,
        mode="lines+markers",
        line=dict(color=ACCENT_GREEN, width=3),
        marker=dict(size=10, color=ACCENT_GREEN, line=dict(width=2, color="white")),
        hovertemplate="<b>%{x}</b><br>$/WAR: $%{y:.2f}M<extra></extra>",
        name="$/WAR",
    ))
    fig.update_layout(**base_layout(
        xaxis=dict(title="Year", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="Cost per WAR ($M)", gridcolor=GRID, zeroline=False),
        height=320, showlegend=False,
    ))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
    <div class="footer">
      ⚾ MLB FA Value Calculator · ACC102 Mini Assignment · Track 4<br>
      Data: Lahman Baseball Database (1985–2016 salaries) · Methodology: simplified WAR (wOBA / FIP),
      delta-method aging curve, league $/WAR market price · Built with Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)

