import os
import html as html_lib
import json
import hashlib
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from model import (
    compute_all, curves_vs_N, escape_trap_solver, growth_trajectory,
    PAPER_BASELINE, FORTUNE_500, INDUSTRY_PRESETS, GOVERNANCE_COSTS, PARAM_BOUNDS,
)

st.set_page_config(
    page_title="Data Hydration Gap Model",
    page_icon="💧",
    layout="wide",
)

# ── Simple-mode questionnaire → parameter mapping ─────────────────────────────

SIMPLE_Q = {
    "org_size": {
        "label": "How many business units own data in your org?",
        "options": {
            "Small  —  2 to 6 teams":           {"N": 4,  "M": 3},
            "Medium  —  7 to 12 teams":          {"N": 10, "M": 8},
            "Large  —  13 to 20 teams":          {"N": 16, "M": 15},
            "Enterprise  —  more than 20 teams": {"N": 25, "M": 25},
        },
    },
    "cross_domain": {
        "label": "How often do projects need data from multiple departments?",
        "options": {
            "Rarely  —  most analytics stays inside one team":         {"lmbda": 0.15, "omega_bar": 0.15},
            "Sometimes  —  roughly 30% of projects":                   {"lmbda": 0.35, "omega_bar": 0.25},
            "Often  —  most projects combine data from several teams":  {"lmbda": 0.55, "omega_bar": 0.35},
            "Always  —  we're building a customer 360 or unified view": {"lmbda": 0.75, "omega_bar": 0.45},
        },
    },
    "governance": {
        "label": "How mature is your data governance today?",
        "options": {
            "None  —  no catalog, no contracts, everything is manual":    {"kappa": 0.5,  "gamma_g": 0.7,  "beta": 0.05},
            "Basic  —  some documentation and shared guidelines":          {"kappa": 0.3,  "gamma_g": 0.5,  "beta": 0.10},
            "Intermediate  —  data catalog in place, some contracts":      {"kappa": 0.2,  "gamma_g": 0.35, "beta": 0.18},
            "Advanced  —  full platform, automated quality, data mesh":    {"kappa": 0.1,  "gamma_g": 0.2,  "beta": 0.30},
        },
    },
    "data_quality": {
        "label": "How good is your raw / source data quality?",
        "options": {
            "Poor  —  lots of nulls, broken schemas, inconsistent formats": {"q_star": 0.3, "alpha": 0.30},
            "Fair  —  mostly usable but with known gaps":                   {"q_star": 0.55,"alpha": 0.50},
            "Good  —  mostly clean, mostly documented":                     {"q_star": 0.75,"alpha": 0.65},
            "Excellent  —  data contracts enforced, high completeness":     {"q_star": 0.9, "alpha": 0.80},
        },
    },
    "pipeline_cost": {
        "label": "How expensive is it to build one custom data pipeline between two teams?",
        "options": {
            "Cheap  —  reusable patterns, less than 2 weeks":          {"tau": 0.02, "P_bar": 0.40},
            "Moderate  —  3 to 5 weeks of engineering time":           {"tau": 0.06, "P_bar": 0.50},
            "Expensive  —  6 to 10 weeks, legacy or compliance issues": {"tau": 0.12, "P_bar": 0.65},
            "Very expensive  —  more than 10 weeks, heavy regulation": {"tau": 0.18, "P_bar": 0.75},
        },
    },
}


def simple_answers_to_params(answers: dict) -> dict:
    p = {}
    for key, answer in answers.items():
        p.update(SIMPLE_Q[key]["options"][answer])
    return p


# ── CSS ────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .hero-pill {
        display: inline-flex; align-items: center; padding: 0.25rem 0.8rem;
        border-radius: 999px; background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.3); color: #7DD3FC;
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
        font-weight: 600; margin-bottom: 0.7rem;
    }
    .hero-title {
        font-size: 2.3rem; font-weight: 800; letter-spacing: -0.04em;
        line-height: 1.15; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #F9FAFB 0%, #93C5FD 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #6B7280; font-size: 0.95rem; line-height: 1.6; max-width: 680px; margin-bottom: 1.8rem; }
    .section-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: #4B5563; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.6rem; }
    .kpi-card { padding: 1.15rem 1.3rem; border-radius: 1rem;
        background: linear-gradient(135deg, #0F172A 0%, #030712 100%);
        border: 1px solid rgba(148,163,184,0.12); height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35); }
    .kpi-label { color: #6B7280; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.25rem; }
    .kpi-value      { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #F9FAFB;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-warn { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #F87171;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-good { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #34D399;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-blue { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #60A5FA;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-sub    { color: #4B5563; font-size: 0.76rem; line-height: 1.45; }
    .kpi-interp { color: #9CA3AF; font-size: 0.78rem; line-height: 1.45; margin-top: 0.4rem;
        border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.4rem; }
    .badge-trap    { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3);
        color:#FCA5A5; font-size:0.68rem; font-weight:700; text-transform:uppercase; }
    .badge-ok      { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
        color:#6EE7B7; font-size:0.68rem; font-weight:700; text-transform:uppercase; }
    .badge-partial { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(252,211,77,0.12); border:1px solid rgba(252,211,77,0.3);
        color:#FDE68A; font-size:0.68rem; font-weight:700; text-transform:uppercase; }
    .divider { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:1.5rem 0; }
    .insight-box { padding: 0.9rem 1.1rem; border-radius: 0.75rem;
        background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.2);
        color: #93C5FD; font-size: 0.82rem; line-height: 1.55; margin-bottom: 1rem; }
    .solver-card { padding: 0.9rem 1.1rem; border-radius: 0.75rem; margin-bottom: 0.6rem;
        background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.1); }
    .simple-diag { padding: 1.4rem 1.6rem; border-radius: 1.2rem; margin-bottom: 1.5rem;
        font-size: 1rem; line-height: 1.65; }
    .simple-rec { padding: 1.2rem 1.4rem; border-radius: 1rem; margin-top: 1rem;
        background: rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.12); }
    </style>
    """, unsafe_allow_html=True)


PLOT_BG  = "rgba(10,15,28,1)"
PAPER_BG = "rgba(5,8,20,1)"
GRID_CLR = "rgba(255,255,255,0.05)"
FONT_CLR = "#9CA3AF"


def base_layout(title_text, h=300, x_title="N (number of domains)"):
    return dict(
        title=dict(text=title_text, font=dict(size=12, color="#E5E7EB"), x=0.01),
        height=h, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, size=11),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, title_text=x_title),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )


inject_css()

# ── URL params — load ONCE before any widget renders ──────────────────────────
if "url_params_loaded" not in st.session_state:
    st.session_state.url_params_loaded = True
    qp = dict(st.query_params)
    if qp:
        _casts = dict(
            N=int, M=int, alpha=float, beta=float, lmbda=float, omega_bar=float,
            gamma_g=float, kappa=float, q_star=float, tau=float, P_bar=float,
            cost_centralized=float, cost_hybrid=float,
        )
        for k, cast in _casts.items():
            if k in qp:
                try:
                    st.session_state[k] = cast(qp[k])
                except (ValueError, TypeError):
                    pass

if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div>
  <div class="hero-pill">💧 Data Hydration Gap Model · Besanson 2026</div>
  <div class="hero-title">Will your data mesh produce a silver layer?</div>
  <div class="hero-sub">
    Answer five questions about your organization and instantly see whether your teams will
    build shared data products on their own — or fall into the data mesh trap.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — mode toggle + inputs ────────────────────────────────────────────
with st.sidebar:
    mode = st.radio("", ["🎯 Simple", "🔬 Expert"], horizontal=True,
                    label_visibility="collapsed",
                    index=0 if st.session_state.get("mode", "Simple") == "Simple" else 1)
    st.session_state.mode = "Simple" if mode.startswith("🎯") else "Expert"
    st.markdown("---")

    # ── SIMPLE MODE SIDEBAR ────────────────────────────────────────────────────
    if st.session_state.mode == "Simple":
        st.markdown("### Tell us about your org")
        st.caption("We'll handle the math — just pick the option that best describes you.")
        st.markdown("")

        answers = {}
        for key, q in SIMPLE_Q.items():
            answers[key] = st.selectbox(
                q["label"],
                list(q["options"].keys()),
                key=f"simple_{key}",
            )
            st.markdown("")

        simple_params = simple_answers_to_params(answers)
        params = simple_params

        st.markdown("---")
        st.markdown("**Industry shortcut**")
        ind_cols = st.columns(2)
        preset_items = list(INDUSTRY_PRESETS.items())
        for idx, (name, preset) in enumerate(preset_items):
            with ind_cols[idx % 2]:
                if st.button(name, use_container_width=True, key=f"preset_s_{idx}"):
                    st.session_state.update(**preset)

    # ── EXPERT MODE SIDEBAR ────────────────────────────────────────────────────
    else:
        st.markdown("## ⚙️ Your Organization")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📄 Paper baseline", use_container_width=True):
                st.session_state.update(**PAPER_BASELINE)
        with col_b:
            if st.button("🏢 Fortune 500", use_container_width=True):
                st.session_state.update(**FORTUNE_500)

        ind_cols = st.columns(2)
        for idx, (name, preset) in enumerate(list(INDUSTRY_PRESETS.items())):
            with ind_cols[idx % 2]:
                if st.button(name, use_container_width=True, key=f"preset_e_{idx}"):
                    st.session_state.update(**preset)

        st.markdown("---")

        with st.expander("🏗️ Organization structure", expanded=True):
            N = st.slider("Number of domains N", 2, 40, st.session_state.get("N", 12),
                help="Count your distinct business domains that own data products.")
            M = st.slider("Cross‑domain consumers M", 0, 50, st.session_state.get("M", 10),
                help="Analytics teams or ML projects consuming data from more than one domain.")

        with st.expander("📈 Incentives & value", expanded=True):
            alpha = st.slider("Domain analytics value α", 0.1, 1.0,
                              st.session_state.get("alpha", 0.5), 0.05)
            beta  = st.slider("Generality–quality synergy β", 0.0, 0.5,
                              st.session_state.get("beta", 0.15), 0.01)
            lmbda = st.slider("Cross‑domain data value λ", 0.0, 1.0,
                              st.session_state.get("lmbda", 0.4), 0.01)
            omega_bar = st.slider("Avg. consumer weight ω̄", 0.0, 1.0,
                                  st.session_state.get("omega_bar", 0.3), 0.05)

        with st.expander("💰 Cost structure", expanded=False):
            gamma_g = st.slider("Generality cost γ_g", 0.1, 1.0,
                                st.session_state.get("gamma_g", 0.4), 0.05)
            kappa   = st.slider("Fixed standardisation cost κ", 0.0, 0.6,
                                st.session_state.get("kappa", 0.25), 0.01)
            q_star  = st.slider("Baseline quality q*", 0.1, 1.0,
                                st.session_state.get("q_star", 0.6), 0.05)

        with st.expander("🏦 Technical debt", expanded=False):
            tau   = st.slider("Integration cost per pair τ (M$)", 0.0, 0.2,
                              st.session_state.get("tau", 0.05), 0.01)
            P_bar = st.slider("Avg. prob. needing another domain P̄", 0.0, 1.0,
                              st.session_state.get("P_bar", 0.5), 0.05)

        with st.expander("🏛️ Governance costs", expanded=False):
            cost_centralized = st.slider("Centralized team cost (M$/yr)", 0.5, 5.0,
                float(st.session_state.get("cost_centralized", GOVERNANCE_COSTS["centralized"])), 0.1)
            cost_hybrid = st.slider("Hybrid approach cost (M$/yr)", 0.5, 3.0,
                float(st.session_state.get("cost_hybrid", GOVERNANCE_COSTS["hybrid"])), 0.1)

        st.markdown("---")
        st.markdown("**💾 Save scenario**")
        scenario_name = st.text_input("Scenario name", placeholder="e.g. Current state")
        if st.button("Save current scenario", use_container_width=True):
            if scenario_name.strip():
                st.session_state.scenarios.append({
                    "name": scenario_name.strip(),
                    "N": N, "M": M, "alpha": alpha, "beta": beta, "lmbda": lmbda,
                    "omega_bar": omega_bar, "gamma_g": gamma_g, "kappa": kappa,
                    "q_star": q_star, "tau": tau, "P_bar": P_bar,
                })
                st.success(f'Saved "{scenario_name.strip()}"')
            else:
                st.warning("Enter a scenario name first.")

        st.markdown("---")
        if st.button("🔗 Update URL to share", use_container_width=True):
            st.query_params.update({
                "N": N, "M": M, "alpha": alpha, "beta": beta, "lmbda": lmbda,
                "omega_bar": omega_bar, "gamma_g": gamma_g, "kappa": kappa,
                "q_star": q_star, "tau": tau, "P_bar": P_bar,
                "cost_centralized": cost_centralized, "cost_hybrid": cost_hybrid,
            })
            st.success("URL updated — copy from your browser's address bar.")

        params = dict(
            N=N, M=M, alpha=alpha, beta=beta, lmbda=lmbda, omega_bar=omega_bar,
            gamma_g=gamma_g, kappa=kappa, q_star=q_star, tau=tau, P_bar=P_bar,
        )


# ── Model — computed once for both modes ──────────────────────────────────────
r               = compute_all(params)
g_ne            = r["g_ne"]
g_so            = r["g_so"]
delta_g         = r["delta_g"]
delta_W         = r["delta_W"]
td_total        = r["td_total"]
subsidy         = r["subsidy"]
pct_of_so       = r["pct_of_so"]
in_trap         = r["in_trap"]
welfare_annual_m = params["N"] * 0.75   # §6.3 calibration ~$750K/domain

# Defaults for expert-only vars when in simple mode
if st.session_state.mode == "Simple":
    cost_centralized = GOVERNANCE_COSTS["centralized"]
    cost_hybrid      = GOVERNANCE_COSTS["hybrid"]

invest_needed = round(subsidy * params["N"], 2)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE MODE — visual dashboard
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "Simple":

    # ── Diagnosis banner ───────────────────────────────────────────────────────
    if in_trap:
        diag_color  = "#F87171"
        diag_bg     = "rgba(239,68,68,0.08)"
        diag_border = "rgba(239,68,68,0.25)"
        diag_icon   = "🔴"
        diag_title  = "Data Mesh Trap — your teams are building data silos"
        diag_body   = (
            f"With <strong>{params['N']} teams</strong>, no one has an incentive to build data products "
            f"that others can reuse. Every cross-department analytics project will require a "
            f"<strong>custom integration built from scratch</strong>. "
            f"The good news: this is fixable with the right governance model."
        )
    elif pct_of_so < 40:
        diag_color  = "#FB923C"
        diag_bg     = "rgba(251,146,60,0.08)"
        diag_border = "rgba(251,146,60,0.25)"
        diag_icon   = "🟠"
        diag_title  = "Severe underinvestment — data silos are forming"
        diag_body   = (
            f"Your teams invest <em>some</em> effort in shared data, but only "
            f"<strong>{pct_of_so:.0f}%</strong> of what the organization actually needs. "
            f"Cross-department projects hit roadblocks frequently. "
            f"A targeted governance investment could unlock significant savings."
        )
    elif pct_of_so < 75:
        diag_color  = "#FCD34D"
        diag_bg     = "rgba(252,211,77,0.08)"
        diag_border = "rgba(252,211,77,0.25)"
        diag_icon   = "🟡"
        diag_title  = "Moderate underinvestment — almost there"
        diag_body   = (
            f"Your teams reach <strong>{pct_of_so:.0f}%</strong> of the ideal data sharing level. "
            f"A real silver layer exists but gaps remain. "
            f"A relatively small governance push could close the remaining gap and unlock full value."
        )
    else:
        diag_color  = "#34D399"
        diag_bg     = "rgba(52,211,153,0.08)"
        diag_border = "rgba(52,211,153,0.25)"
        diag_icon   = "🟢"
        diag_title  = "Near-optimal — your data sharing culture is working"
        diag_body   = (
            f"Your teams are investing at <strong>{pct_of_so:.0f}%</strong> of the theoretical ideal. "
            f"Strong incentive alignment — teams find it in their own interest to build reusable data. "
            f"Marginal improvements are still possible."
        )

    st.markdown(f"""
    <div class="simple-diag" style="background:{diag_bg};border:1px solid {diag_border};color:{diag_color};">
      <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.5rem;">{diag_icon} {diag_title}</div>
      <div style="font-size:0.9rem;line-height:1.65;color:#D1D5DB;">{diag_body}</div>
    </div>""", unsafe_allow_html=True)

    # ── Gauge + key numbers ────────────────────────────────────────────────────
    gauge_col, metrics_col = st.columns([1, 1])

    with gauge_col:
        bar_color = "#34D399" if pct_of_so >= 75 else "#FCD34D" if pct_of_so >= 40 else "#F87171"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_of_so,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Data Sharing Score", "font": {"size": 15, "color": "#E5E7EB"}},
            number={"suffix": " / 100", "font": {"size": 36, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4B5563",
                          "tickfont": {"color": "#6B7280"}, "dtick": 25},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.06)",
                "steps": [
                    {"range": [0,  40], "color": "rgba(239,68,68,0.12)"},
                    {"range": [40, 75], "color": "rgba(252,211,77,0.10)"},
                    {"range": [75,100], "color": "rgba(52,211,153,0.10)"},
                ],
            },
        ))
        fig_gauge.update_layout(
            height=250, margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with metrics_col:
        st.markdown("")
        st.metric(
            label="💸 Estimated annual loss from underinvestment",
            value=f"${welfare_annual_m:.1f}M / year",
            delta=None,
            help="Based on §6.3 calibration of ~$750K per domain in duplicated pipelines and data quality debt.",
        )
        st.markdown("")
        st.metric(
            label="🏗️ Technical debt accumulating",
            value=f"${td_total:.2f}M",
            help=f"Cost to build custom integrations across all {params['N'] * (params['N']-1)} potential domain pairs.",
        )
        st.markdown("")
        st.metric(
            label="📊 Shared data coverage",
            value=f"{pct_of_so:.0f}% of ideal",
            delta=f"{pct_of_so - 100:.0f}% gap to close" if pct_of_so < 100 else "Optimal",
            delta_color="inverse",
        )

    # ── Recommendation ─────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### What should you do?")

    options = [
        ("💰 Federated incentives",   invest_needed,       welfare_annual_m - invest_needed,       "#60A5FA",
         f"Pay each of your {params['N']} teams a reusability bonus when they build data products others can use. "
         f"Total annual investment: <strong>${invest_needed:.1f}M</strong>. "
         f"This is the lowest-cost way to close the gap and works best when teams have capacity but lack motivation."),
        ("🏗️ Centralized data team",  cost_centralized,    welfare_annual_m - cost_centralized,    "#34D399",
         f"Create a central team that builds and maintains shared data products on behalf of all departments. "
         f"Annual cost: <strong>${cost_centralized:.1f}M</strong> (team + tooling). "
         f"Best when domain teams are overloaded or lack data engineering skills."),
        ("🤝 Hybrid approach",        cost_hybrid,         welfare_annual_m - cost_hybrid,         "#A78BFA",
         f"A central team sets the standards and schemas; domain teams execute within those guardrails. "
         f"Annual cost: <strong>${cost_hybrid:.1f}M</strong>. "
         f"Best balance of autonomy and alignment for most organizations."),
    ]

    best_idx = max(range(len(options)), key=lambda i: options[i][2])
    best_name, best_cost, best_net, best_color, best_desc = options[best_idx]
    sign = "+" if best_net >= 0 else ""

    st.markdown(f"""
    <div class="simple-rec">
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#4B5563;margin-bottom:0.4rem;">
        Recommended for your organization
      </div>
      <div style="font-size:1.15rem;font-weight:700;color:{best_color};margin-bottom:0.6rem;">{best_name}</div>
      <div style="color:#9CA3AF;font-size:0.88rem;line-height:1.65;">{best_desc}</div>
      <div style="margin-top:0.8rem;display:flex;gap:1.5rem;">
        <div><span style="color:#4B5563;font-size:0.75rem;">Annual cost</span><br>
             <span style="color:#F9FAFB;font-weight:700;">${best_cost:.1f}M</span></div>
        <div><span style="color:#4B5563;font-size:0.75rem;">Net savings vs. doing nothing</span><br>
             <span style="color:{best_color};font-weight:700;">{sign}${best_net:.1f}M / year</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # show the other two as lighter options
    st.markdown("")
    other_cols = st.columns(2)
    other_options = [o for i, o in enumerate(options) if i != best_idx]
    for col, (name, cost, net, color, desc) in zip(other_cols, other_options):
        sign = "+" if net >= 0 else ""
        col.markdown(f"""
        <div class="kpi-card" style="opacity:0.8;">
          <div style="font-size:0.9rem;font-weight:700;color:{color};margin-bottom:0.3rem;">{name}</div>
          <div style="color:#6B7280;font-size:0.78rem;line-height:1.5;">Cost: ${cost:.1f}M/yr &nbsp;·&nbsp;
            Net: <span style="color:{color};">{sign}${net:.1f}M/yr</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Fastest fixes (inverse solver in plain language) ───────────────────────
    if in_trap or pct_of_so < 75:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### What is the single fastest fix?")
        st.caption(
            "These are the minimum changes to your organization that would "
            "meaningfully improve your data sharing score, ranked by effort."
        )

        target = max(0.01, g_so * 0.5)
        interventions = escape_trap_solver(params, g_ne_target=target)

        PLAIN_LABELS = {
            "kappa":   ("📋 Deploy a data catalog or governance templates",
                        "Right now, the fixed overhead of starting any standardisation effort is too high. "
                        "A data catalog, contract templates, and a governance team would cut this cost significantly."),
            "beta":    ("🛠️ Invest in platform tooling that rewards standardisation",
                        "When teams standardise their data, they should automatically see their own data quality improve. "
                        "Better platform tooling (auto-docs, schema validation, quality scoring) creates this feedback loop."),
            "q_star":  ("🔍 Improve source data quality",
                        "Better raw data makes standardisation worthwhile. "
                        "Data contracts and completeness checks at ingestion raise the baseline so teams have something worth sharing."),
            "alpha":   ("📊 Build a data-driven culture in domain teams",
                        "Teams that rely heavily on their own data are more motivated to invest in data quality. "
                        "Embedding analytics in team OKRs and hiring domain analysts can shift this."),
            "gamma_g": ("⚙️ Make it cheaper to build general data products",
                        "Reusable schema templates, a schema registry, and a self-service SDK would cut the cost "
                        "of building a general data product vs. a narrow one."),
        }

        effort_order = {"Low–Medium": 0, "Medium": 1, "High": 2}
        if interventions:
            sorted_iv = sorted(interventions.items(), key=lambda x: effort_order.get(x[1]["effort"], 9))
            for key, iv in sorted_iv[:3]:
                plain_title, plain_desc = PLAIN_LABELS.get(key, (iv["label"], iv["action"]))
                effort_color = {"Low–Medium": "#34D399", "Medium": "#60A5FA", "High": "#F87171"}.get(iv["effort"], "#9CA3AF")
                st.markdown(f"""
                <div class="solver-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;">
                    <div style="font-size:0.9rem;font-weight:600;color:#E5E7EB;">{plain_title}</div>
                    <span style="font-size:0.65rem;padding:0.15rem 0.5rem;border-radius:999px;
                      background:rgba(255,255,255,0.06);color:{effort_color};font-weight:700;white-space:nowrap;margin-left:0.5rem;">
                      Effort: {iv['effort']}
                    </span>
                  </div>
                  <div style="font-size:0.8rem;color:#6B7280;line-height:1.55;">{plain_desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── Technical details expander ─────────────────────────────────────────────
    with st.expander("🔬 Show technical model details"):
        st.caption("These are the underlying model values for readers of the Besanson (2026) paper.")
        detail_df = pd.DataFrame([{
            "Parameter": "N (domains)", "Value": params["N"]}, {"Parameter": "M (consumers)", "Value": params["M"]},
            {"Parameter": "α", "Value": round(params["alpha"], 3)},
            {"Parameter": "β", "Value": round(params["beta"],  3)},
            {"Parameter": "λ", "Value": round(params["lmbda"], 3)},
            {"Parameter": "γ_g", "Value": round(params["gamma_g"], 3)},
            {"Parameter": "κ",  "Value": round(params["kappa"],   3)},
            {"Parameter": "q*", "Value": round(params["q_star"],  3)},
            {"Parameter": "τ",  "Value": round(params["tau"],     3)},
            {"Parameter": "P̄", "Value": round(params["P_bar"],   3)},
        ])
        out_df = pd.DataFrame([{
            "Output": "gⁿᵉ — equilibrium generality (Eq. 7)",    "Value": round(g_ne, 4)},
            {"Output": "gˢᵒ — social optimum (Prop. 1)",         "Value": round(g_so, 4)},
            {"Output": "Δg — generality gap",                     "Value": round(delta_g, 4)},
            {"Output": "ΔW — welfare loss (Eq. 10)",              "Value": round(delta_W, 4)},
            {"Output": "TD_total — technical debt (Eq. 13, $M)",  "Value": round(td_total, 4)},
            {"Output": "sᵢ — Pigouvian subsidy (Eq. 19)",         "Value": round(subsidy, 4)},
        ])
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Inputs**")
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
        with tc2:
            st.markdown("**Outputs**")
            st.dataframe(out_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERT MODE — full technical app
# ═══════════════════════════════════════════════════════════════════════════════
else:
    # ── Insight banner ─────────────────────────────────────────────────────────
    if in_trap:
        insight_msg = (
            f"⚠️ <strong>Data mesh trap detected.</strong> No domain will voluntarily invest in general "
            f"data products (gⁿᵉ = 0). The silver layer will <em>not</em> emerge organically. "
            f"Socially optimal level: gˢᵒ = {g_so:.2f}. Reusability bonus needed: <strong>{subsidy:.3f}</strong> per domain."
        )
    elif pct_of_so < 40:
        insight_msg = (
            f"🟡 <strong>Severe underinvestment.</strong> Domains reach only "
            f"<strong>{pct_of_so:.0f}%</strong> of the social optimum (gⁿᵉ = {g_ne:.2f} vs gˢᵒ = {g_so:.2f})."
        )
    elif pct_of_so < 75:
        insight_msg = (
            f"🔵 <strong>Moderate underinvestment.</strong> Domains self-invest at {pct_of_so:.0f}% of "
            f"the optimum (gⁿᵉ = {g_ne:.2f} vs gˢᵒ = {g_so:.2f}). A light federated incentive could close the gap."
        )
    else:
        insight_msg = (
            f"✅ <strong>Near-optimal.</strong> Domains are self-investing at {pct_of_so:.0f}% "
            f"of the social optimum (gⁿᵉ = {g_ne:.2f} vs gˢᵒ = {g_so:.2f}). Your governance is working."
        )
    st.markdown(f'<div class="insight-box">{insight_msg}</div>', unsafe_allow_html=True)

    # ── KPI Row 1 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Generality: what domains choose vs. what is optimal</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        val_class = "kpi-value-warn" if in_trap else "kpi-value"
        interp = ("Domains produce purely narrow products. No silver layer without intervention."
                  if in_trap else f"Domains self-invest at <strong>{pct_of_so:.0f}%</strong> of the social optimum.")
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">What domains actually do — gⁿᵉ (eq. 7)</div>
          <div class="{val_class}">{g_ne:.3f}</div>
          <div class="kpi-sub">Equilibrium generality: rational choice when externalities are ignored.</div>
          <div class="kpi-interp">{interp}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">What society needs — gˢᵒ (Prop. 1)</div>
          <div class="kpi-value-good">{g_so:.3f}</div>
          <div class="kpi-sub">Social optimum: generality a central planner would choose.</div>
          <div class="kpi-interp">Includes externality (N−1)λ and consumer term Mω̄ that private domains ignore.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        gap_pct = (delta_g / g_so * 100) if g_so > 0 else 0
        badge = ('<span class="badge-trap">⚠️ Data mesh trap</span>' if in_trap else
                 '<span class="badge-partial">⚡ Large gap</span>' if (delta_g/g_so > 0.5 if g_so > 0 else False) else
                 '<span class="badge-ok">✅ Manageable gap</span>')
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Underinvestment gap — Δg (eq. 8)</div>
          <div class="kpi-value">{delta_g:.3f}</div>
          {badge}
          <div class="kpi-sub">Distance between equilibrium and optimum.</div>
          <div class="kpi-interp">Domains invest at <strong>{pct_of_so:.0f}%</strong> of optimal ({gap_pct:.0f}% gap).</div>
        </div>""", unsafe_allow_html=True)

    # ── KPI Row 2 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Welfare loss, technical debt & corrective mechanism</div>',
                unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Annual welfare loss ΔW (eq. 10)</div>
          <div class="kpi-value-warn">{delta_W:,.1f} <span style="font-size:0.85rem;color:#9CA3AF;">model units</span></div>
          <div class="kpi-sub">Value destroyed annually vs. a central planner. Scales as N².</div>
          <div class="kpi-interp">§6.3 calibration: ≈ <strong>${welfare_annual_m:.1f}M/year</strong> in duplicated effort.</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Technical debt TD_total (eq. 13)</div>
          <div class="kpi-value-warn">${td_total:,.2f}M</div>
          <div class="kpi-sub">Custom integration cost across all domain pairs. Grows quadratically with N.</div>
          <div class="kpi-interp"><strong>{params['N'] * (params['N']-1)} potential integration pairs</strong> with {params['N']} domains.</div>
        </div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Reusability bonus needed — sᵢ (eq. 19)</div>
          <div class="kpi-value-blue">{subsidy:.3f}</div>
          <div class="kpi-sub">Per-domain reward that corrects the externality and aligns incentives with social optimum.</div>
          <div class="kpi-interp">Total: <strong>${invest_needed:.2f}M/yr</strong> across {params['N']} domains vs ${welfare_annual_m:.1f}M/yr loss.</div>
        </div>""", unsafe_allow_html=True)

    # ── Inverse solver ─────────────────────────────────────────────────────────
    if in_trap or pct_of_so < 75:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-label">{"How to escape the trap" if in_trap else "How to close the gap"} — minimum parameter changes needed</div>',
            unsafe_allow_html=True)
        target_val   = 0.01 if in_trap else g_so * 0.75
        interventions = escape_trap_solver(params, g_ne_target=target_val)
        if interventions:
            effort_color = {"Low–Medium": "#34D399", "Medium": "#60A5FA", "High": "#F87171"}
            for key, iv in interventions.items():
                direction_arrow = "↑" if iv["direction"] == "increase" else "↓"
                color = effort_color.get(iv["effort"], "#9CA3AF")
                st.markdown(f"""
                <div class="solver-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="font-size:0.88rem;font-weight:600;color:#E5E7EB;">{iv['label']}</div>
                    <span style="font-size:0.68rem;padding:0.15rem 0.5rem;border-radius:999px;
                      background:rgba(255,255,255,0.06);color:{color};font-weight:700;">Effort: {iv['effort']}</span>
                  </div>
                  <div style="margin:0.4rem 0;font-size:1.1rem;font-weight:700;">
                    <span style="color:#F87171;">{iv['current']}</span>
                    <span style="color:#4B5563;margin:0 0.5rem;">→</span>
                    <span style="color:#34D399;">{iv['required']}</span>
                    <span style="color:#6B7280;font-size:0.78rem;margin-left:0.4rem;">({direction_arrow}{iv['delta']:.4f})</span>
                  </div>
                  <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">{iv['action']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No feasible single-parameter intervention within slider bounds. A combination of smaller changes is needed.")

    # ── Charts ─────────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">How things scale with the number of domains N</div>',
                unsafe_allow_html=True)

    N_vals = np.arange(2, 41, dtype=float)
    curves = curves_vs_N(params, N_vals)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=N_vals, y=curves["g_ne"], name="gⁿᵉ (equilibrium)",
            mode="lines", line=dict(color="#F87171", width=2.5, dash="dot")))
        fig1.add_trace(go.Scatter(x=N_vals, y=curves["g_so"], name="gˢᵒ (social optimum)",
            mode="lines", line=dict(color="#34D399", width=2.5),
            fill="tonexty", fillcolor="rgba(52,211,153,0.07)"))
        fig1.add_vline(x=params["N"], line_dash="dot", line_color="#7DD3FC",
            annotation_text=f"  N={params['N']}", annotation_font_color="#7DD3FC")
        fig1.update_layout(**base_layout("Generality gap grows with N — Proposition 1"))
        fig1.update_yaxes(title_text="Generality g", range=[0, 1.05])
        st.plotly_chart(fig1, use_container_width=True)
    with cc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=N_vals, y=curves["td"], name="TD_total",
            mode="lines", line=dict(color="#F87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
        fig2.add_vline(x=params["N"], line_dash="dot", line_color="#7DD3FC",
            annotation_text=f"  N={params['N']}, TD=${td_total:.2f}M", annotation_font_color="#7DD3FC")
        fig2.update_layout(**base_layout("Technical debt TD_total ($M) vs N — eq. (13)"))
        fig2.update_yaxes(title_text="TD_total ($M)")
        st.plotly_chart(fig2, use_container_width=True)

    fig_dw = go.Figure()
    fig_dw.add_trace(go.Scatter(x=N_vals, y=curves["dw"], name="ΔW",
        mode="lines", line=dict(color="#FCD34D", width=2.5),
        fill="tozeroy", fillcolor="rgba(252,211,77,0.07)"))
    fig_dw.add_vline(x=params["N"], line_dash="dot", line_color="#7DD3FC",
        annotation_text=f"  N={params['N']}", annotation_font_color="#7DD3FC")
    fig_dw.update_layout(**base_layout("Annual welfare loss ΔW vs N — eq. (10)", h=260))
    fig_dw.update_yaxes(title_text="ΔW (model units)")
    st.plotly_chart(fig_dw, use_container_width=True)

    # ── Growth trajectory ──────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Growth trajectory — projected cost as your organization scales</div>',
                unsafe_allow_html=True)
    traj_col1, traj_col2 = st.columns([2, 1])
    with traj_col2:
        n_future   = st.number_input("Projected domains in N years", min_value=params["N"], max_value=100,
                                      value=max(params["N"] + 4, int(params["N"] * 1.5)), step=1)
        traj_years = st.number_input("Years to project", min_value=1, max_value=20, value=5, step=1)
    traj = growth_trajectory(params, n_future=n_future, years=int(traj_years))
    year_labels = [f"Year {int(y)}" if y > 0 else "Now" for y in traj["year"]]
    with traj_col1:
        tc1, tc2 = st.columns(2)
        with tc1:
            ftd = go.Figure()
            ftd.add_trace(go.Scatter(x=year_labels, y=traj["td"], mode="lines+markers",
                line=dict(color="#F87171", width=2.5), fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
            ftd.update_layout(**base_layout(f"Technical debt: Now → {int(traj_years)}yr", h=250, x_title=""))
            ftd.update_yaxes(title_text="TD_total ($M)")
            st.plotly_chart(ftd, use_container_width=True)
        with tc2:
            fdw = go.Figure()
            fdw.add_trace(go.Scatter(x=year_labels, y=traj["dw"], mode="lines+markers",
                line=dict(color="#FCD34D", width=2.5), fill="tozeroy", fillcolor="rgba(252,211,77,0.07)"))
            fdw.update_layout(**base_layout(f"Welfare loss: Now → {int(traj_years)}yr", h=250, x_title=""))
            fdw.update_yaxes(title_text="ΔW (model units)")
            st.plotly_chart(fdw, use_container_width=True)
    td_growth = traj["td"][-1] - traj["td"][0]
    st.markdown(
        f'<div class="insight-box">Growing from <strong>{params["N"]}</strong> to <strong>{n_future} domains</strong> '
        f'over {int(traj_years)} years adds <strong>${td_growth:.2f}M</strong> in technical debt '
        f'and <strong>${(n_future - params["N"]) * 0.75:.1f}M/yr</strong> in additional welfare loss — '
        f'if governance does not change.</div>', unsafe_allow_html=True)

    # ── Sensitivity tornado ────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Sensitivity — which parameter drives welfare loss most?</div>',
                unsafe_allow_html=True)
    SENS_PARAMS = ["N","lmbda","alpha","beta","gamma_g","kappa","q_star","tau","P_bar","M","omega_bar"]
    SENS_LABELS = {"N":"Domains N","lmbda":"Cross-domain value λ","alpha":"Analytics value α",
                   "beta":"Synergy β","gamma_g":"Generality cost γ_g","kappa":"Fixed cost κ",
                   "q_star":"Baseline quality q*","tau":"Integration cost τ",
                   "P_bar":"Prob. needing domain P̄","M":"Consumers M","omega_bar":"Consumer weight ω̄"}
    tornado_rows = []
    for pk in SENS_PARAMS:
        lo, hi = PARAM_BOUNDS[pk]
        dw_lo = compute_all(dict(params, **{pk: lo}))["delta_W"]
        dw_hi = compute_all(dict(params, **{pk: hi}))["delta_W"]
        tornado_rows.append((SENS_LABELS[pk], dw_lo, dw_hi, abs(dw_hi - dw_lo)))
    tornado_rows.sort(key=lambda x: x[3])
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(name="Low end", x=[r[1] for r in tornado_rows], y=[r[0] for r in tornado_rows],
        orientation="h", marker_color="rgba(96,165,250,0.7)", marker_line_width=0))
    fig_t.add_trace(go.Bar(name="High end", x=[r[2] for r in tornado_rows], y=[r[0] for r in tornado_rows],
        orientation="h", marker_color="rgba(248,113,113,0.7)", marker_line_width=0))
    fig_t.add_vline(x=delta_W, line_dash="dot", line_color="#FCD34D",
        annotation_text="  Current ΔW", annotation_font_color="#FCD34D")
    fig_t.update_layout(barmode="overlay", height=380, margin=dict(l=10,r=10,t=50,b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR,size=11),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, title_text="ΔW (model units)"),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Tornado — ΔW range when each parameter is swept across its full range",
                   font=dict(size=12,color="#E5E7EB"), x=0.01), showlegend=True)
    st.plotly_chart(fig_t, use_container_width=True)

    # ── gNE vs gSO bar ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Current snapshot — equilibrium vs. optimum</div>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Nash Equilibrium gⁿᵉ", x=["Generality level"], y=[g_ne],
        marker_color="#F87171", text=[f"{g_ne:.3f}"], textposition="outside", textfont=dict(color="#F87171")))
    fig3.add_trace(go.Bar(name="Social Optimum gˢᵒ", x=["Generality level"], y=[g_so],
        marker_color="#34D399", text=[f"{g_so:.3f}"], textposition="outside", textfont=dict(color="#34D399")))
    fig3.update_layout(barmode="group", height=260, margin=dict(l=10,r=10,t=50,b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR,size=11),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, range=[0,1.2], title_text="Generality g"),
        legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="gⁿᵉ vs. gˢᵒ for your current parameters",
                   font=dict(size=12,color="#E5E7EB"), x=0.01), showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Governance comparison ──────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Governance regime comparison — Table 2 from the paper</div>',
                unsafe_allow_html=True)
    regimes = [
        ("Pure Data Mesh",          0.0,              0.0,               -welfare_annual_m,                   "#F87171"),
        ("Centralized Hydration",   round(g_so,2),    cost_centralized,   welfare_annual_m-cost_centralized,   "#FCD34D"),
        ("Federated + Incentives",  round(g_so,2),    round(invest_needed,2), welfare_annual_m-invest_needed,  "#34D399"),
        ("Hybrid",                  round(g_so*0.7,2), cost_hybrid,       welfare_annual_m-cost_hybrid,        "#60A5FA"),
    ]
    rcols = st.columns(4)
    for col, (name, g_val, cost, net, color) in zip(rcols, regimes):
        sign = "+" if net >= 0 else ""
        col.markdown(f"""<div class="kpi-card" style="text-align:center;">
          <div class="kpi-label" style="text-align:center;">{name}</div>
          <div style="font-size:1.3rem;font-weight:800;color:{color};margin:0.3rem 0;">g = {g_val:.2f}</div>
          <div class="kpi-sub" style="text-align:center;">
            Cost: <strong>${cost:.2f}M</strong><br>
            Net: <span style="color:{color};font-weight:700;">{sign}${net:.1f}M</span>
          </div></div>""", unsafe_allow_html=True)

    # ── Plain English interpretation ───────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What this means for your organization</div>',
                unsafe_allow_html=True)
    if in_trap:
        di, dt = "🔴", "Your organization is in the Data Mesh Trap"
        db = (f"With {params['N']} domains, teams have <strong>zero incentive</strong> to build reusable data products. "
              f"Cross-domain analytics requires custom pipelines every time — estimated <strong>${td_total:,.2f}M</strong> in technical debt.")
    elif pct_of_so < 40:
        di, dt = "🟠", "Severe underinvestment — silver layer is thin and fragile"
        db = (f"Domains reach only <strong>{pct_of_so:.0f}%</strong> of what is needed (gⁿᵉ = {g_ne:.2f}). "
              f"Technical debt: <strong>${td_total:,.2f}M</strong> across {params['N']*(params['N']-1)} potential pairs.")
    elif pct_of_so < 75:
        di, dt = "🟡", "Moderate underinvestment — your silver layer needs a push"
        db = (f"Domains invest at {pct_of_so:.0f}% of optimal (gⁿᵉ = {g_ne:.2f}). "
              f"A small governance investment could unlock <strong>${welfare_annual_m:.1f}M/year</strong>.")
    else:
        di, dt = "🟢", "Near-optimal — your governance is working"
        db = f"Domains invest at {pct_of_so:.0f}% of the social optimum (gⁿᵉ = {g_ne:.2f}). Strong incentive alignment."

    st.markdown(f"""<div class="kpi-card" style="margin-bottom:1rem;">
      <div style="font-size:1.1rem;font-weight:700;color:#F9FAFB;margin-bottom:0.5rem;">{di} {dt}</div>
      <div style="color:#9CA3AF;font-size:0.88rem;line-height:1.65;">{db}</div>
    </div>""", unsafe_allow_html=True)

    # ── AI summary ─────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">AI executive summary</div>', unsafe_allow_html=True)
    try:
        import anthropic as _anthropic
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Anthropic API key", type="password",
                placeholder="sk-ant-... (never stored)", help="Enter your API key to generate an AI executive summary.")
        def _ck(p, res):
            return hashlib.md5(json.dumps({**p,**{k:round(v,6) if isinstance(v,float) else v
                                                  for k,v in res.items()}},sort_keys=True).encode()).hexdigest()
        if api_key:
            ck = _ck(params, r)
            cached = st.session_state.get("ai_summary_cache",{}).get(ck)
            if cached:
                st.markdown(f'<div class="insight-box" style="color:#E5E7EB;">{cached}</div>', unsafe_allow_html=True)
                if st.button("↺ Regenerate"):
                    st.session_state.setdefault("ai_summary_cache",{}).pop(ck,None); st.rerun()
            else:
                if st.button("✨ Generate executive summary"):
                    prompt = (
                        f"You are a data governance consultant advising a CDO.\n\n"
                        f"Organization: {params['N']} domains, {params['M']} cross-domain consumers.\n"
                        f"g_NE={r['g_ne']:.3f}{'  ⚠️ TRAP' if in_trap else ''}, g_SO={r['g_so']:.3f}, "
                        f"{r['pct_of_so']:.0f}% of optimum, ΔW≈${welfare_annual_m:.1f}M/yr, TD=${r['td_total']:.2f}M.\n\n"
                        f"Write exactly 3 paragraphs for a C-suite audience:\n"
                        f"1. Diagnosis and root cause.\n2. Financial impact.\n3. Recommended action and first 90-day step.\n"
                        f"Be specific, cite numbers, no bullet points."
                    )
                    client = _anthropic.Anthropic(api_key=api_key)
                    placeholder = st.empty(); full_text = ""
                    try:
                        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=550,
                                messages=[{"role":"user","content":prompt}]) as stream:
                            for chunk in stream.text_stream:
                                full_text += chunk
                                placeholder.markdown(f'<div class="insight-box" style="color:#E5E7EB;">{full_text}▌</div>',
                                                     unsafe_allow_html=True)
                        placeholder.markdown(f'<div class="insight-box" style="color:#E5E7EB;">{full_text}</div>',
                                             unsafe_allow_html=True)
                        st.session_state.setdefault("ai_summary_cache",{})[ck] = full_text
                    except Exception as e:
                        st.error(f"API error: {e}")
        else:
            st.caption("Enter an Anthropic API key above to enable AI-generated executive summaries.")
    except ImportError:
        st.caption("Install `anthropic` to enable AI-generated executive summaries.")

    # ── Scenario comparison ────────────────────────────────────────────────────
    if st.session_state.scenarios:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Scenario comparison</div>', unsafe_allow_html=True)
        rows = []
        for s in st.session_state.scenarios:
            sr = compute_all({k: s[k] for k in ["N","M","alpha","beta","lmbda","omega_bar","gamma_g","kappa","q_star","tau","P_bar"]})
            rows.append({"Scenario":s["name"],"N":s["N"],"λ":s["lmbda"],"γ_g":s["gamma_g"],"κ":s["kappa"],
                         "gⁿᵉ":round(sr["g_ne"],3),"gˢᵒ":round(sr["g_so"],3),
                         "% of SO":f"{sr['pct_of_so']:.0f}%","ΔW":round(sr["delta_W"],2),"TD ($M)":round(sr["td_total"],2),
                         "Status":"🔴 Trap" if sr["in_trap"] else ("🟡 Low" if sr["pct_of_so"]<40 else ("🟠 Mid" if sr["pct_of_so"]<75 else "🟢 Good"))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        sc1, sc2 = st.columns([1,4])
        with sc1:
            if st.button("🗑️ Clear all scenarios"):
                st.session_state.scenarios = []; st.rerun()
        with sc2:
            st.download_button("⬇️ Download CSV", pd.DataFrame(rows).to_csv(index=False).encode(),
                               "scenarios.csv", "text/csv")

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    export_df = pd.DataFrame([{"N":params["N"],"M":params["M"],"α":params["alpha"],"β":params["beta"],
        "λ":params["lmbda"],"ω̄":params["omega_bar"],"γ_g":params["gamma_g"],"κ":params["kappa"],
        "q*":params["q_star"],"τ":params["tau"],"P̄":params["P_bar"],
        "g_NE":round(g_ne,4),"g_SO":round(g_so,4),"Δg":round(delta_g,4),"ΔW":round(delta_W,4),
        "TD($M)":round(td_total,4),"s_i":round(subsidy,4),"% SO":round(pct_of_so,1),"Trap":in_trap}])
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button("⬇️ Download CSV", export_df.to_csv(index=False).encode(),
                           "dhg_scenario.csv","text/csv")

    with st.expander("📖 Plain English guide to every number"):
        st.markdown("""
| Number | What it means |
|--------|---------------|
| **gⁿᵉ** | How reusable domain data products will be if teams are left alone. 0 = siloed, 1 = fully standardised. |
| **gˢᵒ** | The reusability level that maximises value for the whole organisation. Your target. |
| **Δg** | How far short of the ideal when domains act in self-interest. |
| **ΔW** | Total value destroyed annually by underinvestment. |
| **TD** | Hidden future cost from narrow products — grows as N². |
| **sᵢ** | Reward each domain needs to make general products worth building. |
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#374151;font-size:0.7rem;text-align:center;">
Besanson (2026) · The Data Hydration Gap · A Formal Model of Underinvestment in General-Purpose Data Products<br>
gⁿᵉ → eq.(7) · gˢᵒ → Proposition 1 · ΔW → eq.(10) · TD_total → eq.(13) · sᵢ → eq.(19)
</p>
""", unsafe_allow_html=True)
