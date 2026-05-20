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
        color:#FCA5A5; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .badge-ok      { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
        color:#6EE7B7; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .badge-partial { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(252,211,77,0.12); border:1px solid rgba(252,211,77,0.3);
        color:#FDE68A; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .divider { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:1.5rem 0; }
    .insight-box { padding: 0.9rem 1.1rem; border-radius: 0.75rem;
        background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.2);
        color: #93C5FD; font-size: 0.82rem; line-height: 1.55; margin-bottom: 1rem; }
    .solver-card { padding: 0.9rem 1.1rem; border-radius: 0.75rem; margin-bottom: 0.6rem;
        background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.1); }
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

# ── URL params — load ONCE before any slider renders ──────────────────────────
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

# ── Scenario store ─────────────────────────────────────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div>
  <div class="hero-pill">💧 Data Hydration Gap Model · Besanson 2026</div>
  <div class="hero-title">Will your data mesh produce a silver layer?</div>
  <div class="hero-sub">
    Set your organization's parameters and instantly see whether domains will invest in general
    data products on their own — or fall into the data mesh trap. Based on the formal game-theoretic
    model of the data hydration gap.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Your Organization")

    # Presets row 1: paper + fortune 500
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📄 Paper baseline", use_container_width=True):
            st.session_state.update(**PAPER_BASELINE)
    with col_b:
        if st.button("🏢 Fortune 500", use_container_width=True):
            st.session_state.update(**FORTUNE_500)

    # Presets row 2: industry verticals
    ind_cols = st.columns(2)
    preset_items = list(INDUSTRY_PRESETS.items())
    for idx, (name, preset) in enumerate(preset_items):
        with ind_cols[idx % 2]:
            if st.button(name, use_container_width=True, key=f"preset_{idx}"):
                st.session_state.update(**preset)

    st.markdown("---")

    with st.expander("🏗️ Organization structure", expanded=True):
        N = st.slider("Number of domains N", 2, 40, st.session_state.get("N", 12),
            help="📌 Count your distinct business domains.\n\nExample: Sales, Marketing, Finance, Supply Chain = 4 domains.")
        M = st.slider("Cross‑domain consumers M", 0, 50, st.session_state.get("M", 10),
            help="📌 How many analytics teams or ML projects consume data from more than one domain?")

    with st.expander("📈 Incentives & value", expanded=True):
        alpha = st.slider("Domain analytics value α", 0.1, 1.0,
                          st.session_state.get("alpha", 0.5), 0.05,
            help="📌 How much does a typical domain rely on its own data for daily decisions?\n\nLow (0.3): e.g. Legal, Compliance\nHigh (0.8): Sales, Marketing with dashboards")
        beta = st.slider("Generality–quality synergy β", 0.0, 0.5,
                         st.session_state.get("beta", 0.15), 0.01,
            help="📌 When a domain standardises its data, how much does its own quality improve as a side effect?")
        lmbda = st.slider("Cross‑domain data value λ", 0.0, 1.0,
                          st.session_state.get("lmbda", 0.4), 0.01,
            help="📌 What share of analytical value comes from combining data across domains?")
        omega_bar = st.slider("Avg. consumer weight ω̄", 0.0, 1.0,
                              st.session_state.get("omega_bar", 0.3), 0.05,
            help="📌 How important is each domain's data to cross-domain consumers on average?")

    with st.expander("💰 Cost structure", expanded=False):
        gamma_g = st.slider("Generality cost γ_g", 0.1, 1.0,
                            st.session_state.get("gamma_g", 0.4), 0.05,
            help="📌 How costly is making a data product general vs. narrow, per unit of quality?")
        kappa = st.slider("Fixed standardisation cost κ", 0.0, 0.6,
                          st.session_state.get("kappa", 0.25), 0.01,
            help="📌 Fixed overhead just to START standardising a data product.")
        q_star = st.slider("Baseline quality q*", 0.1, 1.0,
                           st.session_state.get("q_star", 0.6), 0.05,
            help="📌 Typical quality of raw/bronze data before generalisation decisions.")

    with st.expander("🏦 Technical debt", expanded=False):
        tau = st.slider("Integration cost per pair τ (M$)", 0.0, 0.2,
                        st.session_state.get("tau", 0.05), 0.01,
            help="📌 Cost to build one custom pipeline between two domains when no general product exists.")
        P_bar = st.slider("Avg. prob. needing another domain P̄", 0.0, 1.0,
                          st.session_state.get("P_bar", 0.5), 0.05,
            help="📌 Probability that any domain will need data from another specific domain.")

    with st.expander("🏛️ Governance costs", expanded=False):
        cost_centralized = st.slider(
            "Centralized team cost (M$/yr)", 0.5, 5.0,
            float(st.session_state.get("cost_centralized", GOVERNANCE_COSTS["centralized"])), 0.1,
            help="Annual cost of a central data hydration team.")
        cost_hybrid = st.slider(
            "Hybrid approach cost (M$/yr)", 0.5, 3.0,
            float(st.session_state.get("cost_hybrid", GOVERNANCE_COSTS["hybrid"])), 0.1,
            help="Annual cost of a standards team + lighter incentive mechanism.")

    st.markdown("---")

    # Scenario save
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
            st.success(f'Saved \"{scenario_name.strip()}\"')
        else:
            st.warning("Enter a scenario name first.")

    # Share URL
    st.markdown("---")
    if st.button("🔗 Update URL to share", use_container_width=True):
        st.query_params.update({
            "N": N, "M": M, "alpha": alpha, "beta": beta, "lmbda": lmbda,
            "omega_bar": omega_bar, "gamma_g": gamma_g, "kappa": kappa,
            "q_star": q_star, "tau": tau, "P_bar": P_bar,
            "cost_centralized": cost_centralized, "cost_hybrid": cost_hybrid,
        })
        st.success("URL updated — copy from your browser's address bar.")

# ── Model ─────────────────────────────────────────────────────────────────────
params = dict(
    N=N, M=M, alpha=alpha, beta=beta, lmbda=lmbda, omega_bar=omega_bar,
    gamma_g=gamma_g, kappa=kappa, q_star=q_star, tau=tau, P_bar=P_bar,
)
r             = compute_all(params)
g_ne          = r["g_ne"]
g_so          = r["g_so"]
delta_g       = r["delta_g"]
delta_W       = r["delta_W"]
td_total      = r["td_total"]
subsidy       = r["subsidy"]
pct_of_so     = r["pct_of_so"]
in_trap       = r["in_trap"]
welfare_annual_m = N * 0.75  # §6.3 calibration ~$750K/domain

# ── Insight banner ────────────────────────────────────────────────────────────
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

# ── KPI Row 1 — generality ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Generality: what domains choose vs. what is optimal</div>',
            unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    val_class = "kpi-value-warn" if in_trap else "kpi-value"
    interp = ("Domains produce purely narrow products. No silver layer will emerge without intervention."
              if in_trap else
              f"Domains self-invest at <strong>{pct_of_so:.0f}%</strong> of the social optimum.")
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">What domains actually do — gⁿᵉ (eq. 7)</div>
      <div class="{val_class}">{g_ne:.3f}</div>
      <div class="kpi-sub">Equilibrium generality: what a domain rationally chooses when it ignores cross-domain externalities.</div>
      <div class="kpi-interp">{interp}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">What society needs — gˢᵒ (Prop. 1)</div>
      <div class="kpi-value-good">{g_so:.3f}</div>
      <div class="kpi-sub">Social optimum: the generality a central planner would choose, accounting for cross-domain value and consumer benefit.</div>
      <div class="kpi-interp">Includes externality (N−1)λ and consumer term Mω̄ that private domains ignore.</div>
    </div>""", unsafe_allow_html=True)

with col3:
    gap_pct = (delta_g / g_so * 100) if g_so > 0 else 0
    if in_trap:
        badge = '<span class="badge-trap">⚠️ Data mesh trap</span>'
    elif delta_g / g_so > 0.5 if g_so > 0 else False:
        badge = '<span class="badge-partial">⚡ Large gap</span>'
    else:
        badge = '<span class="badge-ok">✅ Manageable gap</span>'
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Underinvestment gap — Δg (eq. 8)</div>
      <div class="kpi-value">{delta_g:.3f}</div>
      {badge}
      <div class="kpi-sub">Distance between equilibrium and optimum.</div>
      <div class="kpi-interp">Domains invest at <strong>{pct_of_so:.0f}%</strong> of what is socially optimal ({gap_pct:.0f}% gap).</div>
    </div>""", unsafe_allow_html=True)

# ── KPI Row 2 — welfare, debt, subsidy ───────────────────────────────────────
st.markdown('<div class="section-label">Welfare loss, technical debt & corrective mechanism</div>',
            unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Annual welfare loss ΔW (eq. 10)</div>
      <div class="kpi-value-warn">{delta_W:,.1f} <span style="font-size:0.85rem;color:#9CA3AF;">model units</span></div>
      <div class="kpi-sub">Value destroyed annually by decentralised governance vs. a central planner.</div>
      <div class="kpi-interp">Using §6.3 calibration (~$750K/domain): ≈ <strong>${welfare_annual_m:.1f}M/year</strong> in duplicated effort and data quality debt.</div>
    </div>""", unsafe_allow_html=True)

with col5:
    td_pairs = N * (N - 1)
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Technical debt TD_total (eq. 13)</div>
      <div class="kpi-value-warn">${td_total:,.2f}M</div>
      <div class="kpi-sub">Custom integration cost across all domain pairs. Grows quadratically with N.</div>
      <div class="kpi-interp">With {N} domains: <strong>{td_pairs} potential integration pairs</strong>. Each narrow product creates debt for every domain that might need it.</div>
    </div>""", unsafe_allow_html=True)

with col6:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Reusability bonus needed — sᵢ (eq. 19)</div>
      <div class="kpi-value-blue">{subsidy:.3f}</div>
      <div class="kpi-sub">The per-domain reward that exactly corrects the externality and aligns domain incentives with the social optimum.</div>
      <div class="kpi-interp">Paying each domain <strong>{subsidy:.3f}</strong>/unit of generality closes the gap at ~$1M/yr governance cost vs ${welfare_annual_m:.1f}M/yr welfare loss.</div>
    </div>""", unsafe_allow_html=True)

# ── Inverse solver ────────────────────────────────────────────────────────────
if in_trap or pct_of_so < 75:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    target_label = "escape the trap" if in_trap else "reach 75% of social optimum"
    target_val   = 0.01 if in_trap else g_so * 0.75
    solver_title = (
        f"🔧 How to escape the data mesh trap" if in_trap
        else f"🔧 How to close the investment gap"
    )
    st.markdown(f'<div class="section-label">{solver_title} — minimum parameter changes needed</div>',
                unsafe_allow_html=True)

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
                  background:rgba(255,255,255,0.06);color:{color};font-weight:700;">
                  Effort: {iv['effort']}
                </span>
              </div>
              <div style="margin:0.4rem 0;font-size:1.1rem;font-weight:700;">
                <span style="color:#F87171;">{iv['current']}</span>
                <span style="color:#4B5563;margin:0 0.5rem;">→</span>
                <span style="color:#34D399;">{iv['required']}</span>
                <span style="color:#6B7280;font-size:0.78rem;margin-left:0.4rem;">
                  ({direction_arrow}{iv['delta']:.4f})
                </span>
              </div>
              <div style="font-size:0.78rem;color:#6B7280;line-height:1.5;">{iv['action']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No feasible single-parameter intervention found within slider bounds. "
                "A combination of smaller changes across multiple parameters is needed.")

# ── Charts: generality, tech debt, welfare loss vs N ─────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">How things scale with the number of domains N</div>',
            unsafe_allow_html=True)

N_vals = np.arange(2, 41, dtype=float)
curves = curves_vs_N(params, N_vals)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=N_vals, y=curves["g_ne"], name="gⁿᵉ (equilibrium)",
        mode="lines", line=dict(color="#F87171", width=2.5, dash="dot")))
    fig1.add_trace(go.Scatter(x=N_vals, y=curves["g_so"], name="gˢᵒ (social optimum)",
        mode="lines", line=dict(color="#34D399", width=2.5),
        fill="tonexty", fillcolor="rgba(52,211,153,0.07)"))
    fig1.add_vline(x=N, line_dash="dot", line_color="#7DD3FC",
        annotation_text=f"  N={N}", annotation_font_color="#7DD3FC")
    fig1.update_layout(**base_layout("Generality gap grows with N — Proposition 1"))
    fig1.update_yaxes(title_text="Generality g", range=[0, 1.05])
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=N_vals, y=curves["td"], name="TD_total",
        mode="lines", line=dict(color="#F87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
    fig2.add_vline(x=N, line_dash="dot", line_color="#7DD3FC",
        annotation_text=f"  N={N}, TD=${td_total:.2f}M", annotation_font_color="#7DD3FC")
    fig2.update_layout(**base_layout("Technical debt TD_total ($M) vs N — eq. (13)"))
    fig2.update_yaxes(title_text="TD_total ($M)")
    st.plotly_chart(fig2, use_container_width=True)

fig_dw = go.Figure()
fig_dw.add_trace(go.Scatter(x=N_vals, y=curves["dw"], name="ΔW (welfare loss)",
    mode="lines", line=dict(color="#FCD34D", width=2.5),
    fill="tozeroy", fillcolor="rgba(252,211,77,0.07)"))
fig_dw.add_vline(x=N, line_dash="dot", line_color="#7DD3FC",
    annotation_text=f"  N={N}, ΔW={delta_W:.1f}", annotation_font_color="#7DD3FC")
fig_dw.update_layout(**base_layout("Annual welfare loss ΔW vs N — eq. (10)", h=260))
fig_dw.update_yaxes(title_text="ΔW (model units)")
st.plotly_chart(fig_dw, use_container_width=True)

# ── Growth trajectory ─────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Growth trajectory — projected cost as your organization scales</div>',
            unsafe_allow_html=True)

traj_col1, traj_col2 = st.columns([2, 1])
with traj_col2:
    n_future = st.number_input(
        "Projected domains in N years", min_value=N, max_value=100,
        value=max(N + 4, int(N * 1.5)), step=1,
        help="How many domains do you expect in the future?",
    )
    traj_years = st.number_input(
        "Years to project", min_value=1, max_value=20, value=5, step=1,
    )

traj = growth_trajectory(params, n_future=n_future, years=int(traj_years))
year_labels = [f"Year {int(y)}" if y > 0 else "Now" for y in traj["year"]]

with traj_col1:
    tc1, tc2 = st.columns(2)
    with tc1:
        fig_traj_td = go.Figure()
        fig_traj_td.add_trace(go.Scatter(
            x=year_labels, y=traj["td"], name="TD_total ($M)",
            mode="lines+markers", line=dict(color="#F87171", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
        ))
        fig_traj_td.update_layout(**base_layout(
            f"Technical debt: Now → {int(traj_years)}yr", h=250, x_title=""))
        fig_traj_td.update_yaxes(title_text="TD_total ($M)")
        st.plotly_chart(fig_traj_td, use_container_width=True)
    with tc2:
        fig_traj_dw = go.Figure()
        fig_traj_dw.add_trace(go.Scatter(
            x=year_labels, y=traj["dw"], name="ΔW (model units)",
            mode="lines+markers", line=dict(color="#FCD34D", width=2.5),
            fill="tozeroy", fillcolor="rgba(252,211,77,0.07)",
        ))
        fig_traj_dw.update_layout(**base_layout(
            f"Welfare loss: Now → {int(traj_years)}yr", h=250, x_title=""))
        fig_traj_dw.update_yaxes(title_text="ΔW (model units)")
        st.plotly_chart(fig_traj_dw, use_container_width=True)

td_growth = traj["td"][-1] - traj["td"][0]
dw_growth = traj["dw"][-1] - traj["dw"][0]
st.markdown(
    f'<div class="insight-box">Growing from <strong>{N}</strong> to <strong>{n_future} domains</strong> '
    f'over {int(traj_years)} years adds <strong>${td_growth:.2f}M</strong> in technical debt and '
    f'<strong>{dw_growth:.1f} model units</strong> (≈<strong>${(n_future - N) * 0.75:.1f}M/yr</strong>) '
    f'in additional annual welfare loss — if governance does not change.</div>',
    unsafe_allow_html=True,
)

# ── Sensitivity tornado ───────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Sensitivity — which parameter drives welfare loss most?</div>',
            unsafe_allow_html=True)

SENSITIVITY_PARAMS = ["N", "lmbda", "alpha", "beta", "gamma_g", "kappa", "q_star", "tau", "P_bar", "M", "omega_bar"]
PARAM_LABELS = {
    "N": "Domains N", "lmbda": "Cross-domain value λ", "alpha": "Analytics value α",
    "beta": "Synergy β", "gamma_g": "Generality cost γ_g", "kappa": "Fixed cost κ",
    "q_star": "Baseline quality q*", "tau": "Integration cost τ",
    "P_bar": "Prob. needing domain P̄", "M": "Consumers M", "omega_bar": "Consumer weight ω̄",
}

tornado_rows = []
for p_key in SENSITIVITY_PARAMS:
    lo, hi = PARAM_BOUNDS[p_key]
    dw_low  = compute_all(dict(params, **{p_key: lo}))["delta_W"]
    dw_high = compute_all(dict(params, **{p_key: hi}))["delta_W"]
    tornado_rows.append((PARAM_LABELS[p_key], dw_low, dw_high, abs(dw_high - dw_low)))

tornado_rows.sort(key=lambda x: x[3])
labels   = [r[0] for r in tornado_rows]
dw_lows  = [r[1] for r in tornado_rows]
dw_highs = [r[2] for r in tornado_rows]

fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(name="Low end of range",  x=dw_lows,  y=labels, orientation="h",
    marker_color="rgba(96,165,250,0.7)", marker_line_width=0))
fig_tornado.add_trace(go.Bar(name="High end of range", x=dw_highs, y=labels, orientation="h",
    marker_color="rgba(248,113,113,0.7)", marker_line_width=0))
fig_tornado.add_vline(x=delta_W, line_dash="dot", line_color="#FCD34D",
    annotation_text="  Current ΔW", annotation_font_color="#FCD34D")
fig_tornado.update_layout(barmode="overlay", height=380,
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_CLR, size=11),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, title_text="ΔW (model units)"),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
    title=dict(text="Tornado — range of ΔW when each parameter is swept across its full range",
               font=dict(size=12, color="#E5E7EB"), x=0.01),
    showlegend=True,
)
st.plotly_chart(fig_tornado, use_container_width=True)

# ── Bar chart: gNE vs gSO ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Current parameter snapshot — equilibrium vs. optimum</div>',
            unsafe_allow_html=True)

fig3 = go.Figure()
fig3.add_trace(go.Bar(name="Nash Equilibrium gⁿᵉ", x=["Generality level"], y=[g_ne],
    marker_color="#F87171", text=[f"{g_ne:.3f}"], textposition="outside", textfont=dict(color="#F87171")))
fig3.add_trace(go.Bar(name="Social Optimum gˢᵒ",  x=["Generality level"], y=[g_so],
    marker_color="#34D399", text=[f"{g_so:.3f}"], textposition="outside", textfont=dict(color="#34D399")))
fig3.update_layout(barmode="group", height=260,
    margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_CLR, size=11),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, range=[0, 1.2], title_text="Generality g"),
    legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)",
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    title=dict(text="gⁿᵉ vs. gˢᵒ for your current parameters — Proposition 1",
               font=dict(size=12, color="#E5E7EB"), x=0.01),
    showlegend=True,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Governance regime comparison ──────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Governance regime comparison — Table 2 from the paper</div>',
            unsafe_allow_html=True)

regimes = [
    ("Pure Data Mesh",          0.0,              0.0,                  -welfare_annual_m,                   "#F87171"),
    ("Centralized Hydration",   round(g_so, 2),   cost_centralized,      welfare_annual_m - cost_centralized, "#FCD34D"),
    ("Federated + Incentives",  round(g_so, 2),   round(subsidy * N, 2), welfare_annual_m - subsidy * N,      "#34D399"),
    ("Hybrid (central silver)", round(g_so*0.7,2), cost_hybrid,          welfare_annual_m - cost_hybrid,      "#60A5FA"),
]

r_cols = st.columns(4)
for col, (name, g_val, cost, net, color) in zip(r_cols, regimes):
    sign = "+" if net >= 0 else ""
    col.markdown(f"""<div class="kpi-card" style="text-align:center;">
      <div class="kpi-label" style="text-align:center;">{name}</div>
      <div style="font-size:1.3rem;font-weight:800;color:{color};margin:0.3rem 0;">g = {g_val:.2f}</div>
      <div class="kpi-sub" style="text-align:center;">
        Platform cost: <strong>${cost:.2f}M</strong><br>
        Net vs. mesh: <span style="color:{color};font-weight:700;">{sign}${net:.1f}M</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Plain English interpretation ──────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">What does all this mean for your organization?</div>',
            unsafe_allow_html=True)

if in_trap:
    diagnosis_icon, diagnosis_title = "🔴", "Your organization is in the Data Mesh Trap"
    diagnosis_body = (
        f"With {N} domains, domain teams have <strong>zero incentive</strong> to build reusable data products. "
        f"No shared silver layer will emerge on its own. Cross-domain analytics will require "
        f"<strong>custom pipelines built from scratch</strong> every time — estimated "
        f"<strong>${td_total:,.2f}M</strong> in accumulated technical debt."
    )
elif pct_of_so < 40:
    diagnosis_icon, diagnosis_title = "🟠", "Severe underinvestment — silver layer is thin and fragile"
    diagnosis_body = (
        f"Domains invest some effort (gⁿᵉ = {g_ne:.2f}) but reach only "
        f"<strong>{pct_of_so:.0f}%</strong> of what is needed. Cross-domain projects hit data contract gaps, "
        f"schema mismatches, and missing documentation. Technical debt: <strong>${td_total:,.2f}M</strong> "
        f"across {N*(N-1)} potential domain pairs."
    )
elif pct_of_so < 75:
    diagnosis_icon, diagnosis_title = "🟡", "Moderate underinvestment — your silver layer needs a push"
    diagnosis_body = (
        f"Domains invest meaningfully (gⁿᵉ = {g_ne:.2f}, {pct_of_so:.0f}% of optimal). "
        f"A real silver layer exists but gaps remain. A small governance investment could close the gap "
        f"and unlock the full <strong>${welfare_annual_m:.1f}M/year</strong> in welfare gains."
    )
else:
    diagnosis_icon, diagnosis_title = "🟢", "Near-optimal — your governance is working"
    diagnosis_body = (
        f"Domains invest at {pct_of_so:.0f}% of the social optimum (gⁿᵉ = {g_ne:.2f}). "
        f"Strong incentive alignment — domain teams find it in their own interest to build reusable products. "
        f"Marginal improvements are still possible but returns are diminishing."
    )

st.markdown(f"""<div class="kpi-card" style="margin-bottom:1rem;">
  <div style="font-size:1.1rem;font-weight:700;color:#F9FAFB;margin-bottom:0.5rem;">
    {diagnosis_icon} {diagnosis_title}
  </div>
  <div style="color:#9CA3AF;font-size:0.88rem;line-height:1.65;">{diagnosis_body}</div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Recommended action</div>', unsafe_allow_html=True)
invest_needed = round(subsidy * N, 2)
rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.markdown(f"""<div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#60A5FA;margin-bottom:0.4rem;">💰 Option 1 — Federated incentives</div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        Pay each domain a <strong>cross-domain bonus of {subsidy:.2f} units</strong> per unit of generality added.<br><br>
        Total annual investment: ~<strong>${invest_needed:.2f}M</strong> across {N} domains.<br>
        Fully closes the gap to gˢᵒ = {g_so:.2f} at the lowest coordination cost.
      </div></div>""", unsafe_allow_html=True)

with rec2:
    st.markdown(f"""<div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#34D399;margin-bottom:0.4rem;">🏗️ Option 2 — Centralized hydration team</div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        Central team owns the silver layer. Domains keep bronze data ownership.<br><br>
        Cost: ~<strong>${cost_centralized:.1f}M/year</strong>.<br>
        Net benefit: ~<strong>${max(0.0, welfare_annual_m - cost_centralized):.1f}M/year</strong> saved.
      </div></div>""", unsafe_allow_html=True)

with rec3:
    st.markdown(f"""<div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#A78BFA;margin-bottom:0.4rem;">🤝 Option 3 — Hybrid approach</div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        Central team sets standards; domains execute within guardrails.<br><br>
        Cost: ~<strong>${cost_hybrid:.1f}M/year</strong>. Achieves ~70% of optimum (g ≈ {g_so*0.7:.2f}).<br>
        Net benefit: ~<strong>${max(0.0, welfare_annual_m - cost_hybrid):.1f}M/year</strong>.
      </div></div>""", unsafe_allow_html=True)

# ── AI executive summary ──────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">AI executive summary</div>', unsafe_allow_html=True)

try:
    import anthropic as _anthropic

    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY", "")
        if hasattr(st, "secrets") else ""
    ) or os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        api_key = st.text_input(
            "Anthropic API key", type="password",
            placeholder="sk-ant-... (your key is never stored)",
            help="Enter your Anthropic API key to generate an AI-written executive summary.",
        )

    def _summary_cache_key(p, res):
        blob = json.dumps({**p, **{k: round(v, 6) if isinstance(v, float) else v
                                   for k, v in res.items()}}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()

    if api_key:
        cache_key = _summary_cache_key(params, r)
        cached = st.session_state.get("ai_summary_cache", {}).get(cache_key)

        if cached:
            st.markdown(f'<div class="insight-box" style="color:#E5E7EB;">{cached}</div>',
                        unsafe_allow_html=True)
            if st.button("↺ Regenerate summary"):
                st.session_state.setdefault("ai_summary_cache", {}).pop(cache_key, None)
                st.rerun()
        else:
            if st.button("✨ Generate executive summary"):
                prompt = f"""You are a data governance consultant advising a Chief Data Officer.

Organization profile (Data Hydration Gap model — Besanson 2026):
- {params['N']} domains, {params['M']} cross-domain consumers
- Cross-domain data value λ={params['lmbda']:.2f}, analytics value α={params['alpha']:.2f}
- Generality cost γ_g={params['gamma_g']:.2f}, fixed cost κ={params['kappa']:.2f}, baseline quality q*={params['q_star']:.2f}

Model outputs:
- Nash Equilibrium g_NE = {r['g_ne']:.3f} {"⚠️ DATA MESH TRAP" if r['in_trap'] else ""}
- Social Optimum g_SO = {r['g_so']:.3f}
- Domains invest at {r['pct_of_so']:.0f}% of social optimum
- Annual welfare loss ΔW ≈ ${welfare_annual_m:.1f}M/year (§6.3 calibration)
- Accumulated technical debt = ${r['td_total']:.2f}M

Write exactly 3 paragraphs:
1. Diagnosis: what is happening and the root cause from a game-theoretic perspective.
2. Financial impact: what this costs the organization in concrete terms.
3. Recommended action: which governance approach, why, and what the first 90-day step should be.

Be specific, cite the numbers, and write for a non-technical C-suite audience. No bullet points."""

                client = _anthropic.Anthropic(api_key=api_key)
                placeholder = st.empty()
                full_text = ""
                try:
                    with client.messages.stream(
                        model="claude-sonnet-4-6", max_tokens=550,
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream:
                        for chunk in stream.text_stream:
                            full_text += chunk
                            placeholder.markdown(
                                f'<div class="insight-box" style="color:#E5E7EB;">{full_text}▌</div>',
                                unsafe_allow_html=True,
                            )
                    placeholder.markdown(
                        f'<div class="insight-box" style="color:#E5E7EB;">{full_text}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.setdefault("ai_summary_cache", {})[cache_key] = full_text
                except Exception as e:
                    st.error(f"API error: {e}")
    else:
        st.caption("Enter an Anthropic API key above to enable AI-generated executive summaries.")

except ImportError:
    st.caption("Install `anthropic` to enable AI-generated executive summaries.")

# ── Scenario comparison ───────────────────────────────────────────────────────
if st.session_state.scenarios:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Scenario comparison</div>', unsafe_allow_html=True)

    rows = []
    for s in st.session_state.scenarios:
        sr = compute_all({k: s[k] for k in [
            "N", "M", "alpha", "beta", "lmbda", "omega_bar",
            "gamma_g", "kappa", "q_star", "tau", "P_bar",
        ]})
        rows.append({
            "Scenario":       s["name"],
            "N":              s["N"],
            "λ":              s["lmbda"],
            "γ_g":            s["gamma_g"],
            "κ":              s["kappa"],
            "gⁿᵉ":           round(sr["g_ne"], 3),
            "gˢᵒ":           round(sr["g_so"], 3),
            "% of SO":        f"{sr['pct_of_so']:.0f}%",
            "ΔW":             round(sr["delta_W"], 2),
            "TD ($M)":        round(sr["td_total"], 2),
            "Status":         "🔴 Trap" if sr["in_trap"] else ("🟡 Low" if sr["pct_of_so"] < 40 else ("🟠 Mid" if sr["pct_of_so"] < 75 else "🟢 Good")),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    sc1, sc2 = st.columns([1, 4])
    with sc1:
        if st.button("🗑️ Clear all scenarios"):
            st.session_state.scenarios = []
            st.rerun()
    with sc2:
        multi_csv = pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("⬇️ Download comparison CSV", data=multi_csv,
                           file_name="scenarios_comparison.csv", mime="text/csv")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Export scenario results</div>', unsafe_allow_html=True)

export_df = pd.DataFrame([{
    "N (domains)": N, "M (consumers)": M,
    "α": alpha, "β": beta, "λ": lmbda, "ω̄": omega_bar,
    "γ_g": gamma_g, "κ": kappa, "q*": q_star, "τ ($M)": tau, "P̄": P_bar,
    "g_NE": round(g_ne, 4), "g_SO": round(g_so, 4),
    "Δg": round(delta_g, 4), "ΔW": round(delta_W, 4),
    "TD_total ($M)": round(td_total, 4), "s_i (subsidy)": round(subsidy, 4),
    "% of social optimum": round(pct_of_so, 1), "In trap": in_trap,
}])


def build_html_report():
    e = html_lib.escape

    def table_row(label, value, highlight=False):
        bg = "#1a2332" if highlight else "#0f172a"
        return f'<tr style="background:{bg}"><td style="padding:6px 12px;color:#9CA3AF;">{e(str(label))}</td><td style="padding:6px 12px;color:#F9FAFB;font-weight:600;">{e(str(value))}</td></tr>'

    solver = escape_trap_solver(params, g_ne_target=max(0.01, g_so * 0.5)) if (in_trap or pct_of_so < 75) else None
    solver_rows = ""
    if solver:
        for iv in solver.values():
            solver_rows += f'<tr style="background:#0f172a"><td style="padding:6px 12px;color:#9CA3AF;">{e(iv["label"])}</td><td style="padding:6px 12px;color:#34D399;">{e(str(iv["current"]))} → {e(str(iv["required"]))}</td><td style="padding:6px 12px;color:#60A5FA;">{e(iv["effort"])}</td><td style="padding:6px 12px;color:#6B7280;font-size:0.8em;">{e(iv["action"])}</td></tr>'
    else:
        solver_rows = f'<tr><td colspan="4" style="padding:6px 12px;color:#4B5563;">No intervention needed — g_NE is at or above target.</td></tr>'

    regime_rows = ""
    for name, g_val, cost, net, color in regimes:
        sign = "+" if net >= 0 else ""
        regime_rows += f'<tr style="background:#0f172a"><td style="padding:6px 12px;color:#9CA3AF;">{e(name)}</td><td style="padding:6px 12px;color:{color};font-weight:600;">g = {g_val:.2f}</td><td style="padding:6px 12px;color:#F9FAFB;">${cost:.2f}M</td><td style="padding:6px 12px;color:{color};font-weight:600;">{sign}${net:.1f}M</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Hydration Gap Report</title>
<style>
  body {{ font-family: 'Segoe UI', Inter, sans-serif; background:#030712; color:#F9FAFB; margin:0; padding:2rem; }}
  h1 {{ font-size:1.8rem; font-weight:800; color:#93C5FD; margin-bottom:0.25rem; }}
  h2 {{ font-size:1rem; font-weight:700; color:#60A5FA; text-transform:uppercase; letter-spacing:0.08em; margin-top:2rem; margin-bottom:0.5rem; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:0.4rem; }}
  .subtitle {{ color:#6B7280; font-size:0.85rem; margin-bottom:2rem; }}
  .diagnosis {{ padding:1rem 1.2rem; border-radius:0.75rem; background:rgba(96,165,250,0.07); border:1px solid rgba(96,165,250,0.2); color:#93C5FD; font-size:0.9rem; line-height:1.6; margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85rem; margin-bottom:1.5rem; }}
  th {{ background:#1e3a5f; color:#93C5FD; padding:8px 12px; text-align:left; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; }}
  tr:hover {{ background:#162032 !important; }}
  .footer {{ color:#374151; font-size:0.7rem; text-align:center; margin-top:3rem; border-top:1px solid rgba(255,255,255,0.06); padding-top:1rem; }}
  @media print {{ body {{ background:#fff; color:#111; }} .diagnosis {{ background:#f0f7ff; color:#1e40af; border-color:#93c5fd; }} th {{ background:#1e3a5f; }} }}
</style>
</head>
<body>
<h1>💧 Data Hydration Gap Report</h1>
<div class="subtitle">Generated {date.today().isoformat()} · Based on Besanson (2026)</div>

<div class="diagnosis"><strong>{diagnosis_icon} {e(diagnosis_title)}</strong><br>{diagnosis_body}</div>

<h2>Model Outputs</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{table_row("Equilibrium generality gⁿᵉ (Eq. 7)", f"{g_ne:.4f}" + (" ⚠️ TRAP" if in_trap else ""), highlight=in_trap)}
{table_row("Social optimum gˢᵒ (Prop. 1)", f"{g_so:.4f}")}
{table_row("Underinvestment gap Δg", f"{delta_g:.4f}")}
{table_row("% of social optimum achieved", f"{pct_of_so:.1f}%")}
{table_row("Annual welfare loss ΔW (Eq. 10)", f"{delta_W:.2f} model units ≈ ${welfare_annual_m:.1f}M/yr", highlight=True)}
{table_row("Accumulated technical debt (Eq. 13)", f"${td_total:.2f}M", highlight=True)}
{table_row("Pigouvian subsidy per domain sᵢ (Eq. 19)", f"{subsidy:.4f}")}
</table>

<h2>Input Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
{table_row("N — number of domains", N)}
{table_row("M — cross-domain consumers", M)}
{table_row("α — domain analytics value", alpha)}
{table_row("β — generality–quality synergy", beta)}
{table_row("λ — cross-domain data value", lmbda)}
{table_row("ω̄ — avg. consumer weight", omega_bar)}
{table_row("γ_g — generality cost", gamma_g)}
{table_row("κ — fixed standardisation cost", kappa)}
{table_row("q* — baseline data quality", q_star)}
{table_row("τ — integration cost per pair ($M)", tau)}
{table_row("P̄ — avg. prob. needing another domain", P_bar)}
</table>

<h2>Governance Regime Comparison</h2>
<table>
<tr><th>Regime</th><th>Generality</th><th>Platform Cost</th><th>Net vs. Mesh</th></tr>
{regime_rows}
</table>

<h2>Minimum Interventions to Close the Gap</h2>
<table>
<tr><th>Lever</th><th>Current → Required</th><th>Effort</th><th>Action</th></tr>
{solver_rows}
</table>

<div class="footer">
  Besanson (2026) · The Data Hydration Gap · A Formal Model of Underinvestment in General-Purpose Data Products<br>
  gⁿᵉ → eq.(7) · gˢᵒ → Prop. 1 · ΔW → eq.(10) · TD_total → eq.(13) · sᵢ → eq.(19)
</div>
</body>
</html>"""


exp1, exp2 = st.columns(2)
with exp1:
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download scenario as CSV", data=csv_bytes,
                       file_name="data_hydration_gap_scenario.csv", mime="text/csv")
with exp2:
    html_bytes = build_html_report().encode("utf-8")
    st.download_button("📄 Download report as HTML", data=html_bytes,
                       file_name="data_hydration_gap_report.html", mime="text/html",
                       help="Open in any browser and use File → Print → Save as PDF for a PDF version.")

with st.expander("📖 Plain English guide to every number on this page"):
    st.markdown("""
| Number | What it means in plain English |
|--------|-------------------------------|
| **gⁿᵉ — equilibrium generality** | If you leave domain teams alone, this is how reusable their data products will be. 0 = completely siloed, 1 = fully standardised. |
| **gˢᵒ — social optimum** | The reusability level that maximises value for the whole organisation. This is your target. |
| **Generality gap Δg** | How far short of the ideal your organisation falls when domains act in self-interest. |
| **Welfare loss ΔW** | Total organisational value destroyed annually because of underinvestment in shared data products. |
| **Technical debt TD** | Hidden future cost from narrow data products. Every domain pair without a general product needs a custom integration pipeline. Grows as N². |
| **Reusability bonus sᵢ** | The reward each domain needs to make it worth their while to build general products — like a carbon tax but in reverse. |
| **N × (N−1) pairs** | Potential cross-domain integrations. With 12 domains: 132 pairs. With 20: 380. This is why data debt explodes at scale. |
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#374151;font-size:0.7rem;text-align:center;">
Besanson (2026) · The Data Hydration Gap · A Formal Model of Underinvestment in General-Purpose Data Products<br>
gⁿᵉ → eq.(7) · gˢᵒ → Proposition 1 · ΔW → eq.(10) · TD_total → eq.(13) · sᵢ → eq.(19)
</p>
""", unsafe_allow_html=True)
