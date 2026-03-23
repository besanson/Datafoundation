import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Data Hydration Gap Model",
    page_icon="💧",
    layout="wide",
)

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
    .hero-sub {
        color: #6B7280; font-size: 0.95rem; line-height: 1.6;
        max-width: 680px; margin-bottom: 1.8rem;
    }
    .section-label {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: #4B5563; font-weight: 600; margin-bottom: 0.5rem; margin-top: 1.6rem;
    }
    .kpi-card {
        padding: 1.15rem 1.3rem; border-radius: 1rem;
        background: linear-gradient(135deg, #0F172A 0%, #030712 100%);
        border: 1px solid rgba(148,163,184,0.12); height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .kpi-label { color: #6B7280; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.25rem; }
    .kpi-value      { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #F9FAFB;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-warn { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #F87171;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-good { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #34D399;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-value-blue { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; color: #60A5FA;  margin-bottom: 0.2rem; line-height: 1.1; }
    .kpi-sub   { color: #4B5563; font-size: 0.76rem; line-height: 1.45; }
    .kpi-interp { color: #9CA3AF; font-size: 0.78rem; line-height: 1.45; margin-top: 0.4rem;
        border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.4rem; }
    .badge-trap { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3);
        color:#FCA5A5; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .badge-ok { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
        color:#6EE7B7; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .badge-partial { display:inline-block; padding:0.18rem 0.6rem; border-radius:999px;
        background:rgba(252,211,77,0.12); border:1px solid rgba(252,211,77,0.3);
        color:#FDE68A; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; }
    .divider { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:1.5rem 0; }
    .insight-box {
        padding: 0.9rem 1.1rem; border-radius: 0.75rem;
        background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.2);
        color: #93C5FD; font-size: 0.82rem; line-height: 1.55; margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

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

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📄 Paper baseline", use_container_width=True):
            st.session_state.update(N=12, alpha=0.5, beta=0.15, lmbda=0.4,
                                    gamma_g=0.4, kappa=0.25, q_star=0.6,
                                    M=10, omega=0.3, tau=0.05, P_bar=0.5)
    with col_b:
        if st.button("🏢 Fortune 500", use_container_width=True):
            st.session_state.update(N=20, alpha=0.6, beta=0.2, lmbda=0.5,
                                    gamma_g=0.35, kappa=0.2, q_star=0.7,
                                    M=20, omega=0.3, tau=0.08, P_bar=0.6)

    st.markdown("---")

    with st.expander("🏗️ Organization structure", expanded=True):
        N = st.slider("Number of domains N", 2, 40, st.session_state.get("N", 12),
            help="📌 Count your distinct business domains that own data products.\n\nExample: Sales, Marketing, Finance, Supply Chain, HR, IT = 6 domains. Large enterprises typically have 12–20.")
        M = st.slider("Cross‑domain consumers M", 0, 50, st.session_state.get("M", 10),
            help="📌 How many analytics teams or ML projects consume data from more than one domain?\n\nExample: A central BI team + 3 data science squads + 6 product analytics = ~10 consumers.")

    with st.expander("📈 Incentives & value", expanded=True):
        alpha = st.slider("Domain analytics value α", 0.1, 1.0,
                          st.session_state.get("alpha", 0.5), 0.05,
            help="📌 How much does a typical domain rely on its own data for daily decisions?\n\nLow (0.3): Domain rarely uses its own data (e.g. Legal, Compliance)\nMedium (0.5): Moderate use (e.g. HR, IT)\nHigh (0.8): Data-driven domain (e.g. Sales, Marketing with dashboards)")
        beta = st.slider("Generality–quality synergy β", 0.0, 0.5,
                         st.session_state.get("beta", 0.15), 0.01,
            help="📌 When a domain standardizes its data product, how much does its own data quality improve as a side effect?\n\nLow (0.1): Standardization adds little internal value (domain already has clean data)\nHigh (0.3): Standardization forces the domain to fix schema issues, improve docs, and clean definitions")
        lmbda = st.slider("Cross‑domain data value λ", 0.0, 1.0,
                          st.session_state.get("lmbda", 0.4), 0.01,
            help="📌 What share of analytical value in your org comes from combining data across domains?\n\nLow (0.2): Most analytics stays within one domain (siloed org)\nMedium (0.4): 40% of queries need cross-domain joins (typical enterprise)\nHigh (0.6): Most insights require cross-domain data (e.g. customer 360, supply chain + finance)")
        omega_bar = st.slider("Avg. consumer weight ω̄", 0.0, 1.0,
                              st.session_state.get("omega", 0.3), 0.05,
            help="📌 How important is each individual domain's data to cross-domain consumers on average?\n\nLow (0.1): Consumers depend on only a few key domains\nHigh (0.5): Consumers depend heavily and equally on all domains")

    with st.expander("💰 Cost structure", expanded=False):
        gamma_g = st.slider("Generality cost γ_g", 0.1, 1.0,
                            st.session_state.get("gamma_g", 0.4), 0.05,
            help="📌 How costly is it to make a data product general vs. narrow, per unit of quality?\n\nLow (0.2): Your platform already has strong schema governance — going general is cheap\nMedium (0.4): Typical enterprise with some tooling\nHigh (0.7): No data contracts, no catalog, every standardization is manual effort")
        kappa = st.slider("Fixed standardization cost κ", 0.0, 0.6,
                          st.session_state.get("kappa", 0.25), 0.01,
            help="📌 What is the fixed overhead just to START standardizing a data product, regardless of quality?\n\nLow (0.1): You have templates, a data catalog, and a governance team\nMedium (0.25): Some governance exists but domains still need coordination meetings\nHigh (0.4): No catalog, no contracts — every standardization needs cross-team alignment from scratch")
        q_star = st.slider("Baseline quality q*", 0.1, 1.0,
                           st.session_state.get("q_star", 0.6), 0.05,
            help="📌 What is the typical quality level of your raw/bronze data before generalization decisions?\n\nLow (0.3): Dirty source data, many nulls, inconsistent formats\nMedium (0.6): Moderate quality — some documentation, mostly complete\nHigh (0.9): Near-production quality sources with strong data contracts")

    with st.expander("🏦 Technical debt", expanded=False):
        tau = st.slider("Integration cost per pair τ (M$)", 0.0, 0.2,
                        st.session_state.get("tau", 0.05), 0.01,
            help="📌 How much does it cost (in millions) to build one custom pipeline between two domains when no general product exists?\n\nLow ($0.02M): Lightweight integrations, reusable patterns\nMedium ($0.05M): ~3 weeks of a data engineer's time + infra\nHigh ($0.10M): Complex schemas, legacy systems, significant QA effort")
        P_bar = st.slider("Avg. prob. needing another domain P̄", 0.0, 1.0,
                          st.session_state.get("P_bar", 0.5), 0.05,
            help="📌 On average, what is the probability that any given domain will need data from another specific domain?\n\nLow (0.2): Most domains are independent — few cross-domain dependencies\nHigh (0.7): High interdependency — e.g. Finance needs Sales, Supply Chain needs Manufacturing, etc.")

# ── Model computations ────────────────────────────────────────────────────────

# Eq. (7): g_NE = max{0, (αβ − κ/q*) / γ_g}
g_ne = max(0.0, (alpha * beta - kappa / q_star) / gamma_g) if (q_star > 0 and gamma_g > 0) else 0.0

# Prop 1 / Eq. (8): g_SO = [αβ + (N−1)λ + Mω̄ − κ/q*] / γ_g clipped [0,1]
g_so_raw = (alpha * beta + (N - 1) * lmbda + M * omega_bar - kappa / q_star) / gamma_g \
    if (q_star > 0 and gamma_g > 0) else 0.0
g_so = float(np.clip(g_so_raw, 0.0, 1.0))

delta_g = max(0.0, g_so - g_ne)

# Eq. (10): ΔW
welfare_ext  = ((N - 1) * lmbda + M * omega_bar) * q_star * delta_g
welfare_cost = (gamma_g / 2.0) * q_star * (g_so**2 - g_ne**2)
delta_W      = N * (welfare_ext - welfare_cost)

# Eq. (13): TD_total = τ · q̄ · N(N−1) · P̄
td_total = tau * q_star * N * (N - 1) * P_bar

# Eq. (19): Pigouvian subsidy s_i = (N−1)λ · q*
subsidy = (N - 1) * lmbda * q_star

# Derived helpers
in_trap          = g_ne == 0.0
pct_of_so        = (g_ne / g_so * 100) if g_so > 0 else 0.0
welfare_annual_m = N * 0.75   # §6.3 calibration ~$750K/domain

# ── Insight banner ────────────────────────────────────────────────────────────
if in_trap:
    insight_msg = (
        f"⚠️ <strong>Data mesh trap detected.</strong> With your current parameters, "
        f"no domain will voluntarily invest in general data products (gⁿᵉ = 0). "
        f"The silver layer will <em>not</em> emerge organically. "
        f"The socially optimal level is gˢᵒ = {g_so:.2f} — requiring either centralized hydration "
        f"or a reusability bonus of <strong>{subsidy:.3f}</strong> per domain."
    )
elif pct_of_so < 40:
    insight_msg = (
        f"🟡 <strong>Severe underinvestment.</strong> Domains invest in generality "
        f"(gⁿᵉ = {g_ne:.2f}) but only reach <strong>{pct_of_so:.0f}%</strong> of the social optimum "
        f"(gˢᵒ = {g_so:.2f}). Significant welfare is being left on the table."
    )
elif pct_of_so < 75:
    insight_msg = (
        f"🔵 <strong>Moderate underinvestment.</strong> Domains self-invest at {pct_of_so:.0f}% of "
        f"the optimum (gⁿᵉ = {g_ne:.2f} vs gˢᵒ = {g_so:.2f}). "
        f"A light federated incentive mechanism could close the gap."
    )
else:
    insight_msg = (
        f"✅ <strong>Near-optimal investment.</strong> Domains are self-investing at {pct_of_so:.0f}% "
        f"of the social optimum (gⁿᵉ = {g_ne:.2f} vs gˢᵒ = {g_so:.2f}). "
        f"Your governance structure is performing well."
    )

st.markdown(f'<div class="insight-box">{insight_msg}</div>', unsafe_allow_html=True)

# ── KPI Row 1 — generality ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Generality: what domains choose vs. what is optimal</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    val_class = "kpi-value-warn" if in_trap else "kpi-value"
    interp = ("Domains produce purely narrow products.<br>No silver layer will emerge without intervention."
              if in_trap else
              f"Domains self-invest at <strong>{pct_of_so:.0f}%</strong> of the social optimum.")
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">What domains actually do — gⁿᵉ (eq. 7)</div>
      <div class="{val_class}">{g_ne:.3f}</div>
      <div class="kpi-sub">Equilibrium generality: what a domain rationally chooses when it ignores benefits it creates for other domains.</div>
      <div class="kpi-interp">{interp}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">What society needs — gˢᵒ (Prop. 1)</div>
      <div class="kpi-value-good">{g_so:.3f}</div>
      <div class="kpi-sub">Social optimum: the generality level a central planner would choose, accounting for cross-domain value and consumer benefit.</div>
      <div class="kpi-interp">Includes externality (N−1)λ and consumer term Mω̄ that private domains ignore.</div>
    </div>""", unsafe_allow_html=True)

with col3:
    if in_trap:
        badge = '<span class="badge-trap">⚠️ Data mesh trap</span>'
    elif (delta_g / g_so > 0.5 if g_so > 0 else False):
        badge = '<span class="badge-partial">⚡ Large gap</span>'
    else:
        badge = '<span class="badge-ok">✅ Manageable gap</span>'
    gap_pct = (delta_g / g_so * 100) if g_so > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Underinvestment gap — Δg (eq. 8)</div>
      <div class="kpi-value">{delta_g:.3f}</div>
      {badge}
      <div class="kpi-sub">Distance between equilibrium and optimum = (N−1)λ / γ_g per Proposition 1.</div>
      <div class="kpi-interp">Domains invest at <strong>{pct_of_so:.0f}%</strong> of what is socially optimal ({gap_pct:.0f}% gap).</div>
    </div>""", unsafe_allow_html=True)

# ── KPI Row 2 — welfare, debt, subsidy ───────────────────────────────────────
st.markdown('<div class="section-label">Welfare loss, technical debt & corrective mechanism</div>',
            unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Annual welfare loss ΔW (eq. 10)</div>
      <div class="kpi-value-warn">{delta_W:,.1f} <span style="font-size:0.85rem;color:#9CA3AF;">model units</span></div>
      <div class="kpi-sub">Value destroyed annually by decentralised governance vs. a central planner. Scales as N².</div>
      <div class="kpi-interp">Using paper calibration (~$750K/domain), your {N}-domain org loses ≈ <strong>${welfare_annual_m:.1f}M/year</strong> in duplicated effort, custom pipelines, and data quality debt.</div>
    </div>""", unsafe_allow_html=True)

with col5:
    td_pairs = N * (N - 1)
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Technical debt TD_total (eq. 13)</div>
      <div class="kpi-value-warn">${td_total:,.2f}M</div>
      <div class="kpi-sub">Custom integration cost across all domain pairs when no general product exists. Grows quadratically with N.</div>
      <div class="kpi-interp">With {N} domains there are <strong>{td_pairs} potential integration pairs</strong>. Each narrow product creates debt for every domain that might need it.</div>
    </div>""", unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Reusability bonus needed — sᵢ (eq. 19)</div>
      <div class="kpi-value-blue">{subsidy:.3f}</div>
      <div class="kpi-sub">The per-domain reward that exactly corrects the externality and aligns domain incentives with the social optimum.</div>
      <div class="kpi-interp">Paying each domain <strong>{subsidy:.3f}</strong> per unit of generality closes the gap at ~$1M/yr governance cost vs ${welfare_annual_m:.1f}M/yr welfare loss.</div>
    </div>""", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">How things scale with the number of domains N</div>',
            unsafe_allow_html=True)

N_vals      = np.arange(2, 41, dtype=float)
td_curve    = tau * q_star * N_vals * (N_vals - 1) * P_bar
g_ne_scalar = max(0.0, (alpha * beta - kappa / q_star) / gamma_g) if (q_star > 0 and gamma_g > 0) else 0.0
g_ne_curve  = np.full_like(N_vals, g_ne_scalar)
g_so_curve  = np.clip(
    (alpha * beta + (N_vals - 1) * lmbda + M * omega_bar - kappa / q_star) / gamma_g,
    0.0, 1.0
)
delta_g_curve  = np.maximum(0.0, g_so_curve - g_ne_curve)
welfare_ext_c  = ((N_vals - 1) * lmbda + M * omega_bar) * q_star * delta_g_curve
welfare_cost_c = (gamma_g / 2.0) * q_star * (g_so_curve**2 - g_ne_scalar**2)
dw_curve       = N_vals * (welfare_ext_c - welfare_cost_c)

PLOT_BG  = "rgba(10,15,28,1)"
PAPER_BG = "rgba(5,8,20,1)"
GRID_CLR = "rgba(255,255,255,0.05)"
FONT_CLR = "#9CA3AF"

def base_layout(title_text, h=300):
    return dict(
        title=dict(text=title_text, font=dict(size=12, color="#E5E7EB"), x=0.01),
        height=h,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, size=11),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, title_text="N (number of domains)"),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=N_vals, y=g_ne_curve, name="gⁿᵉ (equilibrium)",
        mode="lines", line=dict(color="#F87171", width=2.5, dash="dot")))
    fig1.add_trace(go.Scatter(x=N_vals, y=g_so_curve, name="gˢᵒ (social optimum)",
        mode="lines", line=dict(color="#34D399", width=2.5),
        fill="tonexty", fillcolor="rgba(52,211,153,0.07)"))
    fig1.add_vline(x=N, line_dash="dot", line_color="#7DD3FC",
        annotation_text=f"  N={N}", annotation_font_color="#7DD3FC")
    fig1.update_layout(**base_layout("Generality gap grows with N — Proposition 1"))
    fig1.update_yaxes(title_text="Generality g", range=[0, 1.05])
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=N_vals, y=td_curve, name="TD_total",
        mode="lines", line=dict(color="#F87171", width=2.5),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
    fig2.add_vline(x=N, line_dash="dot", line_color="#7DD3FC",
        annotation_text=f"  N={N}, TD=${td_total:.2f}M", annotation_font_color="#7DD3FC")
    fig2.update_layout(**base_layout("Technical debt TD_total ($M) vs N — eq. (13)"))
    fig2.update_yaxes(title_text="TD_total ($M)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-label">Current parameter snapshot — equilibrium vs. optimum</div>',
            unsafe_allow_html=True)

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    name="Nash Equilibrium gⁿᵉ", x=["Generality level"], y=[g_ne],
    marker_color="#F87171",
    text=[f"{g_ne:.3f}"], textposition="outside", textfont=dict(color="#F87171"),
))
fig3.add_trace(go.Bar(
    name="Social Optimum gˢᵒ", x=["Generality level"], y=[g_so],
    marker_color="#34D399",
    text=[f"{g_so:.3f}"], textposition="outside", textfont=dict(color="#34D399"),
))
fig3.update_layout(
    barmode="group",
    height=260,
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
    ("Pure Data Mesh",          0.0,              0.0, -welfare_annual_m,      "#F87171"),
    ("Centralized Hydration",   round(g_so, 2),   2.0,  welfare_annual_m-2.0,  "#FCD34D"),
    ("Federated + Incentives",  round(g_so, 2),   1.0,  welfare_annual_m-1.0,  "#34D399"),
    ("Hybrid (central silver)", round(g_so*0.7,2),1.5,  welfare_annual_m-1.5,  "#60A5FA"),
]

r_cols = st.columns(4)
for col, (name, g_val, cost, net, color) in zip(r_cols, regimes):
    sign = "+" if net >= 0 else ""
    col.markdown(f"""
    <div class="kpi-card" style="text-align:center;">
      <div class="kpi-label" style="text-align:center;">{name}</div>
      <div style="font-size:1.3rem;font-weight:800;color:{color};margin:0.3rem 0;">g = {g_val:.2f}</div>
      <div class="kpi-sub" style="text-align:center;">
        Platform cost: <strong>${cost:.1f}M</strong><br>
        Net vs. mesh: <span style="color:{color};font-weight:700;">{sign}${net:.1f}M</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Plain English interpretation ──────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">What does all this mean for your organization?</div>',
            unsafe_allow_html=True)

if in_trap:
    diagnosis_icon  = "🔴"
    diagnosis_title = "Your organization is in the Data Mesh Trap"
    diagnosis_body  = (
        f"With {N} domains, your current setup gives domain teams <strong>zero incentive</strong> "
        f"to build data products that others can reuse. Every team will build the minimum needed "
        f"for their own use case — and stop there. "
        f"No shared silver layer will emerge on its own. "
        f"Cross-domain analytics (customer 360, supply chain visibility, unified reporting) will require "
        f"<strong>custom pipelines built from scratch</strong> every single time, "
        f"at an estimated accumulated cost of <strong>${td_total:,.2f}M</strong> in technical debt."
    )
elif pct_of_so < 40:
    diagnosis_icon  = "🟠"
    diagnosis_title = "Severe underinvestment — silver layer is thin and fragile"
    diagnosis_body  = (
        f"Domains invest some effort in generality (gⁿᵉ = {g_ne:.2f}) but only reach "
        f"<strong>{pct_of_so:.0f}%</strong> of what is needed. "
        f"A partial silver layer exists but is narrow and domain-specific. "
        f"Cross-domain projects will constantly hit data contract gaps, schema mismatches, "
        f"and missing documentation. Technical debt is accumulating at "
        f"<strong>${td_total:,.2f}M</strong> across {N*(N-1)} potential domain pairs."
    )
elif pct_of_so < 75:
    diagnosis_icon  = "🟡"
    diagnosis_title = "Moderate underinvestment — your silver layer needs a push"
    diagnosis_body  = (
        f"Domains are investing meaningfully in generality (gⁿᵉ = {g_ne:.2f}, "
        f"{pct_of_so:.0f}% of optimal). "
        f"A real silver layer exists but gaps remain. Some cross-domain analytics work well; "
        f"others still require custom integration. "
        f"A relatively small governance investment could close the remaining gap and "
        f"unlock the full ${welfare_annual_m:.1f}M in annual welfare gains."
    )
else:
    diagnosis_icon  = "🟢"
    diagnosis_title = "Near-optimal — your governance is working"
    diagnosis_body  = (
        f"Domains are investing at {pct_of_so:.0f}% of the social optimum (gⁿᵉ = {g_ne:.2f}). "
        f"Your organization has strong incentive alignment — domain teams find it in their own "
        f"interest to build reusable data products. Cross-domain analytics is largely self-serving. "
        f"Marginal improvements are still possible but returns are diminishing."
    )

st.markdown(f"""
<div class="kpi-card" style="margin-bottom:1rem;">
  <div style="font-size:1.1rem;font-weight:700;color:#F9FAFB;margin-bottom:0.5rem;">
    {diagnosis_icon} {diagnosis_title}
  </div>
  <div style="color:#9CA3AF;font-size:0.88rem;line-height:1.65;">{diagnosis_body}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Recommended action</div>', unsafe_allow_html=True)

rec_col1, rec_col2, rec_col3 = st.columns(3)
invest_needed = round(subsidy * N, 2)

with rec_col1:
    st.markdown(f"""
    <div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#60A5FA;margin-bottom:0.4rem;">
        💰 Option 1 — Federated incentives
      </div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        Pay each domain team a <strong>cross-domain bonus</strong> of
        <strong>{subsidy:.2f} units</strong> for every unit of generality they add to their data products.
        Think of it as a "reusability reward" funded by a small platform tax on narrow products.<br><br>
        Total annual investment: ~<strong>${invest_needed:.2f}M</strong> across all {N} domains.<br>
        This fully closes the gap to gˢᵒ = {g_so:.2f} at the lowest coordination cost.
      </div>
    </div>""", unsafe_allow_html=True)

with rec_col2:
    st.markdown(f"""
    <div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#34D399;margin-bottom:0.4rem;">
        🏗️ Option 2 — Centralized hydration team
      </div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        Create a <strong>central data hydration team</strong> responsible for building and maintaining
        the silver layer on behalf of all domains. Domains keep ownership of bronze (raw) data;
        the central team owns the standardized general layer.<br><br>
        Estimated cost: ~<strong>$2M/year</strong> (platform team + tooling).<br>
        Net benefit vs. pure mesh: ~<strong>${max(0.0, welfare_annual_m - 2.0):.1f}M/year</strong> saved.<br>
        Best when domain teams have low capacity or motivation to self-standardize.
      </div>
    </div>""", unsafe_allow_html=True)

with rec_col3:
    st.markdown(f"""
    <div class="kpi-card">
      <div style="font-size:1rem;font-weight:700;color:#A78BFA;margin-bottom:0.4rem;">
        🤝 Option 3 — Hybrid approach
      </div>
      <div style="color:#9CA3AF;font-size:0.82rem;line-height:1.6;">
        A <strong>central platform team sets standards</strong> (schemas, contracts, catalog),
        but domain teams execute their own silver layer within those guardrails.
        Combine tooling investment with a lighter incentive mechanism.<br><br>
        Estimated cost: ~<strong>$1.5M/year</strong>.<br>
        Achieves ~70% of social optimum (g ≈ {g_so*0.7:.2f}).<br>
        Net benefit: ~<strong>${max(0.0, welfare_annual_m - 1.5):.1f}M/year</strong>.<br>
        Best balance of autonomy and alignment for most enterprises.
      </div>
    </div>""", unsafe_allow_html=True)

with st.expander("📖 Plain English guide to every number on this page"):
    st.markdown("""
| Number | What it means in plain English |
|--------|-------------------------------|
| **gⁿᵉ — equilibrium generality** | If you leave domain teams alone with no incentives or mandates, this is how "reusable" their data products will be. 0 = completely siloed, 1 = fully standardized and reusable by everyone. |
| **gˢᵒ — social optimum** | The reusability level that would maximize value for the whole organization — if someone could coordinate everyone perfectly. This is your target. |
| **Generality gap Δg** | How far short of the ideal your organization falls when domains act on their own self-interest. The bigger this number, the more value is being left on the table. |
| **Welfare loss ΔW** | The total organizational value destroyed every year because of underinvestment in shared data products. Includes duplicated engineering, redundant pipelines, and inconsistent data definitions. |
| **Technical debt TD** | The hidden future cost your organization is accumulating. Every narrow data product today means a custom integration pipeline tomorrow. This grows explosively as the number of domains increases. |
| **Reusability bonus sᵢ** | The reward each domain team needs to receive to make it worth their while to invest in general data products. Same principle as a carbon tax but in reverse — you reward positive externalities instead of taxing negative ones. |
| **N × (N−1) pairs** | The number of potential cross-domain integrations that become expensive when no general products exist. With 12 domains that is 132 pairs. With 20 it is 380. This is why data debt explodes at scale. |
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#374151;font-size:0.7rem;text-align:center;">
Besanson (2026) · The Data Hydration Gap · A Formal Model of Underinvestment in General-Purpose Data Products<br>
gⁿᵉ → eq.(7) · gˢᵒ → Proposition 1 · ΔW → eq.(10) · TD_total → eq.(13) · sᵢ → eq.(19)
</p>
""", unsafe_allow_html=True)
