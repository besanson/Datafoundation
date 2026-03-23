import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Hydration Gap Model",
    page_icon="💧",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* Base */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* Hero */
        .hero-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #7DD3FC;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.15;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #F9FAFB 0%, #93C5FD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            color: #6B7280;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 680px;
            margin-bottom: 2rem;
        }

        /* Section headings */
        .section-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #4B5563;
            font-weight: 600;
            margin-bottom: 0.6rem;
            margin-top: 1.8rem;
        }

        /* KPI cards */
        .kpi-card {
            padding: 1.2rem 1.4rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #0F172A 0%, #030712 100%);
            border: 1px solid rgba(148, 163, 184, 0.15);
            height: 100%;
            box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        }
        .kpi-label {
            color: #6B7280;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #F9FAFB;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }
        .kpi-value-warn {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #F87171;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }
        .kpi-value-good {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #34D399;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }
        .kpi-sub {
            color: #4B5563;
            font-size: 0.78rem;
            line-height: 1.4;
        }

        /* Status badge */
        .badge-trap {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #FCA5A5;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.4rem;
        }
        .badge-ok {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.3);
            color: #6EE7B7;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.4rem;
        }

        /* Divider */
        .divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin: 1.5rem 0;
        }

        /* Regime table row colors */
        .regime-best { color: #34D399; font-weight: 700; }
        .regime-mid  { color: #FCD34D; }
        .regime-bad  { color: #F87171; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown(
    """
    <div>
      <div class="hero-pill">💧 Model · Data Hydration Gap · Besanson 2026</div>
      <div class="hero-title">See your data mesh trap in real time.</div>
      <div class="hero-sub">
        Interactively explore how equilibrium generality, social optimum, welfare loss,
        and technical debt evolve under the formal game-theoretic model of the data hydration gap.
        All computations follow the paper's equations directly.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar — grouped parameters
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    # Preset button
    if st.button("📄 Load paper baseline (12 domains)", use_container_width=True):
        st.session_state["N"]       = 12
        st.session_state["alpha"]   = 0.5
        st.session_state["beta"]    = 0.15
        st.session_state["lmbda"]   = 0.4
        st.session_state["gamma_g"] = 0.4
        st.session_state["kappa"]   = 0.25
        st.session_state["q_star"]  = 0.6
        st.session_state["M"]       = 10
        st.session_state["omega"]   = 0.3
        st.session_state["tau"]     = 0.05
        st.session_state["P_bar"]   = 0.5

    if st.button("🏢 Fortune 500 preset (20 domains)", use_container_width=True):
        st.session_state["N"]       = 20
        st.session_state["alpha"]   = 0.6
        st.session_state["beta"]    = 0.2
        st.session_state["lmbda"]   = 0.5
        st.session_state["gamma_g"] = 0.35
        st.session_state["kappa"]   = 0.2
        st.session_state["q_star"]  = 0.7
        st.session_state["M"]       = 20
        st.session_state["omega"]   = 0.3
        st.session_state["tau"]     = 0.08
        st.session_state["P_bar"]   = 0.6

    st.markdown("---")

    with st.expander("🏗️ Organization structure", expanded=True):
        N = st.slider("Number of domains N", 2, 40,
                      st.session_state.get("N", 12),
                      help="How many domain teams are in the mesh?")
        M = st.slider("Cross‑domain consumers M", 0, 50,
                      st.session_state.get("M", 10),
                      help="Analytics / ML teams consuming data from multiple domains.")

    with st.expander("📈 Incentives & value", expanded=True):
        alpha = st.slider("Domain analytics value α", 0.1, 1.0,
                          st.session_state.get("alpha", 0.5), 0.05,
                          help="How much a typical domain values its own analytics. Range: 0.3–0.8")
        beta = st.slider("Generality–quality synergy β", 0.0, 0.5,
                         st.session_state.get("beta", 0.15), 0.01,
                         help="How much standardization forces better quality. Range: 0.1–0.3")
        lmbda = st.slider("Cross‑domain data value λ", 0.0, 1.0,
                          st.session_state.get("lmbda", 0.4), 0.01,
                          help="Share of value from cross‑domain use cases. Range: 0.2–0.6")
        omega_bar = st.slider("Average consumer weight ω̄", 0.0, 1.0,
                              st.session_state.get("omega", 0.3), 0.05,
                              help="Importance of each domain's data for cross‑domain consumers.")

    with st.expander("💰 Cost structure", expanded=False):
        gamma_g = st.slider("Generality cost γ_g", 0.1, 1.0,
                            st.session_state.get("gamma_g", 0.4), 0.05,
                            help="Difficulty of making products general. Range: 0.3–0.5")
        kappa = st.slider("Fixed standardization cost κ", 0.0, 0.6,
                          st.session_state.get("kappa", 0.25), 0.01,
                          help="Schema governance / semantic standardization overhead. Range: 0.1–0.4")
        q_star = st.slider("Baseline quality q*", 0.1, 1.0,
                           st.session_state.get("q_star", 0.6), 0.05,
                           help="Typical silver-layer-ready quality level.")

    with st.expander("🏦 Technical debt", expanded=False):
        tau = st.slider("Integration cost per unit τ (M$)", 0.0, 0.2,
                        st.session_state.get("tau", 0.05), 0.01,
                        help="Cost per unit of custom pipeline integration work (millions).")
        P_bar = st.slider("Avg. prob. needing another domain P̄", 0.0, 1.0,
                          st.session_state.get("P_bar", 0.5), 0.05,
                          help="Probability that domain i needs domain j's data.")

# ─────────────────────────────────────────────
# Core model — paper equations
# ─────────────────────────────────────────────

# Eq. (7): g_NE = max{0, (αβ − κ/q*) / γ_g}
g_ne = max(0.0, (alpha * beta - kappa / q_star) / gamma_g) if (q_star > 0 and gamma_g > 0) else 0.0

# Eq. (8) + consumer term: g_SO = [αβ + (N−1)λ + Mω̄ − κ/q*] / γ_g, clipped to [0,1]
g_so_raw = (alpha * beta + (N - 1) * lmbda + M * omega_bar - kappa / q_star) / gamma_g \
    if (q_star > 0 and gamma_g > 0) else 0.0
g_so = min(max(g_so_raw, 0.0), 1.0)

delta_g = max(0.0, g_so - g_ne)

# Eq. (10): ΔW = N·[((N−1)λ + Mω̄)·q*·Δg] − (γ_g/2)·N·q*·(g_SO² − g_NE²)
welfare_ext  = ((N - 1) * lmbda + M * omega_bar) * q_star * delta_g
welfare_cost = (gamma_g / 2.0) * q_star * (g_so**2 - g_ne**2)
delta_W      = N * (welfare_ext - welfare_cost)

# Eq. (13): TD_total = τ · q̄ · N(N−1) · P̄
td_total = tau * q_star * N * (N - 1) * P_bar

# Pigouvian subsidy (eq. 19): s_i = (N−1)λ·q*
subsidy = (N - 1) * lmbda * q_star

# Data mesh trap flag
in_trap = g_ne == 0.0

# ─────────────────────────────────────────────
# KPI Row 1 — generality
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Equilibrium vs. social optimum</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Equilibrium generality gⁿᵉ</div>
          <div class="{'kpi-value-warn' if in_trap else 'kpi-value'}">{g_ne:.3f}</div>
          <div class="kpi-sub">What domains choose on their own — FOC eq. (7). Ignores cross-domain externalities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Social optimum gˢᵒ</div>
          <div class="kpi-value-good">{g_so:.3f}</div>
          <div class="kpi-sub">Planner choice internalising (N−1)λ and Mω̄ — Proposition 1, eq. (8).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    badge = '<div class="badge-trap">⚠️ Data mesh trap — gⁿᵉ = 0</div>' if in_trap \
        else '<div class="badge-ok">✅ Partial underinvestment</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Generality gap Δg</div>
          <div class="kpi-value">{delta_g:.3f}</div>
          {badge}
          <div class="kpi-sub">Gap = (N−1)λ / γ_g per Proposition 1.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# KPI Row 2 — welfare & debt
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Welfare and technical debt</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Welfare loss ΔW (model units)</div>
          <div class="kpi-value-warn">{delta_W:,.1f}</div>
          <div class="kpi-sub">Decentralised equilibrium vs. planner — eq. (10). Scales as N².</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Technical debt TD_total (M$)</div>
          <div class="kpi-value-warn">{td_total:,.2f}</div>
          <div class="kpi-sub">τ · q̄ · N(N−1) · P̄ — eq. (13). Quadratic growth in N.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col6:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Pigouvian subsi
