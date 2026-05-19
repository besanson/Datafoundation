import numpy as np


# ── Core equations ─────────────────────────────────────────────────────────────

def equilibrium_generality(alpha, beta, kappa, q_star, gamma_g):
    """Eq. (7): g_NE = max{0, (αβ − κ/q*) / γ_g}"""
    if q_star <= 0 or gamma_g <= 0:
        return 0.0
    return max(0.0, (alpha * beta - kappa / q_star) / gamma_g)


def social_optimum(alpha, beta, N, lmbda, M, omega_bar, kappa, q_star, gamma_g):
    """Proposition 1 / Eq. (8): g_SO clipped to [0, 1]"""
    if q_star <= 0 or gamma_g <= 0:
        return 0.0
    raw = (alpha * beta + (N - 1) * lmbda + M * omega_bar - kappa / q_star) / gamma_g
    return float(np.clip(raw, 0.0, 1.0))


def welfare_loss(N, lmbda, M, omega_bar, q_star, gamma_g, g_ne, g_so):
    """Eq. (10): ΔW — total annual welfare loss from decentralised underinvestment.
    Returns 0 when g_so <= g_ne (no gap; model constraint always holds in practice)."""
    if g_so <= g_ne:
        return 0.0
    delta_g = g_so - g_ne
    ext = ((N - 1) * lmbda + M * omega_bar) * q_star * delta_g
    cost = (gamma_g / 2.0) * q_star * (g_so ** 2 - g_ne ** 2)
    return N * (ext - cost)


def technical_debt(tau, q_star, N, P_bar):
    """Eq. (13): TD_total = τ · q* · N(N−1) · P̄  (in $M)"""
    return tau * q_star * N * (N - 1) * P_bar


def pigouvian_subsidy(N, lmbda, q_star):
    """Eq. (19): s_i = (N−1)λ · q* — per-domain reusability bonus"""
    return (N - 1) * lmbda * q_star


# ── Composite helpers ──────────────────────────────────────────────────────────

def compute_all(params):
    """Compute every model output from a parameter dict. Returns a result dict."""
    N         = params["N"]
    M         = params["M"]
    alpha     = params["alpha"]
    beta      = params["beta"]
    lmbda     = params["lmbda"]
    omega_bar = params["omega_bar"]
    gamma_g   = params["gamma_g"]
    kappa     = params["kappa"]
    q_star    = params["q_star"]
    tau       = params["tau"]
    P_bar     = params["P_bar"]

    g_ne    = equilibrium_generality(alpha, beta, kappa, q_star, gamma_g)
    g_so    = social_optimum(alpha, beta, N, lmbda, M, omega_bar, kappa, q_star, gamma_g)
    delta_g = max(0.0, g_so - g_ne)
    delta_W = welfare_loss(N, lmbda, M, omega_bar, q_star, gamma_g, g_ne, g_so)
    td      = technical_debt(tau, q_star, N, P_bar)
    subsidy = pigouvian_subsidy(N, lmbda, q_star)
    pct_of_so = (g_ne / g_so * 100) if g_so > 0 else 0.0

    return dict(
        g_ne=g_ne,
        g_so=g_so,
        delta_g=delta_g,
        delta_W=delta_W,
        td_total=td,
        subsidy=subsidy,
        pct_of_so=pct_of_so,
        in_trap=(g_ne == 0.0),
    )


def curves_vs_N(params, N_vals):
    """Return arrays of model outputs across a range of N values (all other params fixed)."""
    p = dict(params)
    g_ne_scalar = equilibrium_generality(
        p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"]
    )
    g_ne_curve = np.full_like(N_vals, g_ne_scalar, dtype=float)

    g_so_curve = np.array([
        social_optimum(p["alpha"], p["beta"], n, p["lmbda"], p["M"],
                       p["omega_bar"], p["kappa"], p["q_star"], p["gamma_g"])
        for n in N_vals
    ])

    delta_g_curve = np.maximum(0.0, g_so_curve - g_ne_curve)
    td_curve      = np.array([technical_debt(p["tau"], p["q_star"], n, p["P_bar"]) for n in N_vals])
    dw_curve      = np.array([
        welfare_loss(n, p["lmbda"], p["M"], p["omega_bar"],
                     p["q_star"], p["gamma_g"], g_ne_scalar, g_so)
        for n, g_so in zip(N_vals, g_so_curve)
    ])

    return dict(
        g_ne=g_ne_curve,
        g_so=g_so_curve,
        delta_g=delta_g_curve,
        td=td_curve,
        dw=dw_curve,
    )


# ── Inverse solver ─────────────────────────────────────────────────────────────

def escape_trap_solver(params, g_ne_target):
    """Find minimum parameter changes to achieve g_NE = g_ne_target.

    Solves (αβ − κ/q*) / γ_g = g_ne_target for each of five levers independently.
    Returns a dict of feasible interventions, or None if already at/above target.
    Each entry: {label, current, required, delta, direction, action, effort}
    """
    alpha   = params["alpha"]
    beta    = params["beta"]
    kappa   = params["kappa"]
    q_star  = params["q_star"]
    gamma_g = params["gamma_g"]

    if equilibrium_generality(alpha, beta, kappa, q_star, gamma_g) >= g_ne_target:
        return None

    needed_num   = gamma_g * g_ne_target          # target numerator: αβ − κ/q*
    current_num  = alpha * beta - (kappa / q_star if q_star > 0 else float("inf"))
    interventions = {}

    # Increase α
    if beta > 0 and q_star > 0:
        alpha_new = (needed_num + kappa / q_star) / beta
        if alpha < alpha_new <= PARAM_BOUNDS["alpha"][1]:
            interventions["alpha"] = dict(
                label="Increase domain analytics value α",
                current=alpha, required=round(alpha_new, 4),
                delta=round(alpha_new - alpha, 4), direction="increase",
                action="Build a data-driven culture: embed analytics in domain OKRs so teams rely on their own data products daily.",
                effort="High",
            )

    # Increase β
    if alpha > 0 and q_star > 0:
        beta_new = (needed_num + kappa / q_star) / alpha
        if beta < beta_new <= PARAM_BOUNDS["beta"][1]:
            interventions["beta"] = dict(
                label="Increase generality–quality synergy β",
                current=beta, required=round(beta_new, 4),
                delta=round(beta_new - beta, 4), direction="increase",
                action="Invest in platform tooling (schema validation, auto-docs, quality scoring) so standardising a product automatically improves its internal quality.",
                effort="Medium",
            )

    # Reduce κ
    kappa_new = q_star * (alpha * beta - needed_num)
    if PARAM_BOUNDS["kappa"][0] <= kappa_new < kappa:
        interventions["kappa"] = dict(
            label="Reduce fixed standardisation cost κ",
            current=kappa, required=round(kappa_new, 4),
            delta=round(kappa - kappa_new, 4), direction="reduce",
            action="Deploy a data catalog, contract templates, and a governance office to cut the fixed overhead of every standardisation effort.",
            effort="Medium",
        )

    # Increase q*
    denom = alpha * beta - needed_num
    if denom > 0:
        q_star_new = kappa / denom
        if q_star < q_star_new <= PARAM_BOUNDS["q_star"][1]:
            interventions["q_star"] = dict(
                label="Improve baseline data quality q*",
                current=q_star, required=round(q_star_new, 4),
                delta=round(q_star_new - q_star, 4), direction="increase",
                action="Enforce data contracts and completeness checks at ingestion so bronze-layer quality rises before generalisation begins.",
                effort="High",
            )

    # Reduce γ_g (only useful when numerator is already positive)
    if current_num > 0 and g_ne_target > 0:
        gamma_g_new = current_num / g_ne_target
        if PARAM_BOUNDS["gamma_g"][0] <= gamma_g_new < gamma_g:
            interventions["gamma_g"] = dict(
                label="Reduce generality cost γ_g",
                current=gamma_g, required=round(gamma_g_new, 4),
                delta=round(gamma_g - gamma_g_new, 4), direction="reduce",
                action="Build a schema registry, reusable contract templates, and a domain-facing SDK so producing a general product costs less engineering time per unit.",
                effort="Low–Medium",
            )

    return interventions if interventions else None


# ── Growth trajectory ──────────────────────────────────────────────────────────

def growth_trajectory(params, n_future, years=5):
    """Project TD and ΔW as N grows linearly from params['N'] to n_future over `years` years.

    Returns a dict of arrays: year, N, td, dw.
    """
    n_vals    = np.round(np.linspace(params["N"], n_future, years + 1)).astype(int)
    year_vals = np.arange(0, years + 1, dtype=float)

    g_ne_fixed = equilibrium_generality(
        params["alpha"], params["beta"], params["kappa"], params["q_star"], params["gamma_g"]
    )

    td_vals = np.array([technical_debt(params["tau"], params["q_star"], n, params["P_bar"])
                        for n in n_vals])

    dw_vals = np.array([
        welfare_loss(
            n, params["lmbda"], params["M"], params["omega_bar"],
            params["q_star"], params["gamma_g"],
            g_ne_fixed,
            social_optimum(params["alpha"], params["beta"], n, params["lmbda"], params["M"],
                           params["omega_bar"], params["kappa"], params["q_star"], params["gamma_g"]),
        ) for n in n_vals
    ])

    return dict(year=year_vals, N=n_vals, td=td_vals, dw=dw_vals)


# ── Presets ────────────────────────────────────────────────────────────────────

PAPER_BASELINE = dict(
    N=12, M=10, alpha=0.5, beta=0.15, lmbda=0.4, omega_bar=0.3,
    gamma_g=0.4, kappa=0.25, q_star=0.6, tau=0.05, P_bar=0.5,
)

FORTUNE_500 = dict(
    N=20, M=20, alpha=0.6, beta=0.2, lmbda=0.5, omega_bar=0.3,
    gamma_g=0.35, kappa=0.2, q_star=0.7, tau=0.08, P_bar=0.6,
)

INDUSTRY_PRESETS = {
    "🛒 Retail": dict(
        N=15, M=15, alpha=0.55, beta=0.15, lmbda=0.45, omega_bar=0.3,
        gamma_g=0.35, kappa=0.2, q_star=0.65, tau=0.07, P_bar=0.65,
    ),
    "🏦 FinServ": dict(
        N=10, M=8, alpha=0.5, beta=0.1, lmbda=0.5, omega_bar=0.25,
        gamma_g=0.6, kappa=0.4, q_star=0.75, tau=0.12, P_bar=0.5,
    ),
    "🏥 Healthcare": dict(
        N=8, M=5, alpha=0.4, beta=0.1, lmbda=0.3, omega_bar=0.2,
        gamma_g=0.65, kappa=0.45, q_star=0.5, tau=0.15, P_bar=0.45,
    ),
    "💻 SaaS / Tech": dict(
        N=6, M=18, alpha=0.75, beta=0.3, lmbda=0.45, omega_bar=0.4,
        gamma_g=0.2, kappa=0.1, q_star=0.8, tau=0.02, P_bar=0.4,
    ),
}

GOVERNANCE_COSTS = dict(
    centralized=2.0,
    hybrid=1.5,
)

PARAM_BOUNDS = {
    "N":         (2,    40),
    "M":         (0,    50),
    "alpha":     (0.1,  1.0),
    "beta":      (0.0,  0.5),
    "lmbda":     (0.0,  1.0),
    "omega_bar": (0.0,  1.0),
    "gamma_g":   (0.1,  1.0),
    "kappa":     (0.0,  0.6),
    "q_star":    (0.1,  1.0),
    "tau":       (0.0,  0.2),
    "P_bar":     (0.0,  1.0),
}
