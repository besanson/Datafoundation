import numpy as np


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


def compute_all(params):
    """Compute every model output from a parameter dict. Returns a result dict."""
    N        = params["N"]
    M        = params["M"]
    alpha    = params["alpha"]
    beta     = params["beta"]
    lmbda    = params["lmbda"]
    omega_bar = params["omega_bar"]
    gamma_g  = params["gamma_g"]
    kappa    = params["kappa"]
    q_star   = params["q_star"]
    tau      = params["tau"]
    P_bar    = params["P_bar"]

    g_ne   = equilibrium_generality(alpha, beta, kappa, q_star, gamma_g)
    g_so   = social_optimum(alpha, beta, N, lmbda, M, omega_bar, kappa, q_star, gamma_g)
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

    td_curve = np.array([technical_debt(p["tau"], p["q_star"], n, p["P_bar"]) for n in N_vals])

    dw_curve = np.array([
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


PAPER_BASELINE = dict(
    N=12, M=10, alpha=0.5, beta=0.15, lmbda=0.4, omega_bar=0.3,
    gamma_g=0.4, kappa=0.25, q_star=0.6, tau=0.05, P_bar=0.5,
)

FORTUNE_500 = dict(
    N=20, M=20, alpha=0.6, beta=0.2, lmbda=0.5, omega_bar=0.3,
    gamma_g=0.35, kappa=0.2, q_star=0.7, tau=0.08, P_bar=0.6,
)

GOVERNANCE_COSTS = dict(
    centralized=2.0,   # $M/year — central hydration team
    hybrid=1.5,        # $M/year — standards team + lighter incentives
)
