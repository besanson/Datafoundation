import pytest
import numpy as np
from model import (
    equilibrium_generality,
    social_optimum,
    welfare_loss,
    technical_debt,
    pigouvian_subsidy,
    compute_all,
    curves_vs_N,
    escape_trap_solver,
    growth_trajectory,
    PAPER_BASELINE,
    FORTUNE_500,
    INDUSTRY_PRESETS,
    PARAM_BOUNDS,
)


# ── Eq. (7): equilibrium generality ──────────────────────────────────────────

class TestEquilibriumGenerality:
    def test_paper_baseline_is_trap(self):
        p = PAPER_BASELINE
        # αβ = 0.5*0.15 = 0.075, κ/q* = 0.25/0.6 ≈ 0.417 → negative → 0
        g = equilibrium_generality(p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"])
        assert g == 0.0

    def test_positive_investment_when_synergy_high(self):
        g = equilibrium_generality(alpha=1.0, beta=1.0, kappa=0.0, q_star=1.0, gamma_g=1.0)
        assert g == pytest.approx(1.0)

    def test_zero_when_fixed_cost_dominates(self):
        g = equilibrium_generality(alpha=0.1, beta=0.1, kappa=1.0, q_star=0.5, gamma_g=0.5)
        assert g == 0.0

    def test_zero_on_bad_inputs(self):
        assert equilibrium_generality(0.5, 0.15, 0.25, 0.0, 0.4) == 0.0
        assert equilibrium_generality(0.5, 0.15, 0.25, 0.6, 0.0) == 0.0

    def test_increases_with_alpha(self):
        base = dict(beta=0.5, kappa=0.0, q_star=1.0, gamma_g=1.0)
        g_low  = equilibrium_generality(alpha=0.3, **base)
        g_high = equilibrium_generality(alpha=0.8, **base)
        assert g_high > g_low

    def test_decreases_with_gamma_g(self):
        base = dict(alpha=1.0, beta=1.0, kappa=0.0, q_star=1.0)
        g_cheap = equilibrium_generality(gamma_g=0.5, **base)
        g_costly = equilibrium_generality(gamma_g=2.0, **base)
        assert g_cheap > g_costly


# ── Proposition 1 / Eq. (8): social optimum ──────────────────────────────────

class TestSocialOptimum:
    def test_paper_baseline(self):
        p = PAPER_BASELINE
        g = social_optimum(p["alpha"], p["beta"], p["N"], p["lmbda"], p["M"],
                           p["omega_bar"], p["kappa"], p["q_star"], p["gamma_g"])
        # αβ=0.075, (N-1)λ=11*0.4=4.4, Mω̄=10*0.3=3, κ/q*≈0.417 → raw=7.058/0.4=17.6 → clipped to 1
        assert g == pytest.approx(1.0)

    def test_exceeds_equilibrium(self):
        p = PAPER_BASELINE
        g_ne = equilibrium_generality(p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"])
        g_so = social_optimum(p["alpha"], p["beta"], p["N"], p["lmbda"], p["M"],
                              p["omega_bar"], p["kappa"], p["q_star"], p["gamma_g"])
        assert g_so >= g_ne

    def test_clipped_to_one(self):
        g = social_optimum(alpha=1.0, beta=1.0, N=40, lmbda=1.0, M=50,
                           omega_bar=1.0, kappa=0.0, q_star=1.0, gamma_g=0.1)
        assert g == pytest.approx(1.0)

    def test_clipped_to_zero(self):
        g = social_optimum(alpha=0.0, beta=0.0, N=2, lmbda=0.0, M=0,
                           omega_bar=0.0, kappa=1.0, q_star=0.5, gamma_g=1.0)
        assert g == 0.0

    def test_increases_with_N(self):
        base = dict(alpha=0.5, beta=0.15, lmbda=0.4, M=10, omega_bar=0.3,
                    kappa=0.1, q_star=0.8, gamma_g=0.4)
        g_small = social_optimum(N=3,  **base)
        g_large = social_optimum(N=20, **base)
        assert g_large >= g_small

    def test_bad_inputs_return_zero(self):
        assert social_optimum(0.5, 0.15, 12, 0.4, 10, 0.3, 0.25, 0.0, 0.4) == 0.0


# ── Eq. (10): welfare loss ────────────────────────────────────────────────────

class TestWelfareLoss:
    def test_zero_when_no_gap(self):
        dw = welfare_loss(N=10, lmbda=0.4, M=5, omega_bar=0.3,
                          q_star=0.6, gamma_g=0.4, g_ne=0.5, g_so=0.5)
        assert dw == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_gap_exists(self):
        dw = welfare_loss(N=12, lmbda=0.4, M=10, omega_bar=0.3,
                          q_star=0.6, gamma_g=0.4, g_ne=0.0, g_so=1.0)
        assert dw > 0

    def test_scales_with_N(self):
        kwargs = dict(lmbda=0.4, M=10, omega_bar=0.3, q_star=0.6, gamma_g=0.4, g_ne=0.0, g_so=0.8)
        dw_small = welfare_loss(N=5,  **kwargs)
        dw_large = welfare_loss(N=20, **kwargs)
        assert dw_large > dw_small

    def test_g_so_less_than_g_ne_treated_as_zero_gap(self):
        # delta_g = max(0, g_so - g_ne) so this should behave as zero gap
        dw = welfare_loss(N=10, lmbda=0.4, M=5, omega_bar=0.3,
                          q_star=0.6, gamma_g=0.4, g_ne=0.8, g_so=0.5)
        assert dw == pytest.approx(0.0, abs=1e-9)


# ── Eq. (13): technical debt ──────────────────────────────────────────────────

class TestTechnicalDebt:
    def test_paper_baseline(self):
        p = PAPER_BASELINE
        # τ·q*·N(N-1)·P̄ = 0.05*0.6*12*11*0.5 = 1.98
        td = technical_debt(p["tau"], p["q_star"], p["N"], p["P_bar"])
        assert td == pytest.approx(1.98, rel=1e-6)

    def test_zero_with_no_cost(self):
        assert technical_debt(tau=0.0, q_star=0.6, N=12, P_bar=0.5) == 0.0

    def test_quadratic_in_N(self):
        td_6  = technical_debt(tau=0.05, q_star=0.6, N=6,  P_bar=0.5)
        td_12 = technical_debt(tau=0.05, q_star=0.6, N=12, P_bar=0.5)
        # N(N-1): 6*5=30, 12*11=132 → ratio ≈ 4.4
        assert td_12 / td_6 == pytest.approx(132 / 30, rel=1e-6)

    def test_single_domain_is_zero(self):
        assert technical_debt(tau=0.1, q_star=0.8, N=1, P_bar=0.9) == 0.0


# ── Eq. (19): Pigouvian subsidy ───────────────────────────────────────────────

class TestPigouvianSubsidy:
    def test_paper_baseline(self):
        p = PAPER_BASELINE
        # (N-1)λq* = 11*0.4*0.6 = 2.64
        s = pigouvian_subsidy(p["N"], p["lmbda"], p["q_star"])
        assert s == pytest.approx(2.64, rel=1e-6)

    def test_increases_with_N(self):
        s_small = pigouvian_subsidy(N=5,  lmbda=0.4, q_star=0.6)
        s_large = pigouvian_subsidy(N=20, lmbda=0.4, q_star=0.6)
        assert s_large > s_small

    def test_zero_lambda(self):
        assert pigouvian_subsidy(N=12, lmbda=0.0, q_star=0.6) == 0.0


# ── compute_all ───────────────────────────────────────────────────────────────

class TestComputeAll:
    def test_paper_baseline_keys(self):
        result = compute_all(PAPER_BASELINE)
        assert set(result.keys()) == {"g_ne", "g_so", "delta_g", "delta_W",
                                      "td_total", "subsidy", "pct_of_so", "in_trap"}

    def test_paper_baseline_in_trap(self):
        result = compute_all(PAPER_BASELINE)
        assert result["in_trap"] is True
        assert result["g_ne"] == 0.0

    def test_fortune_500(self):
        result = compute_all(FORTUNE_500)
        assert result["g_so"] > 0
        assert result["td_total"] > 0

    def test_delta_g_non_negative(self):
        result = compute_all(PAPER_BASELINE)
        assert result["delta_g"] >= 0

    def test_pct_of_so_zero_in_trap(self):
        result = compute_all(PAPER_BASELINE)
        assert result["pct_of_so"] == 0.0


# ── curves_vs_N ───────────────────────────────────────────────────────────────

class TestCurvesVsN:
    def test_output_shapes(self):
        N_vals = np.arange(2, 21, dtype=float)
        curves = curves_vs_N(PAPER_BASELINE, N_vals)
        for key in ("g_ne", "g_so", "delta_g", "td", "dw"):
            assert curves[key].shape == N_vals.shape

    def test_td_increases_with_N(self):
        N_vals = np.arange(2, 21, dtype=float)
        td = curves_vs_N(PAPER_BASELINE, N_vals)["td"]
        assert np.all(np.diff(td) >= 0)

    def test_g_so_increases_with_N(self):
        N_vals = np.arange(2, 10, dtype=float)
        g_so = curves_vs_N(PAPER_BASELINE, N_vals)["g_so"]
        assert np.all(np.diff(g_so) >= 0)

    def test_g_ne_constant_across_N(self):
        # g_NE does not depend on N
        N_vals = np.arange(2, 21, dtype=float)
        g_ne = curves_vs_N(PAPER_BASELINE, N_vals)["g_ne"]
        assert np.all(g_ne == g_ne[0])


# ── escape_trap_solver ────────────────────────────────────────────────────────

class TestEscapeTrapSolver:
    def test_returns_none_when_already_at_target(self):
        # Paper baseline is in the trap (g_NE=0), so target=0 means already there
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.0)
        assert result is None

    def test_returns_interventions_for_trap(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.1)
        assert result is not None
        assert len(result) >= 1

    def test_kappa_reduction_achieves_target(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.1)
        assert "kappa" in result
        lever = result["kappa"]
        # Verify: plugging required kappa back in gives g_NE >= target
        p = dict(PAPER_BASELINE, kappa=lever["required"])
        g = equilibrium_generality(p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"])
        assert g >= 0.1 - 1e-3

    def test_beta_increase_achieves_target(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.1)
        if "beta" in result:
            lever = result["beta"]
            p = dict(PAPER_BASELINE, beta=lever["required"])
            g = equilibrium_generality(p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"])
            assert g >= 0.1 - 1e-3

    def test_alpha_increase_achieves_target(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.1)
        if "alpha" in result:
            lever = result["alpha"]
            p = dict(PAPER_BASELINE, alpha=lever["required"])
            g = equilibrium_generality(p["alpha"], p["beta"], p["kappa"], p["q_star"], p["gamma_g"])
            assert g >= 0.1 - 1e-3

    def test_required_values_within_param_bounds(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.2)
        if result:
            for key, lever in result.items():
                if key in PARAM_BOUNDS:
                    lo, hi = PARAM_BOUNDS[key]
                    assert lo <= lever["required"] <= hi, f"{key} required={lever['required']} out of [{lo}, {hi}]"

    def test_returns_none_when_already_exceeds_target(self):
        # Build params where g_NE is already 0.5
        p = dict(PAPER_BASELINE, alpha=1.0, beta=0.5, kappa=0.0, gamma_g=1.0)
        result = escape_trap_solver(p, g_ne_target=0.3)
        assert result is None

    def test_all_interventions_have_required_keys(self):
        result = escape_trap_solver(PAPER_BASELINE, g_ne_target=0.15)
        if result:
            required_keys = {"label", "current", "required", "delta", "direction", "action", "effort"}
            for lever in result.values():
                assert required_keys.issubset(lever.keys())


# ── growth_trajectory ─────────────────────────────────────────────────────────

class TestGrowthTrajectory:
    def test_output_keys(self):
        traj = growth_trajectory(PAPER_BASELINE, n_future=20, years=5)
        assert set(traj.keys()) == {"year", "N", "td", "dw"}

    def test_output_length(self):
        traj = growth_trajectory(PAPER_BASELINE, n_future=20, years=5)
        assert len(traj["year"]) == 6   # years+1 points
        assert len(traj["N"])    == 6
        assert len(traj["td"])   == 6
        assert len(traj["dw"])   == 6

    def test_first_year_matches_compute_all(self):
        traj = growth_trajectory(PAPER_BASELINE, n_future=20, years=5)
        expected_td = technical_debt(
            PAPER_BASELINE["tau"], PAPER_BASELINE["q_star"],
            PAPER_BASELINE["N"], PAPER_BASELINE["P_bar"]
        )
        assert traj["td"][0] == pytest.approx(expected_td, rel=1e-5)

    def test_td_increases_as_N_grows(self):
        traj = growth_trajectory(PAPER_BASELINE, n_future=20, years=5)
        # TD should be non-decreasing as N grows
        assert np.all(np.diff(traj["td"]) >= 0)

    def test_year_axis_starts_at_zero(self):
        traj = growth_trajectory(PAPER_BASELINE, n_future=20, years=3)
        assert traj["year"][0] == 0.0
        assert traj["year"][-1] == 3.0

    def test_n_future_equals_current_gives_flat_trajectory(self):
        n = PAPER_BASELINE["N"]
        traj = growth_trajectory(PAPER_BASELINE, n_future=n, years=4)
        assert np.all(traj["N"] == n)
        assert np.all(np.diff(traj["td"]) == 0)

    def test_industry_presets_run_without_error(self):
        for name, preset in INDUSTRY_PRESETS.items():
            traj = growth_trajectory(preset, n_future=preset["N"] + 5, years=3)
            assert traj["td"].shape == (4,), f"Failed for {name}"
