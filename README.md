# Data Hydration Gap Model

Interactive decision-support tool for the game-theoretic model introduced in:

> **Besanson (2026) — The Data Hydration Gap: A Formal Model of Underinvestment in General-Purpose Data Products Under Decentralized Governance**
> [arXiv:2604.00218](https://arxiv.org/abs/2604.00218)

## What it does

The app lets you configure your organization's parameters and instantly see:

- **gⁿᵉ** — the generality level domains will *voluntarily* invest in (Nash equilibrium, Eq. 7)
- **gˢᵒ** — the generality level that maximizes organizational welfare (social optimum, Prop. 1)
- **ΔW** — annual welfare loss from decentralized underinvestment (Eq. 10)
- **TD** — accumulated technical debt across domain pairs (Eq. 13)
- **sᵢ** — the Pigouvian subsidy that closes the gap (Eq. 19)
- Sensitivity analysis showing which parameters drive welfare loss most
- Governance regime comparison: Pure Mesh vs. Centralized vs. Federated vs. Hybrid

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
pytest tests/
```

## Project structure

```
app.py          # Streamlit UI
model.py        # Pure-Python implementation of all model equations
tests/
  test_model.py # 32 unit tests covering every equation
requirements.txt
```

## Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Number of domains | N | Distinct business units owning data products |
| Cross-domain consumers | M | Teams consuming data from more than one domain |
| Domain analytics value | α | How much a domain relies on its own data |
| Generality–quality synergy | β | Internal quality gain from standardization |
| Cross-domain data value | λ | Share of analytical value requiring cross-domain joins |
| Avg. consumer weight | ω̄ | Importance of each domain to cross-domain consumers |
| Generality cost | γ_g | Cost per unit of generality added |
| Fixed standardization cost | κ | One-time overhead to begin standardizing |
| Baseline quality | q* | Quality of raw/bronze data before generalization |
| Integration cost per pair | τ | Cost to build one custom pipeline between two domains |
| Avg. prob. needing another domain | P̄ | Probability any domain needs data from another |
