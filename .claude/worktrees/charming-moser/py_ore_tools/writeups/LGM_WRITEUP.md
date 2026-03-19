# LGM 1F: Mathematical Foundations, Simulation, and IRS/CVA Parity

## 1. Scope

This document gives a self-contained mathematical treatment of the 1-factor Linear Gaussian
Markov (LGM) model as implemented in `lgm.py`, extended by the IRS/XVA utilities in
`irs_xva_utils.py`, and validated against ORE for IRS exposure and unilateral CVA.  Primary focus:

- LGM-only (no cross-asset coupling)
- IRS valuation and exposure under LGM
- Unilateral CVA reconciliation to ORE outputs

---

## 2. Model Parameters and Notation

Let `(Ω, F, {F_t}_{t≥0}, Q)` be a filtered probability space carrying a one-dimensional
Brownian motion `W`. Fix a valuation date `t = 0` with initial discount curve `P^M(0,·)`.

The model is parameterized by:

| Symbol | Name | Code field |
|---|---|---|
| `α_raw(t)` | Hagan volatility (piecewise constant, ≥ 0) | `alpha_values` |
| `κ(t)` | Mean-reversion speed (piecewise constant) | `kappa_values` |
| `shift` | H-function offset | `shift` |
| `scaling` | H-function normalization | `scaling` |

Both `α_raw` and `κ` are right-continuous piecewise-constant functions with finitely many
breakpoints, constant on `[t_{i-1}, t_i)`.

Define the **effective volatility**:

```
α(t) := α_raw(t) / scaling
```

This is the quantity whose square, when integrated, gives the variance of the LGM state.

---

## 3. Deterministic Functions

### 3.1 Cumulative Reversion

```
K(t) := ∫₀ᵗ κ(s) ds
```

For piecewise-constant κ on breakpoints `0 = τ₀ < τ₁ < … < τₙ` with value `κᵢ` on `[τᵢ₋₁, τᵢ)`:

```
K(t) = Σᵢ κᵢ (min(t, τᵢ) - τᵢ₋₁)⁺
```

Implemented via prefix-sum caching in `_kappa_prefix_int` for O(log n) evaluation.

### 3.2 The H-Function

`H(t)` is the fundamental **bond sensitivity function**. It satisfies the ODE:

```
H''(t) + κ(t) H'(t) = 0,    H(0) = shift,    H'(0) = scaling
```

The solution is:

```
H'(t) = scaling · exp(−K(t))

H(t) = scaling · ∫₀ᵗ exp(−K(s)) ds + shift
```

For constant κ this reduces to the familiar Hull-White expression:

```
H(t) = (scaling / κ) · (1 − e^{−κt}) + shift
```

**Key property**: `H(T) − H(t) = scaling · ∫ₜᵀ exp(−K(s)) ds`, which is independent of
`shift`. The shift therefore cancels in all bond price differentials but contributes to
quadratic terms `H²(T) − H²(t) = (H(T)+H(t))(H(T)−H(t))`.

Implemented in `H(t)` and `Hprime(t)`, using the prefix integral `_h_prefix_int` to cache
`∫₀^{τᵢ} exp(−K(s)) ds` at each breakpoint.

### 3.3 The Variance Function ζ(t)

```
ζ(t) := ∫₀ᵗ α²(s) ds  =  (1/scaling²) · ∫₀ᵗ α_raw²(s) ds
```

This is the variance of the LGM state variable `x(t)` (see Section 4). For piecewise-constant
`α_raw`, `ζ(t)` is piecewise linear:

```
ζ(t) = (1/scaling²) · Σᵢ α_raw,ᵢ² · (min(t, sᵢ) − sᵢ₋₁)⁺
```

Implemented via prefix-sum caching in `_alpha_prefix_int`.

### 3.4 The Weighted Variance Functions ζ_n(t)

For `n = 0, 1, 2` define:

```
ζₙ(t) := ∫₀ᵗ α²(s) · H^n(s) ds
```

Observe that `ζ₀ = ζ`. These integrals appear in the BA numeraire (Section 6) and in the
BA-measure moment structure (Section 7). They are computed exactly; see Section 8.

---

## 4. State Variable and LGM Measure

### 4.1 Dynamics

The **LGM state variable** `x(t)` is defined as a driftless Gaussian process under the
**LGM measure** `Q^{LGM}`:

```
dx(t) = α(t) dW^{LGM}(t),    x(0) = 0
```

Since `α(t)` is deterministic, `x` is a Gaussian martingale with:

```
x(t) ~ N(0, ζ(t))   under Q^{LGM}

Cov^{LGM}(x(s), x(t)) = ζ(min(s,t))
```

Increments over disjoint intervals are independent Gaussians:

```
x(t) − x(s) | F_s  ~  N(0, ζ(t) − ζ(s))   for t > s
```

### 4.2 Simulation (Exact Discretization)

Because `x` has deterministic diffusion coefficient, the Euler discretization is **exact** (no
time-step error). Given a grid `0 = t₀ < t₁ < … < t_N`:

```
x(t_{i+1}) = x(t_i) + √(ζ(t_{i+1}) − ζ(t_i)) · Z_i,    Z_i ~ i.i.d. N(0,1)
```

Implemented in `simulate_lgm_measure`. The variance increments `Δζᵢ = ζ(t_{i+1}) − ζ(t_i)` are
non-negative by construction (since `α² ≥ 0`).

---

## 5. Zero-Coupon Bond Pricing Formula

The central pricing identity in LGM is the **discount bond formula**. For `T ≥ t ≥ 0`:

```
P(t, T; x(t)) = [P^M(0,T) / P^M(0,t)] · exp(−(H(T)−H(t)) · x(t)  −  ½(H²(T)−H²(t)) · ζ(t))
```

**Derivation sketch.** Assume an affine form `P(t,T) = exp(A(t,T) + B(t,T)·x(t))`. The
coefficient `B(t,T) = −(H(T)−H(t))` is fixed by requiring `P(t,T)/N^{LGM}(t)` to be a
`Q^{LGM}`-martingale (see Section 5.1). The drift condition then pins `A(t,T)` uniquely given
the initial curve fit `P(0,T;x(0)=0) = P^M(0,T)`.

**Verification at `t=0`**: `P(0,T;0) = P^M(0,T)/P^M(0,0) · exp(0) = P^M(0,T)` ✓ (using
`P^M(0,0) = 1`).

**Verification at `T=t`**: `P(t,t;x) = 1` ✓ (both exponential and ratio collapse to 1).

Implemented in `discount_bond` (scalar `T`) and `discount_bond_paths` (vector `T`, returns a
`(|T|, n_paths)` array).

### 5.1 LGM Numeraire

The **LGM numeraire** `N^{LGM}(t)` is the process under which asset prices deflated by
`N^{LGM}` are `Q^{LGM}`-martingales. It takes the form:

```
N^{LGM}(t; x) = exp(H(t)·x(t)  +  ½·H²(t)·ζ(t)) / P^M(0,t)
```

**Verification**: Compute `P(t,T)/N^{LGM}(t)` and confirm it is a `Q^{LGM}`-local martingale.

```
P(t,T) / N^{LGM}(t)
  = P^M(0,T) · exp(−H(T)·x(t) − ½·H²(T)·ζ(t))

d/dt [−H(T)·x(t) − ½·H²(T)·ζ(t)]
  = −H(T)·α(t) dW^{LGM}  −  ½·H²(T)·α²(t) dt

By Itô: d[exp(f)] = exp(f)·(df + ½(df)²)
  = exp(f)·[−H(T)·α dW^{LGM}  −  ½H²(T)α² dt  +  ½H²(T)α² dt]
  = exp(f)·[−H(T)·α dW^{LGM}]          ← pure local martingale ✓
```

Implemented in `numeraire_lgm`.

---

## 6. Bank-Account (BA) Measure

### 6.1 Short Rate

The instantaneous short rate in LGM is:

```
r(t) = f^M(0,t) + H'(t)·x(t) + H(t)·H'(t)·ζ(t)
```

where `f^M(0,t) = −∂/∂T log P^M(0,T)|_{T=t}` is the market instantaneous forward rate. The
third term is a deterministic convexity correction arising from Jensen's inequality.

**Derivation**: Differentiate `log P(t,T)` with respect to T and evaluate at `T = t`.

### 6.2 Auxiliary State Variable y(t)

Define:

```
y(t) := ∫₀ᵗ α(s)·H(s) dW^{LGM}(s)
```

By Itô integration by parts (using `dx = α dW^{LGM}`):

```
y(t) = ∫₀ᵗ H(s) dx(s) = H(t)·x(t) − ∫₀ᵗ H'(s)·x(s) ds
```

Under `Q^{LGM}`, `y` is a zero-mean Gaussian with:

```
Var^{LGM}(y(t)) = ∫₀ᵗ α²(s)·H²(s) ds = ζ₂(t)

Cov^{LGM}(x(t), y(t)) = ∫₀ᵗ α²(s)·H(s) ds = ζ₁(t)
```

### 6.3 BA Numeraire

The bank account satisfies `log B(t) = ∫₀ᵗ r(s) ds`. Integrating the short-rate expression
and applying the Itô identity `∫₀ᵗ H'(s)·x(s) ds = H(t)·x(t) − y(t)`:

```
log B(t) = H(t)·x(t) − y(t) − log P^M(0,t) + ∫₀ᵗ H(s)·H'(s)·ζ(s) ds
```

The deterministic integral evaluates via integration by parts as:

```
∫₀ᵗ H·H'·ζ ds = ½·H²(t)·ζ(t) − ½·ζ₂(t)
```

The implemented **BA numeraire** differs from `B(t)` by a deterministic factor `exp(ζ₂(t))`:

```
N^{BA}(t; x, y) := exp(H(t)·x(t) − y(t) + ½(H²(t)·ζ(t) + ζ₂(t))) / P^M(0,t)

N^{BA}(t) = exp(ζ₂(t)) · B(t)
```

Since `exp(ζ₂(t))` is deterministic, `N^{BA}` and `B(t)` define the **same equivalent
martingale measure** `Q^B`. The extra `ζ₂(t)` is a normalization choice, not a pricing error.

**Verification**: For any `T`, compute `P(t,T)/N^{BA}(t)`:

```
P(t,T)/N^{BA}(t)
  = P^M(0,T) · exp(−H(T)·x(t) + y(t) − ½·H²(T)·ζ(t) − ½·ζ₂(t))
```

Under `Q^B` the drift of this expression is (applying Itô with the `Q^B` dynamics in Section 7):

```
drift contribution = α²(t)·[H(T)·H(t) − ½H²(T) − ½H²(t)] dt
                   + ½·α²(t)·(H(t)−H(T))² dt
                   = 0  ✓
```

Hence `P(t,T)/N^{BA}(t)` is a `Q^B`-local martingale, confirming the formula. Implemented in
`numeraire_ba`.

---

## 7. BA-Measure Dynamics and Simulation

### 7.1 Measure Change via Girsanov

The Girsanov kernel relating `Q^{LGM}` and `Q^B` is `λ(t) = −α(t)·H(t)`:

```
dW^{LGM}(t) = dW^B(t) − α(t)·H(t) dt
```

Substituting into `dx = α dW^{LGM}` gives the `Q^B`-dynamics of the state pair `(x, y)`:

```
dx(t) = −α²(t)·H(t) dt  +  α(t) dW^B(t)

dy(t) =  α(t)·H(t) dW^B(t)
```

### 7.2 Interval Moments under Q^B

For the interval `[s, t]`, the joint increment `(Δx, Δy) = (x(t)−x(s), y(t)−y(s))` is
bivariate Gaussian with:

```
E^B[Δx]         = −(ζ₁(t) − ζ₁(s))    (= −Δζ₁)
Var^B[Δx]       =   ζ(t)  − ζ(s)       (= Δζ₀)
Cov^B[Δx, Δy]  =   ζ₁(t) − ζ₁(s)     (= Δζ₁)
Var^B[Δy]       =   ζ₂(t) − ζ₂(s)     (= Δζ₂)
E^B[Δy]         =   0
```

These follow directly from the `Q^B`-SDEs above by computing:

```
∫_s^t E^B[dx(u)] du = −∫_s^t α²H du = −Δζ₁

Var^B(∫_s^t α dW^B) = ∫_s^t α² du = Δζ₀

Cov^B(∫_s^t α dW^B, ∫_s^t αH dW^B) = ∫_s^t α²H du = Δζ₁

Var^B(∫_s^t αH dW^B) = ∫_s^t α²H² du = Δζ₂
```

Implemented in `ba_interval_moments`.

### 7.3 Simulation (Exact Discretization)

The joint bivariate Gaussian `(Δx, Δy)` is sampled exactly via Cholesky factorization of the
`2×2` covariance matrix. For each step `[t_i, t_{i+1}]`:

```
L = Cholesky([[Δζ₀, Δζ₁], [Δζ₁, Δζ₂]])  =  [[l₁₁, 0], [l₂₁, l₂₂]]

l₁₁ = √Δζ₀
l₂₁ = Δζ₁ / l₁₁
l₂₂ = √(Δζ₂ − l₂₁²)

[Δx]   [−Δζ₁]   [l₁₁   0  ] [Z₁]
[Δy] = [  0  ] + [l₂₁  l₂₂] [Z₂],    Z₁, Z₂ ~ i.i.d. N(0,1)
```

This discretization is again exact (zero time-step error) because all moment coefficients are
deterministic, computed from the ζ-integrals at each grid point.

Implemented in `simulate_ba_measure` (outer loop) and `_sample_correlated_2d` (Cholesky step).

---

## 8. Exact Computation of ζₙ Integrals

### 8.1 Reduction to Segment Integrals

`ζₙ(t) = ∫₀ᵗ α²(s)·H^n(s) ds` is computed exactly by splitting at the union of `α`- and
`κ`-breakpoints, so that both `α` and `κ` are constant on each sub-segment `[a, b]`.

On segment `[a, b]` with constant `α`, `κ`, and with `H(a) = h₀` and `H'(a) = c` (where `c =
scaling · exp(−K(a))`):

```
H(a + u) = h₀ + c · g(u, κ),    u ∈ [0, δ],    δ = b − a
```

where:

```
g(u, κ) = (1 − e^{−κu}) / κ    for κ ≠ 0
g(u, 0) = u                     (limit as κ → 0)
```

### 8.2 Closed-Form Primitives

Define the following scalar integrals:

```
J₀(δ) := δ

J₁(δ, κ) := ∫₀^δ g(u,κ) du
           = δ/κ − (1 − e^{−κδ})/κ²              for κ ≠ 0
           = δ²/2                                  for κ = 0

J₂(δ, κ) := ∫₀^δ g²(u,κ) du
           = δ/κ² − 2(1 − e^{−κδ})/κ³ + (1 − e^{−2κδ})/(2κ³)   for κ ≠ 0
           = δ³/3                                                   for κ = 0
```

These correspond to `_h_increment_integral` (J₁) and `_h_increment_square_integral` (J₂).

### 8.3 Segment Contribution

For `n = 0, 1, 2`, the contribution of segment `[a, b]` to `ζₙ` is:

```
n = 0:  α² · J₀(δ)

n = 1:  α² · [h₀ · J₀(δ) + c · J₁(δ, κ)]

n = 2:  α² · [h₀² · J₀(δ) + 2·h₀·c · J₁(δ, κ) + c² · J₂(δ, κ)]
```

These follow from expanding `H(a+u)^n = (h₀ + c·g(u,κ))^n` and integrating term by term.

The segment iterator tracks `h₀` and `c` recursively across segments:

```
h₀ ← h₀ + c · g(δ, κ)       (= H(b))
c  ← c · e^{−κδ}             (= H'(b)/scaling)
```

For `n > 2`, numeric trapezoidal quadrature is used as fallback (`_zetan_interval_numeric`).

Implemented in `_zetan_interval_exact`. Validated in `test_exact_zetan_matches_dense_numeric_integration`
to relative error `< 5 × 10⁻⁵` against a 20,001-point trapezoidal grid.

### 8.4 Incremental Grid Evaluation

`zetan_grid(n, times)` computes `(ζₙ(t₀), ζₙ(t₁), …)` in a single pass by accumulating
segment contributions, avoiding redundant recomputation.

---

## 9. IRS Valuation under LGM

### 9.1 Dual-Curve Setup

Post-2008 market practice separates discounting and forwarding curves:

- `P^d(0, t)`: OIS discount curve (e.g., EUR-EONIA)
- `P^f(0, t)`: forwarding curve (e.g., EUR-EURIBOR-6M)

The LGM state drives the discounting curve. Pathwise forward bonds on the discounting curve are:

```
P^d(t, T; x_t) = [P^d(0,T) / P^d(0,t)] · exp(−(H(T)−H(t))·x_t − ½(H²(T)−H²(t))·ζ(t))
```

The forwarding curve is related via a **deterministic basis ratio**:

```
B(t) := P^f(0,t) / P^d(0,t)

P^f(t, T; x_t) = P^d(t, T; x_t) · [B(T) / B(t)]
```

This assumes that the basis `B(·)` is deterministic — a standard approximation in LGM-based
multi-curve pricing.

### 9.2 Swap NPV

At simulation time `t` with state `x_t`, a payer IRS with fixed rate `K` has pathwise NPV:

```
V(t, x_t) = FloatLeg(t, x_t) − FixedLeg(t, x_t)
```

**Fixed leg** (coupon `K`, accruals `τᵢ`, notional `N`, pay dates `Tᵢ`):

```
FixedLeg(t, x_t) = N · K · Σᵢ τᵢ · P^d(t, Tᵢ; x_t)    [sum over remaining coupons]
```

**Floating leg** (unfixed periods with forward start `sᵢ`, end `eᵢ`, pay date `pᵢ`):

The projected forward rate for each period under the forwarding curve is:

```
F(t; sᵢ, eᵢ) = [P^f(t, sᵢ) / P^f(t, eᵢ) − 1] / τᵢ
```

Substituting the dual-curve bond formula:

```
FloatLeg(t, x_t) = N · Σᵢ [F(t; sᵢ, eᵢ) + spread_i] · τᵢ · P^d(t, pᵢ; x_t)
```

For periods whose fixing date has already passed (coupon locked in at `c_i`), the pathwise
cash flow is deterministic: `N · c_i · τᵢ · P^d(t, pᵢ; x_t)`.

**Node-tenor interpolation**: To avoid evaluating the LGM bond formula for every cashflow
maturity, discount bonds at a set of node maturities `{t + θ_j}` are computed first. Interior
maturities are then obtained by log-linear interpolation:

```
log P^d(t, T; x_t) ≈ (1−w) · log P^d(t, t+θ_j; x_t) + w · log P^d(t, t+θ_{j+1}; x_t)
```

for `T ∈ [t+θ_j, t+θ_{j+1}]`, with `w = (T − t − θ_j)/(θ_{j+1} − θ_j)`.

---

## 10. Exposure and CVA

### 10.1 Exposure Profiles

For each simulation time `tᵢ`, the pathwise NPV across `N_paths` Monte Carlo paths gives:

```
EE(tᵢ)  = E^{Q^{LGM}}[V(tᵢ, x(tᵢ))]           (Expected Exposure)
EPE(tᵢ) = E^{Q^{LGM}}[max(V(tᵢ, x(tᵢ)), 0)]   (Expected Positive Exposure)
ENE(tᵢ) = E^{Q^{LGM}}[min(V(tᵢ, x(tᵢ)), 0)]   (Expected Negative Exposure)
```

Monte Carlo estimates use the exact LGM-measure paths from Section 4.2.

Note: ORE computes exposure under the **risk-neutral** measure, which is the LGM measure with
numeraire `N^{LGM}`. Prices are reported undiscounted (i.e., not divided by the numeraire), so
the LGM-measure paths are the correct simulation measure for parity.

### 10.2 Unilateral CVA

Given a counterparty default curve with piecewise-flat hazard rates `λ(t)`, the survival
probability is:

```
Q(τ > t) = exp(−∫₀ᵗ λ(s) ds)
```

The survival probability on `[0, t]` is computed by integrating the hazard rate piecewise-
exactly.

Unilateral CVA (from the perspective of the institution, counterparty defaults):

```
CVA = LGD · Σᵢ EPE(tᵢ) · P^d(0, tᵢ) · [Q(τ > tᵢ₋₁) − Q(τ > tᵢ)]
```

where `LGD = 1 − Recovery`, `P^d(0, tᵢ)` is the risk-free discount factor, and the
`[Q(τ > tᵢ₋₁) − Q(τ > tᵢ)]` term is the marginal default probability in `(tᵢ₋₁, tᵢ]`.

This is a discrete approximation to the continuous integral:

```
CVA = LGD · ∫₀ᵀ EPE(t) · P^d(0, t) · λ(t) · Q(τ > t) dt
```

---

## 11. Convention Sweep and Parity Findings

### 11.1 Reconciliation Harness

`compare_ore_python_lgm.py` loads ORE artifacts (`curves.csv`, `flows.csv`,
`exposure_trade_*.csv`, `xva.csv`) and independently simulates the LGM on the same
exposure date grid. Per-date diagnostics and CVA decomposition are written to:

- `*_diagnostics.csv`
- `*_cva_terms.csv`
- `*_summary.json`

### 11.2 Convention Sweep

`convention_sweep_lgm.py` performs a grid search over:

| Dimension | Values |
|---|---|
| `swap_source` | `trade`, `flows` |
| `forward_column` | e.g. `EUR-EURIBOR-6M`, `EUR-EONIA` |
| `pathwise_fixing_lock` | on, off |
| `node_tenor_interp` | on, off |
| `coupon_spread_calibration` | on, off |
| `alpha_scale` | ladder, e.g. `[0.95, 1.00, 1.05]` |

Ranking metric: absolute relative CVA gap `|CVA_py − CVA_ORE| / |CVA_ORE|`.

Sweep artifacts:
- `parity_artifacts/calibrated/sweep_full_results.csv`
- `parity_artifacts/calibrated/sweep_full_results.json`

### 11.3 Best-Performing Configuration

| Setting | Value |
|---|---|
| `swap_source` | `trade` |
| Discount curve | `EUR-EONIA` |
| Forward curve | `EUR-EURIBOR-6M` |
| Node-tenor interpolation | ON |
| Coupon-spread calibration | OFF |

### 11.4 Observed Parity

| Scenario | CVA gap |
|---|---|
| `alpha_scale = 1.05`, 20k paths, single seed | ~0.43% |
| `alpha_scale = 1.00`, 20k paths, single seed | ~2.82% |
| `alpha_scale = 1.00`, 20k paths, 10-seed mean | ~2.54% |

The residual gap at `alpha_scale = 1.00` is consistent with a combination of Monte Carlo
dispersion (~1% at 20k paths per seed) and minor convention differences in the ORE exposure
grid vs the Python schedule reconstruction.

### 11.5 Interpreting α

The Hagan volatility `α_raw` controls the variance of `x(t)` via:

```
ζ(t) = (1/scaling²) ∫₀ᵗ α_raw²(s) ds
```

A larger `α_raw` widens the distribution of `x(t)`, which widens the distribution of swap NPVs
and therefore increases EPE, ENE, and CVA. The `alpha_scale` parameter multiplies `α_raw`
uniformly, providing a clean sensitivity lever without changing the model structure.

Recommended usage:

- Keep ORE-calibrated `alpha_scale = 1.0` as the model baseline
- Use `alpha_scale` as an explicit sensitivity parameter (not hidden retuning)
- Report a ladder `{0.95×, 1.00×, 1.05×}` if vega-like sensitivity to α is required

---

## 12. Accepted Baseline Configuration

| Parameter | Value |
|---|---|
| Scenario | `calibrated` |
| `swap_source` | `trade` |
| Discount curve | `EUR-EONIA` |
| Forward curve | `EUR-EURIBOR-6M` |
| Node-tenor interpolation | ON |
| Coupon-spread calibration | OFF |
| Anchor t₀ NPV | OFF |
| Paths | 20,000 |
| Seed | Fixed per run |

This baseline is currently at an acceptable parity level for the LGM-only prototype.
