# Physically Anchored Adaptive PUEA Detection — Methodology

This document describes the experimental framework added on top of the
original PUEA pipeline. It does **not** replace that pipeline: the original
dataset, `models/xgboost.pkl` and stages 1–5 are unchanged. The only change
to the stage-6 demo is an added "Adaptive PUEA Detection" page. The demo
also now skips missing git-ignored models instead of refusing to start.

**Research question.** Can a PUEA detector tell legitimate drift in the
primary user's (PU's) signal apart from adversarial poisoning, while it
adapts its model of legitimate PU behaviour?

**Mechanism under test.** An observation may update the adaptive PU profile
only if it is (a) *PU-like* according to the classifier stage **and**
(b) *physically consistent* with the known PU location. Consistency means the
spatial RSS pattern across the secondary users (SUs) could have come from a
transmitter at the PU's location.

- Naive adaptation: PU-like → update.
- Proposed: PU-like **and** physically consistent → update.
- PU-like but physically inconsistent → reject the update and flag PUEA.

---

## 1. Why the original dataset could not be used

The original CSV has one aggregated row per transmission, with the columns
`RSS_dBm, SNR_dB, Transmission_Power_dBm, X/Y_Coordinate_km,
Channel_Occupancy_Time_s, Frequency_Hz, RSS_Deviation, SNR_Deviation,
Distance_Mismatch_km, SINR_dB`.

- **Missing data.** There is no SU identity or position, no per-SU sensing
  vector, no known PU location, and no time or slot index. A multi-SU
  physical-consistency check and a drift or poisoning timeline therefore
  cannot be computed from it.
- **Unobservable columns.** `Transmission_Power_dBm` is not observable at
  the fusion center for an unknown transmitter. `Distance_Mismatch_km` is
  precomputed and already encodes "the transmitter is in the wrong place".

Everything the new experiments need is simulated separately under
`data/simulation/`. The original data files are never read or written by the
new code.

Because the original XGBoost expects those 11 columns, it cannot be applied
to simulated multi-SU streams. The *Baseline-XGBoost* of the new experiments
uses the **same learner and hyperparameters** (copied from
`src/train_ensemble.py`). It is trained on simulated fusion-center-observable
features (§3).

## 2. Simulator (`src/simulation/`)

| Module | Contents |
|---|---|
| `config.py` | Loads `configs/experiments.json`, applies regime overrides, fingerprints the simulation settings |
| `crn_simulator.py` | Geometry, propagation, per-slot measurement, `StreamData` container |
| `scenarios.py` | Drift schedules, attacker policies, the slot-by-slot `ScenarioStream` driver |

- **Propagation** (Chhetry & Marchang, 2021):
  `Pr = Pt · r^(-α) · exp(a·β)`, with `a = ln(10)/10` and `β ~ N(0, σ²)` in dB.
- **Measurement.** Each SU measures energy = Pr + noise, and keeps an
  estimate of its own noise floor.
- **Geometry.** 40 SUs uniform in a 100 × 100 m area, fixed per seed. The PU
  is at (15, 85).
- **Transmitter per slot.** Exactly one transmitter per sensing slot: the PU,
  or the attacker if it is active in that slot (PUEA is launched while the PU
  is idle).
- **Randomness.** All random draws for a stream are made up front. Every
  detector in a closed-loop run therefore sees identical PU observations,
  which makes the comparison paired.

**Regimes.** Each regime re-runs every scenario with the listed overrides.

- `default`: 40 SUs, σ = 4 dB. The physical anchor is strong here.
- `weak_anchor`: 20 SUs, σ = 8 dB. The anchor's spatial resolution is poor
  here.

### Scenarios

All scenarios are configured in `configs/experiments.json → scenarios`.

| Scenario | What changes |
|---|---|
| `stable` | Nothing |
| `power_drift` | PU power +6 dB (ramp, slots 400–1400) |
| `shadowing_drift` | σ 4 → 5.5 dB |
| `noise_drift` | Noise floor +4 dB |
| `combined_drift` | +4 dB power, +1 dB σ, +3 dB noise |
| `basic_puea` | Attacker at (75, 25), 22 dBm (not matched), 30% of slots after slot 500 |
| `power_matching_puea` | Same position; power chosen so the mean RSS over the SUs equals the PU's |
| `gradual_puea` | Open-loop evasion. PU-likeness stages 0.2 → 0.95 move the attacker toward a point 10 m from the PU and its power toward the matched power |
| `stealthy_puea` | Attacker 20 m from the PU re-matches the PU's drifting power every slot; 15% of slots |
| `poisoning` | **Closed loop.** Starts as a PU mimic 12 m from the PU and moves toward its target (the basic attacker), advancing when not flagged and retreating when flagged. Then it attacks at the target |
| `near_pu_poisoning` | Same, but the path runs from 3 m to 8 m from the PU: **inside** the anchor's resolution limit |

**Attacker knowledge.** The power-matching, stealthy and poisoning attackers
are strong: they know the SU positions and the path-loss exponent. The
poisoning attacker's feedback is whether the SUs vacated the channel, which
is observable in practice.

## 3. What the detector may use at slot t

The detector may use only:

- The per-SU RSS vector and per-SU noise-floor estimates for slots 0..t.
- The static SU positions and the known PU location (for example, from a
  PU registry).
- The trained classifiers, and the calibration outputs derived from the
  training and calibration streams.

The detector never sees:

- Labels.
- Transmitter identity, position or power.
- The path-loss exponent, shadowing or noise ground truth.
- Future slots.
- The oracle PU observations.

`StreamData.observable()` is the only accessor the detectors are given. Three
evaluation-only quantities are computed outside the detector by
`run_adaptive_experiment.py`: adaptation error (from oracle PU observations),
profile contamination (from labels) and the metrics. They are never fed back
into the detector.

**Features** (`src/fc_features.py`): mean, variance, median, 75th and 25th
percentile of the RSS vector (the five statistics of Chhetry & Marchang),
plus mean SNR. These are permutation-invariant and discard *which* SU saw
*what*.

## 4. Physical consistency (`src/physical_consistency.py`)

**Signal estimate.** For each SU with measured SNR ≥ 3 dB:
`s_i = 10·log10(E_i − N̂_i)`.

**Model at a candidate location x.** `s_i = P − 10·α̂·log10 d_i(x) + e_i`.

- P is fitted per slot, so legitimate power drift is absorbed.
- α̂ is fixed from calibration, so a displaced transmitter cannot hide by
  bending α.

**Statistic (default `glrt`).** `Λ = n · ln(SSR(x_PU) / min_x SSR(x))`, with
the minimum taken over a 2.5 m grid plus x_PU itself.

- This is the generalised likelihood ratio for "source at the PU" against
  "source anywhere".
- Its null distribution depends only weakly on σ. The 99th percentile moves
  from about 9.8 to about 8.8 as σ goes from 4 to 8 dB for 40 SUs. So
  legitimate shadowing drift does not by itself raise the false-rejection
  rate.
- The alternative `chi2` statistic (SSR/σ̂²/(n−1)) grows with σ² and is kept
  for comparison.

**Outputs.**

- `physical_residual`: Λ.
- `physical_consistency_score`: the empirical survival probability of Λ
  under the PU's calibration distribution. 1 means typical of the PU. It is
  **not** a probability that the transmitter is the PU.
- `physically_consistent`: Λ ≤ threshold.
- The best-fit source location.

**Unverifiable slots.** If fewer than 8 SUs are above the SNR floor, the slot
is unverifiable. It never raises a physical alarm and never passes the
update gate (fail closed).

## 5. Threshold calibration (no test data)

All thresholds come from a PU-only `calibration` stream (1,000 slots) with
its own random stream. No test data is used.

| Quantity | Procedure |
|---|---|
| α̂, σ̂ | Pooled least squares with a free power intercept per slot, on the first half |
| Physical threshold | Empirical `1 − 0.01` quantile of Λ on the held-out second half (quantile method `higher`). The χ² quantile is recorded as `theoretical_threshold` for reference only |
| Profile | MinCovDet mean and covariance on the first half. Mahalanobis threshold = empirical `1 − 0.01` quantile of d² on the second half |
| XGBoost | Decision threshold 0.5, the standard classifier default and not tuned. Trained on a separate `xgb_train` stream: stable PU slots plus generic attackers at random positions ≥ 10 m from the PU, with power = matched ± 10 dB |

The saved thresholds are in
`reports/adaptive/calibration/<regime>_seed_<k>.json`.

## 6. Adaptive profile and systems

**Profile** (`src/adaptive_profile.py`). A Gaussian (μ, Σ) over the six
features, updated by EWMA with λ = 0.01. The update gate is
`update_allowed(pu_like, physically_consistent, require_physical)`.

**Systems** (`src/adaptive_systems.py`). "PU-like" means NOT (XGBoost flags
AND the profile flags).

| System | Ablation | Alarm | Profile update |
|---|---|---|---|
| Baseline-XGBoost | A | xgb | — |
| Naive-Adaptive | B | xgb ∧ prof_t | if PU-like |
| XGB+Physical | C | (xgb ∧ prof_0) ∨ ¬phys | — (frozen) |
| Physically-Anchored-Adaptive | D | (xgb ∧ prof_t) ∨ ¬phys | if PU-like ∧ phys |
| XGB+StaticProfile | ref | xgb ∧ prof_0 | — |
| Anchored-Update-Only | ref | xgb ∧ prof_t | if PU-like ∧ phys |
| Physical-Only | ref | ¬phys | — |
| Spatial-XGBoost | ref | XGBoost on the raw per-SU RSS vector | — |

Why the rules look like this:

- **Why "xgb ∧ prof".** A static classifier cannot learn legitimate drift it
  never saw. Letting the adaptive profile veto its alarms is the only way
  adaptation can reduce drift false alarms. It is also exactly the opening a
  poisoner exploits.
- **Anchored-Update-Only.** It separates the anchor's role as an update gate
  from its role as an extra detector.
- **Physical-Only.** It tests whether physics alone explains every result.
- **Spatial-XGBoost.** It tests whether a supervised classifier given SU
  identity learns location by itself.

**ROC-AUC.** Each system's continuous score is built so that its sign matches
the thresholded decision:

- logit for XGBoost;
- `ln(d²/τ)` for the profile;
- `ln((Λ+1)/(τ+1))` for physics;
- min() for AND, max() for OR.

AUCs over these composite scores depend on how the components are scaled
against each other. Treat them as secondary to the operating-point metrics.

## 7. Metrics (`src/evaluate_adaptive.py`)

**Standard metrics:** accuracy, precision, recall, F1, FPR, FNR and ROC-AUC,
per stream. These are then averaged over 5 seeds, with the standard deviation
reported.

**Research-specific metrics:**

| Metric | Definition |
|---|---|
| Adaptation error | Mahalanobis distance, in the initial-profile metric, between the profile mean and a trailing-window mean of the oracle PU features |
| Poisoning rate | Fraction of attacker slots accepted for update |
| Profile contamination | Attacker share of the EWMA mean's weight (exact) |
| Drift false alarms | FPR on legitimate slots from drift onset onward |
| Attack detection | Recall |
| Detection delay | Slots from the first attacker slot to the first detection |
| Adaptation recovery | Settling time: slots from drift onset to the last time the rolling FPR (100 slots) exceeds 0.10, plus the fraction of seeds that recovered at all |
| Poisoning outcome | Recall in the attack phase after poisoning, and the maximum poisoning progress reached |
| Contamination at attack start | Profile contamination at the first attack-phase slot, after any rollback (§11) |
| Sequential alarm delay | Slots from the first attacker slot to the first sequential alarm (§11) |
| Sequential false-alarm burden | Fraction of slots in sequential alarm before any attack (§11) |

**Anchor-resolution sweep.** For a power-matched attacker at distance d from
the PU (8 random directions, 100 slots each), the sweep measures the fraction
of slots rejected by the physical check. It covers σ ∈ {4, 6, 8} dB and
{10, 20, 40} SUs, and is written to `reports/adaptive/anchor_resolution.csv`.

## 8. Reproduce

Run from the repository root:

```bash
python puea_detection/src/run_simulation.py          # ~1 min: data/simulation/
python puea_detection/src/run_adaptive_experiment.py # calibration, traces, sweep (slower with §11 systems)
python puea_detection/src/evaluate_adaptive.py       # metrics, report, figures
python puea_detection/src/check_theory.py            # single-slot resolution: theory vs sweep
python puea_detection/src/check_theory.py --sequential  # sequential delay: theory vs simulation (~4 min)
python -m unittest discover -s puea_detection/tests  # unit tests
streamlit run puea_detection/app/demo_app.py         # sidebar → "Adaptive PUEA Detection"
```

The outputs are:

- `reports/adaptive/adaptive_report.md`: every table.
- `reports/adaptive/metrics_summary.csv` and `metrics_by_seed.csv`.
- `reports/figures/adaptive_*.png`.

If the simulation settings in the config change, the experiment script
refuses to run on the stale data.

## 9. Results

All numbers below are from `reports/adaptive/adaptive_report.md`, as
produced by the commands in §8 (5 seeds per regime; mean over seeds). Two
independent end-to-end runs gave identical numbers. These are
**simulation-only** results under the assumptions in §2–§3.

## 10. Findings and limitations

### What the experiments support

1. **Naive adaptation is poisonable, as expected.**

   | Scenario (default regime) | Poisoning rate | Profile contamination | Recall |
   |---|---|---|---|
   | `poisoning` | 0.95 | up to 0.37 | 0.001 (attack phase) |
   | `near_pu_poisoning` | 0.97 | up to 0.37 | 0.017 (attack phase) |

   Naive-Adaptive also absorbs *non-adaptive* attackers:
   - `power_matching_puea`: F1 0.03
   - `stealthy_puea`: F1 0.05

   The reason is structural. The "xgb ∧ profile" alarm lets the profile veto
   the classifier, and every accepted attacker slot moves the profile toward
   the attacker.

2. **The physical gate prevents that corruption when the attacker is
   resolvable.**
   - In `poisoning` (default regime), the anchored system's contamination
     stays ≤ 0.012 with a poisoning rate of 0.004.
   - The attacker's progress never leaves its PU-mimic starting point.
   - Anchored-Update-Only reaches the same poisoning rate (0.001) using
     physics **only as an update gate**. That isolates the gate's
     contribution to protecting the profile.

3. **Adaptation removes drift false alarms that a static classifier cannot
   handle.** False-positive rate after drift onset, default regime:

   | System | power drift | noise drift | combined drift | Recovered (all four drift scenarios) |
   |---|---|---|---|---|
   | Baseline-XGBoost | 0.83 | 0.36 | 0.70 | 0/5 seeds (power, combined) |
   | XGB+Physical (frozen profile) | 0.77 | 0.36 | 0.71 | 0/5 seeds (power, combined) |
   | Physically-Anchored-Adaptive | 0.018 | 0.014 | 0.019 | 5/5 seeds |
   | Naive-Adaptive | ≈0.01 or lower | ≈0.01 or lower | ≈0.01 or lower | 5/5 seeds |

   The adaptive systems' error from the true PU profile is 0.3–0.4
   Mahalanobis units. For the frozen profile it is 10–86.

### What the experiments do NOT support (read before claiming novelty)

4. **Physics alone explains most of the detection results.**
   - In the default regime, *Physical-Only* matches or slightly beats the
     proposed system on every attack scenario except `near_pu_poisoning`.
     Examples (F1): power-matching 0.980 vs 0.976, stealthy 0.957 vs 0.935.
   - Physical-Only also has *lower* false alarms (≈0.012 vs ≈0.018).
   - The anchored system's value over Physical-Only shows up only in the
     combined decision for `near_pu_poisoning`: attack-phase recall 0.995 vs
     0.853.
   - A static **XGB+Physical** (no adaptation) is within ±0.005 F1 of the
     proposed system on every attack scenario *without* PU drift. It
     collapses where the PU also drifts: `stealthy_puea` F1 0.30 vs 0.94,
     through false alarms. The measurable benefit of adaptation is therefore
     drift tolerance, not better attack detection.

5. **Where the anchor is weak, adaptation becomes a liability.** In
   `weak_anchor` `near_pu_poisoning`:

   | System | Poisoning rate | Contamination | Attack-phase recall |
   |---|---|---|---|
   | Physically-Anchored-Adaptive | 0.88 | 0.35 | **0.13** |
   | Baseline-XGBoost (static) | — | — | 0.99 |
   | XGB+Physical (static) | — | — | 0.77 |

   Even in `weak_anchor` `poisoning`, the anchored profile reaches
   contamination 0.27 with a poisoning rate of 0.43. Detection survives
   there only because the target point is physically resolvable.

   **The gate is only as good as the anchor's spatial resolution. Once an
   attacker is inside that resolution, a gated adaptive profile can be
   dragged exactly like a naive one.**

6. **The anchor's resolution is limited and depends strongly on SU density
   and shadowing.** From the anchor-resolution sweep, for a power-matched
   attacker:

   | Setting | Attacker slots rejected |
   |---|---|
   | 40 SUs, σ = 4 dB, d ≤ 3 m | ≤ 10% |
   | 40 SUs, σ = 4 dB, d = 5 m | ~50% |
   | 40 SUs, σ = 4 dB, d ≥ 10 m | ≥ 95% |
   | 10 SUs, σ = 8 dB, d = 20 m | 16% |

   Close-in or co-located attackers, and multi-transmitter attackers (not
   simulated), defeat it.

7. **A supervised classifier given SU identity learns location by itself.**
   - Spatial-XGBoost (XGBoost on the raw per-SU RSS vector, no physics
     model) matches the anchored system on most attacks and beats it on
     `near_pu_poisoning` in the weak regime (F1 0.56 vs 0.20).
   - It fails under legitimate drift: FPR 0.26 under power drift, never
     recovered in 5/5 seeds.
   - A fair paper must include it as a baseline. It is the strongest one
     here.

### Honest reading

In this simulation, the defensible claim is narrow:

> A physically anchored update gate lets an adaptive PU profile absorb
> legitimate drift (low false alarms) without being poisoned, **provided the
> attacker is outside the anchor's spatial resolution**. Inside it, the gate
> provides little protection, and a static detector is safer.

The simulation does not support the claim that the adaptive profile adds
detection power beyond the physical check itself.

### Limitations of the framework

- **Everything is simulated.** Propagation matches the detector's model
  (path loss + log-normal shadowing), which favours the physical check. Real
  RF effects are absent: multipath, antenna patterns, non-log-normal
  shadowing, SU position errors and time-correlated shadowing
  (`shadow_time_correlation` = 0 by default).
- **Known locations are assumed.** The PU location and the SU locations are
  assumed known exactly. The PU is static; mobile PUs remove the anchor.
- **Attacker capability is bounded.** The attackers use one transmitter and
  know α and the SU positions. Multi-transmitter or beam-shaping attackers
  are not modelled.
- **The "xgb ∧ profile" decision rule is a design choice.** It is what makes
  adaptation useful and what makes poisoning possible. Other rules (for
  example, profile-only decisions) were not evaluated.
- **The profile is simple.** It is an EWMA Gaussian with λ = 0.01 and no
  tether or step clipping. A robust or rate-limited update might narrow the
  gap in finding 5. That has not been tested.
- **Recovery is coarse.** It is a settling time with tolerance 0.10 and a
  100-slot window, so it is insensitive to FPR changes below that
  tolerance. Use the drift-false-alarm-rate column for finer comparisons.
- **AUCs use composite scores** whose relative scaling is arbitrary (§6).
- **Five seeds per regime.** Standard deviations are in the report, and some
  differences above are within them.

## 11. Extension: sequential drift-invariant location test (2026-10-05)

**Why.** §10 finding 5: once an attacker is inside the anchor's single-slot
resolution (a few metres from the PU), the anchored profile can be poisoned
like a naive one. But poisoning needs many attacker slots, and each one
carries a small, consistently signed location mismatch. The extension
accumulates that evidence across slots. The prior-art check behind this
design is in [novelty_lit_review.md](novelty_lit_review.md).

**Premise.** For a static PU, legitimate drift changes the transmit power P,
the noise floor and the shadowing spread σ. It never moves the source.
P is re-fitted every slot, σ is tracked, and the PU location and α are never
adapted.

### Components (`src/sequential_spatial.py`)

1. **Location score per slot.** This is the efficient score of the source
   location at x_PU, with P profiled out:
   - `u = Σ w_i r_i (g_i − ḡ)/σ²`, where `g_i = ∂(α x_i)/∂p`;
   - Fisher information `J = Σ w_i (g_i − ḡ)(g_i − ḡ)ᵀ/σ²`;
   - standardised score `v = J^{-1/2} u`, which is N(0, I₂) under H0.

   Drift behaviour:
   - power drift cancels exactly (unit-tested);
   - σ is tracked by EWMA from the residual at the *best-fit* location, which
     does not depend on where the source is (unit-tested);
   - the noise floor enters only through the per-slot noise estimates and the
     SNR mask.
2. **Calibration (PU-only, first half of the calibration stream).**
   - An AR(1) coefficient φ of v is estimated and used to prewhiten,
     `e_t = (v_t − φ v_{t−1})/√(1−φ²)`. This is needed when shadowing is
     correlated in time.
   - An empirical whitening of e absorbs SNR-mask and noise-estimate model
     mismatch.
   - The GLR-CUSUM threshold h comes from Monte Carlo on N(0, I₂) for a
     per-slot false-alarm probability of 10⁻⁴ (window 500 slots).
3. **Window-limited GLR-CUSUM.**
   - `W_t = max_k ‖Σ_{k..t} e‖² / (2(t−k+1))`.
   - The argmax gives the estimated change point k̂.
4. **Rollback.** The profile is checkpointed every 10 slots. When an alarm
   starts, it is restored to the last checkpoint before k̂. The contamination
   metric follows the rollback.
5. **Attribution.**
   - During an alarm, a two-source mixture is fitted to the slots since k̂:
     the PU, or one fixed unknown location x_A with probability ρ, with P
     profiled per slot. The fit uses a grid over x_A and EM for ρ, and is
     refitted every 25 slots.
   - **Updates:** each slot's profile update is weighted by its posterior
     probability of coming from the PU (the EM-style soft update).
   - **Decisions:** a slot is flagged if its LLR (x̂_A against PU) exceeds the
     Neyman–Pearson threshold for the same 1% per-slot false-alarm budget as
     the other components. That threshold is `τ = −λ_A/2 + z₀.₉₉ √λ_A`, in
     closed form from the noncentrality λ_A of x̂_A.

### Systems added (`src/adaptive_systems.py`)

| System | Ablation | Behaviour during a sequential alarm |
|---|---|---|
| Sequential-Anchored | E | D, with profile updates frozen |
| Sequential-Anchored+Rollback | F | E, and the profile is rolled back to before k̂ when the alarm starts |
| Sequential-Anchored-Full | G | F, but with posterior-weighted (soft) updates instead of freezing, plus NP attribution in the decision |
| Physical-Only+Sequential | ref | Physical check plus NP attribution. No classifier and no profile (kill criterion 3) |

### Simulator change

The attacker's per-link shadowing is now correlated with the PU's:
`c = exp(−|x_att − x_PU| / 20 m)`, a Gudmundson-type exponential model
(`channel.shadow_decorrelation_distance`). Before this change, a co-located
attacker saw shadowing independent of the PU's.

- That was physically wrong.
- With shadowing correlated in time, it let the sequential test detect every
  switch between PU and attacker. Detection 2 m from the PU happened within
  2–26 slots, which was an artifact.

With i.i.d. shadowing over time the change has no statistical effect, because
the PU's realisation in an attacker slot is never observed. This changes the
realisations of every attack stream, so all regimes were re-simulated.

### Added regime and scenario

- `correlated_shadowing`: AR(1) shadowing with ρ = 0.9 across slots
  (kill criterion 2).
- `slow_near_pu_poisoning`: like `near_pu_poisoning`, but poison is injected
  in only 5% of slots (kill criterion 4).

### Theory checks (`src/check_theory.py`)

- **Single-slot resolution.** The exact GLRT noncentrality, used as
  `P(ncχ²₂(λ) > threshold)`, predicts the measured anchor-resolution sweep
  (243 cells, 3 σ × 3 SU counts × 9 distances × 5 seeds). Linearised
  `δᵀJδ` is accurate up to about 5 m and overestimates beyond that.
  (Agreement numbers: §11 results.)
- **Sequential delay.**
  - The score noncentrality ‖J^{-1/2}E[u]‖² comes in closed form from the
    geometry.
  - The first-passage time of the same GLR-CUSUM on the idealised score model
    (v = noise + Bernoulli(ρ)·μ) predicts the delay. That model uses only
    geometry, ρ, φ and h, with no RF simulation.
  - It is compared with the measured delay for a fixed attacker at
    d ∈ {2, 3, 5, 8} m and ρ ∈ {0.1, 0.3}, in all three regimes.
  - The purely closed-form drift approximation `h / (½ρ²‖μ‖²)` does **not**
    predict the delay (it ignores the noise in W and the effect of
    intermittency). It is reported only as a scaling law.

### §11 results

All numbers are means ± std over 5 seeds, from
`reports/adaptive/metrics_summary.csv` (the full tables are in
`adaptive_report.md`). Simulation only.

**Theory.**

- **Single-slot resolution.** The exact-noncentrality prediction matches the
  re-run anchor-resolution sweep with MAE **0.020** (max |err| 0.22,
  correlation **0.997**, 243 cells). The linearised Fisher prediction has
  MAE 0.066.
- **Sequential delay.** The model-based prediction against the measured
  first-alarm delay (120 cells) gives a median measured/predicted ratio of
  **1.00**, IQR 0.40–2.00, and a log-correlation of **0.87**. Twelve cells did
  not alarm within 3,000 slots (weak regime, ρ = 0.1). The first-order
  closed-form formula fails (log-correlation 0.21).

**Kill criteria (fixed in advance, docs/novelty_lit_review.md).**

| # | Test | Outcome |
|---|---|---|
| 1 | Theory predicts measured resolution | **Pass** (above). The delay is predicted only semi-analytically |
| 2 | Survives time-correlated shadowing | **Pass, with caveats.** No drift false alarms (≤ 0.4% of slots in alarm before any attack). Detection is *faster*: near-PU alarm in 6 ± 4 slots, against 40 ± 25 in the default regime, because intermittent attacker slots break the PU's temporal continuity. The size of this effect depends on the assumed 20 m decorrelation distance, which was not varied. With the post-hoc hysteresis fix, contamination at attack start is 0.006 ± 0.01 (F) |
| 3 | Full system beats Physical-Only + sequential | **Fail.** Physical-Only+Sequential is within one std of G on almost every scenario and has a slightly lower FPR. The adaptive profile adds little once the sequential test is present |
| 4 | Slow poisoner (5% duty) caught in time | **Partial.** Default regime: alarm after 317 ± 307 slots, contamination at attack start 0.03. Weak regime: 1,031 ± 517 slots, often near the end of the 1,200-slot poison phase. Contamination stays low (0.03) anyway, because a slow poisoner also poisons slowly. Attack-phase recall improves from 0.17 (D) to 0.44 (F) |
| 5 | σ-inflation attacker | **Not tested.** σ is estimated from the best-fit residual, which a single transmitter cannot change through its position or power. Multi-transmitter attackers are out of scope |

**Where the sequential layer helps (weak anchor: 20 SUs, σ = 8 dB).**

| `near_pu_poisoning` | D (§6) | F freeze+rollback | G full | Phys+Seq | Spatial-XGB |
|---|---|---|---|---|---|
| F1 | 0.19 ± 0.14 | **0.54 ± 0.06** | 0.35 ± 0.14 | 0.32 ± 0.15 | 0.56 ± 0.02 |
| Attack-phase recall | 0.12 ± 0.11 | **0.63 ± 0.14** | 0.30 ± 0.23 | 0.28 ± 0.21 | 0.69 ± 0.09 |
| Contamination at attack start | 0.29 ± 0.05 | **0.08 ± 0.08** | 0.20 ± 0.04 | – | – |
| FPR under power drift | 0.01 | 0.01 | 0.01 | 0.01 | **0.12 ± 0.04** |

- **Default regime.** Contamination at attack start drops from 0.24 ± 0.03
  (D) to 0.05 ± 0.03 (F). Attack-phase recall was already about 1.0 for D,
  so F1 barely moves.
- **Drift.** None of the sequential systems raised a sequential alarm before
  any attack in any drift scenario, in any regime. Their drift FPRs equal D's.

**What the results do *not* support.**

1. **No single variant wins.**
   - Freezing (E/F) resists poisoning best, but collapses when an attacker is
     present throughout *and* the PU drifts: `stealthy_puea` F1 is 0.32 ± 0.02
     (default) against 0.93 for D.
   - Soft updates (G) keep drift tracking (stealthy 0.92 ± 0.03) but leak
     poison inside the resolution limit (contamination 0.15–0.20).
   - Inside the resolution limit, the poisoning-versus-drift tradeoff is
     narrowed, not removed.
2. **The adaptive profile is not where the gain comes from** (kill criterion
   3). Physical-Only+Sequential is as good as G.
3. **Spatial-XGBoost matches F** on near-PU poisoning in the weak regime. It
   fails under legitimate drift (FPR 0.12–0.25), which F does not.
4. **Attribution costs a little on easy attacks.** G's F1 on
   basic/power-matching attacks falls by about 0.01–0.02 compared with D,
   because the NP attribution spends some of its 1% budget.

**Limitations specific to §11.**

1. Everything in §10's limitations still applies.
2. **Alarm flicker (fixed post hoc; see "Post-hoc change" below).** In the
   first evaluation the alarm had no hysteresis. Under correlated shadowing
   it switched on and off during poisoning, with up to 181 rollbacks per
   stream, so rollback gave no gain over freezing. The numbers above are
   from the re-run with the fix. The pre-fix results are kept in
   `reports/adaptive/prefix_flicker_fix/`.
3. **Decorrelation distance.** The correlated-shadowing benefit depends on
   d_corr = 20 m, and a co-located attacker (d ≈ 0) remains undetectable by
   construction.
4. **Design choices.** The window (500), per-slot alarm probability (10⁻⁴)
   and refit interval (25) were set before the runs and not tuned. They were
   also not varied.

**Defensible claim (simulation only).**

> A drift-invariant sequential location test, with change-point rollback,
> detects attackers inside the single-slot resolution limit without raising
> false alarms under legitimate drift. In the weak-anchor regime it raises
> attack-phase recall against near-PU poisoning from 0.12 to 0.63 and cuts
> profile contamination from 0.29 to 0.08. The geometry predicts its
> single-slot resolution (MAE 0.02), and a geometry-only score model
> predicts its detection delay (median ratio 1.0). The benefit comes from the
> sequential location test, not from the adaptive profile.

### Post-hoc change: alarm hysteresis (2026-10-06)

**What changed.**

- An alarm now opens an *episode* that stays open until the statistic has
  been below the threshold for `window` (500) consecutive verifiable slots.
  This reuses the existing window length, so no new parameter was added.
- Rollback happens once per episode, to the change point estimated at the
  episode's start.
- σ tracking pauses for the whole episode.

**Why it is post hoc.** The flicker was found by inspecting the first
evaluation's traces. The fix was then designed and the whole experiment
re-run (all regimes, all seeds, same data). Nothing else changed between the
two runs. Pre-fix outputs are in `reports/adaptive/prefix_flicker_fix/`.

**Effect (mean over 5 seeds, before → after).**

| Regime | Metric (scenario) | E | F | G | Phys+Seq |
|---|---|---|---|---|---|
| correlated_shadowing | Rollbacks per stream (near-PU) | – | 151.6 → **1.0** | – | – |
| correlated_shadowing | Contamination at attack start (near-PU) | 0.157 → **0.019** | 0.158 → **0.006** | 0.150 → 0.127 | – |
| correlated_shadowing | Attack-phase recall (near-PU) | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.94 → **0.99** |
| correlated_shadowing | F1 (stealthy + PU drift) | 0.70 → **0.32** | 0.64 → **0.32** | 0.88 → 0.90 | 0.93 → 0.92 |
| weak_anchor | Contamination (near-PU) | 0.220 → 0.219 | 0.079 → 0.079 | 0.204 → 0.206 | – |
| weak_anchor | Attack-phase recall (near-PU) | 0.27 → 0.28 | 0.63 → 0.63 | 0.30 → 0.30 | 0.28 → 0.29 |
| default | Contamination (near-PU) | 0.108 → 0.081 | 0.039 → 0.050 | 0.150 → 0.151 | – |
| all | FPR under drift | unchanged | unchanged | unchanged | unchanged |

**Reading.**

- **The fix does what it was meant to.** Under correlated shadowing,
  rollback now undoes the poisoning: contamination falls from 0.158 to
  0.006, with a single rollback.
- **Elsewhere it is neutral.** Weak and default regimes change by less than
  one std.
- **It has a cost.** With a persistent attacker and PU drift (`stealthy_puea`,
  correlated regime), the freeze-based systems E and F now stay frozen for
  the whole attack. Their F1 drops to 0.32, the same collapse they already
  showed in the other regimes. Before the fix, the flicker had accidentally
  let them adapt between alarms. G (soft updates) and Physical-Only+Sequential
  are unaffected.
- This sharpens the conclusion of finding 1: **freezing is the right response
  to poisoning, and the wrong one to a persistent attacker during drift.**

