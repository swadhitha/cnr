# PUEA Detection for Cognitive Radio Networks

This repository detects **Primary User Emulation Attacks (PUEA)** in cognitive radio networks. In a PUEA, an attacker imitates the licensed primary user (PU) so that secondary users (SUs) vacate the channel.

It contains two parts:

1. **Baseline pipeline (stages 1–6).** Six classifiers (KNN, SVM, ANN, Random Forest, XGBoost, and a stacking ensemble), SHAP explainability, and a Streamlit demo. All of it runs on a labelled tabular dataset.
2. **Research extension: Physically Anchored Adaptive PUEA Detection.** A temporal multi-SU simulator and an experimental framework. They test whether a detector can adapt to legitimate PU signal drift without being poisoned by an attacker. See [below](#research-extension-physically-anchored-adaptive-puea-detection).

## Project Structure

```
cnr/
├── puea_detection/
│   ├── app/
│   │   ├── demo_app.py              # Streamlit demo (all pages)
│   │   └── adaptive_page.py         # "Adaptive PUEA Detection" page (extension)
│   ├── configs/experiments.json     # extension: every experimental constant
│   ├── data/
│   │   ├── puea_dataset_{train,test,full}.csv   # baseline dataset
│   │   └── simulation/              # extension: generated, git-ignored
│   ├── docs/adaptive_puea.md        # extension: methodology, findings, limitations
│   ├── models/
│   │   ├── scaler.pkl, knn.pkl, svm.pkl, ann.pkl, xgboost.pkl, xgboost.json
│   │   ├── random_forest.pkl        # git-ignored (>100 MB) - regenerate with stage 3
│   │   ├── voting_ensemble.pkl      # git-ignored - final stacking ensemble
│   │   └── simulation/              # extension: generated, git-ignored
│   ├── reports/
│   │   ├── metrics_summary.csv, comparison_report.md, explainability_report.md
│   │   ├── adaptive/                # extension: metrics, report, calibration
│   │   └── figures/                 # baseline figures + adaptive_*.png
│   ├── src/
│   │   ├── preprocessing.py         # Stage 1
│   │   ├── train_baselines.py       # Stage 2: KNN, SVM, ANN
│   │   ├── train_ensemble.py        # Stage 3: RF, XGBoost, stacking
│   │   ├── evaluate.py              # Stage 4
│   │   ├── explain.py               # Stage 5: SHAP
│   │   ├── utils.py
│   │   ├── simulation/              # extension: CRN simulator + scenarios
│   │   ├── fc_features.py, physical_consistency.py, adaptive_profile.py, adaptive_systems.py
│   │   ├── run_simulation.py, run_adaptive_experiment.py, evaluate_adaptive.py, adaptive_plots.py
│   ├── tests/                       # extension: unit tests
│   └── requirements.txt
└── README.md
```

## Setup

Python 3.9+.

```bash
cd cnr
python -m venv .venv && source .venv/bin/activate
pip install -r puea_detection/requirements.txt
```

`random_forest.pkl` and `voting_ensemble.pkl` are excluded from git because they exceed GitHub's 100 MB limit. The demo still runs without them and uses the other four models. To restore them, run:

```bash
python puea_detection/src/train_ensemble.py
```

## Baseline Pipeline

Run from the repository root (`cnr/`).

**Dataset.**
- Training set: 80,000 rows (40,000 genuine, 40,000 attack).
- Test set: 20,000 rows (10,000 genuine, 10,000 attack).
- 11 features: `RSS_dBm, SNR_dB, Transmission_Power_dBm, X_Coordinate_km, Y_Coordinate_km, Channel_Occupancy_Time_s, Frequency_Hz, RSS_Deviation, SNR_Deviation, Distance_Mismatch_km, SINR_dB`.

| Stage | Command | What it does | Outputs |
|---|---|---|---|
| 1 | `python puea_detection/src/preprocessing.py` | Loads the existing train/test CSVs. Fits a `StandardScaler` on the training set only | `models/scaler.pkl` |
| 2 | `python puea_detection/src/train_baselines.py` | KNN, SVM, ANN (scikit-learn MLP) on scaled features | `models/{knn,svm,ann}.pkl` |
| 3 | `python puea_detection/src/train_ensemble.py` | Random Forest and XGBoost on raw features. The ensemble strategy is chosen by 5-fold CV on the training set only | `models/{random_forest,xgboost,voting_ensemble}.pkl` |
| 4 | `python puea_detection/src/evaluate.py` | Evaluates every model once on the untouched test set | `reports/metrics_summary.csv`, `reports/comparison_report.md`, ROC / confusion / accuracy figures |
| 5 | `python puea_detection/src/explain.py` | SHAP for RF, XGBoost and the ensemble | `reports/explainability_report.md`, `figures/shap_*.png` |
| 6 | `streamlit run puea_detection/app/demo_app.py` | Interactive demo (see [Demo](#demo)) | — |

### Baseline results

Test set, 20,000 rows, from [`reports/metrics_summary.csv`](puea_detection/reports/metrics_summary.csv):

| Model | Accuracy | Precision | Recall | F1 | False alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.8872 | 0.9643 | 0.8042 | 0.8770 | **0.0298** | 0.9356 |
| SVM | 0.9094 | 0.9582 | 0.8563 | 0.9044 | 0.0374 | 0.9606 |
| ANN | 0.9125 | 0.9504 | 0.8704 | 0.9087 | 0.0454 | 0.9623 |
| Random Forest | 0.9113 | 0.9538 | 0.8646 | 0.9070 | 0.0419 | 0.9656 |
| XGBoost | **0.9216** | **0.9600** | 0.8799 | **0.9182** | 0.0367 | 0.9704 |
| Stacking ensemble (RF + XGBoost → LR) | 0.9204 | 0.9479 | **0.8897** | 0.9179 | 0.0489 | **0.9704** |

**Observations.**
- **XGBoost vs ensemble.** XGBoost has the highest accuracy and F1. The ensemble matches its ROC-AUC (0.9704 vs 0.9704) with higher recall but a higher false alarm rate. It does not outperform XGBoost overall.
- **Ensemble weighting.** The ensemble relies almost entirely on XGBoost: meta-learner coefficients are 7.02 for XGBoost and 0.44 for RF.
- **Random Forest vs ANN.** Random Forest does not beat the ANN on accuracy or F1.
- **Dominant feature.** `Distance_Mismatch_km` dominates the SHAP rankings.
- **Caution on two features.** `Transmission_Power_dBm` and `Distance_Mismatch_km` are not directly observable by a real fusion center for an unknown transmitter. That is one motivation for the simulated extension.

## Demo

```bash
streamlit run puea_detection/app/demo_app.py      # opens http://localhost:8501
```

Pages, chosen from the sidebar:
- **Predict:** enter features or load a test row, see every loaded model's vote and an XGBoost SHAP waterfall. If the git-ignored RF and ensemble models are missing, the page says so and uses the rest.
- **Leaderboard:** the test-set metrics table with ROC and accuracy charts.
- **Explainability:** global SHAP plots.
- **Adaptive PUEA Detection:** replays the extension's experiments slot by slot. You pick a scenario, regime, seed and two systems to compare. For each slot it shows:
  - the XGBoost prediction and its confidence;
  - the physical consistency score and residual;
  - whether the profile update was allowed or blocked;
  - the current profile compared with the true PU profile;
  - attack warnings.

  It also shows the spatial RSS map with the best-fit source location. It needs the extension outputs (see [Reproduce](#reproduce)); otherwise it lists the commands to run.

## Research Extension: Physically Anchored Adaptive PUEA Detection

**Question.** Can a PUEA detector adapt its model of legitimate PU behaviour to real signal drift (power, shadowing, noise) without being poisoned by an attacker who injects increasingly PU-like signals?

**Mechanism under test.**
- **Naive adaptation:** PU-like → update the profile.
- **Proposed:** PU-like **and** physically consistent → update. Physically consistent means the spatial RSS pattern across the SUs could have come from the PU's known location.
- **PU-like but physically inconsistent:** reject the update and flag PUEA.

Stages 1–5, the dataset and `models/xgboost.pkl` are unchanged. The demo gained one page. Methodology, assumptions, leakage rules and the full findings are in [`docs/adaptive_puea.md`](puea_detection/docs/adaptive_puea.md). All generated tables are in [`reports/adaptive/adaptive_report.md`](puea_detection/reports/adaptive/adaptive_report.md).

### Why a separate simulator

The baseline CSV has one aggregated row per transmission. It has no per-SU sensing vector, no SU or PU positions and no time index, so neither a multi-SU physical check nor drift and poisoning timelines can be computed from it.

`src/simulation/` generates that data separately under `data/simulation/`:
- 40 SUs in a 100 × 100 m area.
- One PU and one attacker.
- Repeated sensing slots.
- The path-loss + log-normal shadowing model of Chhetry & Marchang (2021): `Pr = Pt · r^(-α) · exp(a·β)`.

In these experiments, *Baseline-XGBoost* uses the same learner and hyperparameters as stage 3. It is trained on features a fusion center can actually compute: the five RSS statistics from Chhetry & Marchang plus mean SNR.

### Systems compared

| System | Ablation | Alarm rule | Profile update |
|---|---|---|---|
| Baseline-XGBoost | A | XGBoost | none |
| Naive-Adaptive | B | XGBoost ∧ adaptive profile | every PU-like slot |
| XGB+Physical | C | (XGBoost ∧ frozen profile) ∨ physically inconsistent | none |
| **Physically-Anchored-Adaptive** (proposed) | D | (XGBoost ∧ adaptive profile) ∨ physically inconsistent | PU-like ∧ physically consistent |

There are also four reference systems:
- XGB+StaticProfile.
- Anchored-Update-Only: physics gates updates but is not used in decisions.
- Physical-Only.
- Spatial-XGBoost: XGBoost on the raw per-SU RSS vector.

### Scenarios

- **Legitimate PU:** `stable`, `power_drift` (+6 dB), `shadowing_drift` (σ 4 → 5.5 dB), `noise_drift` (+4 dB), `combined_drift`.
- **Attacks:** `basic_puea`, `power_matching_puea`, `gradual_puea` (PU-likeness 0.2 → 0.95), `stealthy_puea` (tracks the PU's drifting power), `poisoning` (closed-loop boiling frog), `near_pu_poisoning` (poisoning inside the physical check's resolution limit).
- **Regimes:** everything runs in `default` (40 SUs, σ = 4 dB) and `weak_anchor` (20 SUs, σ = 8 dB).
- **Anchor-resolution sweep:** measures how close to the PU an attacker must be before the physical check stops rejecting it.

### Findings at a glance

Simulation only; mean over 5 seeds. The details and caveats are in [`docs/adaptive_puea.md` §10](puea_detection/docs/adaptive_puea.md).

- **Naive adaptation is poisonable.** In `poisoning` it absorbs 95% of attacker slots, and recall after poisoning is 0.001.
- **The physical gate blocks poisoning by a resolvable attacker.** In the same scenario, profile contamination stays ≤ 0.012.
- **Adaptation handles legitimate drift.** False alarms after drift onset are about 0.02, against 0.36–0.83 for static XGBoost.
- **But the physical check does most of the detection work.** *Physical-Only* matches the proposed system on almost every attack. A static XGB+Physical ties it wherever the PU does not drift.
- **Inside the anchor's resolution, adaptation is a liability.** In `weak_anchor` / `near_pu_poisoning`, the proposed system's recall after poisoning is **0.13**, against 0.99 for static XGBoost.
- **Spatial-XGBoost is the strongest competing baseline.** It learns location from SU identity but fails under drift.

The claim these results support is narrow: the gate enables drift-tolerant adaptation without poisoning **only when the attacker is outside the anchor's spatial resolution**. That resolution is roughly ≥ 10 m with 40 SUs at 4 dB shadowing and much worse with fewer SUs or more shadowing.

### Reproduce

Run from the repository root; about 8 minutes in total:

```bash
python puea_detection/src/run_simulation.py           # 1. simulation data (5 seeds × 2 regimes)
python puea_detection/src/run_adaptive_experiment.py  # 2. train + calibrate + run every system and scenario, sweep
python puea_detection/src/evaluate_adaptive.py        # 3. metrics, report, figures (reports/figures/adaptive_*.png)
python -m unittest discover -s puea_detection/tests   # unit tests
streamlit run puea_detection/app/demo_app.py          # sidebar → "Adaptive PUEA Detection"
```

Use `--seeds 0 --regimes default` for a quick subset. The experiment script refuses to run if the simulation settings in `configs/experiments.json` have changed since the data was generated.

### How thresholds are selected

All thresholds are learnt from a **PU-only calibration stream** that is separate from every test stream. No test data or labels are used.

- **Physical consistency:** the path-loss exponent and shadowing σ are estimated on the first half of the stream. The threshold is the empirical 99th percentile of the GLRT residual on the held-out second half.
- **Adaptive profile:** the initial mean and covariance come from MinCovDet on the first half. The threshold is the empirical 99th percentile of the Mahalanobis distance on the second half.
- **XGBoost:** the standard 0.5 threshold.

The saved values are in `reports/adaptive/calibration/<regime>_seed_<k>.json`. The profile only ever sees what the fusion center observes up to the current slot. Labels, attacker parameters and future slots are never used to update it.

## Troubleshooting

- **"Not loaded (file missing): random_forest.pkl, voting_ensemble.pkl"** on the Predict page. These are git-ignored; run stage 3 to regenerate them.
- **Adaptive page says "No experiment traces found".** Run the three commands under [Reproduce](#reproduce).
- **`ModuleNotFoundError: pyarrow`.** Streamlit's tables and charts need it. It is in `requirements.txt`; run `pip install pyarrow` if your environment predates that.
- **SHAP errors.** SHAP rendering can fail for some model configurations. The app shows a warning instead of crashing.

## License

This project is for research and educational purposes.
