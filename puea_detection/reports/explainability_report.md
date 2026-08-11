# PUEA Detection — Explainability Report (SHAP)

Global feature attributions computed with SHAP on held-out test subsamples.

- **Random Forest:** `TreeExplainer` (approximate=True, n=400).
- **XGBoost:** `TreeExplainer` (n=1000).
- **Ensemble (Stacking):** `KernelExplainer` on `predict_proba` (background=40, explain=60, nsamples=50).

## Mean |SHAP| Feature Rankings

### Random Forest

| Rank | Feature | Description | Mean |SHAP| |
|---:|---|---|---:|
| 1 | `Distance_Mismatch_km` | Location / Distance Mismatch (km) | 0.246700 |
| 2 | `Channel_Occupancy_Time_s` | Channel Occupancy Time (s) | 0.155718 |
| 3 | `Transmission_Power_dBm` | Transmission Power (dBm) | 0.042388 |
| 4 | `Frequency_Hz` | Operating Frequency (Hz) | 0.031368 |
| 5 | `X_Coordinate_km` | Transmitter X Position (km) | 0.027869 |
| 6 | `SINR_dB` | Signal-to-Interference-plus-Noise Ratio (dB) | 0.027168 |
| 7 | `Y_Coordinate_km` | Transmitter Y Position (km) | 0.013606 |
| 8 | `SNR_dB` | Signal-to-Noise Ratio (dB) | 0.010954 |
| 9 | `SNR_Deviation` | SNR Deviation from Expected Value | 0.008056 |
| 10 | `RSS_dBm` | Received Signal Strength (dBm) | 0.007701 |
| 11 | `RSS_Deviation` | RSS Deviation from PU Profile | 0.007487 |

### XGBoost

| Rank | Feature | Description | Mean |SHAP| |
|---:|---|---|---:|
| 1 | `Distance_Mismatch_km` | Location / Distance Mismatch (km) | 2.415315 |
| 2 | `Channel_Occupancy_Time_s` | Channel Occupancy Time (s) | 0.969321 |
| 3 | `Frequency_Hz` | Operating Frequency (Hz) | 0.827989 |
| 4 | `SINR_dB` | Signal-to-Interference-plus-Noise Ratio (dB) | 0.730966 |
| 5 | `SNR_dB` | Signal-to-Noise Ratio (dB) | 0.544214 |
| 6 | `Transmission_Power_dBm` | Transmission Power (dBm) | 0.411531 |
| 7 | `Y_Coordinate_km` | Transmitter Y Position (km) | 0.204434 |
| 8 | `X_Coordinate_km` | Transmitter X Position (km) | 0.203624 |
| 9 | `RSS_dBm` | Received Signal Strength (dBm) | 0.065788 |
| 10 | `SNR_Deviation` | SNR Deviation from Expected Value | 0.054458 |
| 11 | `RSS_Deviation` | RSS Deviation from PU Profile | 0.027806 |

### Ensemble (Stacking)

| Rank | Feature | Description | Mean |SHAP| |
|---:|---|---|---:|
| 1 | `Distance_Mismatch_km` | Location / Distance Mismatch (km) | 0.155885 |
| 2 | `SINR_dB` | Signal-to-Interference-plus-Noise Ratio (dB) | 0.099924 |
| 3 | `Channel_Occupancy_Time_s` | Channel Occupancy Time (s) | 0.084757 |
| 4 | `Frequency_Hz` | Operating Frequency (Hz) | 0.076300 |
| 5 | `Transmission_Power_dBm` | Transmission Power (dBm) | 0.069493 |
| 6 | `SNR_dB` | Signal-to-Noise Ratio (dB) | 0.064012 |
| 7 | `X_Coordinate_km` | Transmitter X Position (km) | 0.022899 |
| 8 | `Y_Coordinate_km` | Transmitter Y Position (km) | 0.022087 |
| 9 | `RSS_Deviation` | RSS Deviation from PU Profile | 0.015884 |
| 10 | `SNR_Deviation` | SNR Deviation from Expected Value | 0.015528 |
| 11 | `RSS_dBm` | Received Signal Strength (dBm) | 0.011849 |

## Consensus Ranking (mean rank across explained models)

| Rank | Feature | Mean Rank |
|---:|---|---:|
| 1 | `Distance_Mismatch_km` | 1.00 |
| 2 | `Channel_Occupancy_Time_s` | 2.33 |
| 3 | `Frequency_Hz` | 3.67 |
| 4 | `SINR_dB` | 4.00 |
| 5 | `Transmission_Power_dBm` | 4.67 |
| 6 | `SNR_dB` | 6.33 |
| 7 | `X_Coordinate_km` | 6.67 |
| 8 | `Y_Coordinate_km` | 7.33 |
| 9 | `SNR_Deviation` | 9.67 |
| 10 | `RSS_dBm` | 10.00 |
| 11 | `RSS_Deviation` | 10.33 |

## Stacking Meta-Learner Weights

LogisticRegression coefficients on stacked base-model outputs (positive → pushes toward PUEA class).

- Intercept: `-3.273734`

| Base Estimator | Meta Coefficient |
|---|---:|
| `rf` | 0.437207 |
| `xgb` | 7.017241 |

## Figures

Saved under `reports/figures/`:

- `shap_summary_*.png` — beeswarm global attributions
- `shap_bar_*.png` — mean |SHAP| importance
- `shap_dependence_*.png` — dependence for top RF/XGB feature
- `shap_waterfall_*_puea.png` / `*_benign.png` — local explanations
