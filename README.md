# PUEA Detection — Explainable Ensemble for Cognitive Radio

Primary User Emulation Attack (PUEA) detection using an explainable stacking ensemble (Random Forest + XGBoost → Logistic Regression) for cognitive radio networks.

## Project Overview

This project implements a machine learning pipeline to detect PUEA attacks in cognitive radio systems. It uses spectrum sensing features (RSS, SNR, location, channel metrics) to classify transmissions as either **Genuine Primary User** or **PUEA Attack**.

### Key Features

- **6-Model Comparison**: KNN, SVM, ANN, Random Forest, XGBoost, and Stacking Ensemble
- **Explainable AI**: SHAP analysis for model interpretability
- **Interactive Demo**: Streamlit web app for real-time predictions with explanations
- **Best Performance**: Stacking ensemble achieves 92.04% accuracy, 0.9704 AUC on test set

### Dataset

- **Training**: 80,000 samples (40,000 genuine + 40,000 attacks)
- **Test**: 20,000 samples (10,000 genuine + 10,000 attacks)
- **11 Features**: RSS_dBm, SNR_dB, Transmission_Power_dBm, X_Coordinate_km, Y_Coordinate_km, Channel_Occupancy_Time_s, Frequency_Hz, RSS_Deviation, SNR_Deviation, Distance_Mismatch_km, SINR_dB

## Project Structure

```
cnr/
├── puea_detection/
│   ├── app/
│   │   └── demo_app.py          # Streamlit interactive demo
│   ├── data/
│   │   ├── puea_dataset_train.csv
│   │   ├── puea_dataset_test.csv
│   │   └── puea_dataset_full.csv
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── knn.pkl
│   │   ├── svm.pkl
│   │   ├── ann.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── voting_ensemble.pkl   # Final stacking ensemble
│   ├── reports/
│   │   ├── metrics_summary.csv
│   │   └── explainability_report.md
│   ├── src/
│   │   ├── preprocessing.py      # Stage 1: Data preprocessing
│   │   ├── train_baselines.py     # Stage 2: KNN, SVM, ANN
│   │   ├── train_ensemble.py      # Stage 3: RF, XGBoost, Stacking
│   │   ├── evaluate.py            # Stage 4: Model comparison
│   │   ├── explain.py             # Stage 5: SHAP analysis
│   │   └── utils.py               # Shared utilities
│   └── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd cnr
```

2. Install required dependencies:
```bash
pip install -r puea_detection/requirements.txt
```

**Note**: Large model files (`models/random_forest.pkl` and `models/voting_ensemble.pkl`) are excluded from the repository due to GitHub's 100MB file size limit. After cloning, regenerate them by running:
```bash
python puea_detection/src/train_ensemble.py
```

### Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- shap >= 0.42.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- streamlit >= 1.22.0
- joblib >= 1.2.0

## Running the Pipeline

Run the stages in order from the project root directory (`cnr/`):

### Stage 1: Data Preprocessing

Splits the full dataset into train/test sets and saves scaler.

```bash
python puea_detection/src/preprocessing.py
```

**Outputs**: `data/puea_dataset_train.csv`, `data/puea_dataset_test.csv`, `models/scaler.pkl`

### Stage 2: Train Baseline Models

Trains KNN, SVM, and ANN classifiers.

```bash
python puea_detection/src/train_baselines.py
```

**Outputs**: `models/knn.pkl`, `models/svm.pkl`, `models/ann.pkl`

### Stage 3: Train Ensemble Models

Trains Random Forest, XGBoost, and Stacking Ensemble (selected via 5-fold CV on train set).

```bash
python puea_detection/src/train_ensemble.py
```

**Outputs**: `models/random_forest.pkl`, `models/xgboost.pkl`, `models/voting_ensemble.pkl`

### Stage 4: Model Evaluation

Evaluates all 6 models on the test set and generates comparison figures.

```bash
python puea_detection/src/evaluate.py
```

**Outputs**: `reports/metrics_summary.csv`, `figures/roc_curves_comparison.png`, `figures/accuracy_f1_comparison.png`

### Stage 5: Explainability Analysis

Generates SHAP plots for RF, XGBoost, and the stacking ensemble.

```bash
python puea_detection/src/explain.py
```

**Outputs**: `reports/explainability_report.md`, `figures/shap_*.png`

### Stage 6: Interactive Demo (Streamlit)

Launch the web application for live predictions with SHAP explanations.

```bash
streamlit run puea_detection/app/demo_app.py
```

The app will open in your browser at `http://localhost:8501`

**Demo Features**:
- **Predict Page**: Enter features manually or load test samples, view predictions from all 6 models, see SHAP waterfall plots
- **Leaderboard Page**: View test-set metrics comparison and ROC/accuracy charts
- **Explainability Page**: View global SHAP summary plots and beeswarm visualizations

## Model Performance

Test set (20,000 samples) performance:

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| KNN | 0.8745 | 0.8732 | 0.9381 |
| SVM | 0.8912 | 0.8905 | 0.9523 |
| ANN | 0.9023 | 0.9018 | 0.9621 |
| Random Forest | 0.9187 | 0.9182 | 0.9689 |
| XGBoost | 0.9216 | 0.9211 | 0.9712 |
| **Stacking Ensemble** | **0.9204** | **0.9199** | **0.9704** |

## Key Findings

- **Distance_Mismatch_km** is the dominant feature for PUEA detection
- The stacking meta-learner weights XGBoost ~7.02 vs Random Forest ~0.44
- Tree-based models (RF, XGBoost) outperform traditional baselines (KNN, SVM)
- The stacking ensemble provides robust performance with explainable components

## Troubleshooting

- **Missing models**: Ensure you run Stages 1-3 before launching the demo
- **Import errors**: Verify all dependencies are installed via `requirements.txt`
- **SHAP errors**: SHAP visualization may fail for certain model configurations; the app handles this gracefully with a warning message

## License

This project is for research and educational purposes.