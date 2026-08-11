"""
Tree Ensembles & Methodologically Sound Ensemble Selection via Training Cross-Validation.
Trains Random Forest & XGBoost, selects the optimal ensemble strategy strictly via 5-fold CV on X_train,
and evaluates the selected winner ONCE on the untouched test set.
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from utils import save_object, get_project_paths
from preprocessing import load_and_preprocess_data


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains RandomForestClassifier on raw features.
    """
    print("\n" + "=" * 50)
    print("1. Training Ensemble Component: Random Forest Classifier")
    print("=" * 50)

    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=250, max_depth=25, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"Random Forest Training completed in {elapsed:.2f}s")
    print(f"Random Forest Test Accuracy : {test_acc:.4f}")
    print(f"Random Forest Test F1-Score : {test_f1:.4f}")
    print(f"Random Forest Test ROC-AUC  : {test_auc:.4f}")

    save_object(rf_model, 'random_forest.pkl')
    return rf_model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains XGBClassifier on raw features.
    Saves model to models/xgboost.json & models/xgboost.pkl.
    """
    print("\n" + "=" * 50)
    print("2. Training Ensemble Component: XGBoost Classifier")
    print("=" * 50)

    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )

    print("Fitting XGBClassifier on X_train_raw...")
    xgb_model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"XGBoost Training completed in {elapsed:.2f}s")
    print(f"XGBoost Test Accuracy : {test_acc:.4f}")
    print(f"XGBoost Test F1-Score : {test_f1:.4f}")
    print(f"XGBoost Test ROC-AUC  : {test_auc:.4f}")

    paths = get_project_paths()
    json_path = os.path.join(paths['models'], 'xgboost.json')
    xgb_model.save_model(json_path)
    print(f"[utils] Native XGBoost model saved to {json_path}")
    save_object(xgb_model, 'xgboost.pkl')

    return xgb_model


def select_and_evaluate_ensemble(rf_model, xgb_model, X_train, y_train, X_test, y_test):
    """
    Selects optimal ensemble strategy STRICTLY using 5-fold CV on X_train.
    Fits winning strategy on full X_train and evaluates once on untouched X_test.
    """
    print("\n" + "=" * 50)
    print("3. Methodological Ensemble Selection via 5-Fold Cross-Validation on X_train")
    print("=" * 50)

    # Fresh estimator pairs per candidate so CV clones do not share state
    def _make_estimators():
        return [('rf', clone(rf_model)), ('xgb', clone(xgb_model))]

    candidates = {
        'Equal-Weight Soft Voting [1, 1]': VotingClassifier(
            estimators=_make_estimators(), voting='soft', weights=[1, 1], n_jobs=-1
        ),
        'Weighted Soft Voting [1, 2]': VotingClassifier(
            estimators=_make_estimators(), voting='soft', weights=[1, 2], n_jobs=-1
        ),
        'Weighted Soft Voting [1, 3]': VotingClassifier(
            estimators=_make_estimators(), voting='soft', weights=[1, 3], n_jobs=-1
        ),
        'Stacking Classifier (LogisticRegression)': StackingClassifier(
            estimators=_make_estimators(),
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1,
        ),
    }

    cv_results = []
    print("Running 5-fold CV on X_train for candidate ensemble strategies...")

    for name, model in candidates.items():
        start_cv = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
        mean_cv = np.mean(scores)
        std_cv = np.std(scores)
        elapsed_cv = time.time() - start_cv
        print(f" -> {name:<42} | 5-Fold CV Accuracy: {mean_cv:.5f} (+/- {std_cv:.5f}) [{elapsed_cv:.1f}s]")
        cv_results.append((name, model, mean_cv, std_cv))

    # Pick winner strictly based on highest training CV accuracy
    cv_results.sort(key=lambda x: x[2], reverse=True)
    best_name, winning_model_template, best_cv_mean, best_cv_std = cv_results[0]

    print("\n" + "-" * 70)
    print(f"WINNING ENSEMBLE STRATEGY SELECTED BY TRAIN CV: '{best_name}'")
    print(f"Mean Train 5-Fold CV Accuracy: {best_cv_mean:.5f} (+/- {best_cv_std:.5f})")
    print("-" * 70)

    # Retrain single chosen strategy on full X_train
    print(f"\nRetraining winning strategy ('{best_name}') on full X_train_raw...")
    start_train = time.time()
    winning_model = candidates[best_name]
    winning_model.fit(X_train, y_train)
    elapsed_train = time.time() - start_train
    print(f"Training completed in {elapsed_train:.2f}s")

    # One-time final evaluation on untouched X_test
    print("\nPerforming single final evaluation on untouched test set (X_test_raw)...")
    y_pred = winning_model.predict(X_test)
    y_proba = winning_model.predict_proba(X_test)[:, 1]
    final_test_acc = accuracy_score(y_test, y_pred)
    final_test_f1 = f1_score(y_test, y_pred)
    final_test_auc = roc_auc_score(y_test, y_proba)

    print("\n=== FINAL ENSEMBLE TEST SET RESULTS ===")
    print(f"Selected Model   : {best_name}")
    print(f"Test Accuracy    : {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"Test F1-Score    : {final_test_f1:.4f}")
    print(f"Test ROC-AUC     : {final_test_auc:.4f}")

    # Overwrite models/voting_ensemble.pkl with winning retrained model
    save_object(winning_model, 'voting_ensemble.pkl')

    return cv_results, best_name, winning_model, final_test_acc, final_test_f1, final_test_auc


def _fresh_rf_template():
    """Unfitted RF with the same hyperparameters used for the saved component."""
    return RandomForestClassifier(
        n_estimators=250, max_depth=25, n_jobs=-1, random_state=42
    )


def _fresh_xgb_template():
    """Unfitted XGBoost with the same hyperparameters used for the saved component."""
    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )


def main(skip_component_training=True):
    print("Loading preprocessed dataset (raw features for tree ensembles)...")
    X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(save_scaler=False)

    if skip_component_training:
        # RF / XGBoost already trained & saved in a prior pass — reuse hyperparams via fresh templates
        print("Skipping RF/XGBoost retraining (artifacts already saved). Using fresh templates for ensemble CV.")
        rf_model = _fresh_rf_template()
        xgb_model = _fresh_xgb_template()
    else:
        rf_model = train_random_forest(X_train_raw, y_train, X_test_raw, y_test)
        xgb_model = train_xgboost(X_train_raw, y_train, X_test_raw, y_test)

    # Methodologically sound ensemble selection via train CV only
    cv_results, best_name, winning_model, test_acc, test_f1, test_auc = select_and_evaluate_ensemble(
        rf_model, xgb_model, X_train_raw, y_train, X_test_raw, y_test
    )

    print("\n" + "=" * 60)
    print("Stage 3 Methodological Ensemble Selection & Finalization Complete!")
    print(f"Selected Strategy: {best_name}")
    for name, _, mean_cv, std_cv in cv_results:
        if name == best_name:
            print(f"Train CV Accuracy : {mean_cv:.5f} (+/- {std_cv:.5f})")
            break
    print(f"Test Accuracy     : {test_acc:.4f}")
    print(f"Test F1-Score     : {test_f1:.4f}")
    print(f"Test ROC-AUC      : {test_auc:.4f}")
    print(f"Saved artifact to models/voting_ensemble.pkl")
    print("=" * 60)


if __name__ == '__main__':
    main(skip_component_training=True)
