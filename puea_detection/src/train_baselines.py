"""
Baseline Models Training Script for PUEA Detection Framework.
Trains KNN, SVM, and ANN models on SCALED features and saves artifacts.
"""

import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import save_object
from preprocessing import load_and_preprocess_data


def train_knn(X_train, y_train, X_test, y_test):
    """
    Tunes n_neighbors for KNN using a validation split, trains best KNN model.
    """
    print("\n" + "=" * 50)
    print("1. Training Baseline: K-Nearest Neighbors (KNN)")
    print("=" * 50)

    # Stratified validation split for quick hyperparameter selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    candidate_k = [3, 5, 7, 9, 11]
    best_k = 5
    best_val_score = 0.0

    print("Tuning n_neighbors across [3, 5, 7, 9, 11]...")
    for k in candidate_k:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, knn.predict(X_val))
        print(f" -> k={k:2d}: Validation Accuracy = {val_acc:.5f}")
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_k = k

    print(f"Optimal n_neighbors selected: k={best_k}")

    # Retrain on full X_train with optimal k
    start_time = time.time()
    best_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    best_knn.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # Evaluate on test set
    y_pred = best_knn.predict(X_test)
    y_proba = best_knn.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"KNN Training completed in {elapsed:.2f}s")
    print(f"KNN Test Accuracy : {test_acc:.4f}")
    print(f"KNN Test F1-Score : {test_f1:.4f}")
    print(f"KNN Test ROC-AUC  : {test_auc:.4f}")

    save_object(best_knn, 'knn.pkl')
    return best_knn


def train_svm(X_train, y_train, X_test, y_test):
    """
    Compares RBF vs Linear kernels, trains final SVC with probability=True.
    """
    print("\n" + "=" * 50)
    print("2. Training Baseline: Support Vector Machine (SVM)")
    print("=" * 50)

    # Subsample for fast kernel comparison to avoid bottleneck
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=15000, random_state=42, stratify=y_train
    )

    kernels = ['rbf', 'linear']
    best_kernel = 'rbf'
    best_acc = 0.0

    print("Comparing SVM kernels on 15,000 stratified samples...")
    for kernel in kernels:
        svm_temp = SVC(kernel=kernel, random_state=42)
        svm_temp.fit(X_sub, y_sub)
        acc = accuracy_score(y_test, svm_temp.predict(X_test))
        print(f" -> Kernel '{kernel}': Holdout Accuracy = {acc:.5f}")
        if acc > best_acc:
            best_acc = acc
            best_kernel = kernel

    print(f"Selected SVM Kernel: '{best_kernel}'")

    # Train final SVM model on 25,000 stratified sample for efficient probability calibration & scaling
    train_size_svm = min(30000, len(X_train))
    X_tr_svm, _, y_tr_svm, _ = train_test_split(
        X_train, y_train, train_size=train_size_svm, random_state=42, stratify=y_train
    )

    print(f"Training final SVC(kernel='{best_kernel}', probability=True) on {train_size_svm} samples...")
    start_time = time.time()
    final_svm = SVC(kernel=best_kernel, probability=True, random_state=42)
    final_svm.fit(X_tr_svm, y_tr_svm)
    elapsed = time.time() - start_time

    # Evaluate on test set
    y_pred = final_svm.predict(X_test)
    y_proba = final_svm.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"SVM Training completed in {elapsed:.2f}s")
    print(f"SVM Test Accuracy : {test_acc:.4f}")
    print(f"SVM Test F1-Score : {test_f1:.4f}")
    print(f"SVM Test ROC-AUC  : {test_auc:.4f}")

    save_object(final_svm, 'svm.pkl')
    return final_svm


def train_ann(X_train, y_train, X_test, y_test):
    """
    Trains Artificial Neural Network (MLPClassifier) with hidden_layer_sizes=(64, 32).
    """
    print("\n" + "=" * 50)
    print("3. Training Baseline: Artificial Neural Network (ANN / MLP)")
    print("=" * 50)

    start_time = time.time()
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42
    )

    print("Fitting MLPClassifier on X_train_scaled...")
    ann.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # Evaluate on test set
    y_pred = ann.predict(X_test)
    y_proba = ann.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"ANN Training completed in {elapsed:.2f}s (Iterations: {ann.n_iter_})")
    print(f"ANN Test Accuracy : {test_acc:.4f}")
    print(f"ANN Test F1-Score : {test_f1:.4f}")
    print(f"ANN Test ROC-AUC  : {test_auc:.4f}")

    save_object(ann, 'ann.pkl')
    return ann


def main():
    print("Loading preprocessed dataset (scaled features for baseline models)...")
    X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(save_scaler=False)

    # 1. Train KNN
    knn_model = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)

    # 2. Train SVM
    svm_model = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)

    # 3. Train ANN
    ann_model = train_ann(X_train_scaled, y_train, X_test_scaled, y_test)

    print("\n" + "=" * 60)
    print("Stage 2 Baseline Training Finished Successfully!")
    print("All baseline model artifacts saved in models/ (knn.pkl, svm.pkl, ann.pkl)")
    print("=" * 60)


if __name__ == '__main__':
    main()
