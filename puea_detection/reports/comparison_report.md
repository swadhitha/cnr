# PUEA Detection — Model Comparison Report

All metrics computed **once** on the untouched test set (20,000 samples).

| Model | Accuracy | Precision | Recall | F1 | False Alarm Rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| KNN | 0.8872 | 0.9643 | 0.8042 | 0.8770 | 0.0298 | 0.9356 |
| SVM | 0.9094 | 0.9582 | 0.8563 | 0.9044 | 0.0374 | 0.9606 |
| ANN | 0.9125 | 0.9504 | 0.8704 | 0.9087 | 0.0454 | 0.9623 |
| Random Forest | 0.9113 | 0.9538 | 0.8646 | 0.9070 | 0.0419 | 0.9656 |
| XGBoost | 0.9216 | 0.9600 | 0.8799 | 0.9182 | 0.0367 | 0.9704 |
| Ensemble (Stacking) | 0.9204 | 0.9479 | 0.8897 | 0.9179 | 0.0489 | 0.9704 |

## Highlights

- **Highest Accuracy:** XGBoost (0.9216)
- **Highest F1:** XGBoost (0.9182)
- **Highest ROC-AUC:** Ensemble (Stacking) (0.9704)
- **Lowest False Alarm Rate:** KNN (0.0298)

## Notes

- KNN, SVM, and ANN were evaluated on **scaled** features.
- Random Forest, XGBoost, and the Ensemble were evaluated on **raw** features.
- The Ensemble is a StackingClassifier (RF + XGBoost → LogisticRegression),
  selected via 5-fold CV accuracy on the training set only.
