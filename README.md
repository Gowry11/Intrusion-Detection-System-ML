# Explainable & Optimized Hybrid Intrusion Detection System

A production-style ML project for network intrusion detection using the NSL-KDD dataset. Supports binary (normal vs attack) and multi-class (DoS, Probe, U2R, R2L, Normal) classification.

## Project Structure

```
IDS_Project/
├── data/              # Place NSL-KDD CSV files here
├── models/            # Saved trained models
├── outputs/
│   ├── plots/         # Confusion matrices, ROC curves, SHAP plots
│   └── reports/       # Evaluation reports, comparison tables
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── stacking_model.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── robustness.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset

1. Download NSL-KDD (CSV/TXT) from:
   - Kaggle: https://www.kaggle.com/datasets/hassan06/nslkdd
   - GitHub: https://github.com/Jehuty4949/NSL_KDD or https://github.com/jmnwong/NSL-KDD-Dataset
2. Place in `data/` folder:
   - `KDDTrain+.csv` or `KDDTrain+.txt`
   - `KDDTest+.csv` or `KDDTest+.txt`

## Setup

```bash
cd IDS_Project
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Pipeline

1. **Preprocessing**: Load data, encode categoricals, scale, stratified 80-20 split  
2. **Feature Engineering**: Low variance removal, correlation filter, SelectKBest (mutual info)  
3. **SMOTE**: Class imbalance handling (training set only)  
4. **Base Models**: LR, Random Forest, XGBoost, LightGBM, SVM  
5. **Hyperparameter Tuning**: RandomizedSearchCV for RF, XGBoost, LightGBM  
6. **Stacking**: Tuned RF + Tuned XGBoost → Meta Logistic Regression  
7. **Evaluation**: Confusion matrix, ROC curve, class-wise F1, accuracy comparison  
8. **Explainability**: SHAP summary, feature importance, force plot  
9. **Robustness**: Gaussian noise test, performance drop analysis  

## Outputs

- **Plots**: `outputs/plots/` — confusion matrices, ROC curves, SHAP plots  
- **Reports**: `outputs/reports/` — accuracy comparison, robustness table  
- **Models**: `models/` — saved `.joblib` models  

## Target Metrics

- Final accuracy: 95%+  
- Class-wise recall for U2R and R2L  
- Accuracy comparison table  
- Robustness comparison  
