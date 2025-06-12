# models.py
import numpy as np
from xgboost import XGBClassifier
import shap
import config

def train_xgboost(X_train, y_train):
    """Trains a standard XGBoost classifier."""
    print("Training XGBoost model...")
    xgb_model = XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def get_shap_importance(xgb_model, X_test):
    """Calculates SHAP feature importance for the trained XGBoost model."""
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values can be a list of two arrays.
    # We are interested in the importance for the "positive" class (class 1).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    return np.mean(np.abs(shap_values), axis=0)