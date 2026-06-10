import numpy as np
import shap
import joblib

def _safe_proba(model, X):
    X = np.atleast_2d(np.array(X, dtype=np.float64))
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    return proba.astype(np.float64)

# Load a dummy model from the directory
model = joblib.load(Path(__file__).parent / "models" / "svm.pkl")
from pathlib import Path
X_bg = np.random.rand(10, 287) # Assuming 287 features from PCA
features_pca = np.random.rand(1, 287)

# Pre-cache Background
exp = shap.KernelExplainer(lambda X: _safe_proba(model, X), shap.kmeans(X_bg, 5))
sv = exp.shap_values(features_pca, nsamples=100, l1_reg='num_features(10)', silent=True)
ev = exp.expected_value

print("sv type:", type(sv))
if isinstance(sv, list):
    print("sv length:", len(sv))
else:
    print("sv shape:", np.array(sv).shape)

print("ev type:", type(ev))
if hasattr(ev, "__len__"):
    print("ev length/shape:", np.array(ev).shape)
print("ev value:", ev)
