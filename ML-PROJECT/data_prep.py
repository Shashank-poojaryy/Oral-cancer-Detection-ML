import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROC   = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed")
MODELS = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\models")
PLOTS  = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\plots")
MODELS.mkdir(parents=True, exist_ok=True)

# LOAD DATA
X = np.load(PROC / 'X_features.npy')
y = np.load(PROC / 'y_labels.npy')
print(f"Loaded X: {X.shape}, y: {y.shape}")
print(f"Class distribution: {Counter(y)}")

# STEP 1 — STRATIFIED SPLIT (75% train / 25% test)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

print("\nSTEP 1 — STRATIFIED SPLIT:")
print(f"  X_train: {X_train.shape} | y_train: {Counter(y_train)}")
print(f"  X_test : {X_test.shape}  | y_test : {Counter(y_test)}")

# STEP 2 — SMOTE (Training Only)
print("\nSTEP 2 — SMOTE:")
print(f"  Before SMOTE: {Counter(y_train)}")

smote = SMOTE(k_neighbors=5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"  After  SMOTE: {Counter(y_train)}")
print(f"  X_train shape after SMOTE: {X_train.shape}")

# STEP 3 — STANDARD SCALING
print("\nSTEP 3 — STANDARD SCALING:")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     # same scaler, no refit
joblib.dump(scaler, MODELS / 'scaler.pkl')

print(f"  X_train mean (should be ~0): {X_train.mean():.6f}")
print(f"  X_train std  (should be ~1): {X_train.std():.6f}")
print(f"  Scaler saved ✓")

# STEP 4 — VARIANCE THRESHOLD
print("\nSTEP 4 — VARIANCE THRESHOLD:")
vt = VarianceThreshold(threshold=0.01)
X_train = vt.fit_transform(X_train)
X_test  = vt.transform(X_test)
joblib.dump(vt, MODELS / 'variance_threshold.pkl')

removed = 1867 - X_train.shape[1]
print(f"  Features before : 1867")
print(f"  Features removed: {removed}")
print(f"  Features after  : {X_train.shape[1]}")

# STEP 5 — PCA (95% variance)
print("\nSTEP 5 — PCA:")
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)
joblib.dump(pca, MODELS / 'pca.pkl')

print(f"  Features before PCA : {vt.get_support().sum()}")
print(f"  Features after  PCA : {X_train.shape[1]}")
print(f"  Variance retained   : {sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"  PCA saved ✓")

# Plot explained variance curve
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
plt.figure(figsize=(10,5))
plt.plot(cumvar, color='steelblue', linewidth=2)
plt.axhline(y=95, color='red', linestyle='--', label='95% threshold')
plt.axvline(x=X_train.shape[1], color='green', linestyle='--', label=f'{X_train.shape[1]} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA — Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS / '11_pca_variance.png', dpi=150)
plt.close()
print(f"  PCA variance plot saved ✓")

# STEP 6 — SAVE ALL SPLITS
np.save(PROC / 'X_train.npy', X_train)
np.save(PROC / 'X_test.npy',  X_test)
np.save(PROC / 'y_train.npy', y_train)
np.save(PROC / 'y_test.npy',  y_test)

print("\nAll splits saved ✓")
print(f"  X_train.npy : {X_train.shape}")
print(f"  X_test.npy  : {X_test.shape}")
print(f"  y_train.npy : {y_train.shape}")
print(f"  y_test.npy  : {y_test.shape}")

# STEP 7 — FINAL REPORT
print(f"""
═══════════════════════════════════════════
       DATA PREPARATION COMPLETE
═══════════════════════════════════════════
SPLIT RESULTS:
  X_train shape        : {X_train.shape}
  X_test  shape        : {X_test.shape}
  y_train CANCER       : {Counter(y_train)[1]}
  y_train NON-CANCER   : {Counter(y_train)[0]}
  y_test  CANCER       : {Counter(y_test)[1]}
  y_test  NON-CANCER   : {Counter(y_test)[0]}
───────────────────────────────────────────
FEATURE REDUCTION:
  Raw features         : 1867
  After Var Threshold  : {vt.get_support().sum()}
  After PCA (95% var)  : {X_train.shape[1]}
  Reduction ratio      : {((1 - X_train.shape[1]/1867)*100):.1f}% smaller
───────────────────────────────────────────
SAVED ARTIFACTS:
  scaler.pkl           : ✓
  variance_threshold.pkl: ✓
  pca.pkl              : ✓
  X_train.npy          : ✓
  X_test.npy           : ✓
  y_train.npy          : ✓
  y_test.npy           : ✓
═══════════════════════════════════════════
✓ Ready for model training
""")
