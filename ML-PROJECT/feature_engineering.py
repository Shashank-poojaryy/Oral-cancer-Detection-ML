import csv
import cv2, os, numpy as np, joblib
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import hog, local_binary_pattern
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import warnings
import sys

# Ensure UTF-8 printing for console formatting
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

CANCER_PATH    = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed\CANCER")
NONCANCER_PATH = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed\NON_CANCER")
SAVE_PATH      = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed")
PLOTS_PATH     = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\plots")
CSV_PATH       = SAVE_PATH / "features_dataset.csv"

# STEP 1 — IMAGE LOADING FUNCTION
def load_image_gray(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize to float32 [0,1] for feature extraction
    return gray.astype(np.float32) / 255.0

# STEP 2 — FEATURE EXTRACTION FUNCTIONS
def extract_glcm(img_float):
    img_uint8 = (img_float * 255).astype(np.uint8)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1, 2, 3]
    glcm = graycomatrix(img_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    features = []
    for prop in props:
        result = graycoprops(glcm, prop)  # shape: (3,4)
        features.extend(result.flatten())  # 12 values per prop
    return np.array(features, dtype=np.float32)  # 72 total

def extract_hog(img_float):
    features = hog(img_float,
                   orientations=9,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys',
                   feature_vector=True)
    return features.astype(np.float32)

def extract_lbp(img_float):
    img_uint8 = (img_float * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26), density=True)
    return hist.astype(np.float32)

def extract_stats(img_float):
    flat = img_float.ravel()
    return np.array([
        np.mean(flat),
        np.std(flat),
        skew(flat),
        kurtosis(flat),
        shannon_entropy(img_float)
    ], dtype=np.float32)

def extract_all(img_float):
    glcm_f  = extract_glcm(img_float)
    hog_f   = extract_hog(img_float)
    lbp_f   = extract_lbp(img_float)
    stats_f = extract_stats(img_float)
    return np.concatenate([glcm_f, hog_f, lbp_f, stats_f]), glcm_f.size, hog_f.size, lbp_f.size, stats_f.size

# STEP 3 — RUN EXTRACTION ON ALL 1000 IMAGES
X = []
y = []
failed = 0
sizes = None

print("Process CANCER folder (label=1):")
for i, fpath in enumerate(sorted(CANCER_PATH.glob("*.jpg"))):
    img = load_image_gray(fpath)
    if img is None:
        failed += 1
        continue
    features, s_g, s_h, s_l, s_s = extract_all(img)
    if sizes is None: sizes = (s_g, s_h, s_l, s_s)
    X.append(features)
    y.append(1)
    if (i+1) % 100 == 0:
        print(f"  CANCER: {i+1}/500 processed...")

print("Process NON_CANCER folder (label=0):")
for i, fpath in enumerate(sorted(NONCANCER_PATH.glob("*.jpg"))):
    img = load_image_gray(fpath)
    if img is None:
        failed += 1
        continue
    features, _, _, _, _ = extract_all(img)
    X.append(features)
    y.append(0)
    if (i+1) % 100 == 0:
        print(f"  NON-CANCER: {i+1}/500 processed...")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
c_samples = np.sum(y == 1)
nc_samples = np.sum(y == 0)

# STEP 4 — RECORD FEATURE GROUP INDICES
glcm_size, hog_size, lbp_size, stats_size = sizes
glcm_start  = 0
glcm_end    = glcm_size
hog_start   = glcm_end
hog_end     = glcm_end + hog_size
lbp_start   = hog_end
lbp_end     = hog_end + lbp_size
stats_start = lbp_end
stats_end   = lbp_end + stats_size

feature_groups = {
    'GLCM':  (glcm_start,  glcm_end),
    'HOG':   (hog_start,   hog_end),
    'LBP':   (lbp_start,   lbp_end),
    'STATS': (stats_start, stats_end)
}
print("Feature Groups:", feature_groups)

# STEP 5 — DATA CLEANING
raw_total = X.shape[1]

# Check 1: NaN values
nan_count = np.isnan(X).sum()
print(f"NaN values found: {nan_count}")
X = np.nan_to_num(X, nan=0.0)

# Check 2: Inf values
inf_count = np.isinf(X).sum()
print(f"Inf values found: {inf_count}")
X = np.nan_to_num(X, posinf=0.0, neginf=0.0)

# Check 3: Zero variance features
selector = VarianceThreshold(threshold=0.0)
X_clean = selector.fit_transform(X)
removed = X.shape[1] - X_clean.shape[1]
print(f"Zero-variance features removed: {removed}")
X = X_clean
final_count = X.shape[1]

# STEP 6 — FEATURE DISTRIBUTION PLOT
c_means = np.mean(X[y == 1], axis=0) if c_samples > 0 else np.zeros(final_count)
nc_means = np.mean(X[y == 0], axis=0) if nc_samples > 0 else np.zeros(final_count)

plt.figure(figsize=(12, 6))
plt.plot(c_means, label='Cancer Mean', color='tomato', alpha=0.8)
plt.plot(nc_means, label='Non-Cancer Mean', color='steelblue', alpha=0.8)

for name, (start, end) in feature_groups.items():
    if start < final_count:
        plt.axvline(x=start, color='black', linestyle='--', alpha=0.5)
        # Shift slightly right of the line for text
        # Y position could be high inside plot bounds
        max_val = max(np.max(c_means), np.max(nc_means)) if (c_samples > 0 or nc_samples > 0) else 1
        plt.text(start + (end-start)*0.02 if (end-start) > 0 else start, max_val * 0.9, name, rotation=90, verticalalignment='center')

plt.title("Mean Feature Values per Class")
plt.xlabel("Feature Index")
plt.ylabel("Mean Value")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_PATH / "10_feature_distributions.png", dpi=150)
plt.close()

# STEP 7 — SAVE EVERYTHING
np.save(SAVE_PATH / 'X_features.npy', X)
np.save(SAVE_PATH / 'y_labels.npy', y)
joblib.dump(feature_groups, SAVE_PATH / 'feature_groups.pkl')

# Save extracted features to CSV for tabular inspection/export
csv_header = [f"feature_{i+1}" for i in range(X.shape[1])] + ["label", "label_name"]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    for i in range(X.shape[0]):
        label_int = int(y[i])
        label_name = "cancer" if label_int == 1 else "no_cancer"
        writer.writerow(X[i].tolist() + [label_int, label_name])

print("✓ X_features.npy saved")
print("✓ y_labels.npy saved")
print("✓ feature_groups.pkl saved")
print(f"✓ {CSV_PATH.name} saved")

# STEP 8 — FINAL REPORT
print(f"""
═══════════════════════════════════════════
       FEATURE EXTRACTION COMPLETE
═══════════════════════════════════════════
GLCM features        : {glcm_size}
HOG features         : {hog_size}
LBP features         : {lbp_size}
Statistical features : {stats_size}
───────────────────────────────────────────
Raw feature total    : {raw_total}
NaN values fixed     : {nan_count}
Inf values fixed     : {inf_count}
Zero-variance removed: {removed}
Final feature count  : {final_count}
───────────────────────────────────────────
Feature matrix shape : ({X.shape[0]} samples, {final_count} features)
CANCER samples       : {c_samples}
NON-CANCER samples   : {nc_samples}
Failed to load       : {failed}
───────────────────────────────────────────
Feature group indices:
  GLCM  : [{glcm_start}  → {glcm_end}]
  HOG   : [{hog_start} → {hog_end}]
  LBP   : [{lbp_start}  → {lbp_end}]
  STATS : [{stats_start}  → {stats_end}]
═══════════════════════════════════════════
✓ Features saved. Ready for ML preparation.
""")
