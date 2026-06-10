import os
import sys
import pandas as pd
import warnings
from pathlib import Path
from PIL import Image

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TERMINAL DISPLAY LOGIC (Run once on startup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_terminal_comparison():
    if os.environ.get("COMPARISON_DISPLAYED") == "1":
        return
    try:
        csv_path = Path(__file__).parent / "outputs" / "model_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Map Specificity (True Negative Rate) extracted from the full classification reports
            spec_map = {
                'SVM (Tuned)': 83.00,
                'Voting Ensemble': 90.00,
                'Logistic Regression': 88.00,
                'Decision Tree': 72.00,
                'Random Forest (Tuned)': 82.00,
                'Naive Bayes': 82.00,
                'KNN (k=9)': 87.00
            }
            df['Specificity'] = df['Model'].map(lambda x: spec_map.get(x, 0.0))
            
            # Sort by Recall descending to find the best cancer detection model
            df = df.sort_values(by='Recall', ascending=False).reset_index(drop=True)
            
            print("\n╔" + "═"*92 + "╗")
            print(f"║{'FINAL MODEL COMPARISON - Sorted by Cancer Recall':^92}║")
            print("╠═════╦═══════════════════════════╦══════════╦═══════════╦══════════╦═════════════╦══════════╣")
            print("║  #  ║ Model                     ║ Accuracy ║ Precision ║ Recall   ║ Specificity ║    F1    ║")
            print("╠═════╬═══════════════════════════╬══════════╬═══════════╬══════════╬═════════════╬══════════╣")
            for i, row in df.iterrows():
                name = str(row['Model'])[:25]
                print(f"║ {i+1:<3} ║ {name:<25} ║ {row['Accuracy']:>7.2f}% ║ {row['Precision']:>8.2f}% ║ {row['Recall']:>7.2f}% ║ {row['Specificity']:>10.2f}% ║ {row['F1_Score']:>7.2f}% ║")
            print("╚═════╩═══════════════════════════╩══════════╩═══════════╩══════════╩═════════════╩══════════╝")
            print(f"✓ Comparison table saved → outputs/model_comparison.csv")
            
            best = df.iloc[0]
            print(f"\n🏆 Best Model : {best['Model']}")
            print(f"   Accuracy    : {best['Accuracy']}%")
            print(f"   Precision   : {best['Precision']}%")
            print(f"   Recall      : {best['Recall']}%")
            print(f"   Specificity : {best['Specificity']}%")
            print(f"   F1 Score    : {best['F1_Score']}%")
            print(f"\nMeaning: {best['Recall']/10:.1f} out of 10 cancer cases correctly identified by this model.")
            print("═"*94 + "\n")
            
            # --- ASCII Confusion Matrices ---
            print("╔" + "═"*92 + "╗")
            print(f"║{'CONFUSION MATRICES (Test Support: 100 Each)':^92}║")
            print("╠" + "═"*45 + "╦" + "═"*46 + "╣")
            
            for index in range(0, len(df), 2):
                row1 = df.iloc[index]
                name1 = str(row1['Model'])[:25]
                tn1, fp1 = int(row1['Specificity']), int(100 - row1['Specificity'])
                fn1, tp1 = int(100 - row1['Recall']), int(row1['Recall'])
                
                if index + 1 < len(df):
                    row2 = df.iloc[index + 1]
                    name2 = str(row2['Model'])[:25]
                    tn2, fp2 = int(row2['Specificity']), int(100 - row2['Specificity'])
                    fn2, tp2 = int(100 - row2['Recall']), int(row2['Recall'])
                else:
                    row2 = None
                    name2 = ""
                    tn2, fp2 = 0, 0
                    fn2, tp2 = 0, 0
                
                col1 = f"{name1:^45}"
                col2 = f"{name2:^46}" if row2 is not None else " " * 46
                
                h1 = f"{'Pred NC   Pred C':^45}"
                h2 = f"{'Pred NC   Pred C':^46}" if row2 is not None else " " * 46
                
                l3_c1 = f"NC  [ TN:{tn1:<2}  ,  FP:{fp1:<2} ]"
                l3_c2 = f"NC  [ TN:{tn2:<2}  ,  FP:{fp2:<2} ]" if row2 is not None else ""
                s3_1 = f"{l3_c1:^45}"
                s3_2 = f"{l3_c2:^46}" if row2 is not None else " " * 46
                
                l4_c1 = f"C   [ FN:{fn1:<2}  ,  TP:{tp1:<2} ]"
                l4_c2 = f"C   [ FN:{fn2:<2}  ,  TP:{tp2:<2} ]" if row2 is not None else ""
                s4_1 = f"{l4_c1:^45}"
                s4_2 = f"{l4_c2:^46}" if row2 is not None else " " * 46
                
                print(f"║{col1}│{col2}║")
                print(f"║{h1}│{h2}║")
                print(f"║{s3_1}│{s3_2}║")
                print(f"║{s4_1}│{s4_2}║")
                
                if index + 2 < len(df):
                    print("╠" + "═"*45 + "┼" + "═"*46 + "╣")
                else:
                    print("╚" + "═"*45 + "╧" + "═"*46 + "╝")
                    
            print(f"✓ Matrix images correctly labeled TN/FP/FN/TP saved → outputs/plots/\n", flush=True)

            # --- Save the plots dynamically ---
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plots_dir = Path(__file__).parent / "outputs" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            for i, row in df.iterrows():
                name_model = str(row['Model'])
                safe_name = name_model.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("⭐", "").strip()
                cm_path = plots_dir / f"cm_{safe_name}.png"
                
                tn_v, fp_v = int(row['Specificity']), int(100 - row['Specificity'])
                fn_v, tp_v = int(100 - row['Recall']), int(row['Recall'])
                cm = np.array([[tn_v, fp_v], [fn_v, tp_v]])
                
                fig, ax = plt.subplots(figsize=(5, 4))
                cax = ax.matshow(cm, cmap='Blues', alpha=0.9)
                plt.colorbar(cax)
                for (m, x), z in np.ndenumerate(cm):
                    f_col = 'white' if z > 40 else 'black'
                    if m == 0 and x == 0: lbl = "TN"
                    elif m == 0 and x == 1: lbl = "FP"
                    elif m == 1 and x == 0: lbl = "FN"
                    else: lbl = "TP"
                    ax.text(x, m, f"{lbl}\n{z}", ha='center', va='center', color=f_col, fontweight='bold', fontsize=12)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Non-Cancer', 'Cancer'], fontsize=10)
                ax.set_yticklabels(['Non-Cancer', 'Cancer'], fontsize=10)
                ax.xaxis.set_ticks_position('bottom')
                plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
                plt.ylabel('True Label', fontsize=11, fontweight='bold')
                plt.title(f'{name_model}\n', pad=10, fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(cm_path, dpi=120, bbox_inches='tight')
                plt.close(fig)

            os.environ["COMPARISON_DISPLAYED"] = "1"
    except Exception:
        pass


def print_terminal_kfold_summary():
    """
    Reads the K-Fold results from CSV and displays a premium styled table in the terminal.
    """
    try:
        csv_path = Path(__file__).parent / "outputs" / "kfold_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Sort by Recall Mean descending (Primary metric)
            df = df.sort_values(by='Recall Mean %', ascending=False).reset_index(drop=True)
            
            print("\n╔" + "═"*92 + "╗")
            print(f"║{'K-FOLD CROSS-VALIDATION SUMMARY (Robustness Assessment)':^92}║")
            print("╠═════╦═══════════════════════════╦══════════════════════╦══════════════════════╣")
            print("║  #  ║ Model                     ║ Accuracy (Mean ±Std) ║ Recall (Mean ±Std)   ║")
            print("╠═════╬═══════════════════════════╬══════════════════════╬══════════════════════╣")
            for i, row in df.iterrows():
                name = str(row['Model'])[:25]
                acc_str = f"{row['Acc Mean %']:>6.2f}% ±{row['Acc Std']:>5.2f}"
                rec_str = f"{row['Recall Mean %']:>6.2f}% ±{row['Recall Std']:>5.2f}"
                print(f"║ {i+1:<3} ║ {name:<25} ║ {acc_str:^20} ║ {rec_str:^20} ║")
            print("╚═════╩═══════════════════════════╩══════════════════════╩══════════════════════╝")
            print(f"✓ K-Fold summary loaded from outputs/kfold_comparison.csv\n")
    except Exception:
        pass

# Execute terminal prints
print_terminal_comparison()
print_terminal_kfold_summary()

import streamlit as st
import cv2
import numpy as np
import joblib
import shap
import streamlit as st

# Initialize session state for batch results
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = []
if 'active_shap_index' not in st.session_state:
    st.session_state['active_shap_index'] = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
warnings.filterwarnings('ignore')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Oral Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — CUSTOM CSS STYLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    .result-cancer {
        background: linear-gradient(135deg, #2d1b1b, #1c2333);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .result-normal {
        background: linear-gradient(135deg, #1b2d1b, #1c2333);
        border: 2px solid #44ff44;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #1c2333;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        color: white;
    }
    .stApp { background-color: #0e1117; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed steelblue;
        border-radius: 8px;
        padding: 10px;
    }
    h1, h2, h3 { color: white; font-weight: bold; }
    .shap-header {
        background: linear-gradient(90deg, #1a1f35, #1c2333);
        border-left: 4px solid #4f8ef7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .shap-intro-box {
        background: #141926;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 14px 18px;
        color: #a0b4d6;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 10px;
    }
    .shap-model-agree {
        background: #111827;
        border: 1px solid #1f2d45;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #c9d8f0;
        margin-top: 8px;
    }
    .clinical-note {
        background: #12191f;
        border: 1px solid #1e3a2f;
        border-left: 4px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        color: #a7c8b4;
        font-size: 0.88rem;
        line-height: 1.7;
        margin-top: 10px;
    }
    .clinical-note-cancer {
        background: #1f1212;
        border: 1px solid #3a1e1e;
        border-left: 4px solid #ef4444;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        color: #c8a7a7;
        font-size: 0.88rem;
        line-height: 1.7;
        margin-top: 10px;
    }
    .shap-status-box {
        background: #1a1f35;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 10px 14px;
        color: #a0b4d6;
        font-size: 0.85rem;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — PATHS  (update if needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELS_DIR = Path(__file__).parent / "outputs" / "models"
PROC_DIR   = Path(__file__).parent / "data" / "processed"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — MODEL LOADING (Cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_pipeline():
    try:
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        vt     = joblib.load(MODELS_DIR / 'variance_threshold.pkl')
        pca    = joblib.load(MODELS_DIR / 'pca.pkl')
        return scaler, vt, pca
    except FileNotFoundError as e:
        st.error(f"Pipeline file missing: {e}")
        return None, None, None

@st.cache_resource
def load_models():
    try:
        return {
            'SVM (Tuned) ⭐'      : joblib.load(MODELS_DIR / 'svm.pkl'),
            'Voting Ensemble'     : joblib.load(MODELS_DIR / 'voting_ensemble.pkl'),
            'Random Forest'       : joblib.load(MODELS_DIR / 'random_forest.pkl'),
            'Logistic Regression' : joblib.load(MODELS_DIR / 'logistic_regression.pkl'),
            'Decision Tree'       : joblib.load(MODELS_DIR / 'decision_tree.pkl'),
            'KNN (k=9)'           : joblib.load(MODELS_DIR / 'knn.pkl'),
            'Naive Bayes'         : joblib.load(MODELS_DIR / 'naive_bayes.pkl'),
        }
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}")
        return {}

@st.cache_resource
def load_shap_background():
    """
    Load a balanced 100-sample background set from X_test.npy.
    This is used as the reference distribution for KernelExplainer
    and LinearExplainer. Falls back gracefully if files are missing.
    """
    try:
        X_test = np.load(PROC_DIR / 'X_test.npy')
        y_test = np.load(PROC_DIR / 'y_test.npy')
        np.random.seed(42)
        c_idx  = np.where(y_test == 1)[0]
        nc_idx = np.where(y_test == 0)[0]
        bg_idx = np.concatenate([
            np.random.choice(c_idx,  min(50, len(c_idx)),  replace=False),
            np.random.choice(nc_idx, min(50, len(nc_idx)), replace=False),
        ])
        return X_test[bg_idx]
    except Exception:
        return None

scaler, vt, pca = load_pipeline()
models          = load_models()
X_shap_bg       = load_shap_background()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — PREPROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preprocess_image(pil_image):
    try:
        img = np.array(pil_image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img        = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced   = clahe.apply(gray)
        denoised   = cv2.GaussianBlur(enhanced, (3, 3), 0)
        normalized = denoised.astype(np.float32) / 255.0
        sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
        return normalized, sharpness, enhanced
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None, 0.0, None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — FEATURE EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_features(img_float):
    try:
        img_uint8 = (img_float * 255).astype(np.uint8)
        angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm      = graycomatrix(img_uint8, distances=[1, 2, 3], angles=angles,
                                 levels=256, symmetric=True, normed=True)
        props  = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        glcm_f = []
        for prop in props:
            glcm_f.extend(graycoprops(glcm, prop).flatten())
        glcm_f = np.array(glcm_f, dtype=np.float32)

        hog_f = hog(img_float, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), block_norm='L2-Hys',
                    feature_vector=True).astype(np.float32)

        lbp         = local_binary_pattern(img_uint8, P=24, R=3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        lbp_f       = lbp_hist.astype(np.float32)

        flat    = img_float.ravel()
        stats_f = np.array([
            np.mean(flat), np.std(flat),
            float(skew(flat)), float(kurtosis(flat)),
            float(shannon_entropy(img_float))
        ], dtype=np.float32)

        features = np.concatenate([glcm_f, hog_f, lbp_f, stats_f])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features, len(glcm_f), len(hog_f), len(lbp_f), len(stats_f)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None, 0, 0, 0, 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — PREDICTION PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_prediction(pil_image, model, scaler, vt, pca):
    processed, sharpness, enhanced = preprocess_image(pil_image)
    if processed is None:
        return None

    feat_result = extract_features(processed)
    if feat_result[0] is None:
        return None

    features, n_glcm, n_hog, n_lbp, n_stats = feat_result
    features        = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_vt     = vt.transform(features_scaled)
    features_pca    = pca.transform(features_vt)

    prediction = model.predict(features_pca)[0]

    if hasattr(model, 'predict_proba'):
        proba          = model.predict_proba(features_pca)[0]
        confidence     = float(proba[prediction]) * 100
        cancer_prob    = float(proba[1]) * 100
        noncancer_prob = float(proba[0]) * 100
    else:
        score          = float(model.decision_function(features_pca)[0])
        cancer_prob    = float(min(max(50.0 + score * 10.0, 0.0), 100.0))
        noncancer_prob = 100.0 - cancer_prob
        confidence     = cancer_prob if prediction == 1 else noncancer_prob

    return {
        'prediction'    : int(prediction),
        'label'         : 'CANCER' if prediction == 1 else 'NON-CANCER',
        'confidence'    : confidence,
        'cancer_prob'   : cancer_prob,
        'noncancer_prob': noncancer_prob,
        'processed_img' : processed,
        'enhanced_img'  : enhanced,
        'sharpness'     : sharpness,
        'features_pca'  : features_pca,
        'n_glcm'        : n_glcm,
        'n_hog'         : n_hog,
        'n_lbp'         : n_lbp,
        'n_stats'       : n_stats,
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — SHAP HELPERS (FIXED FOR ALL 7 MODELS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_sv1_and_base(sv, expected_value, want_class=1):
    """
    Safely extract the 1-D SHAP vector for the cancer class (class index 1)
    and the corresponding scalar base value from any SHAP output format.

    SHAP can return:
      - list of arrays  → [sv_class0, sv_class1]  each (n_samples, n_features)
      - 2-D array       → (n_samples, n_features)          (single-output)
      - 3-D array       → (n_samples, n_features, n_classes)

    expected_value mirrors the same pattern (list, scalar, or array).
    """
    # ── unwrap list format ────────────────────────────────────────
    if isinstance(sv, list):
        sv_out  = np.array(sv[want_class], dtype=float)
        if isinstance(expected_value, (list, np.ndarray)):
            bv = float(expected_value[want_class])
        else:
            bv = float(expected_value)
    # ── 3-D array: (samples, features, classes) ──────────────────
    elif sv.ndim == 3:
        sv_out = sv[:, :, want_class].astype(float)
        if isinstance(expected_value, (list, np.ndarray)):
            bv = float(expected_value[want_class])
        else:
            bv = float(expected_value)
    # ── 2-D array: (samples, features) ───────────────────────────
    else:
        sv_out = sv.astype(float)
        if isinstance(expected_value, (list, np.ndarray)):
            # try to pick class 1; fall back to index 0
            try:
                bv = float(expected_value[want_class])
            except (IndexError, TypeError):
                bv = float(expected_value[0])
        else:
            bv = float(expected_value)

    # We only want the single-sample row (index 0)
    if sv_out.ndim == 2:
        sv_out = sv_out[0]

    return sv_out.flatten(), bv


def get_shap_values(model, model_name, features_pca, X_bg):
    """
    Compute SHAP values for a single PCA feature vector.
    Works for all 7 trained models:
      - Random Forest       → TreeExplainer  (fast, exact)
      - Logistic Regression → LinearExplainer (fast, exact)
      - Decision Tree       → TreeExplainer  (fast, exact)
      - SVM                 → KernelExplainer (slow ~20-30 s)
      - KNN                 → KernelExplainer (slow ~20-30 s)
      - Naive Bayes         → KernelExplainer (slow ~20-30 s)
      - Voting Ensemble     → KernelExplainer (slow ~20-30 s)

    Returns: (sv1d, base_val, component_names)
      sv1d          – 1-D numpy array of SHAP values for cancer class
      base_val      – scalar float base value
      component_names – list of PC label strings
    """
    n      = features_pca.shape[1]
    cnames = [f"PC{i+1}" for i in range(n)]
    clean  = model_name.replace(' ⭐', '').strip()

    # Ensure background is always available; fall back to the query itself
    bg = X_bg if (X_bg is not None) else features_pca

    try:
        # ── TREE-BASED: Random Forest ─────────────────────────────
        if 'Random Forest' in clean:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(features_pca)
            sv1, bv = _extract_sv1_and_base(sv, exp.expected_value, want_class=1)

        # ── TREE-BASED: Decision Tree ─────────────────────────────
        elif 'Decision Tree' in clean:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(features_pca)
            sv1, bv = _extract_sv1_and_base(sv, exp.expected_value, want_class=1)

        # ── LINEAR: Logistic Regression ───────────────────────────
        elif 'Logistic Regression' in clean:
            exp = shap.LinearExplainer(
                model, bg,
                feature_perturbation='correlation_dependent'
            )
            sv  = exp.shap_values(features_pca)
            sv1, bv = _extract_sv1_and_base(sv, exp.expected_value, want_class=1)

        # ── KERNEL: SVM, KNN, Naive Bayes, Voting Ensemble ────────
        else:
            # Use predict_proba when available (SVM has probability=True,
            # KNN/NB/Voting all have it). SVM without proba falls back
            # to decision_function.
            if hasattr(model, 'predict_proba'):
                def _pred_fn(x):
                    return model.predict_proba(x)

                exp = shap.KernelExplainer(_pred_fn, bg)
                sv  = exp.shap_values(features_pca, nsamples=100)
                sv1, bv = _extract_sv1_and_base(sv, exp.expected_value, want_class=1)

            else:
                # SVM trained without probability=True
                def _pred_fn(x):
                    scores = model.decision_function(x)
                    # Return as 2-column array so we can treat col 1 as cancer
                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)
                    return np.hstack([-scores, scores])   # col0=non-cancer, col1=cancer

                exp = shap.KernelExplainer(_pred_fn, bg)
                sv  = exp.shap_values(features_pca, nsamples=100)
                sv1, bv = _extract_sv1_and_base(sv, exp.expected_value, want_class=1)

        return sv1, float(bv), cnames

    except Exception as ex:
        # Surface the real error message so it can be debugged
        st.warning(f"⚠️ SHAP computation failed for **{clean}**: `{ex}`")
        return None, None, cnames


# ── Plot helpers (unchanged from original) ──────────────────────────────────

def _dark_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2333')
    return fig, ax


def plot_waterfall(sv, base_val, cnames, top_n=15):
    idx    = np.argsort(np.abs(sv))[::-1][:top_n]
    names  = [cnames[i] for i in idx]
    vals   = sv[idx]
    run    = base_val
    starts = []
    for v in vals:
        starts.append(run)
        run += float(v)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2333')
    for i, (name, val, s) in enumerate(zip(names, vals, starts)):
        col = '#ef4444' if val > 0 else '#3b82f6'
        ax.barh(i, val, left=s, color=col, alpha=0.85,
                edgecolor='#2a3550', linewidth=0.5, height=0.65)
        offset = 0.0002 if val >= 0 else -0.0002
        ax.text(s + val + offset, i, f'{val:+.4f}',
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=8, color='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=9, color='#a0b4d6')
    ax.set_xlabel('SHAP value', color='#a0b4d6', fontsize=10)
    ax.set_title(
        f'SHAP Waterfall — top {top_n} PCA components\n'
        f'Base: {base_val:.4f}  →  f(x): {base_val + float(sv.sum()):.4f}',
        color='white', fontsize=11, fontweight='bold')
    ax.axvline(0, color='#4a5568', linewidth=0.8, linestyle='--')
    ax.tick_params(colors='#a0b4d6')
    for sp in ax.spines.values():
        sp.set_color('#2a3550')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    p1 = mpatches.Patch(color='#ef4444', label='→ CANCER')
    p2 = mpatches.Patch(color='#3b82f6', label='→ NON-CANCER')
    ax.legend(handles=[p1, p2], loc='lower right',
              facecolor='#1c2333', edgecolor='#2a3550', labelcolor='white', fontsize=8)
    plt.tight_layout()
    return fig


def plot_summary_bar(sv, cnames, top_n=20):
    idx    = np.argsort(np.abs(sv))[::-1][:top_n]
    names  = [cnames[i] for i in idx]
    absv   = np.abs(sv[idx])
    colors = ['#ef4444' if sv[i] > 0 else '#3b82f6' for i in idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2333')
    bars = ax.barh(range(top_n), absv[::-1], color=colors[::-1],
                   alpha=0.85, edgecolor='#2a3550', linewidth=0.5)
    for bar, val in zip(bars, absv[::-1]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8, color='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9, color='#a0b4d6')
    ax.set_xlabel('|SHAP value|', color='#a0b4d6', fontsize=10)
    ax.set_title(
        f'Top {top_n} PCA components by SHAP impact\n'
        'Red = cancer driver  ·  Blue = normal driver',
        color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#a0b4d6')
    for sp in ax.spines.values():
        sp.set_color('#2a3550')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_force(sv, base_val, cnames, top_n=10):
    pos_total = float(sv[sv > 0].sum())
    neg_total = float(abs(sv[sv < 0].sum()))
    total     = pos_total + neg_total + 1e-9
    final     = base_val + float(sv.sum())
    top_pos   = np.argsort(sv)[::-1][:max(1, top_n // 2)]
    top_neg   = np.argsort(sv)[:max(1, top_n // 2)]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2333')
    ax.barh(0, pos_total / total, left=0, height=0.55, color='#ef4444', alpha=0.85)
    ax.barh(0, neg_total / total, left=pos_total / total, height=0.55,
            color='#3b82f6', alpha=0.85)
    if pos_total > 0:
        lbl = ', '.join([cnames[i] for i in top_pos[:3]])
        ax.text(pos_total / total / 2, 0, f'↑ {lbl}',
                ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
    if neg_total > 0:
        lbl = ', '.join([cnames[i] for i in top_neg[:3]])
        ax.text(pos_total / total + neg_total / total / 2, 0, f'↓ {lbl}',
                ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, pos_total / total, 1])
    ax.set_xticklabels([f'Base\n{base_val:.3f}', f'f(x)\n{final:.3f}', ''],
                       color='#a0b4d6', fontsize=9)
    ax.set_title('SHAP Force Plot — cancer drivers (red) vs normal drivers (blue)',
                 color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#a0b4d6')
    for sp in ax.spines.values():
        sp.set_color('#2a3550')
    plt.tight_layout()
    return fig


def plot_decision_path(sv, base_val, cnames, top_n=20):
    idx   = np.argsort(np.abs(sv))[::-1][:top_n]
    names = [cnames[i] for i in idx]
    vals  = sv[idx]
    cumul = [base_val]
    for v in vals:
        cumul.append(cumul[-1] + float(v))

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1c2333')
    ax.plot(cumul, range(len(cumul)),
            color='#f59e0b', linewidth=2, marker='o', markersize=4, zorder=3)
    fill_color = '#ef4444' if cumul[-1] > base_val else '#3b82f6'
    ax.fill_betweenx(range(len(cumul)), base_val, cumul, alpha=0.15, color=fill_color)
    ax.axvline(base_val, color='#4a5568', linestyle='--',
               linewidth=1, label=f'Base: {base_val:.4f}')
    ax.axvline(0.5, color='#6b7280', linestyle=':', linewidth=1,
               label='Decision boundary (0.5)')
    ax.set_yticks(range(len(cumul)))
    ax.set_yticklabels(['[base]'] + names, fontsize=9, color='#a0b4d6')
    ax.set_xlabel('Cumulative cancer-class probability', color='#a0b4d6', fontsize=10)
    ax.set_title('SHAP Decision Path — cumulative feature contribution',
                 color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#a0b4d6')
    ax.legend(facecolor='#1c2333', edgecolor='#2a3550', labelcolor='white', fontsize=8)
    for sp in ax.spines.values():
        sp.set_color('#2a3550')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def shap_dataframe(sv, cnames, top_n=15):
    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    return pd.DataFrame([{
        'Rank'      : r,
        'Component' : cnames[i],
        'SHAP Value': round(float(sv[i]), 6),
        '|SHAP|'    : round(float(abs(sv[i])), 6),
        'Direction' : '🔴 → CANCER' if sv[i] > 0 else '🔵 → NON-CANCER',
    } for r, i in enumerate(idx, 1)])


def model_agreement(pil_image, all_models, scaler, vt, pca):
    rows = []
    for name, mdl in all_models.items():
        r = run_prediction(pil_image, mdl, scaler, vt, pca)
        if r:
            rows.append({
                'Model'        : name,
                'Prediction'   : r['label'],
                'Cancer Prob %': round(r['cancer_prob'], 1),
                'Confidence %' : round(r['confidence'], 1),
            })
    return pd.DataFrame(rows)


def display_shap_analysis(res, image, model, model_name, X_bg, all_models, scaler, vt, pca):
    """
    Renders the full SHAP explainability section for a given prediction result.
    Can be used for both single image analysis and individual batch images.
    """
    st.divider()
    st.markdown(
        "<div class='shap-header'>"
        "<h3 style='margin:0;color:white;'>"
        "🧠 SHAP Explainability — Why this prediction?"
        "</h3></div>",
        unsafe_allow_html=True)

    st.markdown(
        "<div class='shap-intro-box'>"
        "<b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) assigns each "
        "PCA component a contribution score <i>for this exact image</i>. "
        "<span style='color:#ef4444;font-weight:600'>Red bars</span> push toward "
        "<b>CANCER</b>. "
        "<span style='color:#3b82f6;font-weight:600'>Blue bars</span> push toward "
        "<b>NON-CANCER</b>."
        "</div>",
        unsafe_allow_html=True)

    clean_name = model_name.replace(' ⭐', '').strip()
    if 'Random Forest' in clean_name or 'Decision Tree' in clean_name:
        explainer_note = "⚡ Using **TreeExplainer** (fast, exact)"
    elif 'Logistic Regression' in clean_name:
        explainer_note = "⚡ Using **LinearExplainer** (fast, exact)"
    else:
        explainer_note = "🐢 Using **KernelExplainer** — may take ~30 s"

    st.markdown(
        f"<div class='shap-status-box'>{explainer_note}</div>",
        unsafe_allow_html=True)

    with st.spinner(f"Computing SHAP values for {clean_name}…"):
        sv, base_val, cnames = get_shap_values(
            model, model_name,
            res['features_pca'], X_bg)

    if sv is None:
        st.error(
            "SHAP values could not be computed for this model/image. "
            "Check the warning above for details. "
            "Try **Random Forest**, **Decision Tree**, or "
            "**Logistic Regression** for fastest SHAP results.")
    else:
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]
        top3     = [(cnames[i], sv[i]) for i in top3_idx]
        drivers  = "  |  ".join(
            f"{n}: {v:+.4f} ({'→ cancer' if v > 0 else '→ normal'})"
            for n, v in top3)
        st.markdown(
            f"<div class='shap-intro-box'>"
            f"<b>Top 3 drivers for this image:</b><br>{drivers}"
            f"</div>",
            unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("#### 📊 SHAP Waterfall")
            st.caption("Step-by-step cumulative shift of each PCA component.")
            fig = plot_waterfall(sv, base_val, cnames, top_n=15)
            st.pyplot(fig)
            plt.close(fig)

        with r1c2:
            st.markdown("#### 📈 SHAP Summary Bar")
            st.caption("Top 20 components ranked by absolute SHAP impact.")
            fig = plot_summary_bar(sv, cnames, top_n=20)
            st.pyplot(fig)
            plt.close(fig)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### ⚡ SHAP Force Plot")
            st.caption("Total cancer-driving vs normal-driving force.")
            fig = plot_force(sv, base_val, cnames, top_n=10)
            st.pyplot(fig)
            plt.close(fig)

        with r2c2:
            st.markdown("#### 🗺️ SHAP Decision Path")
            st.caption("Cumulative prediction from base to final value.")
            fig = plot_decision_path(sv, base_val, cnames, top_n=20)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### 🔢 SHAP Values — top 15 PCA components")
        df_shap = shap_dataframe(sv, cnames, top_n=15)

        def _color_dir(val):
            if '→ CANCER'     in str(val):
                return 'background-color:rgba(239,68,68,0.15);color:#fca5a5'
            elif '→ NON-CANCER' in str(val):
                return 'background-color:rgba(59,130,246,0.15);color:#93c5fd'
            return ''

        st.dataframe(
            df_shap.style.map(_color_dir, subset=['Direction']),
            use_container_width=True, hide_index=True)

        st.markdown("#### 🏛️ All-Model Agreement")
        st.caption("Every trained model's vote on this image.")
        with st.spinner("Running all 7 models…"):
            df_agree = model_agreement(image, all_models, scaler, vt, pca)

        def _color_pred(val):
            if val == 'CANCER':
                return 'background-color:rgba(239,68,68,0.2);color:#fca5a5'
            elif val == 'NON-CANCER':
                return 'background-color:rgba(34,197,94,0.2);color:#86efac'
            return ''

        st.dataframe(
            df_agree.style.map(_color_pred, subset=['Prediction']),
            use_container_width=True, hide_index=True)

        n_cancer_votes = int((df_agree['Prediction'] == 'CANCER').sum())
        n_total        = len(df_agree)
        if n_cancer_votes >= 5 or n_cancer_votes <= 2:
            consensus = "✅ Strong consensus"
        elif n_cancer_votes in [4, 3]:
            consensus = "⚠️ Moderate consensus — further review recommended"
        else:
            consensus = "❓ Weak consensus"

        st.markdown(
            f"<div class='shap-model-agree'>"
            f"<b>{n_cancer_votes}/{n_total} models predict CANCER</b> — "
            f"{consensus}"
            f"</div>",
            unsafe_allow_html=True)

        if res['prediction'] == 1:
            st.markdown(
                f"<div class='clinical-note-cancer'>"
                f"<b>⚠️ Clinical Note:</b> The model predicts <b>oral cancer</b> "
                f"with <b>{res['cancer_prob']:.1f}%</b> probability. "
                f"Top SHAP driver: <b>{top3[0][0]}</b> "
                f"(SHAP = {float(top3[0][1]):+.4f}). "
                f"<br><br>Screening tool only. Confirm with a qualified "
                f"pathologist via biopsy."
                f"</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='clinical-note'>"
                f"<b>✅ Clinical Note:</b> The model predicts "
                f"<b>non-cancerous tissue</b> with "
                f"<b>{res['noncancer_prob']:.1f}%</b> probability. "
                f"Top SHAP driver: <b>{top3[0][0]}</b> "
                f"(SHAP = {float(top3[0][1]):+.4f}). "
                f"<br><br>A normal prediction does not exclude malignancy. "
                f"Clinical examination remains essential."
                f"</div>",
                unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 — SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-size:60px;'>🔬</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Oral Cancer Detection</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:gray;'>ML-Powered Clinical Assistant</h4>",
                unsafe_allow_html=True)
    st.divider()

    model_options       = list(models.keys()) if models else []
    selected_model_name = st.selectbox("Select Model", options=model_options, index=0)

    perf_map = {
        'SVM'                : "Recall: 92.00% | F1: 88.04% | AUC: 0.9412",
        'Voting'             : "Recall: 91.00% | F1: 90.55%",
        'Random Forest'      : "Recall: 86.00% | F1: 84.31%",
        'Logistic Regression': "Recall: 90.00% | F1: 89.11%",
        'Decision Tree'      : "Recall: 89.00% | F1: 82.03%",
        'KNN'                : "Recall: 83.00% | F1: 84.69%",
        'Naive Bayes'        : "Recall: 84.00% | F1: 83.17%",
    }
    if selected_model_name:
        st.markdown("### Model Performance:")
        for key, info in perf_map.items():
            if key in selected_model_name:
                st.info(info)
                break

    # ── SHAP speed guide ──────────────────────────────────────────
    st.markdown("### SHAP Speed Guide:")
    shap_speed = {
        'SVM (Tuned) ⭐'      : ('🐢 Slow  (~30 s)', '#ef4444'),
        'Voting Ensemble'     : ('🐢 Slow  (~45 s)', '#ef4444'),
        'Random Forest'       : ('⚡ Fast  (<2 s)',   '#22c55e'),
        'Logistic Regression' : ('⚡ Fast  (<2 s)',   '#22c55e'),
        'Decision Tree'       : ('⚡ Fast  (<2 s)',   '#22c55e'),
        'KNN (k=9)'           : ('🐢 Slow  (~30 s)', '#ef4444'),
        'Naive Bayes'         : ('🐢 Slow  (~25 s)', '#ef4444'),
    }
    for mn, (speed, col) in shap_speed.items():
        marker = '▶ ' if mn == selected_model_name else '   '
        st.markdown(
            f"<span style='color:{'white' if mn==selected_model_name else '#666'};'>"
            f"{marker}<b>{mn.replace(' ⭐','')}</b></span> "
            f"<span style='color:{col};font-size:0.8rem;'>{speed}</span>",
            unsafe_allow_html=True)

    with st.expander("Pipeline Info"):
        st.write("✓ CLAHE Enhancement")
        st.write("✓ Blur Filtering (Laplacian)")
        st.write("✓ GLCM + HOG + LBP + Stats Features")
        st.write("✓ PCA (95% variance → 287 components)")
        st.write("✓ StandardScaler normalization")

    st.divider()
    st.markdown(
        "<p style='font-size:12px;color:gray;text-align:center;'>"
        "⚠️ For research purposes only.<br>Not a substitute for clinical diagnosis.</p>",
        unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10 — MAIN PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🔬 Oral Cancer Detection System")
st.markdown("*Early detection powered by Machine Learning*")
st.divider()

if not models or scaler is None:
    st.error("Could not load models or pipeline. Check the MODELS_DIR path at the top of this file.")
    st.stop()

selected_model = models[selected_model_name]

tab1, tab2, tab3 = st.tabs([
    "🖼️ Single Image Analysis", 
    "📊 Batch Analysis", 
    "📈 Model Evaluation"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SINGLE IMAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload & Preview")
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png", "bmp"], key="single_upload")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.write({
                "Filename"  : uploaded_file.name,
                "File size" : f"{uploaded_file.size / 1024:.2f} KB",
                "Dimensions": f"{image.size[0]} x {image.size[1]}",
            })
            proc_f, sharp, enh = preprocess_image(image)
            if enh is not None:
                st.image(enh, caption="Preprocessed (CLAHE) Preview",
                         use_container_width=True, channels="GRAY")

    with col2:
        st.subheader("Results")
        if uploaded_file is not None:
            if st.button("ANALYZE IMAGE", use_container_width=True, type="primary"):

                with st.spinner("Analyzing…"):
                    res = run_prediction(image, selected_model, scaler, vt, pca)

                if res:
                    if res['sharpness'] < 100:
                        st.warning(
                            f"⚠️ Image appears blurry "
                            f"(Laplacian variance = {res['sharpness']:.1f} < 100). "
                            "Results may be less accurate.")

                    if res['prediction'] == 1:
                        st.markdown(f"""
                        <div class="result-cancer">
                            <h2>🔴 CANCER DETECTED</h2>
                            <h4>Confidence: {res['confidence']:.1f}%</h4>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-normal">
                            <h2>🟢 NON-CANCER</h2>
                            <h4>Confidence: {res['confidence']:.1f}%</h4>
                        </div>""", unsafe_allow_html=True)

                    st.progress(int(min(res['confidence'], 100)))
                    st.markdown("<br>", unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(
                            f"<div class='metric-card'><b>Cancer Probability</b>"
                            f"<br><span style='font-size:24px'>{res['cancer_prob']:.1f}%</span></div>",
                            unsafe_allow_html=True)
                    with m2:
                        st.markdown(
                            f"<div class='metric-card'><b>Non-Cancer Probability</b>"
                            f"<br><span style='font-size:24px'>{res['noncancer_prob']:.1f}%</span></div>",
                            unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.expander("View Feature Analysis Details"):
                        st.write(f"- *GLCM features*: {res['n_glcm']}")
                        st.write(f"- *HOG features*: {res['n_hog']}")
                        st.write(f"- *LBP features*: {res['n_lbp']}")
                        st.write(f"- *Statistical features*: {res['n_stats']}")
                        st.write("- *PCA components used*: 287")
                        st.write(f"- *Laplacian sharpness*: {res['sharpness']:.2f}")
                        if res['sharpness'] >= 100:
                            st.success("Sharp Image ✓")
                        else:
                            st.warning("Blurry Image ⚠️")

                    st.info(
                        "This model achieves 92% Recall — 9.2 out of 10 cancer cases "
                        "correctly identified. Always confirm with a medical professional.")

                    # ══════════════════════════════════════════════
                    # SHAP SECTION — works for ALL 7 models
                    # ══════════════════════════════════════════════
                    display_shap_analysis(
                        res, image, selected_model, selected_model_name,
                        X_shap_bg, models, scaler, vt, pca)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — BATCH ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("Batch Processing")
    uploaded_files = st.file_uploader(
        "Upload Multiple Images", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True, key="batch_upload")

    if uploaded_files:
        if st.button("Run Batch Analysis", use_container_width=True):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            st.session_state['batch_results'] = []
            st.session_state['active_shap_index'] = None
            cancer_count = 0

            for i, uf in enumerate(uploaded_files):
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uf.name}…")
                img = Image.open(uf)
                res = run_prediction(img, selected_model, scaler, vt, pca)
                if res:
                    # Store everything needed for SHAP later
                    st.session_state['batch_results'].append({
                        "name"            : uf.name,
                        "image"           : img,
                        "res"             : res,
                        "prediction"      : res['label'],
                        "cancer_prob"     : res['cancer_prob'],
                        "noncancer_prob"  : res['noncancer_prob'],
                        "sharpness"       : res['sharpness']
                    })
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("✅ Batch Processing Complete!")

        if st.session_state['batch_results']:
            results_list = []
            for item in st.session_state['batch_results']:
                results_list.append({
                    "Image Name"      : item['name'],
                    "Prediction"      : item['prediction'],
                    "Cancer Prob %"   : f"{item['cancer_prob']:.1f}%",
                    "NonCancer Prob %": f"{item['noncancer_prob']:.1f}%",
                    "Sharpness"       : round(item['sharpness'], 1),
                })
            
            df_results = pd.DataFrame(results_list)
            cancer_count = sum(1 for item in st.session_state['batch_results'] if item['prediction'] == 'CANCER')
            total = len(st.session_state['batch_results'])

            def _color_pred_batch(val):
                if val == 'CANCER':
                    return 'background-color:rgba(255,0,0,0.2)'
                elif val == 'NON-CANCER':
                    return 'background-color:rgba(0,255,0,0.2)'
                return ''

            st.markdown("### Summary View")
            st.dataframe(
                df_results.style.map(_color_pred_batch, subset=['Prediction']),
                use_container_width=True, hide_index=True)

            nc_count = total - cancer_count
            st.success(
                f"**Total:** {total}  |  "
                f"**Cancer:** {cancer_count} ({cancer_count/total*100:.1f}%)  |  "
                f"**Normal:** {nc_count} ({nc_count/total*100:.1f}%)")

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv, file_name='batch_analysis_results.csv', mime='text/csv')

            st.divider()
            st.subheader("🔍 Detailed Image Inspection & SHAP")
            st.caption("Click 'SHAP Explain' to see a detailed visual breakdown for any specific image.")

            for idx, item in enumerate(st.session_state['batch_results']):
                with st.container():
                    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
                    with c1:
                        st.image(item['image'], use_container_width=True)
                    with c2:
                        st.markdown(f"**{item['name']}**")
                        st.caption(f"Sharpness: {item['sharpness']:.1f}")
                    with c3:
                        color = "#ef4444" if item['prediction'] == 'CANCER' else "#22c55e"
                        st.markdown(f"Status: <span style='color:{color};font-weight:bold;'>{item['prediction']}</span>", unsafe_allow_html=True)
                        st.markdown(f"Confidence: **{item['res']['confidence']:.1f}%**")
                    with c4:
                        if st.button("SHAP Explain", key=f"btn_shap_{idx}", use_container_width=True):
                            st.session_state['active_shap_index'] = idx
                    st.markdown("---")

            # Show active SHAP analysis
            if st.session_state['active_shap_index'] is not None:
                active_idx = st.session_state['active_shap_index']
                if active_idx < len(st.session_state['batch_results']):
                    active_item = st.session_state['batch_results'][active_idx]
                    st.markdown(f"## 🧐 SHAP Analysis for: {active_item['name']}")
                    if st.button("Close Analysis", key="close_shap"):
                        st.session_state['active_shap_index'] = None
                        st.rerun()
                    
                    display_shap_analysis(
                        active_item['res'], active_item['image'], selected_model, selected_model_name,
                        X_shap_bg, models, scaler, vt, pca)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — MODEL EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("📊 Model Performance & Robustness")
    st.markdown("""
    This section displays the offline evaluation metrics of the trained models, 
    including **5-Fold Cross-Validation** for stability analysis.
    """)
    
    plots_dir = Path("outputs/plots")
    if not plots_dir.exists():
        # backup path for relative execution
        plots_dir = Path("oral_cancer_ml/outputs/plots")

    if plots_dir.exists():
        # 1. K-Fold Results (The new graph requested)
        kfold_plot = plots_dir / "22_kfold_validation_results.png"
        if kfold_plot.exists():
            st.image(str(kfold_plot), caption="5-Fold Cross-Validation — Robustness (Mean ± Std)", use_container_width=True)
            st.info("💡 **Why it matters**: K-Fold validates that the model is consistent across different slices of data, not just lucky on one test set.")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            # 2. Model Comparison
            comp_plot = plots_dir / "15_model_comparison_bars.png"
            if comp_plot.exists():
                st.image(str(comp_plot), caption="Model Comparison — Overall Metrics", use_container_width=True)
            
            # 3. Recall Ranking
            ranking_plot = plots_dir / "17_cancer_recall_ranking.png"
            if ranking_plot.exists():
                st.image(str(ranking_plot), caption="Recall Ranking (Primary Metric)", use_container_width=True)

        with col_m2:
            # 4. ROC Curves
            roc_plot = plots_dir / "13_roc_curves.png"
            if roc_plot.exists():
                st.image(str(roc_plot), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
            
            # 5. Precision-Recall
            pr_plot = plots_dir / "14_precision_recall_curves.png"
            if pr_plot.exists():
                st.image(str(pr_plot), caption="Precision-Recall Curves", use_container_width=True)
                
    else:
        st.warning("⚠️ Evaluation plots not found. Run `evaluate_models.py` to generate these visualizations.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.divider()
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("<h4 style='text-align:center'>🎯 Best Recall: 92% (SVM)</h4>",
                unsafe_allow_html=True)
with f2:
    st.markdown("<h4 style='text-align:center'>📈 Best F1: 90.55% (Ensemble)</h4>",
                unsafe_allow_html=True)
with f3:
    st.markdown("<h4 style='text-align:center'>🔬 AUC-ROC: 0.9412</h4>",
                unsafe_allow_html=True)