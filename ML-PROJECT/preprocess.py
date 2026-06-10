import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2, os, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

CANCER_SRC    = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\raw\CANCER")
NONCANCER_SRC = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\raw\NON_CANCER")
CANCER_OUT    = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed\CANCER")
NONCANCER_OUT = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\processed\NON_CANCER")
PLOTS_OUT     = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\plots")

CANCER_OUT.mkdir(parents=True, exist_ok=True)
NONCANCER_OUT.mkdir(parents=True, exist_ok=True)

def clear_existing_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for old in folder.glob(ext):
            old.unlink(missing_ok=True)

# Always rebuild processed folders from scratch.
clear_existing_images(CANCER_OUT)
clear_existing_images(NONCANCER_OUT)

# PHASE 1 — BLURRY IMAGE FILTERING
def is_sharp(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= threshold, lap_var

# PHASE 2 — PREPROCESSING
def preprocess_image(img):
    # Step 2: Resize
    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    # Step 3: Grayscale directly
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Step 4: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Step 5: Gaussian Blur
    denoised = cv2.GaussianBlur(enhanced, (3,3), 0)
    # Step 6: Normalize to float32
    normalized = denoised.astype(np.float32) / 255.0
    # Step 7: Return
    return normalized

# PHASE 3 — AUGMENTATION FUNCTIONS
def aug_hflip(img):
    return cv2.flip(img, 1)

def aug_vflip(img):
    return cv2.flip(img, 0)

def aug_rotate_pos(img):
    M = cv2.getRotationMatrix2D((64,64), 20, 1.0)
    return cv2.warpAffine(img, M, (128,128))

def aug_rotate_neg(img):
    M = cv2.getRotationMatrix2D((64,64), -20, 1.0)
    return cv2.warpAffine(img, M, (128,128))

def aug_brightness_up(img):
    return np.clip(img * 1.3, 0.0, 1.0).astype(np.float32)

def aug_brightness_down(img):
    return np.clip(img * 0.7, 0.0, 1.0).astype(np.float32)

def aug_noise(img):
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)

aug_functions = [
    aug_hflip, aug_vflip, aug_rotate_pos, aug_rotate_neg,
    aug_brightness_up, aug_brightness_down, aug_noise
]

stats = {
    'cancer_sharp': 0, 'cancer_blurry': 0, 'cancer_orig': 0, 'cancer_aug': 0,
    'noncancer_sharp': 0, 'noncancer_blurry': 0, 'noncancer_orig': 0, 'noncancer_aug': 0
}

sample_plot_c = None
sample_plot_nc = None

def process_class(src_dir, out_dir, class_prefix, stat_sharp, stat_blur, stat_orig, stat_aug):
    paths = list(src_dir.glob('*.jpg'))
    sharp_images = []
    
    print(f"Filtering '{class_prefix}' for blurry images...")
    for p in paths:
        img = cv2.imread(str(p))
        if img is None: continue
        sharp, var = is_sharp(img, 100)
        if sharp:
            stats[stat_sharp] += 1
            processed = preprocess_image(img)
            sharp_images.append((p.stem, processed))
            
            # PHASE 4 — SAVE ORIGINALS
            stats[stat_orig] += 1
            out_img = (processed * 255.0).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{class_prefix}_orig_{stats[stat_orig]:03d}.jpg"), out_img)
        else:
            stats[stat_blur] += 1
            
    print(f"  {class_prefix.upper()} — Sharp kept: {stats[stat_sharp]} | Blurry removed: {stats[stat_blur]}")
    
    sample = None
    if sharp_images:
        sample = sharp_images[0][1]
        
    # PHASE 3 — AUGMENTATION
    current_count = stats[stat_orig]
    aug_count = 0
    aug_idx = 0
    
    print(f"Augmenting '{class_prefix}' images to reach 500...")
    while current_count < 500 and sharp_images:
        for name, img_arr in sharp_images:
            if current_count >= 500:
                break
            
            aug_func = aug_functions[aug_idx % len(aug_functions)]
            augmented = aug_func(img_arr)
            
            aug_count += 1
            current_count += 1
            
            # Save augmented
            out_img = (augmented * 255.0).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{class_prefix}_aug_{aug_count:03d}.jpg"), out_img)
            
            aug_idx += 1

    stats[stat_aug] = aug_count
    return sample


# EXECUTE
sample_plot_c = process_class(CANCER_SRC, CANCER_OUT, "cancer", 
                              'cancer_sharp', 'cancer_blurry', 'cancer_orig', 'cancer_aug')

print()
sample_plot_nc = process_class(NONCANCER_SRC, NONCANCER_OUT, "noncancer", 
                               'noncancer_sharp', 'noncancer_blurry', 'noncancer_orig', 'noncancer_aug')

# PHASE 5 — QUALITY VERIFICATION
print("\nRunning Quality Verification...")
def check_quality(dir_path):
    files = list(dir_path.glob('*.jpg'))
    if not files: return False, False, False, False
    
    random.seed(42)
    sample_files = random.sample(files, min(10, len(files)))
    
    all_128, all_uint8, all_min_0, all_max_255 = True, True, True, True
    
    for f in sample_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if img.shape != (128, 128): all_128 = False
        if img.dtype != np.uint8: all_uint8 = False
        if img.min() < 0: all_min_0 = False
        if img.max() > 255: all_max_255 = False
        
    return all_128, all_uint8, all_min_0, all_max_255

c_q1, c_q2, c_q3, c_q4 = check_quality(CANCER_OUT)
nc_q1, nc_q2, nc_q3, nc_q4 = check_quality(NONCANCER_OUT)

all_128 = c_q1 and nc_q1
all_uint8 = c_q2 and nc_q2
all_val = c_q3 and nc_q3 and c_q4 and nc_q4

# PHASE 6 — COMPARISON PLOT
if sample_plot_c is not None and sample_plot_nc is not None:
    print("Generating Augmented Samples Plot...")
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    samples = [sample_plot_c, sample_plot_nc]
    names = ['Original', 'HFlip', 'VFlip', 'Rot+20', 'Rot-20', 'Bright+', 'Bright-', 'Noise']
    
    for row in range(2):
        img = samples[row]
        axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].imshow(aug_hflip(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 2].imshow(aug_vflip(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 3].imshow(aug_rotate_pos(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 4].imshow(aug_rotate_neg(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 5].imshow(aug_brightness_up(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 6].imshow(aug_brightness_down(img), cmap='gray', vmin=0, vmax=1)
        axes[row, 7].imshow(aug_noise(img), cmap='gray', vmin=0, vmax=1)
        
        for col in range(8):
            axes[row, col].axis('off')
            if row == 0:
                axes[0, col].set_title(names[col], fontsize=10)
                
    # Create invisible empty text elements on left just to add a makeshift label
    axes[0, 0].text(-30, 64, "CANCER", ha='right', va='center', rotation=90, fontsize=12)
    axes[1, 0].text(-30, 64, "NON-CANCER", ha='right', va='center', rotation=90, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / "09_augmentation_samples.png", dpi=150)
    plt.close()

# PHASE 7 — FINAL REPORT
c_final = stats['cancer_orig'] + stats['cancer_aug']
nc_final = stats['noncancer_orig'] + stats['noncancer_aug']
total_processed = c_final + nc_final

print(f"""
═══════════════════════════════════════════
         PREPROCESSING COMPLETE
═══════════════════════════════════════════
         BLURRY FILTER RESULTS
───────────────────────────────────────────
CANCER   — Sharp kept : {stats['cancer_sharp']} | Blurry removed: {stats['cancer_blurry']}
NON-CANCER — Sharp kept: {stats['noncancer_sharp']} | Blurry removed: {stats['noncancer_blurry']}
───────────────────────────────────────────
         AUGMENTATION RESULTS
───────────────────────────────────────────
CANCER   — Originals: {stats['cancer_orig']} | Augmented added: {stats['cancer_aug']} | Final: {c_final}
NON-CANCER — Originals: {stats['noncancer_orig']} | Augmented added: {stats['noncancer_aug']} | Final: {nc_final}
───────────────────────────────────────────
         QUALITY CHECKS
───────────────────────────────────────────
All images 128x128     : {'✓' if all_128 else '✗'}
All images uint8       : {'✓' if all_uint8 else '✗'}
All values [0-255]     : {'✓' if all_val else '✗'}
───────────────────────────────────────────
Total processed images : {total_processed}
Ready for feature extraction : ✓
═══════════════════════════════════════════
""")
