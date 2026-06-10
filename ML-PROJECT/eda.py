import cv2, os, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys

# Ensure UTF-8 printing for console formatting
sys.stdout.reconfigure(encoding='utf-8')

sns.set_theme(style='darkgrid')

CANCER_PATH    = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\raw\CANCER")
NONCANCER_PATH = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\data\raw\NON_CANCER")
PLOTS_PATH     = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC\oral_cancer_ml\outputs\plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# LOAD ALL IMAGES FIRST
cancer_imgs = []
noncancer_imgs = []

print("Loading CANCER images...")
for i, file in enumerate(CANCER_PATH.glob('*.jpg')):
    if i > 0 and i % 100 == 0:
        print(f"  Loaded {i} CANCER images...")
    img = cv2.imread(str(file))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cancer_imgs.append(img_rgb)
print(f"Total CANCER images loaded: {len(cancer_imgs)}")

print("Loading NON-CANCER images...")
for i, file in enumerate(NONCANCER_PATH.glob('*.jpg')):
    if i > 0 and i % 100 == 0:
        print(f"  Loaded {i} NON-CANCER images...")
    img = cv2.imread(str(file))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        noncancer_imgs.append(img_rgb)
print(f"Total NON-CANCER images loaded: {len(noncancer_imgs)}")

cancer_count = len(cancer_imgs)
noncancer_count = len(noncancer_imgs)
total_count = cancer_count + noncancer_count
cancer_pct = cancer_count / total_count * 100 if total_count > 0 else 0
noncancer_pct = noncancer_count / total_count * 100 if total_count > 0 else 0

# PLOT 1 — CLASS DISTRIBUTION
classes = ['NON-CANCER', 'CANCER']
counts = [noncancer_count, cancer_count]
pcts = [noncancer_pct, cancer_pct]

plt.figure(figsize=(8, 6))
bars = plt.bar(classes, counts, color=['steelblue', 'tomato'])
plt.title("Class Distribution — Oral Cancer Dataset", fontsize=14)
plt.ylabel("Image Count")

for bar, count, pct in zip(bars, counts, pcts):
    y_val = bar.get_height()
    x_val = bar.get_x() + bar.get_width() / 2
    # Count on top
    plt.text(x_val, y_val + max(counts) * 0.01, f"{int(count)}", 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
    # Percentage at bottom
    if y_val > 0:
        plt.text(x_val, y_val / 2, f"{pct:.1f}%", 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_PATH / "01_class_distribution.png", dpi=150)
plt.close()


# PLOT 2 — RESOLUTION SCATTER
cancer_sizes = [(img.shape[1], img.shape[0]) for img in cancer_imgs]
noncancer_sizes = [(img.shape[1], img.shape[0]) for img in noncancer_imgs]
all_sizes = cancer_sizes + noncancer_sizes

unique_res = Counter(all_sizes)
dominant_res = unique_res.most_common(1)[0][0] if unique_res else (0,0)

c_w, c_h = zip(*cancer_sizes) if cancer_sizes else ([], [])
nc_w, nc_h = zip(*noncancer_sizes) if noncancer_sizes else ([], [])

plt.figure(figsize=(8, 6))
plt.scatter(c_w, c_h, color='tomato', alpha=0.5, label='CANCER')
plt.scatter(nc_w, nc_h, color='steelblue', alpha=0.5, label='NON-CANCER')
plt.title("Image Resolution Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_PATH / "02_resolution_scatter.png", dpi=150)
plt.close()


# PLOT 3 — SAMPLE IMAGE GRID
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Sample Images from Each Class", fontsize=16)

import random
random.seed(42)
c_idx = random.sample(range(len(cancer_imgs)), min(8, len(cancer_imgs))) if cancer_imgs else []
nc_idx = random.sample(range(len(noncancer_imgs)), min(8, len(noncancer_imgs))) if noncancer_imgs else []

for i in range(8):
    if i < len(c_idx):
        axes[0, i].imshow(cancer_imgs[c_idx[i]])
    axes[0, i].axis('off')

    if i < len(nc_idx):
        axes[1, i].imshow(noncancer_imgs[nc_idx[i]])
    axes[1, i].axis('off')

# Labels on left side
axes[0, 0].set_title("CANCER", loc='left', pad=10, fontsize=12, y=0.3, x=-0.2)
axes[1, 0].set_title("NON-CANCER", loc='left', pad=10, fontsize=12, y=0.3, x=-0.2)

plt.tight_layout()
plt.savefig(PLOTS_PATH / "03_sample_grid.png", dpi=150)
plt.close()


# PLOT 4 — PIXEL INTENSITY HISTOGRAM
print("Computing pixel intensities...")
c_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten() for img in cancer_imgs]
nc_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten() for img in noncancer_imgs]

c_pixels = np.concatenate(c_gray) if c_gray else np.array([])
nc_pixels = np.concatenate(nc_gray) if nc_gray else np.array([])

c_mean_int = c_pixels.mean() if len(c_pixels) else 0
nc_mean_int = nc_pixels.mean() if len(nc_pixels) else 0

plt.figure(figsize=(10, 6))
if len(c_pixels):
    plt.hist(c_pixels, bins=50, alpha=0.6, color='tomato', label=f'CANCER (mean={c_mean_int:.1f})', density=True)
if len(nc_pixels):
    plt.hist(nc_pixels, bins=50, alpha=0.6, color='steelblue', label=f'NON-CANCER (mean={nc_mean_int:.1f})', density=True)

plt.axvline(c_mean_int, color='red', linestyle='dashed', linewidth=2)
plt.axvline(nc_mean_int, color='blue', linestyle='dashed', linewidth=2)
plt.title("Pixel Intensity Distribution by Class")
plt.xlabel("Pixel Intensity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_PATH / "04_intensity_histogram.png", dpi=150)
plt.close()


# PLOT 5 — MEAN IMAGE PER CLASS
print("Computing mean images...")
IMG_SIZE = (128, 128)
c_resized = [cv2.resize(img, IMG_SIZE) for img in cancer_imgs]
nc_resized = [cv2.resize(img, IMG_SIZE) for img in noncancer_imgs]

c_mean_img = np.mean(c_resized, axis=0).astype(np.uint8) if c_resized else np.zeros((128,128,3), dtype=np.uint8)
nc_mean_img = np.mean(nc_resized, axis=0).astype(np.uint8) if nc_resized else np.zeros((128,128,3), dtype=np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(c_mean_img)
axes[0].set_title("Mean CANCER Image")
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(nc_mean_img)
axes[1].set_title("Mean NON-CANCER Image")
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(PLOTS_PATH / "05_mean_images.png", dpi=150)
plt.close()


# PLOT 6 — BRIGHTNESS & CONTRAST BOXPLOTS
print("Computing brightness and contrast...")
c_bright = [np.mean(pixels) for pixels in c_gray] if c_gray else []
c_contrast = [np.std(pixels) for pixels in c_gray] if c_gray else []

nc_bright = [np.mean(pixels) for pixels in nc_gray] if nc_gray else []
nc_contrast = [np.std(pixels) for pixels in nc_gray] if nc_gray else []

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

bplot1 = axes[0].boxplot([c_bright, nc_bright], patch_artist=True, labels=['CANCER', 'NON-CANCER'])
colors = ['tomato', 'steelblue']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_title("Brightness (Mean)")

bplot2 = axes[1].boxplot([c_contrast, nc_contrast], patch_artist=True, labels=['CANCER', 'NON-CANCER'])
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_title("Contrast (Std Dev)")

plt.tight_layout()
plt.savefig(PLOTS_PATH / "06_brightness_contrast.png", dpi=150)
plt.close()


# PLOT 7 — COLOR CHANNEL ANALYSIS
print("Analyzing color channels...")
c_rgb_means = np.mean([np.mean(img, axis=(0,1)) for img in cancer_imgs], axis=0) if cancer_imgs else [0,0,0]
nc_rgb_means = np.mean([np.mean(img, axis=(0,1)) for img in noncancer_imgs], axis=0) if noncancer_imgs else [0,0,0]

channels = ['Red', 'Green', 'Blue']
x = np.arange(len(channels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, c_rgb_means, width, label='CANCER', color='tomato')
plt.bar(x + width/2, nc_rgb_means, width, label='NON-CANCER', color='steelblue')
plt.xticks(x, channels)
plt.title("Mean RGB Channel Values by Class")
plt.ylabel("Mean Intensity")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_PATH / "07_color_channels.png", dpi=150)
plt.close()


# PLOT 8 — SHARPNESS ANALYSIS
print("Analyzing sharpness...")
c_sharp = [cv2.Laplacian(img, cv2.CV_64F).var() for img in c_gray] if c_gray else []
nc_sharp = [cv2.Laplacian(img, cv2.CV_64F).var() for img in nc_gray] if nc_gray else []

c_blurry = sum(1 for s in c_sharp if s < 100)
nc_blurry = sum(1 for s in nc_sharp if s < 100)

plt.figure(figsize=(8, 6))
bplot3 = plt.boxplot([c_sharp, nc_sharp], patch_artist=True, labels=['CANCER', 'NON-CANCER'])
for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)
plt.yscale('log')
plt.title("Image Sharpness (Laplacian Variance) by Class")
plt.ylabel("Laplacian Variance (Log Scale)")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "08_sharpness.png", dpi=150)
plt.close()


# FINAL EDA SUMMARY
print("Generating summary...")

c_mean_b = np.mean(c_bright) if c_bright else 0
nc_mean_b = np.mean(nc_bright) if nc_bright else 0
c_mean_c = np.mean(c_contrast) if c_contrast else 0
nc_mean_c = np.mean(nc_contrast) if nc_contrast else 0

unique_res_dict = dict(unique_res.most_common(5))
unique_res_str = str(unique_res_dict)
if len(unique_res) > 5:
    unique_res_str += f" ... (+{len(unique_res)-5} more)"

b_diff = f"Cancer images are on average {'brighter' if c_mean_b > nc_mean_b else 'darker'} ({c_mean_b:.1f} vs {nc_mean_b:.1f})."
c_diff = f"Red channel is {'higher' if c_rgb_means[0] > nc_rgb_means[0] else 'lower'} in CANCER ({c_rgb_means[0]:.1f} vs {nc_rgb_means[0]:.1f})."
s_diff = f"Sharpness variance shows {c_blurry} blurry CANCER vs {nc_blurry} NON-CANCER images."

ratio = max(cancer_pct, noncancer_pct) / min(cancer_pct, noncancer_pct) if min(cancer_pct, noncancer_pct) > 0 else 0

print(f"""
═══════════════════════════════════════════
            EDA SUMMARY REPORT
═══════════════════════════════════════════
Total Images         : {total_count}
CANCER               : {cancer_count}  ({cancer_pct:.1f}%)
NON-CANCER           : {noncancer_count}  ({noncancer_pct:.1f}%)
Imbalance Ratio      : {ratio:.2f} : 1
───────────────────────────────────────────
Unique Resolutions   : {unique_res_str}
Dominant Size        : {dominant_res[0]}x{dominant_res[1]}
───────────────────────────────────────────
Mean Brightness      : Cancer={c_mean_b:.2f} | NonCancer={nc_mean_b:.2f}
Mean Contrast(std)   : Cancer={c_mean_c:.2f} | NonCancer={nc_mean_c:.2f}
───────────────────────────────────────────
Blurry Images        : Cancer={c_blurry} | NonCancer={nc_blurry}
───────────────────────────────────────────
Key Findings:
 - {b_diff}
 - {c_diff}
 - {s_diff}
═══════════════════════════════════════════
✓ EDA Complete — 8 plots saved to outputs/plots/
""")
