import cv2
import shutil
import hashlib
from pathlib import Path

# STEP 1 — DEFINE PATHS (Kaggle dataset only)
PROJECT_ROOT = Path(r"C:\Users\Admin\OneDrive\Desktop\Andrew\MKC")
DATASET_ROOT = PROJECT_ROOT / "archive" / "Oral cancer Dataset 2.0" / "OC Dataset kaggle new"

output_cancer = PROJECT_ROOT / "oral_cancer_ml" / "data" / "raw" / "CANCER"
output_noncancer = PROJECT_ROOT / "oral_cancer_ml" / "data" / "raw" / "NON_CANCER"

output_cancer.mkdir(parents=True, exist_ok=True)
output_noncancer.mkdir(parents=True, exist_ok=True)

# STEP 2 — FIND CLASS FOLDERS ROBUSTLY
def normalize_name(name: str) -> str:
    return name.strip().replace("_", " ").replace("-", " ").upper()

def discover_class_dirs(dataset_root: Path):
    cancer_dirs = []
    noncancer_dirs = []

    for p in dataset_root.rglob("*"):
        if not p.is_dir():
            continue
        n = normalize_name(p.name)
        if n == "CANCER":
            cancer_dirs.append(p)
        elif n in {"NON CANCER", "NONCANCER", "NO CANCER", "NORMAL"}:
            noncancer_dirs.append(p)

    return sorted(cancer_dirs), sorted(noncancer_dirs)

cancer_sources, noncancer_sources = discover_class_dirs(DATASET_ROOT)

print("Checking Kaggle source directories...")
print(f"Dataset root: {DATASET_ROOT}")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        "Kaggle dataset root not found. Place extracted dataset at: "
        f"{DATASET_ROOT}"
    )

if not cancer_sources or not noncancer_sources:
    raise FileNotFoundError(
        "Could not find both class folders (CANCER and NON CANCER/NON_CANCER) "
        f"under {DATASET_ROOT}"
    )

for p in cancer_sources:
    count = len([f for f in p.iterdir() if f.is_file()])
    print(f"  ✓ CANCER source: {p} ({count} files)")
for p in noncancer_sources:
    count = len([f for f in p.iterdir() if f.is_file()])
    print(f"  ✓ NO CANCER source: {p} ({count} files)")

# STEP 3 — HELPERS
def get_md5(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_image(filepath: Path) -> bool:
    img = cv2.imread(str(filepath))
    return img is not None

def clear_existing_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for old in folder.glob(ext):
            old.unlink(missing_ok=True)

valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

# Reset raw folders so only this dataset is used
clear_existing_images(output_cancer)
clear_existing_images(output_noncancer)

# Shared hash set prevents cross-class duplicates
seen_hashes = set()

def copy_class_images(class_name: str, src_dirs, out_dir: Path, prefix: str):
    stats = {
        "source_files": 0,
        "duplicates": 0,
        "corrupt": 0,
        "copied": 0,
    }

    print(f"\nProcessing {class_name} images...")
    for src_path in src_dirs:
        for file in src_path.iterdir():
            if not file.is_file() or file.suffix.lower() not in valid_exts:
                continue

            stats["source_files"] += 1
            file_hash = get_md5(file)

            if file_hash in seen_hashes:
                stats["duplicates"] += 1
                continue

            if not check_image(file):
                stats["corrupt"] += 1
                continue

            seen_hashes.add(file_hash)
            stats["copied"] += 1
            new_name = f"{prefix}_{stats['copied']:04d}.jpg"
            shutil.copy2(file, out_dir / new_name)

    print(f"{class_name} processing stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats

# STEP 4 — COPY BOTH CLASSES
cancer_stats = copy_class_images(
    class_name="CANCER",
    src_dirs=cancer_sources,
    out_dir=output_cancer,
    prefix="cancer",
)

noncancer_stats = copy_class_images(
    class_name="NO CANCER",
    src_dirs=noncancer_sources,
    out_dir=output_noncancer,
    prefix="noncancer",
)

print("\n✓ Kaggle dataset ingestion completed successfully!")
print("Summary:")
print(f"  CANCER copied    : {cancer_stats['copied']}")
print(f"  NO CANCER copied : {noncancer_stats['copied']}")
print(f"  TOTAL copied     : {cancer_stats['copied'] + noncancer_stats['copied']}")
