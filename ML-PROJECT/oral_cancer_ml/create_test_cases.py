import os, shutil, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

C_SRC = Path(__file__).parent / "data" / "processed" / "CANCER"
NC_SRC = Path(__file__).parent / "data" / "processed" / "NON_CANCER"
TEST_DIR = Path(__file__).parent / "test_cases"

folders = [
    TEST_DIR / "cancer_tests",
    TEST_DIR / "noncancer_tests",
    TEST_DIR / "edge_cases"
]
for f in folders:
    f.mkdir(parents=True, exist_ok=True)

c_files = sorted(list(C_SRC.glob("*.jpg")))
nc_files = sorted(list(NC_SRC.glob("*.jpg")))

def get_nearest_file(target_name, file_list):
    # Try exact match
    for f in file_list:
        if f.name == target_name:
            return f
    # Not found, grab a random near one (fallback)
    return file_list[random.randint(0, len(file_list)-1)]

# STEP 2
cancer_targets = [
    ("cancer_orig_001.jpg", "TC01_cancer.jpg"),
    ("cancer_orig_010.jpg", "TC02_cancer.jpg"),
    ("cancer_orig_025.jpg", "TC03_cancer.jpg"),
    ("cancer_orig_050.jpg", "TC04_cancer.jpg"),
    ("cancer_orig_075.jpg", "TC05_cancer.jpg"),
    ("cancer_orig_100.jpg", "TC06_cancer.jpg"),
    ("cancer_orig_150.jpg", "TC07_cancer.jpg"),
    ("cancer_orig_200.jpg", "TC08_cancer.jpg"),
    ("cancer_orig_250.jpg", "TC09_cancer.jpg"),
    ("cancer_orig_300.jpg", "TC10_cancer.jpg")
]

for src, dst in cancer_targets:
    f_src = get_nearest_file(src, c_files)
    shutil.copy(str(f_src), str(TEST_DIR / "cancer_tests" / dst))

# STEP 3
noncancer_targets = [
    ("non_cancer_orig_001.jpg", "TC11_noncancer.jpg"),
    ("non_cancer_orig_010.jpg", "TC12_noncancer.jpg"),
    ("non_cancer_orig_025.jpg", "TC13_noncancer.jpg"),
    ("non_cancer_orig_050.jpg", "TC14_noncancer.jpg"),
    ("non_cancer_orig_075.jpg", "TC15_noncancer.jpg"),
    ("non_cancer_orig_100.jpg", "TC16_noncancer.jpg"),
    ("non_cancer_orig_125.jpg", "TC17_noncancer.jpg"),
    ("non_cancer_orig_150.jpg", "TC18_noncancer.jpg")
]

for src, dst in noncancer_targets:
    f_src = get_nearest_file(src, nc_files)
    shutil.copy(str(f_src), str(TEST_DIR / "noncancer_tests" / dst))

# STEP 4
img = cv2.imread(str(c_files[0]))
blurry = cv2.GaussianBlur(img, (25, 25), 0)
cv2.imwrite(str(TEST_DIR / "edge_cases" / "TC19_blurry.jpg"), blurry)
print("✓ TC19 blurry edge case created")

noise = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
cv2.imwrite(str(TEST_DIR / "edge_cases" / "TC20_random_noise.jpg"), noise)
print("✓ TC20 random noise edge case created")

# STEP 5
test_manifest = [
    {"TC": "TC01", "File": "cancer_tests/TC01_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC02", "File": "cancer_tests/TC02_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC03", "File": "cancer_tests/TC03_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC04", "File": "cancer_tests/TC04_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC05", "File": "cancer_tests/TC05_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC06", "File": "cancer_tests/TC06_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC07", "File": "cancer_tests/TC07_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC08", "File": "cancer_tests/TC08_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC09", "File": "cancer_tests/TC09_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC10", "File": "cancer_tests/TC10_cancer.jpg", "True_Label": "CANCER", "Expected": "CANCER", "Type": "Standard"},
    {"TC": "TC11", "File": "noncancer_tests/TC11_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC12", "File": "noncancer_tests/TC12_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC13", "File": "noncancer_tests/TC13_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC14", "File": "noncancer_tests/TC14_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC15", "File": "noncancer_tests/TC15_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC16", "File": "noncancer_tests/TC16_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC17", "File": "noncancer_tests/TC17_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC18", "File": "noncancer_tests/TC18_noncancer.jpg", "True_Label": "NON-CANCER", "Expected": "NON-CANCER", "Type": "Standard"},
    {"TC": "TC19", "File": "edge_cases/TC19_blurry.jpg", "True_Label": "CANCER", "Expected": "WARNING shown", "Type": "Edge Case - Blur"},
    {"TC": "TC20", "File": "edge_cases/TC20_random_noise.jpg", "True_Label": "N/A", "Expected": "Any + Low confidence", "Type": "Edge Case - Noise"},
]

df = pd.DataFrame(test_manifest)
df['Result'] = ''
df['Confidence'] = ''
df['Pass_Fail'] = ''

df.to_csv(TEST_DIR / 'test_manifest.csv', index=False)
print("✓ test_manifest.csv saved")

# STEP 6
print(f"""
╔══════════════════════════════════════╗
║      TEST CASES CREATED              ║
╠══════════════════════════════════════╣
║ Cancer test images     : 10          ║
║ Non-cancer test images : 8           ║
║ Edge case images       : 2           ║
║ Total test cases       : 20          ║
╠══════════════════════════════════════╣
║ Location:                            ║
║ oral_cancer_ml/test_cases/           ║
║   ├── cancer_tests/    (TC01-TC10)   ║
║   ├── noncancer_tests/ (TC11-TC18)   ║
║   └── edge_cases/      (TC19-TC20)   ║
╠══════════════════════════════════════╣
║ Manifest saved → test_manifest.csv   ║
╚══════════════════════════════════════╝

HOW TO USE:
1. Run: python -m streamlit run app.py
2. Go to Batch Analysis tab
3. Upload all 20 images at once
4. Record results in test_manifest.csv
5. Calculate final real-world accuracy
""")
