import os
import sys
import datetime

# Step 2: Import and verify libraries
print("Verifying libraries:")
try:
    import cv2
    print(f"cv2 version: {cv2.__version__}")
    import numpy as np
    print(f"numpy version: {np.__version__}")
    import pandas as pd
    print(f"pandas version: {pd.__version__}")
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"matplotlib version: {matplotlib.__version__}")
    import seaborn as sns
    print(f"seaborn version: {sns.__version__}")
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
    import skimage
    print(f"skimage version: {skimage.__version__}")
    import imblearn
    print(f"imblearn version: {imblearn.__version__}")
    import joblib
    print(f"joblib version: {joblib.__version__}")
    import PIL
    from PIL import Image
    print(f"PIL version: {PIL.__version__}")
    import scipy
    print(f"scipy version: {scipy.__version__}")
except ImportError as e:
    print(f"Error importing library: {e}")
    sys.exit(1)

# Step 3: Create folder structure
base_dir = "oral_cancer_ml"
directories = [
    f"{base_dir}/data/raw/CANCER",
    f"{base_dir}/data/raw/NON_CANCER",
    f"{base_dir}/data/processed/CANCER",
    f"{base_dir}/data/processed/NON_CANCER",
    f"{base_dir}/src",
    f"{base_dir}/outputs/models",
    f"{base_dir}/outputs/plots",
]

print("\nCreating directories:")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
print("✓ Folder structure created successfully")

# Step 4: Set and print config variables
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET_PER_CLASS = 500
CLASSES = ['NON_CANCER', 'CANCER']
LABELS = {'NON_CANCER': 0, 'CANCER': 1}

print("\nGlobal Config Variables:")
print(f"IMG_SIZE = {IMG_SIZE}")
print(f"RANDOM_STATE = {RANDOM_STATE}")
print(f"TEST_SIZE = {TEST_SIZE}")
print(f"TARGET_PER_CLASS = {TARGET_PER_CLASS}")
print(f"CLASSES = {CLASSES}")
print(f"LABELS = {LABELS}")

# Step 5: Print completion with timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n✓ Environment Ready at {current_time}")
