# setup_project.py
import shutil
from pathlib import Path

# Source (where you extracted the dataset)
SOURCE_DIR = Path("BUU-LSPINE_400/LA")

# Destination (your organized project folder)
DEST_DIR = Path("data/raw")
IMAGES_DEST = DEST_DIR / "images"
ANNOTATIONS_DEST = DEST_DIR / "annotations"

IMAGES_DEST.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DEST.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR.resolve()}")

# Dataset files are flat inside SOURCE_DIR: *.jpg and *.csv
for image_file in SOURCE_DIR.glob("*.jpg"):
    shutil.copy2(image_file, IMAGES_DEST / image_file.name)

for annotation_file in SOURCE_DIR.glob("*.csv"):
    shutil.copy2(annotation_file, ANNOTATIONS_DEST / annotation_file.name)

print(f"Copied {len(list(IMAGES_DEST.glob('*.jpg')))} images")
print(f"Copied {len(list(ANNOTATIONS_DEST.glob('*.csv')))} annotations")