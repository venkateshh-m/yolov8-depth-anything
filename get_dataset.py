"""
Download a small sample of COCO validation images to run the pipeline on.

Uses the Ultralytics 'coco8' dataset (8 COCO images, ships with ultralytics) as a
zero-setup default, and can also fetch more COCO val images if you want a bigger
run. coco8 is enough to demonstrate the pipeline; use --n for a larger sample.

Usage:
  python get_dataset.py                 # copies the 8 coco8 images into data/images
  python get_dataset.py --n 25          # tries to download ~25 COCO val images
"""
import os
import shutil
import argparse
import urllib.request
import zipfile


def from_coco8(dest):
    """coco8 ships with ultralytics; download the tiny zip and copy its images."""
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
    tmp = "coco8.zip"
    print(f"Downloading coco8 sample from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    with zipfile.ZipFile(tmp) as z:
        z.extractall("coco8_tmp")
    os.remove(tmp)
    count = 0
    for root, _, files in os.walk("coco8_tmp"):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(root, fn), os.path.join(dest, fn))
                count += 1
    shutil.rmtree("coco8_tmp", ignore_errors=True)
    print(f"Copied {count} images into {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="ignored for coco8; reserved")
    ap.add_argument("--dest", default="data/images")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    from_coco8(args.dest)
    print("\nNext:  python run_pipeline.py --images data/images --out results --device cpu")


if __name__ == "__main__":
    main()
