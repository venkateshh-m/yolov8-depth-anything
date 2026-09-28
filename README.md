# YOLOv8 + Depth Anything V2 — Detection with Monocular Depth

**PD234 Assignment 1 — Intelligent User Interface, IISc**
Venkateshh Moningi

A two-stage computer-vision pipeline that runs over a **folder of images**:

1. **YOLOv8** (pre-trained on COCO, 80 classes) — detects *what* and *where* (bounding boxes).
2. **Depth Anything V2** (pre-trained) — estimates *how far* (a dense depth map).

For each detected object the depth at its location is read, so every detection
gets a relative-depth value. The system outputs labelled result figures (with a
depth colour bar), a per-detection CSV, and summary charts + a results table.

---

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Model weights download automatically on first run (YOLOv8 from Ultralytics,
Depth Anything V2 from Hugging Face).

## 2. Get the dataset

```bash
python get_dataset.py
```

Downloads the **COCO8 sample** (8 real COCO validation images) into
`data/images/`. This is a genuine subset of the COCO dataset — a standard,
citable benchmark with 80 object classes.

## 3. Run the pipeline

```bash
python run_pipeline.py --images data/images --out results --device cpu
```

Use `--device 0` if you have an NVIDIA GPU (much faster). Outputs land in
`results/`:

```
results/
  figures/        one labelled PNG per image (detections | depth + colour bar)
  detections.csv  every detection: class, confidence, box, closeness, rel-distance
  analysis/
    detections_per_class.png   bar chart of class distribution
    conf_vs_distance.png       confidence vs relative distance scatter
    summary.md                 results table (images, detections, per-class)
    summary.json               same, machine-readable
```

## 4. How it works

1. **Detection** — YOLOv8 returns boxes `(x1,y1,x2,y2)`, class, and confidence.
2. **Depth** — Depth Anything V2 produces a dense depth map, resized to the image.
3. **Per-object depth** — a small central window of each box is sampled and the
   **median** taken (robust to background pixels near the box edges).
4. **Normalisation** — the raw model output is *inverse depth* (higher = closer).
   It is normalised (2nd–98th percentile) to a 0–1 **closeness** score, and
   reported as **relative distance** `rd = 1 − closeness` (0 = nearest, 1 = farthest)
   so the numbers read naturally.

## 5. Answers to the submission form

| Form question | Answer |
|---|---|
| Pre-trained model used | **Depth Anything** (+ YOLOv8) |
| Dataset | **COCO** (COCO8 validation sample) |
| Number of classes | **80** (COCO) |
| Benefit of pre-trained model | No training/GPU/labelled data needed; strong zero-shot detection *and* depth out of the box |

## 6. Limitations (state in report)

- Depth Anything V2 (Small) gives **relative** depth, not metres. For metric
  depth use a metric checkpoint or calibrate against a known reference distance.
- Relative distance is comparable *within* one image, not across images.

## 7. Repository layout

```
run_pipeline.py     main pipeline (detection + depth + figures + analysis)
get_dataset.py      downloads the COCO8 sample
requirements.txt    dependencies
REPORT.md           write-up of method, results, and limitations
data/images/        input images (after get_dataset.py)
results/            generated outputs (after run_pipeline.py)
```
