"""
YOLOv8 + Depth Anything V2  —  Batch Detection with Monocular Depth Estimation
==============================================================================
Assignment 1 (PD234) — Intelligent User Interface, IISc

Runs a two-stage perception pipeline over a FOLDER of images:
  1. YOLOv8  (pre-trained on COCO, 80 classes)  ->  detects WHAT and WHERE
  2. Depth Anything V2 (pre-trained)            ->  estimates HOW FAR (depth map)
For every detected object it reads the depth at that object's location, so each
detection gets a relative-depth value. Results are saved as:
  - annotated figures with a labelled depth colour bar   (results/figures/)
  - a per-detection CSV                                  (results/detections.csv)
  - summary charts + a results table                     (results/analysis/)

Usage
-----
  python run_pipeline.py --images data/images --out results --device cpu

Note on depth: the free Depth Anything V2 model outputs *relative inverse depth*
(higher = closer). This script converts it to a normalised 0..1 "closeness"
score and also to a 0..1 "relative distance" (1 = farthest) so the numbers read
naturally. These are RELATIVE, not metres — see README for how to get metric.
"""

import os
import csv
import glob
import json
import argparse
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from transformers import pipeline


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
#  Depth helpers
# --------------------------------------------------------------------------- #
def get_depth_map(depth_pipe, frame_bgr, out_w, out_h):
    """Run Depth Anything V2 on a BGR frame -> float32 depth map at (out_h,out_w).

    Returns the RAW model output (inverse depth: higher = closer)."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    out = depth_pipe(pil)
    depth = out["depth"] if isinstance(out, dict) else out[0]["depth"]
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[::-1] != (out_w, out_h):
        depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return depth


def normalise_closeness(depth_raw):
    """Raw inverse depth -> 0..1 closeness (1 = closest). Robust to outliers."""
    lo, hi = np.percentile(depth_raw, 2), np.percentile(depth_raw, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((depth_raw - lo) / (hi - lo), 0, 1)


def object_depth(closeness_map, x1, y1, x2, y2, window=7):
    """Median closeness in a small central window of the box (robust)."""
    h, w = closeness_map.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = window // 2
    patch = closeness_map[max(0, cy - half):cy + half + 1,
                          max(0, cx - half):cx + half + 1]
    if patch.size == 0:
        patch = closeness_map[y1:y2 + 1, x1:x2 + 1]
    return float(np.median(patch)) if patch.size else None


# --------------------------------------------------------------------------- #
#  Figure drawing  (properly labelled, with colour bar = the "axes")
# --------------------------------------------------------------------------- #
def save_figure(orig_bgr, boxes, labels, closeness_map, out_path, title):
    """Left: image + boxes. Right: depth map with a labelled near/far colour bar."""
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    annotated = orig_rgb.copy()

    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (int(x1), int(y1) - th - 6),
                      (int(x1) + tw + 2, int(y1)), (0, 255, 0), -1)
        cv2.putText(annotated, lab, (int(x1) + 1, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    axL.imshow(annotated)
    axL.set_title("YOLOv8 detections  (label: class  conf  rel-dist)", fontsize=11)
    axL.set_xlabel("image x (pixels)"); axL.set_ylabel("image y (pixels)")

    im = axR.imshow(closeness_map, cmap="turbo")
    axR.set_title("Depth Anything V2  (relative depth)", fontsize=11)
    axR.set_xlabel("image x (pixels)"); axR.set_ylabel("image y (pixels)")
    cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
    cbar.set_label("relative closeness  (0 = far, 1 = near)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Analysis outputs
# --------------------------------------------------------------------------- #
def write_analysis(rows, class_counter, n_images, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Detections-per-class bar chart
    if class_counter:
        items = class_counter.most_common()
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(names, vals, color="#4C78A8")
        ax.set_title("Detections per class across the dataset")
        ax.set_xlabel("COCO class"); ax.set_ylabel("number of detections")
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "detections_per_class.png"), dpi=120)
        plt.close(fig)

    # 2) Confidence vs relative distance scatter
    confs = [r["conf"] for r in rows if r["rel_distance"] is not None]
    dists = [r["rel_distance"] for r in rows if r["rel_distance"] is not None]
    if confs:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(dists, confs, s=14, alpha=0.5, color="#E45756")
        ax.set_title("Detection confidence vs relative distance")
        ax.set_xlabel("relative distance  (0 = nearest, 1 = farthest)")
        ax.set_ylabel("YOLO confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "conf_vs_distance.png"), dpi=120)
        plt.close(fig)

    # 3) Summary table (markdown)
    total = len(rows)
    avg_per_img = total / n_images if n_images else 0
    with_depth = sum(1 for r in rows if r["rel_distance"] is not None)
    md = [
        "# Results Summary\n",
        f"- Images processed: **{n_images}**",
        f"- Total detections: **{total}**",
        f"- Average detections per image: **{avg_per_img:.2f}**",
        f"- Distinct classes detected: **{len(class_counter)}**",
        f"- Detections with a depth reading: **{with_depth}/{total}**\n",
        "## Detections per class\n",
        "| Class | Count |", "|---|---|",
    ]
    for k, v in class_counter.most_common():
        md.append(f"| {k} | {v} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(images=n_images, detections=total,
                       avg_per_image=avg_per_img,
                       classes=dict(class_counter)), f, indent=2)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of input images")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out, "figures")
    ana_dir = os.path.join(args.out, "analysis")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)

    paths = []
    for e in IMG_EXTS:
        paths += glob.glob(os.path.join(args.images, "*" + e))
        paths += glob.glob(os.path.join(args.images, "*" + e.upper()))
    paths = sorted(set(paths))
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    print(f"Found {len(paths)} images.")

    print("Loading YOLOv8 ...")
    yolo = YOLO(args.yolo_weights)
    try:
        dev = int(args.device)
    except (ValueError, TypeError):
        dev = args.device
    print("Loading Depth Anything V2 ...")
    depth_pipe = pipeline(task="depth-estimation", model=args.depth_model, device=dev)

    rows = []
    class_counter = Counter()
    csv_path = os.path.join(args.out, "detections.csv")
    csv_f = open(csv_path, "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["image", "det_idx", "class", "conf",
                 "x1", "y1", "x2", "y2", "rel_closeness", "rel_distance"])

    for p in tqdm(paths, desc="Processing"):
        frame = cv2.imread(p)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        res = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        depth_raw = get_depth_map(depth_pipe, frame, w, h)
        closeness = normalise_closeness(depth_raw)

        boxes_xy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        scores = res.boxes.conf.cpu().numpy() if len(boxes_xy) else []
        classes = res.boxes.cls.cpu().numpy() if len(boxes_xy) else []
        names = res.names

        draw_boxes, draw_labels = [], []
        base = os.path.basename(p)
        for i, box in enumerate(boxes_xy):
            x1, y1, x2, y2 = box
            cls = names.get(int(classes[i]), str(int(classes[i])))
            conf = float(scores[i])
            c = object_depth(closeness, x1, y1, x2, y2)
            rel_dist = (1.0 - c) if c is not None else None   # 0 near .. 1 far
            class_counter[cls] += 1
            draw_boxes.append((x1, y1, x2, y2))
            draw_labels.append(f"{cls} {conf:.2f} rd:{rel_dist:.2f}" if rel_dist is not None
                               else f"{cls} {conf:.2f}")
            cw.writerow([base, i, cls, f"{conf:.4f}",
                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                         f"{c:.4f}" if c is not None else "",
                         f"{rel_dist:.4f}" if rel_dist is not None else ""])
            rows.append(dict(image=base, cls=cls, conf=conf, rel_distance=rel_dist))

        save_figure(frame, draw_boxes, draw_labels, closeness,
                    os.path.join(fig_dir, os.path.splitext(base)[0] + "_result.png"),
                    title=f"{base}  —  {len(draw_boxes)} detections")

    csv_f.close()
    write_analysis(rows, class_counter, len(paths), ana_dir)
    print(f"\nDone. {len(rows)} detections over {len(paths)} images.")
    print(f"  Figures : {fig_dir}")
    print(f"  CSV     : {csv_path}")
    print(f"  Analysis: {ana_dir}  (charts + summary.md)")


if __name__ == "__main__":
    main()
