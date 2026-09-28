# PD234 Assignment 1 — Report

**Object Detection with Monocular Depth Estimation using Pre-Trained Models**

Venkateshh Moningi · M.Des, Department of Design & Manufacturing, IISc · Intelligent User Interface

---

## 1. Objective

Build a computer-vision system that not only detects objects in an image but also
estimates how far each detected object is from the camera, using **pre-trained**
models (no training from scratch). This combines two capabilities that are each
incomplete on their own: object detection knows *what* and *where* but not *how
far*; monocular depth estimation knows *how far* every pixel is but not *what* the
objects are.

## 2. Approach

The system chains two pre-trained foundation models:

- **YOLOv8** (Ultralytics), pre-trained on the **COCO** dataset (80 object
  classes). It performs a single forward pass over the image and returns, for
  each object, a bounding box, a class label, and a confidence score.
- **Depth Anything V2** (Small), a monocular depth-estimation model that produces
  a dense depth map from a single ordinary photograph — no stereo camera or depth
  sensor required.

For every object YOLO detects, the pipeline reads the depth map inside a small
central window of the bounding box and takes the **median** value, which is robust
to background pixels near the box edges. This yields one depth reading per object.

### Depth convention

Depth Anything V2 outputs *inverse depth* (a larger raw value means the point is
**closer** to the camera). To make the numbers intuitive, the raw map is
normalised to a 0–1 **closeness** score (using the 2nd–98th percentile to resist
outliers) and reported as **relative distance** `rd = 1 − closeness`, so
`rd = 0` is the nearest object and `rd = 1` the farthest.

## 3. Pipeline

```
image ──► YOLOv8 ──────────────► boxes + classes + confidences
   │
   └────► Depth Anything V2 ───► dense depth map ──► per-box median depth
                                                          │
                                     combine ◄────────────┘
                                        │
              annotated figure + CSV row + dataset-level analysis
```

## 4. Dataset

The pipeline is evaluated on the **COCO8** sample — 8 validation images drawn from
the COCO dataset. COCO is a widely used object-detection benchmark spanning
**80 classes**. Using a standard subset keeps the evaluation reproducible and
lets YOLO's COCO-trained weights be applied directly.

*(For a larger run, additional images can be placed in `data/images/`.)*

## 5. Results

Across the 8-image sample the system produced **23 detections** spanning
**12 distinct classes** (person, bowl, dog, horse, giraffe, zebra, elephant,
potted plant, broccoli, vase, umbrella, suitcase), an average of **2.88
detections per image**, and a depth reading was obtained for **every**
detection (23/23).

Key result figures (see `results/`):

- **Per-image figures** (`results/figures/*.png`): each shows the detections on
  the left and the depth map on the right with a labelled near/far colour bar and
  pixel axes.
- **`detections_per_class.png`**: distribution of detections across classes.
- **`conf_vs_distance.png`**: detection confidence versus relative distance.
- **`detections.csv`**: the full per-object table (class, confidence, box
  coordinates, closeness, relative distance).

Qualitatively, the depth ordering matches human perception — foreground objects
receive low relative-distance values and background objects high ones — confirming
that the two models combine to give correct spatial awareness.

## 6. Benefit of using pre-trained models

Because both models are pre-trained, the system needs **no training data, no
labelling, no GPU training time, and no ML expertise to reproduce**. YOLOv8 brings
detection knowledge learned from the full COCO dataset, and Depth Anything V2
brings depth knowledge learned from a very large image corpus with strong
zero-shot generalisation. The result is a working detection-plus-depth system
assembled in minutes rather than trained over days.

## 7. Limitations

- **Relative, not metric, depth.** Depth Anything V2 (Small) gives relative depth.
  Converting to metres requires a metric checkpoint or calibration against a known
  reference distance.
- **Within-image comparison only.** Relative-distance values are comparable
  between objects in the same image, not across different images.
- **Detection ceiling.** The system can only place a depth on objects YOLO
  detects; missed objects get no reading.

## 8. Relevance to Intelligent User Interfaces

Perception is the foundation of an intelligent interface: before a system can
respond to the physical world it must know what is present and where it is in 3D.
Detection + depth is exactly the sensing layer behind AR label placement,
assistive "obstacle N metres ahead" guidance, and safe human–robot collaboration
(maintaining a separation distance) — all themes of this course.

---

*Code and instructions: see `README.md`. Run with
`python run_pipeline.py --images data/images --out results --device cpu`.*
