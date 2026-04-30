# DiT Anomaly Detection — Live Demo

Streamlit-based live demo for the DATA-MSML 612 Group 6 project: DiT-based
anomaly detection via diffusion reconstruction. Drag-and-drop a product
image, the app classifies it as **NORMAL** or **DEFECTIVE** and visualizes
*where* the defect is via the model's reconstruction-difference heatmap.

This folder is **self-contained**: it does not import from `../code/`
and does not modify any other directory in the repo.

---

## Layout

```
demo/
  app.py                       # Streamlit frontend
  demo_inference.py            # core classify() API + model loaders
  extract_checkpoints.py       # one-time: pull .pt from ALL_OUTPUT.zip
  prepare_samples.py           # one-time: copy demo images per category
  compute_thresholds.py        # one-time: per-category Youden's-J threshold
  test_demo.py                 # smoke test: classify all sample images
  README.md                    # this file

  thresholds.json              # AUROC + decision boundary per category
  sample_images/{cat}/{normal,defective}/   # 6 click-to-try images per cat
  checkpoints/{cat}/best.pt    # NOT in git -- recreated locally (~780 MB total)
```

---

## First-time setup (3 commands)

From the `demo/` directory:

```bash
# 1. Extract trained checkpoints from ALL_OUTPUT.zip
#    Pass --zip OR set ALL_OUTPUT_ZIP env var
python extract_checkpoints.py --zip /path/to/ALL_OUTPUT.zip

# 2. Copy demo images from the MVTec test set
#    Pass --data OR set MVTEC_ROOT env var
python prepare_samples.py --data /path/to/mvtec

# 3. Compute per-category thresholds + AUROC (skips already-computed ones)
python compute_thresholds.py
```

By default `extract_checkpoints.py` extracts the 3 strongest demo categories
(`hazelnut`, `leather`, `wood`). For all 15:

```bash
python extract_checkpoints.py --zip /path/to/ALL_OUTPUT.zip --categories \
    bottle cable capsule carpet grid hazelnut leather metal_nut \
    pill screw tile toothbrush transistor wood zipper
python prepare_samples.py --data /path/to/mvtec
python compute_thresholds.py        # ~3-4 min/category on RTX 4060
```

---

## Run the demo

```bash
conda run -n base streamlit run app.py
# -> open http://localhost:8501
```

Models load into VRAM once (~3-5 s the first time the app renders), then
stay cached for the session. Inference is ~1.5 s per image with 5-run
averaging.

The UI:
- **Sidebar:** category dropdown, AUROC card, threshold card, click-to-try
  sample images, inference-runs slider, device indicator
- **Main:** drag-and-drop uploader, big animated NORMAL/DEFECTIVE banner,
  metric strip (score / threshold / inference time), 4-panel comparison
  (input · reconstruction · heatmap · overlay), score-distribution
  histogram showing where the current image falls vs the full test set,
  confidence gauge, and a "Run details" expander

---

## Smoke test (no Streamlit needed)

```bash
python test_demo.py
```

Iterates the 6 demo images per category through `classify()` and reports
label vs ground truth + saves visualizations to `output/e2e/`. Useful to
verify a fresh checkout without launching the UI.

---

## Programmatic use

```python
from demo_inference import classify

result = classify("path/to/image.png", category="hazelnut")
print(result["label"])          # "NORMAL" | "DEFECTIVE"
print(result["score"])          # image-level anomaly score
print(result["threshold"])      # decision boundary (Youden's J)
print(result["confidence"])     # 0..1 sigmoid of margin

result["heatmap"].save("heatmap.png")        # PIL Image, jet colormap
result["overlay"].save("overlay.png")        # heatmap blended on input
result["reconstruction"].save("recon.png")   # what the model "expects"
```

---

## What the model is doing

For each query image:

1. Resize to 128 x 128, normalize to `[-1, 1]`
2. Add `t_partial=250` steps of cosine-schedule noise
3. DDIM denoise (50 steps) -> reconstruction (the "normal" version)
4. Combined anomaly map = 0.5 * pixel(L2) + 0.5 * feature(ResNet-18, layers 1-3)
5. Image score = max of combined map, averaged across 5 deterministic seeds
6. Compare score to per-category Youden's-J threshold

The model never saw a defective image during training -- only "good" examples.
So whatever it can't reconstruct accurately is, by definition, anomalous.

---

## Caveat

This is a reconstruction-based anomaly detector, not a discriminative classifier.
The binary verdict is approximate (AUROC 0.5-0.8 across categories), but the
**heatmap localization is excellent and is the real output**. Lead the demo
with the heatmap; present the verdict as secondary.

The committed L2 baseline (`results/dit_l2_and_baselines/l2_summary.json`) is
the published reference; this demo's per-category numbers are recomputed with
5-run averaging and saved to `thresholds.json`.
