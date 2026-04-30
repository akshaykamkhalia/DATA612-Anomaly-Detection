# Industrial Anomaly Detection via Diffusion Reconstruction

DATA-MSML 612 Group 6 final project. University of Maryland.

We train a Diffusion Transformer (DiT-Tiny) on **only normal** product
images and use reconstruction error to flag and localize defects.
Evaluated end-to-end on all 15 categories of the
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
benchmark.

| Metric | Value |
|---|---|
| Mean image-level AUROC (15 categories, L2 + max) | **0.573** |
| Mean pixel-level AUROC | **0.759** |
| Strongest category (image AUROC) | hazelnut, 0.744 |
| Strongest category (pixel AUROC) | leather, 0.913 |
| Model size | DiT-Tiny, 4.4 M parameters, 52 MB checkpoint |
| Inference latency | ~1.5 s / image on RTX 4060 Laptop |

## Authors

Shrikanth Vilvadrinath · Sivram Sahu · Kevin Rathod · Akshay Kamkhalia

---

## Repo layout

```
.
├── code/                  Source for training, evaluation, ablations
│   ├── src/               PyTorch modules: DiT, UNet, diffusion, dataset, scoring
│   ├── notebooks/         Colab notebooks (train from scratch / quick verify)
│   └── requirements.txt
├── demo/                  Streamlit live demo (see demo/README.md)
│   ├── app.py             Drag-and-drop UI with verdict + heatmap + score chart
│   ├── demo_inference.py  classify() API
│   ├── compute_thresholds.py
│   ├── extract_checkpoints.py / prepare_samples.py
│   ├── DEMO_SCRIPT.md          2-3 minute live walkthrough script
│   └── PRESENTATION_SCRIPT.md  Slide-by-slide narration script
├── presentation/          Final .pptx
├── report/                Final report (.tex + .pdf)
├── proposal/              Original proposal (.tex + .pdf)
├── results/               All committed evaluation JSONs
│   ├── dit_l2_and_baselines/   L2 scoring + PatchCore + ablations
│   └── dit_ssim/                SSIM scoring + Conv-AE/PCA baselines
└── figures/               Figures used in the report
```

## What's in this repo vs. external

| | Where |
|---|---|
| Source code, notebooks, results JSON, figures, report, demo | **In this repo** |
| Trained checkpoints (~780 MB total, 15 × 52 MB) | External: `ALL_OUTPUT.zip`, restored locally via `demo/extract_checkpoints.py` |
| MVTec AD dataset (~5 GB) | External: download from [Kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) or [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| Demo sample images (~87 MB) | Recreated locally via `demo/prepare_samples.py` from MVTec test set |

The repo itself is small (~21 MB). All large artifacts are recreated by
the scripts in `demo/`.

---

## Quick start - run the live demo

You need the trained checkpoints (`ALL_OUTPUT.zip`) and the MVTec dataset
on disk. The repo gives you all the code; the `extract_checkpoints.py`
and `prepare_samples.py` scripts do the rest.

```bash
# 1. Environment
pip install -r code/requirements.txt
pip install streamlit plotly

# 2. Restore checkpoints (15 × ~52 MB) from ALL_OUTPUT.zip
cd demo
python extract_checkpoints.py --zip /path/to/ALL_OUTPUT.zip --categories \
    bottle cable capsule carpet grid hazelnut leather metal_nut \
    pill screw tile toothbrush transistor wood zipper

# 3. Build sample-image gallery from your MVTec test set
python prepare_samples.py --data /path/to/mvtec

# 4. Compute per-category thresholds + AUROC
python compute_thresholds.py    # ~30-40 min on a desktop GPU

# 5. Launch
streamlit run app.py
# -> open http://localhost:8501
```

The UI: pick a category, drag-and-drop an image (or click a sample),
get a NORMAL/DEFECTIVE verdict with a heatmap localization, the
reconstruction the model would expect, and a histogram showing where
this image's score falls inside the full test-set distribution.

See [demo/README.md](demo/README.md) for full details.

---

## Quick start - reproduce the results

To retrain from scratch:

```bash
cd code
pip install -r requirements.txt
python -m src.train --category hazelnut --epochs 100 --backbone dit_tiny
python -m src.evaluate --data_root /path/to/mvtec --category hazelnut \
    --checkpoint output/checkpoints/hazelnut/best.pt --scoring l2
```

The `code/notebooks/02_quick_verify.ipynb` notebook walks through
loading our committed checkpoints and reproducing the published
mean-image-AUROC of 0.573 / mean-pixel-AUROC of 0.759 on Colab T4.

---

## Method

1. Train DiT-Tiny on normal images only, per category, 100 epochs at
   128 × 128 resolution with a cosine noise schedule (T = 1000).
2. At test time: partially noise the image to t = 250, denoise with
   50-step DDIM, get the reconstruction.
3. Anomaly map = pixel-wise L2 difference + ResNet-18 feature distance,
   combined 50/50.
4. Image-level score = max over the spatial map.
5. Decision threshold = Youden's J on the per-category test-set ROC
   curve (computed once and cached in `demo/thresholds.json`).

See `report/report.pdf` for the full writeup, including ablations on
scoring method (SSIM / L2 / LPIPS), partial-noise timestep, feature
weighting, and DiT vs UNet backbone.

---

## Key insight

The single most impactful finding of the project was that **scoring
choices dominated final detection quality more than the architecture**.

- SSIM → L2 (same checkpoint, hazelnut): image AUROC **0.407 → 0.808**
- 95th-percentile aggregation → max aggregation (same map): image
  AUROC **0.417 → 0.744**

See slide 12 of the final presentation and section 4 of the report.

---

## License

Educational use - University of Maryland coursework. MVTec AD dataset
is licensed separately by MVTec Software GmbH.
