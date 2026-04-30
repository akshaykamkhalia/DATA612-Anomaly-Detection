"""
For each category that has a checkpoint, run the trained model on the entire
MVTec test set, collect image-level scores, then compute:

  - Image AUROC
  - Optimal decision threshold (Youden's J on the ROC: argmax TPR - FPR)
  - Mean/std of normal vs defective score distributions
  - The score quantiles that would correspond to common operating points

Saves results to thresholds.json. Re-run to refresh.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from demo_inference import (
    DEMO_DIR, DEVICE, available_categories, score_image,
)

DATA_ROOT = DEMO_DIR.parent / "data" / "mvtec"
OUT_PATH = DEMO_DIR / "thresholds.json"


def collect_scores(category: str, n_repeats: int = 5) -> dict:
    """
    Score every test image n_repeats times with deterministic seeds (0..n-1)
    and average. Averaging dramatically reduces stochastic noise from
    `q_sample`'s random noise draw and brings AUROC close to the published
    baselines.
    """
    test_root = DATA_ROOT / category / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"No MVTec test data for {category} at {test_root}")

    scores = []
    labels = []  # 0 = normal, 1 = defective
    paths = []
    defect_types = []

    subdirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    for sd in subdirs:
        is_good = sd.name == "good"
        for img_path in sorted(sd.glob("*.png")):
            ss = []
            for k in range(n_repeats):
                torch.manual_seed(k)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(k)
                # n_runs=1 here -- the outer n_repeats loop is the averaging.
                r = score_image(img_path, category, n_runs=1)
                ss.append(r["score"])
            scores.append(float(np.mean(ss)))
            labels.append(0 if is_good else 1)
            paths.append(str(img_path.relative_to(DATA_ROOT)))
            defect_types.append(sd.name)

    return {
        "scores": np.asarray(scores),
        "labels": np.asarray(labels),
        "paths": paths,
        "defect_types": defect_types,
    }


def compute(category: str, n_repeats: int = 5) -> dict:
    print(f"\n[{category}] scoring test set (n_repeats={n_repeats})...")
    t0 = time.time()
    data = collect_scores(category, n_repeats=n_repeats)
    elapsed = time.time() - t0

    scores = data["scores"]
    labels = data["labels"]
    n_total = len(scores)
    n_good = int((labels == 0).sum())
    n_bad = int((labels == 1).sum())

    auroc = float(roc_auc_score(labels, scores)) if (n_good > 0 and n_bad > 0) else float("nan")

    # Youden's J: best operating point on the ROC curve
    fpr, tpr, thr_grid = roc_curve(labels, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    threshold = float(thr_grid[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])
    # roc_curve sometimes returns +inf for the degenerate first threshold.
    # Clamp to the median of normal+defective so the demo at least gives a
    # reasonable verdict (instead of flagging everything as NORMAL).
    if not np.isfinite(threshold):
        threshold = float(np.median(scores))

    # 95th percentile of NORMAL scores: a more conservative alternative
    p95_normal = float(np.percentile(scores[labels == 0], 95)) if n_good > 0 else float("nan")

    s_good = scores[labels == 0]
    s_bad = scores[labels == 1]

    print(f"  done in {elapsed:.1f}s ({n_total} images)")
    print(f"  AUROC:           {auroc:.4f}")
    print(f"  threshold (J):   {threshold:.6f}  -> TPR={best_tpr:.3f} FPR={best_fpr:.3f}")
    print(f"  threshold (P95): {p95_normal:.6f}")
    print(f"  normal scores:   mean={s_good.mean():.4f} std={s_good.std():.4f} "
          f"(n={n_good})")
    print(f"  defect scores:   mean={s_bad.mean():.4f} std={s_bad.std():.4f} "
          f"(n={n_bad})")

    return {
        "category": category,
        "n_repeats": n_repeats,
        "n_total": n_total,
        "n_normal": n_good,
        "n_defective": n_bad,
        "auroc": auroc,
        "threshold": threshold,           # the one demo_inference.py reads
        "threshold_method": "youden_j",
        "threshold_p95_normal": p95_normal,
        "tpr_at_threshold": best_tpr,
        "fpr_at_threshold": best_fpr,
        "normal_score_mean": float(s_good.mean()),
        "normal_score_std": float(s_good.std()),
        "defect_score_mean": float(s_bad.mean()),
        "defect_score_std": float(s_bad.std()),
        "elapsed_sec": float(elapsed),
        "all_scores": [
            {"path": p, "label": int(l), "defect_type": d, "score": float(s)}
            for p, l, d, s in zip(data["paths"], data["labels"],
                                  data["defect_types"], data["scores"])
        ],
    }


def main():
    cats = available_categories()
    if not cats:
        raise SystemExit("No checkpoints found. Run extract_checkpoints.py first.")

    print(f"Device: {DEVICE}")
    print(f"Categories with checkpoints: {cats}")

    # Reuse existing thresholds unless recomputation was requested.
    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())

    results = dict(existing)
    for cat in cats:
        if cat in results and results[cat].get("auroc") is not None:
            print(f"\n[{cat}] already computed (AUROC={results[cat]['auroc']:.4f}), skipping. "
                  f"Delete from thresholds.json to recompute.")
            continue
        results[cat] = compute(cat, n_repeats=5)
        # Save incrementally so a crash mid-run doesn't lose work
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nSaved -> {OUT_PATH}")
    print("\nSummary:")
    print(f"{'category':<12} {'AUROC':>7} {'thresh':>10} {'P95':>10}")
    for cat, r in results.items():
        print(f"{cat:<12} {r['auroc']:>7.4f} {r['threshold']:>10.4f} {r['threshold_p95_normal']:>10.4f}")


if __name__ == "__main__":
    main()
