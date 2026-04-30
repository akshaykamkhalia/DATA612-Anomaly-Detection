

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Try Anomalib v1.x first, then fall back to v0.x.
_ANOMALIB_VERSION = None

try:
    import anomalib

    _ANOMALIB_VERSION = getattr(anomalib, "__version__", "unknown")
except ImportError:
    print(
        "ERROR: anomalib is not installed.\n"
        "Install it with:  pip install 'anomalib>=1.0'\n"
        "Or:               pip install anomalib"
    )
    sys.exit(1)

_USE_V1 = True
try:
    from anomalib.data import MVTec  # v1.x datamodule
    from anomalib.models import Patchcore, ReverseDistillation  # v1.x models
    from anomalib.engine import Engine  # v1.x engine (replaces Trainer)
except ImportError:
    _USE_V1 = False

if not _USE_V1:
    try:
        from anomalib.data import MVTec  # same in some v0.x builds
    except ImportError:
        try:
            from anomalib.data.mvtec import MVTec
        except ImportError:
            print(
                f"ERROR: Could not import MVTec datamodule from anomalib "
                f"(version {_ANOMALIB_VERSION}).  Please upgrade:\n"
                "  pip install 'anomalib>=1.0'"
            )
            sys.exit(1)

    try:
        from anomalib.models import Patchcore
    except ImportError:
        try:
            from anomalib.models.patchcore import Patchcore
        except ImportError:
            from anomalib.models import PatchCore as Patchcore  # type: ignore[no-redef]

    try:
        from anomalib.models import ReverseDistillation
    except ImportError:
        try:
            from anomalib.models.reverse_distillation import (
                ReverseDistillation,
            )
        except ImportError:
            print(
                f"ERROR: Could not import ReverseDistillation from anomalib "
                f"(version {_ANOMALIB_VERSION}).  Please upgrade:\n"
                "  pip install 'anomalib>=1.0'"
            )
            sys.exit(1)

    # v0.x uses a Trainer (lightning-based), not Engine
    try:
        from anomalib.engine import Engine
    except ImportError:
        Engine = None  # type: ignore[assignment,misc]

try:
    from tabulate import tabulate
except ImportError:
    print("ERROR: tabulate is not installed.  pip install tabulate")
    sys.exit(1)

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

METHOD_CHOICES = ["patchcore", "reverse_distillation", "all"]

# Default data root -- anomalib will download MVTec here if missing.
_DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "mvtec")

# Output directory for saved results.
_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "results"


def _build_datamodule(category: str, data_root: str, img_size: int = 256):
    """Instantiate the anomalib MVTec datamodule for *category*."""
    try:
        # v1.x signature
        datamodule = MVTec(
            root=data_root,
            category=category,
            image_size=img_size,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,  # safe default; bump on Colab/Kaggle
        )
    except TypeError:
        # Some anomalib builds use slightly different kwarg names.
        datamodule = MVTec(
            root=data_root,
            category=category,
            image_size=(img_size, img_size),
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
        )
    return datamodule


def _build_model(method: str):
    if method == "patchcore":
        model = Patchcore()
        max_epochs = 1  # memory-bank, no gradient training
    elif method == "reverse_distillation":
        model = ReverseDistillation()
        max_epochs = 200
    else:
        raise ValueError(f"Unknown method: {method}")
    return model, max_epochs


def _extract_metrics(test_results) -> dict:

    img_auroc = None
    pix_auroc = None

    # Engine.test may return a list of metric dictionaries.
    if isinstance(test_results, list):
        for entry in test_results:
            if isinstance(entry, dict):
                for key, val in entry.items():
                    k = key.lower()
                    if "image_auroc" in k or "image-level" in k.replace(" ", ""):
                        img_auroc = float(val)
                    if "pixel_auroc" in k or "pixel-level" in k.replace(" ", ""):
                        pix_auroc = float(val)

    # Some versions return a single metric dictionary instead.
    elif isinstance(test_results, dict):
        for key, val in test_results.items():
            k = key.lower()
            if "image_auroc" in k:
                img_auroc = float(val)
            if "pixel_auroc" in k:
                pix_auroc = float(val)

    return {
        "image_AUROC": img_auroc,
        "pixel_AUROC": pix_auroc,
    }


def _run_single(
    method: str,
    category: str,
    data_root: str,
    img_size: int,
    accelerator: str,
) -> dict:
    """Train + test one (method, category) pair.  Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {method.upper()} -- {category}")
    print(f"{'='*60}")

    datamodule = _build_datamodule(category, data_root, img_size)
    model, max_epochs = _build_model(method)

    t0 = time.time()

    if Engine is not None:
        # Anomalib v1.x uses Engine.
        engine = Engine(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            default_root_dir=str(
                _OUTPUT_DIR.parent / "anomalib_runs" / method / category
            ),
        )
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
    else:
        # Anomalib v0.x falls back to the Lightning Trainer API.
        from pytorch_lightning import Trainer

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            default_root_dir=str(
                _OUTPUT_DIR.parent / "anomalib_runs" / method / category
            ),
        )
        trainer.fit(model=model, datamodule=datamodule)
        test_results = trainer.test(model=model, datamodule=datamodule)

    elapsed = time.time() - t0
    metrics = _extract_metrics(test_results)
    metrics["time_sec"] = round(elapsed, 1)
    metrics["category"] = category
    metrics["method"] = method

    print(
        f"  -> image_AUROC={metrics['image_AUROC']}, "
        f"pixel_AUROC={metrics['pixel_AUROC']}, "
        f"time={metrics['time_sec']}s"
    )
    return metrics



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anomalib baselines (PatchCore / Reverse Distillation) on MVTec AD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=METHOD_CHOICES,
        help="Baseline method to run (default: all).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="MVTec category name, or 'all' for all 15 (default: all).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=_DEFAULT_DATA_ROOT,
        help="Root directory of MVTec AD dataset.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size for anomalib baselines (default: 256).",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator for Engine/Trainer: auto, gpu, cpu (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_DIR / "baselines_anomalib.json"),
        help="Path for JSON results file.",
    )
    args = parser.parse_args()

    if args.category.lower() == "all":
        categories = MVTEC_CATEGORIES
    else:
        cat = args.category.lower()
        if cat not in MVTEC_CATEGORIES:
            parser.error(
                f"Unknown category '{cat}'. Choose from: {MVTEC_CATEGORIES}"
            )
        categories = [cat]

    if args.method == "all":
        methods = ["patchcore", "reverse_distillation"]
    else:
        methods = [args.method]

    print(f"anomalib version : {_ANOMALIB_VERSION}")
    print(f"API mode         : {'v1.x (Engine)' if _USE_V1 else 'v0.x (Trainer fallback)'}")
    print(f"Methods          : {methods}")
    print(f"Categories       : {categories}")
    print(f"Image size       : {args.img_size}")
    print(f"Data root        : {args.data_root}")

    all_results: list[dict] = []

    for method in methods:
        for category in categories:
            try:
                metrics = _run_single(
                    method=method,
                    category=category,
                    data_root=args.data_root,
                    img_size=args.img_size,
                    accelerator=args.accelerator,
                )
                all_results.append(metrics)
            except Exception as exc:
                print(f"  !! FAILED: {method}/{category}: {exc}")
                all_results.append(
                    {
                        "method": method,
                        "category": category,
                        "image_AUROC": None,
                        "pixel_AUROC": None,
                        "time_sec": None,
                        "error": str(exc),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    for method in methods:
        rows = [r for r in all_results if r["method"] == method]
        if not rows:
            continue

        table_data = []
        for r in rows:
            img_auc = (
                f"{r['image_AUROC']:.4f}" if r.get("image_AUROC") is not None else "N/A"
            )
            pix_auc = (
                f"{r['pixel_AUROC']:.4f}" if r.get("pixel_AUROC") is not None else "N/A"
            )
            t = f"{r['time_sec']:.1f}" if r.get("time_sec") is not None else "N/A"
            err = r.get("error", "")
            table_data.append([r["category"], img_auc, pix_auc, t, err])

        # Compute mean for numeric columns (skip failures).
        valid = [r for r in rows if r.get("image_AUROC") is not None]
        if valid:
            mean_img = sum(r["image_AUROC"] for r in valid) / len(valid)
            mean_pix_vals = [r["pixel_AUROC"] for r in valid if r.get("pixel_AUROC") is not None]
            mean_pix = sum(mean_pix_vals) / len(mean_pix_vals) if mean_pix_vals else None
            mean_t = sum(r["time_sec"] for r in valid) / len(valid)
            table_data.append([
                "MEAN",
                f"{mean_img:.4f}",
                f"{mean_pix:.4f}" if mean_pix is not None else "N/A",
                f"{mean_t:.1f}",
                "",
            ])

        header = ["Category", "Image AUROC", "Pixel AUROC", "Time (s)", "Error"]
        print(f"\n{'='*60}")
        print(f"  {method.upper()} Results")
        print(f"{'='*60}")
        print(tabulate(table_data, headers=header, tablefmt="grid"))


if __name__ == "__main__":
    main()
