"""
Copy a small set of MVTec test images per category into demo/sample_images/
so the Streamlit demo has self-contained click-to-try examples.

Picks 3 'good' images and 3 defective images (spread across defect types)
per category that has a checkpoint extracted on disk.

Run from demo/:
    python prepare_samples.py --data /path/to/mvtec

Or set the env var MVTEC_ROOT to skip --data on every call.
"""
import argparse
import os
import shutil
from pathlib import Path

DEMO_DIR = Path(__file__).parent
DEFAULT_DATA_ROOT = os.environ.get(
    "MVTEC_ROOT", str(DEMO_DIR.parent / "data" / "mvtec"),
)
OUT_DIR = DEMO_DIR / "sample_images"

GOOD_PER_CAT = 3
BAD_PER_CAT = 3


def main(data_root):
    DATA_ROOT = Path(data_root)
    if not DATA_ROOT.exists():
        raise SystemExit(
            f"MVTec data not found at: {DATA_ROOT}\n"
            f"Pass --data <path> or set MVTEC_ROOT env var."
        )
    cats = sorted([p.parent.name for p in (DEMO_DIR / "checkpoints").glob("*/best.pt")])
    if not cats:
        raise SystemExit("No checkpoints found. Run extract_checkpoints.py first.")

    for cat in cats:
        test_root = DATA_ROOT / cat / "test"
        if not test_root.exists():
            print(f"[skip] {cat}: no test data at {test_root}")
            continue

        good_dst = OUT_DIR / cat / "normal"
        bad_dst = OUT_DIR / cat / "defective"
        good_dst.mkdir(parents=True, exist_ok=True)
        bad_dst.mkdir(parents=True, exist_ok=True)

        # Skip if already populated (idempotent re-run)
        if len(list(good_dst.glob("*.png"))) >= GOOD_PER_CAT and \
           len(list(bad_dst.glob("*.png"))) >= BAD_PER_CAT:
            print(f"[have] {cat}: already populated")
            continue

        good_imgs = sorted((test_root / "good").glob("*.png"))[:GOOD_PER_CAT]
        for img in good_imgs:
            shutil.copy2(img, good_dst / img.name)

        defect_dirs = sorted([d for d in test_root.iterdir()
                              if d.is_dir() and d.name != "good"])
        bad_imgs = []
        for ddir in defect_dirs:
            if len(bad_imgs) >= BAD_PER_CAT:
                break
            for img in sorted(ddir.glob("*.png"))[:1]:
                bad_imgs.append((ddir.name, img))
        # Pad if not enough defect types
        while len(bad_imgs) < BAD_PER_CAT and defect_dirs:
            ddir = defect_dirs[0]
            more = sorted(ddir.glob("*.png"))
            if len(more) > len(bad_imgs):
                bad_imgs.append((ddir.name, more[len(bad_imgs)]))
            else:
                break
        for defect_type, img in bad_imgs:
            shutil.copy2(img, bad_dst / f"{defect_type}_{img.name}")

        print(f"[ok]   {cat}: {len(good_imgs)} good, {len(bad_imgs)} defective")

    print(f"\nSamples in: {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA_ROOT,
                   help="Path to mvtec/ folder (default: $MVTEC_ROOT or repo-root sibling)")
    args = p.parse_args()
    main(args.data)
