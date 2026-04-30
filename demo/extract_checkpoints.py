
import argparse
import os
import zipfile
from pathlib import Path

DEFAULT_CATEGORIES = ["hazelnut", "leather", "wood"]
DEFAULT_ZIP = os.environ.get(
    "ALL_OUTPUT_ZIP",
    str(Path(__file__).resolve().parent.parent / "ALL_OUTPUT.zip"),
)
OUT_DIR = Path(__file__).resolve().parent / "checkpoints"


def extract(categories, zip_path):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Zip not found: {zip_path}\n"
            f"Pass --zip <path> or set ALL_OUTPUT_ZIP env var."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_names = z.namelist()
        for cat in categories:
            target = f"output/checkpoints/{cat}/best.pt"
            if target not in all_names:
                print(f"  [SKIP] {cat}: not in zip")
                continue
            dest = OUT_DIR / cat / "best.pt"
            if dest.exists():
                print(f"  [HAVE] {cat}: {dest.relative_to(OUT_DIR.parent)}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(target) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            size_mb = dest.stat().st_size / 1e6
            print(f"  [OK]   {cat}: {size_mb:.1f} MB -> {dest.relative_to(OUT_DIR.parent)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--zip", default=DEFAULT_ZIP,
                   help=f"Path to ALL_OUTPUT.zip (default: ${{ALL_OUTPUT_ZIP}} or repo-root sibling)")
    args = p.parse_args()
    print(f"Extracting from {args.zip}")
    print(f"Target: {OUT_DIR}")
    print(f"Categories: {args.categories}\n")
    extract(args.categories, args.zip)


if __name__ == "__main__":
    main()
