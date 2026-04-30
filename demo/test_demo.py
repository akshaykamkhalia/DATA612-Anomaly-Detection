"""
End-to-end test of classify(): for each category, run on the 3 good and 3
defective sample images and report label, score, threshold, and accuracy.
Saves visualizations for one good + one defective per category.
"""
import time
from pathlib import Path

from demo_inference import (
    DEMO_DIR, DEVICE, available_categories, classify,
)

SAMPLES = DEMO_DIR / "sample_images"
OUT = DEMO_DIR / "output" / "e2e"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cats = available_categories()
    print(f"Device: {DEVICE}")
    print(f"Categories: {cats}\n")

    summary = {}
    for cat in cats:
        good_dir = SAMPLES / cat / "normal"
        bad_dir = SAMPLES / cat / "defective"
        good_imgs = sorted(good_dir.glob("*.png"))
        bad_imgs = sorted(bad_dir.glob("*.png"))

        results = []
        print(f"=== {cat} ===")
        for tag, paths, expected in [("NORMAL", good_imgs, "NORMAL"),
                                     ("DEFECTIVE", bad_imgs, "DEFECTIVE")]:
            for img in paths:
                t0 = time.time()
                r = classify(img, cat)
                dt = time.time() - t0
                ok = r["label"] == expected
                results.append((expected, r["label"], r["score"], r["threshold"], ok))
                tick = "OK" if ok else "MISS"
                print(f"  [{tick:4s}] {tag:9s} {img.name:25s} "
                      f"score={r['score']:.4f} thr={r['threshold']:.4f} "
                      f"margin={r['margin']:+.4f} ({dt:.2f}s)")
                # Save artifacts for the first image of each tag
                if img == paths[0]:
                    stem = f"{cat}_{tag.lower()}_{img.stem}"
                    r["heatmap"].save(OUT / f"{stem}_heatmap.png")
                    r["overlay"].save(OUT / f"{stem}_overlay.png")
                    r["reconstruction"].save(OUT / f"{stem}_recon.png")
                    r["input_resized"].save(OUT / f"{stem}_input.png")

        n = len(results)
        correct = sum(1 for _, _, _, _, ok in results if ok)
        summary[cat] = (correct, n)
        print(f"  -> {correct}/{n} correct\n")

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for cat, (correct, n) in summary.items():
        print(f"  {cat:<10} {correct}/{n}  ({correct/n*100:.0f}%)")
    total_correct = sum(c for c, _ in summary.values())
    total_n = sum(n for _, n in summary.values())
    print(f"  {'TOTAL':<10} {total_correct}/{total_n}  ({total_correct/total_n*100:.0f}%)")
    print(f"\nVisualizations saved to: {OUT}")


if __name__ == "__main__":
    main()
