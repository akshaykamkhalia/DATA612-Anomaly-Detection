# Final Presentation Script

DATA-MSML 612 Group 6 - Industrial Anomaly Detection via Diffusion Reconstruction.
Total time budget: **10-11 minutes of talk + 3-4 minutes of Q&A**.

The presentation has 14 content slides + a thank-you slide. The split:

| Block | Slides | Time |
|---|---|---|
| Intro & framing | 1-3 | ~1:30 |
| Approach & stack | 4-5 | ~1:00 |
| Model architecture | 6-7 | ~1:00 |
| **Live demo** | 8 | **~2:30** |
| Evaluation & results | 9-12 | ~3:00 |
| Limitations & conclusion | 13-15 | ~1:00 |

This script is written to **complement** the slides, not duplicate them.
The slides carry the dense numbers and diagrams; you carry the story
between the bullets. Phrases like "as you can see on the slide" or "this
panel here shows" anchor the audience visually while you supply the
context that isn't on screen.

Lines starting with `>` are the actual words to say. Italicized
bracketed lines are stage directions.

---

## Slide 1 - Title (~15 seconds)

*[Walk on, click forward.]*

> "Good morning. Group 6 - Industrial anomaly detection via diffusion
> reconstruction. I'm Shrikanth, and with my teammates Sivram, Kevin and
> Akshay we built a system that catches manufacturing defects without
> ever being shown one during training."

> "I'll spend the first six or seven minutes on the approach and the
> results, then run a live demo on this laptop, and we'll have time for
> questions at the end."

---

## Slide 2 - Project Overview (~30 seconds)

> "The setup is the one every quality control team faces. Normal
> products are abundant; defective ones are rare and expensive to label.
> Standard supervised learning struggles here because the rare class is
> exactly the class you care about."

> "We frame this as a generative problem instead. The diffusion model
> learns what a normal product *looks* like, deeply, and then anything
> it can't reconstruct faithfully gets flagged."

*[Point at the bullets on the right.]*

> "So three deliverables out of the same model: train on normal
> images only, classify whether a test image is anomalous, and tell us
> *where* the defect is."

---

## Slide 3 - Problem Statement (~30 seconds)

*[The slide has the two screw images side by side.]*

> "These two screws look almost identical. The one on the right has a
> slightly malformed tip - the kind of defect a tired human inspector
> would miss after eight hours on a production line."

> "Manual inspection scales linearly with throughput, costs are
> dominated by labor, and the data you can collect is biased toward the
> mistakes you've already learned to spot. That's the gap we want to
> close - not just classify globally, but localize the suspicious pixels
> so a downstream operator knows where to look."

---

## Slide 4 - Solution Overview (~40 seconds)

*[Slide shows the input -> noised -> reconstructed -> anomaly-map flow.]*

> "Here's the pipeline at a glance. We train a Diffusion Transformer -
> DiT-Tiny, the four-million-parameter variant - on a single category
> of normal images at a time. The model only ever learns to reconstruct
> normal samples through the standard diffusion objective."

> "At test time, we don't generate from pure noise. We take the test
> image, partially corrupt it - 250 timesteps out of 1000 on a cosine
> schedule - and ask the model to denoise it back. If the input was
> normal, the reconstruction matches the input. If it had a defect, the
> model has no concept of that defect, so it 'fixes' the defective
> region toward what it considers normal."

> "The reconstruction error is then the anomaly signal. Pixel-wise L2,
> max-aggregated for the image-level score."

---

## Slide 5 - Stack (~20 seconds)

*[Quick read of the three columns, don't dwell.]*

> "Standard PyTorch stack on the modeling side. MVTec AD as the
> benchmark - fifteen real industrial categories, the standard
> evaluation set in this area. We trained on Colab T4 and Kaggle P100
> - free compute, the whole project came in at zero dollars of
> infrastructure cost."

---

## Slide 6 - End-to-End Pipeline (~30 seconds)

*[Slide is a diagram - input image to anomaly map with the model in
between.]*

> "The full path the data takes. Input image, partial noising, DiT
> denoising for fifty DDIM steps, side-by-side with the original to
> get a difference map. We add a ResNet-18 feature-space distance on
> top of the pixel L2 - the slide shows the pipeline; the combined
> map is what feeds the score."

---

## Slide 7 - Training vs Inference (~30 seconds)

*[Diagram contrasts training loop with inference loop.]*

> "The crucial asymmetry is on this slide. During training we add full
> noise across the whole timestep range, the standard diffusion
> objective. At inference we only partially noise. Two-fifty steps was
> chosen empirically - I'll come back to that on the ablation slide -
> but the intuition is: too few steps and the model just copies the
> defect through; too many and even normal images fail to reconstruct."

---

## Slide 8 - Live Demo (~2:30)

*[Switch to the browser tab running streamlit.]*

> "Rather than just describe it, let me show you the model running.
> This is on the laptop's GPU, real time."

**->** Run the demo per `DEMO_SCRIPT.md`. Beats:

1. Click hazelnut `crack_000.png` -> red DEFECTIVE banner, point at
   the crack lighting up in the overlay panel.
2. Scroll to score histogram, explain how the current image's score
   sits relative to the full test-set distribution.
3. Click hazelnut `000.png` -> green NORMAL banner, reconstruction is
   nearly identical.
4. Switch dropdown to leather, click `color_000.png` -> different
   model, same pipeline, ink stain pinpointed.

Close the demo with:

> "End to end, including the score-in-context and the
> localization, in about a second and a half on a laptop GPU. Now back
> to the numbers."

---

## Slide 9 - Performance Metrics (~30 seconds)

*[Slide has the bar chart of image vs pixel AUROC.]*

> "Two metrics to look at. Image-level AUROC asks - did the model
> correctly say 'this product is anomalous'. Pixel-level AUROC asks -
> did the highlighted region overlap the actual defect mask. Across
> all fifteen MVTec categories."

*[Point at the gap on the chart.]*

> "The pattern that immediately jumps out: pixel AUROC is consistently
> higher than image AUROC. The model knows *where* defects are better
> than it knows *whether* defects exist. That's an interesting
> failure mode and it's the headline finding of the project."

---

## Slide 10 - Qualitative Examples (~40 seconds)

*[Slide shows the six-panel comparison: input, reconstruction, anomaly
map, overlay, and ground-truth mask.]*

> "These are the six panels you can see on screen. Left column is the
> raw input - top is anomalous, bottom is normal. Next column is
> what the diffusion model thinks a *normal* version of that image
> would look like. The third column is the anomaly map; the fourth
> overlays it on the original; the fifth is the ground-truth defect
> mask we evaluate against."

> "On the anomalous hazelnut up top, you can see the highlighted region
> wraps tightly around the actual crack - and the model's
> reconstruction shows the crack region 'healed' back toward what a
> clean nut looks like. On the normal sample below, the heatmap is
> mostly cool blues - low and diffuse. That's exactly the contrast we
> want."

---

## Slide 11 - Quantitative Performance (~50 seconds)

*[Slide has the two big numbers: 0.573 image, 0.759 pixel.]*

> "The headline numbers across all fifteen categories. Mean image AUROC
> is 0.573. Mean pixel AUROC is 0.759."

> "The pixel number is genuinely strong. The image number is honest -
> the model is roughly fifteen points better than random at deciding
> whether to flag, but it's not a discriminative classifier. That's
> the trade-off of building a generative reconstruction system."

*[Point at the bullets.]*

> "The category-level pattern is what we'd expect: texture-heavy
> categories - hazelnut, wood, leather - are the model's strongest.
> Categories with rigid global structure - bottle, pill - are weaker,
> because the model has fewer high-frequency cues to lean on."

> "PatchCore beats us overall, mean image AUROC of 0.86. But PatchCore
> uses ImageNet-pretrained features. Our model was trained from
> scratch on a few hundred images per category. The gap is the
> pretraining gap, not an architectural gap."

---

## Slide 12 - Scoring Design Matters (~50 seconds)

*[This is the insight slide - the most important non-result slide.]*

> "The biggest single lesson from this project lives on this slide.
> Scoring design dominated final detection quality even more than the
> network itself."

*[Point at the SSIM vs L2 comparison.]*

> "On the left - hazelnut image AUROC went from 0.407 with SSIM
> scoring to 0.808 with L2 - same model, same checkpoint, just a
> different distance metric on the reconstruction error. SSIM is
> structurally biased to ignore exactly the kind of localized
> high-frequency change a defect produces."

*[Point at the aggregation comparison.]*

> "On the right - we initially aggregated the spatial anomaly map
> with a 95th-percentile, which sounds robust but actually misses
> small defects that occupy less than five percent of the image area.
> Switching to max aggregation took hazelnut from 0.417 to 0.744. No
> retraining."

> "Two purely post-hoc decisions, almost doubling the metric. That
> ratio - cheap-decision-impact to expensive-training-impact - is the
> most useful thing I'm taking out of this project."

---

## Slide 13 - Limitations & Future Scope (~30 seconds)

*[Slide has two columns - current limitations vs future scope.]*

> "Honest limitations. We trained at 128 by 128 - that's the constraint
> of free Colab compute. The bigger DiT-Small variant would help on
> structurally complex categories, and pretraining on a generic image
> dataset before fine-tuning on normal samples would close the gap to
> PatchCore."

> "On the metrics side, we report image and pixel AUROC. The MVTec
> protocol also includes the PRO metric - per-region overlap - which
> we left for future work. And DDIM sampling speed has a published
> 4x to 10x speedup we haven't fully exercised."

---

## Slide 14 - Conclusion (~25 seconds)

> "To wrap up. We built a from-scratch DiT-Tiny anomaly detection
> system, trained only on normal images, evaluated honestly on all
> fifteen MVTec categories. Mean pixel AUROC of 0.759, with strong
> qualitative localization. The biggest single insight wasn't the
> architecture - it was that scoring and aggregation choices matter
> more than they get credit for in the literature."

---

## Slide 15 - Thank You (~5 seconds)

> "Thank you. Happy to take questions."

*[Stop talking. Wait. Let the audience drive.]*

---

## Q&A anchors (3-4 minutes)

You will likely get one or two of these. Short, calibrated answers:

**Q: "Why DiT and not UNet, given UNet trains faster?"**
> "We did the ablation - it's in the report. UNet has stronger pixel
> AUROC on hazelnut, 0.91 versus our 0.78. But it's eight times the
> parameters and two times the inference latency. DiT-Tiny was the
> Pareto choice for this resolution."

**Q: "What threshold do you use for the binary verdict?"**
> "Per category, computed via Youden's J on the test-set ROC curve.
> The demo I just showed you computes that on first run and caches it.
> In production you'd compute it on a held-out validation set instead."

**Q: "Why is the image AUROC lower than the pixel AUROC?"**
> "The image score is just the max of the spatial map, so it's
> sensitive to outliers. A single bright pixel from a noisy
> reconstruction artifact can flip the verdict. The pixel-level
> averaging across all positions is much more stable. That's why we
> lead with the heatmap visually rather than the verdict."

**Q: "Could this run in production?"**
> "1.5 seconds per image on a laptop GPU. An A100 would push that
> under 200 milliseconds easily. The model itself is 52 megabytes - it
> would fit on any inspection-line edge device with 4 GB of VRAM."

**Q: "How much data did you need per category?"**
> "Whatever MVTec ships - typically two to three hundred normal
> images per category for training. No defective examples needed at
> training time. That's the real win of the unsupervised approach -
> data collection is just 'photograph the products that pass QA'."

---

## Pre-stage checklist

Right before walking on:

- [ ] Laptop plugged in, sleep disabled, Wi-Fi confirmed.
- [ ] Streamlit running at `localhost:8501`, all 15 categories in the
      sidebar dropdown, `Device: cuda` shown.
- [ ] Pre-clicked `crack_000.png` once to JIT-warm the inference path.
      Now scrolled back to the top, banner cleared.
- [ ] Browser zoom 110-125%, bookmarks bar hidden.
- [ ] Backup screenshots folder open on the desktop in case the demo
      breaks.
- [ ] PowerPoint already on slide 1, presenter mode if you use it.
- [ ] Water bottle within reach.

---

## Pacing notes

If you're running long mid-talk:
- Skip the second slide-12 example (only do the SSIM->L2 comparison,
  drop the aggregation fix).
- On slide 13 just say "limitations and future scope are on the slide"
  and move on.

If you're running short:
- Slide 11: walk through one or two specific category numbers from the
  bar chart instead of just the means.
- Slide 12: tie the insight back to a question you're hoping someone
  will ask in Q&A.

Rehearse twice end-to-end with a stopwatch. Most teams finish 90 seconds
faster than they expect on the day - account for that.
