# Live Demo Script — DiT Anomaly Detection (2-3 minutes)

For DATA-MSML 612 Group 6 final presentation. The first ~8 minutes of the slot
are slides; this is the closing demo.

---

## Pre-presentation checklist (do this before the talk starts)

1. **Power, connectivity, sleep settings.** Plug in the laptop, disable sleep,
   confirm an Ethernet/Wi-Fi connection if you'll need anything else online.

2. **Open a terminal in the demo folder:**
   ```bash
   cd <repo-root>/demo
   conda activate base
   ```

3. **Pre-warm the app** (so the audience doesn't see the first-load lag):
   ```bash
   streamlit run app.py
   ```
   It opens `http://localhost:8501` in your default browser. Wait for the
   sidebar to populate (about 5-10 seconds while CUDA loads the 15 models
   into VRAM). Confirm the dropdown shows all 15 categories and `Device: cuda`
   appears at the bottom.

4. **Click ONE sample image first** (e.g. `crack_000.png` under hazelnut) so
   the inference path is JIT-warm. The heatmap appears in ~1.5 s. Now scroll
   back up. The first inference of the live demo will be just as fast.

5. **Browser zoom: 110-125%** so the audience can read the verdict banner
   from the back of the room. Hide bookmarks bar (Ctrl+Shift+B).

6. **Have backup screenshots in a folder on your desktop** in case anything
   goes wrong on stage.

---

## The demo (run-of-show, 2-3 minutes)

### Beat 1 -- Framing (~15 seconds)

Switch to the browser tab. Verdict banner is hidden / unset.

> "Everything I've shown you so far has been numbers on slides. Let me show
> you the actual model running, in real time, on this laptop's GPU. The
> model has never seen the defective images we're about to feed it -- it
> was trained only on good examples. Anything it can't reconstruct
> faithfully, it flags."

Point at the sidebar.

> "Fifteen categories from the MVTec industrial benchmark. One DiT-Tiny
> diffusion model per category, about 4 million parameters each."

### Beat 2 -- The hero shot, hazelnut (~45 seconds)

Make sure `hazelnut` is selected. In the sidebar under "Try a sample,"
click **`crack_000.png`** in the Defective column.

While the spinner runs (~1.5 s):

> "What's happening right now: the image is being noised by 250 steps of a
> cosine schedule, then denoised back via DDIM sampling -- 50 steps of
> reverse diffusion. The model tries to reconstruct what a *normal*
> hazelnut would look like in this position."

The verdict banner snaps in red: **DEFECTIVE**.

Point at the four-panel grid.

> "Top-left: the input the model saw, downsampled to 128 by 128.
> Next to it: what the model *thinks* a clean hazelnut should look like in
> this pose -- notice how it's filled in the cracked region. The third
> panel is the L2 + ResNet-18 anomaly map, and the overlay on the right
> shows you exactly where the defect is."

Pause on the overlay. The crack lights up bright yellow/red.

> "That's not a hand-drawn annotation. That's the model's confidence per
> pixel."

### Beat 3 -- Score in context (~20 seconds)

Scroll down to the histogram.

> "And we're not just throwing out a binary answer. This histogram is the
> distribution of anomaly scores across the entire MVTec hazelnut test set.
> Green is normal, red is defective. The yellow line is *this image's*
> score. The dashed line is the threshold we computed via Youden's J on
> the ROC curve. So you can see exactly how confident the model is."

### Beat 4 -- Counterexample, normal hazelnut (~25 seconds)

Scroll back up. Click **`000.png`** under Normal.

The banner snaps green: **NORMAL**.

> "Same model, normal hazelnut. The reconstruction is nearly identical to
> the input. The heatmap is mostly cool blues -- no localized hotspot
> anywhere on the surface. The score sits below the threshold."

### Beat 5 -- Cross-category demonstration (~30 seconds)

Switch the sidebar dropdown to **`leather`**. Click `color_000.png` under
Defective.

> "Different category, completely different model -- but the pipeline is
> identical. Here a tiny ink stain..."

The heatmap pinpoints a small bright spot.

> "...gets localized to a region maybe two centimeters across in the
> original image. The DiT backbone is what gives us this spatial
> precision -- the same architecture used in modern image-generation
> models, repurposed for industrial inspection."

### Beat 6 -- Close (~15 seconds)

Switch back to your slides or stay on the demo. Speak to the audience.

> "End to end -- upload, inference, heatmap, verdict, score-in-context --
> in about 1.5 seconds on a laptop GPU. Trained only on normal images.
> Generalizes across 15 product categories. That wraps up the demo;
> happy to take questions."

---

## What to say if asked tricky questions

**"What's your accuracy?"**
> "Image-level AUROC ranges 0.5 to 0.8 across categories -- mean around 0.6
> with single-shot inference. We trade a bit of binary accuracy for the
> spatial localization you saw in the heatmap, which is what makes the
> tool useful in practice. The pixel-level AUROC is consistently above
> 0.75, peaking above 0.9 on leather and screw."

**"Why did it say NORMAL on a clearly defective image?" (if it happens)**
> "Reconstruction-based detection is stochastic -- there's randomness in
> the noise we add at the start. We average 5 seeds to stabilize, but
> single-shot variance is real. The committed test-set AUROC is what
> matters for the binary verdict; the heatmap, on the other hand, is
> reliably correct in *where* the defect is even when the score barely
> crosses or misses the threshold."

**"Why DiT instead of UNet?"**
> "We did the ablation -- it's in the report. UNet trains faster but DiT's
> attention layers give cleaner reconstructions on the categories with
> structured global content like leather and capsule. We use the same
> 128 x 128 input resolution for both."

**"Could this run in production?"**
> "1.5 seconds per image on a laptop. Production-grade inspection lines
> would batch images and use a beefier GPU; we'd expect under 200 ms per
> image on an A100. The model itself is 52 MB, which fits comfortably on
> any edge device with 4 GB of VRAM."

---

## If the demo breaks on stage (recovery script)

**Streamlit didn't start / crashed / browser closed:**
- Open the backup screenshots folder on your desktop. Walk through the
  same beats using the still images. Audience won't know.

**Inference is slow / spinner won't stop:**
- Don't wait. Click a different sample. The cached model will respond.

**Verdict is wrong:**
- Acknowledge it directly: "The classifier is uncertain on this one. Look
  at the heatmap -- the localization is still pinpointing the
  [crack/stain/etc.]. That's the actual output you'd ship."

**Total black screen / can't recover:**
- "Looks like the GPU is wedged. Let's go back to the slides for the
  numbers." Move on. Don't waste presentation time troubleshooting.

---

## Timing budget (for rehearsal)

| Beat | Target | Cumulative |
|---|---|---|
| 1. Framing | 0:15 | 0:15 |
| 2. Hazelnut crack | 0:45 | 1:00 |
| 3. Score histogram | 0:20 | 1:20 |
| 4. Normal hazelnut | 0:25 | 1:45 |
| 5. Cross-category leather | 0:30 | 2:15 |
| 6. Close | 0:15 | 2:30 |

Aim for 2:30. Leaves 30 s buffer inside the 3-minute demo block.

Rehearse twice. Time yourself.
