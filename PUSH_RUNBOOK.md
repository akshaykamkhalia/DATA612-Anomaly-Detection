# Push Runbook (delete this file before pushing if you want)

This document is just for **you** to bootstrap the new GitHub repo. Once
you've pushed once, you can delete it.

The repo is currently a plain folder at `C:/Data612-Final/`. No git history
exists yet. You will run all git commands yourself so the entire commit
history shows your identity.

---

## 1. Sanity-check what's in the folder

```bash
cd C:/Data612-Final
ls
# Expect: README.md, .gitignore, code/, demo/, figures/, presentation/,
#         proposal/, report/, results/, PUSH_RUNBOOK.md (this file)

du -sh .
# Expect: ~21M
```

If `du` reports anything substantially larger than 25 MB, something is
unexpected - investigate before continuing.

---

## 2. Confirm your git identity is correct

```bash
git config --global user.name
git config --global user.email
```

These should be **your** name and email. If they're not, set them:

```bash
git config --global user.name "Shrikanth Vilvadrinath"
git config --global user.email "<your-github-email>"
```

(You can also set these per-repo with `git config user.name "..."` after
the `git init` below if you don't want to change globals.)

---

## 3. Initialize the local repo

```bash
cd C:/Data612-Final
git init
git branch -M main
```

---

## 4. Stage and commit

Stage everything (the .gitignore will keep out `data/`, `checkpoints/`,
`sample_images/`, etc.):

```bash
git add .
git status      # double-check the file list
```

Make the first commit. Suggested message (edit if you'd like):

```bash
git commit -m "Initial commit: DiT-based industrial anomaly detection"
```

If you prefer a longer message, use this one (paste as-is between the
quote markers, including the line breaks):

```bash
git commit -m "Initial commit: DiT-based industrial anomaly detection

Final deliverables for DATA-MSML 612 Group 6.

- Code: PyTorch implementation of DiT-Tiny + UNet backbones, cosine
  diffusion schedule, DDIM sampling, dual-level (pixel L2 + ResNet-18
  feature) anomaly scoring, ablations, classical baselines.
- Notebooks: Colab pipeline for full retraining and quick verification
  of committed results.
- Demo: Streamlit app with drag-and-drop upload, animated verdict
  banner, four-panel reconstruction grid, score-distribution histogram,
  and confidence gauge. Per-category Youden's-J thresholds.
- Reports: final report (.tex + .pdf), proposal, final presentation.
- Results: per-category evaluation JSONs, t_partial / scoring / feature
  / backbone ablations, PatchCore baseline."
```

(If `git commit` complains about a multi-line `-m` argument on Windows,
either use a single short message, or write the message into a file and
use `git commit -F message.txt`.)

---

## 5. Create the GitHub repo and add the remote

### Option A - using GitHub CLI (`gh`):

```bash
# Pick a name. I'll use DATA612-Anomaly-Detection but anything is fine.
gh repo create DATA612-Anomaly-Detection --public --source=. --remote=origin
```

This creates the public GitHub repo and adds `origin` in one shot. Skip
to step 6.

### Option B - manual:

1. Open https://github.com/new in your browser.
2. Repo name: `DATA612-Anomaly-Detection` (or your choice). Public.
3. **Do not** initialize with a README, .gitignore, or license - we
   already have them locally.
4. Click "Create repository".
5. Copy the SSH or HTTPS URL it shows.
6. Add it as a remote:

```bash
git remote add origin <YOUR_REPO_URL>
```

---

## 6. Push

```bash
git push -u origin main
```

That's it. The commit history will show only your name. No metadata
will identify any other contributor or assistant.

---

## 7. Post-push checklist

- [ ] Visit the repo URL on github.com - verify file list looks correct.
- [ ] Open `README.md` on GitHub - confirm it renders cleanly.
- [ ] Click on `code/notebooks/02_quick_verify.ipynb` - confirm it opens.
- [ ] Click on `demo/app.py` - confirm it shows the Streamlit code.
- [ ] Open the .pptx in `presentation/` to download - confirm the file
      is intact.
- [ ] Update the notebook `REPO_URL` placeholders if you want Colab to
      auto-clone. Search the notebooks for `<YOUR_REPO_URL>` and
      replace with your actual URL.
- [ ] (Optional) Delete this `PUSH_RUNBOOK.md` once everything is
      pushed - it's only useful for the first push.

---

## 8. If something goes wrong

**Push rejected because remote isn't empty:**
You probably initialized the GitHub repo with a README on the website.
Run:
```bash
git pull --rebase origin main
git push -u origin main
```

**Push hangs on credential prompt:**
For HTTPS, you'll need a Personal Access Token (PAT) instead of a
password. Settings → Developer settings → Personal access tokens →
Generate new (classic) → at minimum `repo` scope.

**Files larger than 100 MB rejected:**
Should not happen with this layout. If it does, run
`find . -size +50M -type f` to find the culprit and add it to
`.gitignore`.
