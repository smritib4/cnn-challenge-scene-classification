# AI Usage

## Tool

**Cursor** (agent mode, Claude-based model) was used as the primary AI coding
assistant for this assignment. It was used for implementation, refactoring,
debugging and boilerplate. Experimental design, the choice of what to test, and
the interpretation of results are mine.

This file is written incrementally as the project develops, so it reflects what
actually happened rather than a reconstruction at submission time.

---

## 1. Representative ways AI assisted

### 1.1 Project scaffolding and the config system

The starter is a single notebook. I wanted a repository where each experiment is
one file and one command, because the assignment is graded on experimental
reasoning and that is impossible to defend if hyperparameters live in mutated
notebook cells.

I specified the design — a `DEFAULTS` dict, per-experiment YAML merged on top,
plus `--set key=value` CLI overrides — and Cursor implemented
`src/cnn_challenge/config.py`, `train.py` and `evaluate.py` around it. This is
exactly the kind of plumbing where AI is most valuable: the design decisions are
mine, the argparse and dict-merging boilerplate is not interesting to write.

One detail I asked for specifically: overrides must **fail loudly** on an unknown
key. A silent no-op on `--set optim.lrr=1e-4` would produce an experiment that
looks like it tested something and did not. That check is in
`apply_overrides()`.

### 1.2 Recovering the dataset link from the assignment PDF

The dataset URL in the PDF is a hyperlink with the visible text "Download
Dataset", and a plain text extraction of the file did not contain any URL. Cursor
wrote a short script that walks the PDF's binary objects, zlib-decompresses each
stream, and regex-searches the decompressed bytes. That recovered the Google
Drive ID immediately. A genuinely useful five-line throwaway tool that I would
otherwise have solved by clicking around.

### 1.3 Per-family classifier-head replacement

torchvision exposes the final layer inconsistently: ResNet uses `.fc`, DenseNet
uses a `.classifier` that is a bare `Linear`, and EfficientNet/ConvNeXt/MobileNet
use a `.classifier` that is a `Sequential` ending in a `Linear`. Rather than
hardcode one model, I had Cursor write `_replace_head()` in `models.py` to handle
each case explicitly. I insisted it raise on an unrecognised architecture instead
of falling back to a guess, so that adding a backbone cannot silently leave the
1000-way ImageNet head attached.

### 1.4 Pipeline verification before spending GPU time

I asked for a synthetic-data generator (`scripts/make_dummy_data.py`) with the
same directory layout as the real dataset, deliberately made *learnable* (a
distinct colour per class plus noise) rather than random. The point is that
training loss must fall on it; if it does not, the bug is in my pipeline. I ran
the full `train.py` → checkpoint → `evaluate.py` path on this synthetic set,
exercising mixup, cutmix, EMA and flip-TTA, before the real images had even
finished downloading. It caught nothing on the first try, but it means every one
of those code paths has actually executed.

### 1.5 Plotting and reporting utilities

The row-normalised confusion matrix and the per-class "most confused with" report
in `evaluate.py` were AI-written. Row normalisation was my requirement: raw
counts are misleading when comparing classes, and per-class error structure is
what the failure analysis needs.

---

## 2. Where the AI was wrong, ineffective, or questionable

### 2.1 Ineffective: downloading the dataset with `gdown`

Asked to fetch the dataset, the assistant's first approach was
`gdown <file_id>`, which failed because the ID is a *folder*, not a file. The
second attempt, `gdown --folder`, did start working — and that is the
questionable part. It crawls the folder and downloads images one HTTP request at
a time. After six minutes it had retrieved 114 files of several thousand, a rate
that projected to roughly two hours, and Google Drive rate-limits such crawls, so
it might not have completed at all.

The assistant was willing to let this run because it was technically working. I
killed it. Downloading the folder as a single archive from the browser takes under
a minute, and the reproducibility requirement is satisfied just as well by
`scripts/prepare_data.py`, which unpacks whatever archive you downloaded into the
expected layout and *verifies* it (per-class counts, matching train/test class
names, a warning if the total is not 2,400). The verification step is worth more
for reproducibility than a scripted download that is slow and fragile.

**Lesson:** "the code runs" is not the same as "this is the right approach". An
agent optimises for the former.

### 2.2 Questionable default: mixed precision regardless of device

Generated training loops default to enabling `torch.amp.GradScaler` and
`autocast` unconditionally. On this project's development machine (no CUDA GPU)
AMP is pointless at best: `GradScaler` is a no-op outside CUDA and CPU autocast
uses bfloat16, which changes numerics without buying speed on this hardware. Left
as-is, the CPU smoke tests would have run under different numerics than the GPU
runs, for no benefit.

In `engine.fit()` the config's `optim.amp` flag is therefore gated on the device:

```python
amp_enabled = amp_requested and device.type == "cuda"
```

so the same config file is correct on both machines.

### 2.3 To be extended

Further entries will be added as the experiment ladder runs — in particular any
case where an AI-suggested hyperparameter or "improvement" failed to reproduce as
a validation gain.

---

## 3. A decision I made rather than accepting the AI's

### The validation split

The starter uses `torch.utils.data.random_split`, and AI-generated training code
reproduces that pattern by default. I replaced it with a stratified split
(`data.stratified_split`).

The reasoning is about measurement, not accuracy. There are 2,400 images across 16
classes, so roughly 150 per class, and a 20% validation split leaves about 480
validation images — 30 per class if balanced. Under an unstratified random split
the per-class validation count follows a binomial distribution with a standard
deviation of about 4.9 images, so it is entirely ordinary to end up with 22 images
of one class and 38 of another. Every class then carries a different weight in the
validation metric.

That matters because the differences I need to resolve between experiments — label
smoothing, EMA, TTA — are on the order of 1%, which on 480 images is about five
images. An unstratified split adds variance to the metric of the same order as the
effects being measured, and worse, it is *fixed* variance: it biases every
experiment sharing that seed in the same direction, so it does not average out
across the ladder.

Stratified splitting costs nothing and removes that term. I also kept the
unstratified path available (`data.split: random`) purely so that experiment 00
reproduces the starter's number faithfully rather than being confounded by my own
change.

### A second decision: where experiments run

The development laptop has no CUDA GPU. Rather than shrink the research question
to fit the hardware (small models, low resolution), I kept the repository
device-agnostic and moved the full-resolution runs to a Colab T4, with the Colab
notebook as a thin driver that clones this repo and calls the same `train.py`.
The alternative — a separate Colab training path — would have meant the reported
numbers came from code that is not the code in the repository.
