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

### 2.3 Wrong: a resolution-agnostic layer that broke parameter counting

To make `TNet` work at any input resolution, the classifier was written as
`nn.LazyLinear`, which infers its input width on the first forward pass. This is
idiomatic and looks like a clean way to avoid hardcoding a shape.

It crashed both baseline experiments on the GPU. `train.py` logs the trainable
parameter count before training starts, and lazy parameters do not exist until a
forward pass has happened, so `p.numel()` raised `ValueError: Attempted to use an
uninitialized parameter`. Worse, the pipeline smoke test used `resnet18`, so the
`tnet` path was never exercised locally and the failure only appeared after the
dataset was mounted and the GPU session was running.

The fix computes the flattened width from `img_size` — a 3×3 convolution without
padding followed by a 4×4 stride-4 max-pool gives `(img_size - 2) // 4` — which
yields 57,776 parameters at 64px, matching the starter's `16 * 15 * 15` linear
input exactly, so the reproduction stays faithful.

The verification is the more important part. `scripts/check_models.py` builds all
16 model/resolution/grayscale/freeze combinations and deliberately performs the
operations **in the order `train.py` performs them**: count parameters, construct
the optimizer, and only then run a forward pass. A check that ran a forward pass
first — the obvious way to write it — would pass while the bug was still present.

### 2.4 Questionable: silently appending every run to the reported results file

`train.py` appends one row per run to `experiments/results/results.csv`, which is
what the report table is generated from. That is a good design, but the path was
hardcoded, so *every* execution contaminated it — including throwaway pipeline
tests on synthetic data. A row reading `smoke, 93.75%` appeared in the results
table alongside real experiments, and on synthetic data that number means nothing
at all.

The results path is now a `--results-csv` flag and the notebook's smoke test
redirects it. This is a small change, but a report table that silently mixes real
and synthetic results is a correctness problem rather than an inconvenience.

### 2.5 Ineffective: an interrupt that discarded 8 epochs of GPU work

When a Colab run needed to be stopped, the resulting `KeyboardInterrupt`
propagated out of the training loop, so no checkpoint and no results row were
written — discarding a ResNet-50 run that had already reached 93.96% validation
accuracy. The best weights were sitting in memory at the time.

`fit()` now catches the interrupt, keeps the best-so-far weights, and flags the run
with `interrupted` and `epochs_completed`. The report table labels such rows
incomplete, shows the epochs actually run, and excludes them from best-run
selection — because a run cut short is not comparable to a full one, and quietly
treating it as if it were would be worse than losing it.

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

### A second decision: measure the noise floor before believing any result

Nothing suggested this, and it changed how the entire results table should be read.

Because the phase 1 experiments were accidentally run twice at the same seed, I had
unintentional repeats. Experiment 04 produced 94.58% and 93.96% — a 0.62 pp spread
from an identical configuration and an identical seed. GPU training is not bitwise
reproducible: cuDNN benchmarking selects convolution algorithms based on runtime
timing, and floating-point reduction orders vary between those algorithms.

Rather than treat that as an annoyance, I made it the reference point for the whole
table. 0.62 pp is 3 images out of 480, which means the 0.2 pp separating my best
ResNet-18 from ResNet-50-with-strong-augmentation is not a result at all. The
standard workflow — run each configuration once, report the best — would have had
me claim that ResNet-50 and RandAugment mattered, and then write a "secret recipe"
section justifying components that do nothing.

So the experiment table reports repeats instead of best-of, `make_report_table.py`
has a `--seed-summary` mode reporting mean and spread, and the final recipe is run
across three seeds. An AI assistant will happily help you tune to a validation set
well past the point where the differences are real; knowing where that point is
was my responsibility.

### A third decision: where experiments run

The development laptop has no CUDA GPU. Rather than shrink the research question
to fit the hardware (small models, low resolution), I kept the repository
device-agnostic and moved the full-resolution runs to a Colab T4, with the Colab
notebook as a thin driver that clones this repo and calls the same `train.py`.
The alternative — a separate Colab training path — would have meant the reported
numbers came from code that is not the code in the repository.
