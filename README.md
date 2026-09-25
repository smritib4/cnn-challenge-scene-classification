# CNN Challenge — 16-Class Scene Recognition from 2,400 Images

ITCS 6169/8169 Computer Vision, Assignment 1.
Author: Smriti Bhemireddy

Task: classify images into 16 scene categories, training on only 2,400 labelled
images. The provided starter is a single-convolution network on grayscale 64×64
inputs that reaches under ~50% validation accuracy. This repository builds a
substantially stronger CNN system through controlled experiments.

All tables below are generated from `experiments/results/results.csv`, which
`train.py` writes automatically — render them with `scripts/make_report_table.py`.
Per-run epoch histories, merged configs and summaries are archived under
[`experiments/run_logs/`](experiments/run_logs).

---

## 1. Headline result

| Metric | Value |
|---|---|
| **Test accuracy (final model)** | **96.75%** — 387/400 images; top-5 **100%** |
| Validation accuracy (selected checkpoint) | **95.63%** (epoch 31 of 40) |
| Architecture | ConvNeXt-Tiny, ImageNet-1k pretrained, all 27.8M parameters fine-tuned |
| Input resolution | 224 × 224 RGB |
| Validation strategy | Stratified 80/20 split of the 2,400 training images, exactly 30 validation images per class |
| Measured noise floor | **±0.62 pp** between repeats at a fixed seed |
| Config | [`configs/best.yaml`](configs/best.yaml) |
| Checkpoint | `runs/best_seed1/best.pt` (see [§8](#8-checkpoint)) |
| Starter baseline | 49.58% under the same protocol → **+47.2 pp** |

Selection rule, fixed before any test-set contact: highest validation accuracy among
runs that **completed**. `train.py` never touches the test set unless
`--evaluate-test` is passed explicitly.

**Two test evaluations happened, and both are reported.** A fault in the results
logging (see [§6](#6-failure-analysis)) hid the ConvNeXt run from the selection step,
so a fine-tuned ResNet-50 was scored first at **94.25%** before the ConvNeXt
checkpoint was found and scored at **96.75%**. Neither number influenced any
hyperparameter choice. Reporting both is the honest record; reporting only the second
would not be.

### The three results that matter

**A linear probe with 8,208 trainable parameters reaches 91%.** Freezing all of
ResNet-18's convolutional weights and training only the 16-way head takes the
starter's 49.58% to 90.83–91.04%. Generic ImageNet features are very nearly
sufficient for this task; full fine-tuning of 11.2M parameters — 1,360× more —
adds 3.4 points. The from-scratch control makes the same point from the other
direction: a modern 8-conv network with BatchNorm and global average pooling,
trained from random initialisation for 60 epochs, reaches 89.38% — *below* the
frozen probe, with 572× the trainable parameters.

**Accuracy saturates near 95%, and extra machinery stops helping.** ResNet-50 with
strong augmentation (RandAugment, 40 epochs) scored 94.38%, *below* the same
network with basic augmentation at 95.21%. Validation accuracy sat in 0.93–0.94
from epoch 13 to 40 while train loss flattened at the label-smoothing floor.

**The noise floor is 0.62 pp, so most small differences are not real.** Repeating
the same config at the same seed gave 94.58% and 93.96%, a spread of 3 images out
of 480. GPU runs are not bitwise reproducible at fixed seed because cuDNN
benchmarking selects algorithms non-deterministically. Any single-run claim below
roughly 1 pp on this validation set is unsupported, which is why experiments are
reported as repeats rather than best-of.

---

## 2. Quick start

```bash
# 1. Environment
python -m pip install -r requirements.txt

# 2. Data: download the dataset archive from the link in the assignment PDF,
#    then unpack it into the expected layout
python scripts/prepare_data.py --zip /path/to/downloaded.zip

# 3. Reproduce the final model
python train.py --config configs/best.yaml

# 4. Score the selected checkpoint on the test set
python evaluate.py --checkpoint runs/best_seed0/best.pt --split test \
    --confusion-matrix reports/confusion_matrix.png
```

After step 2 the layout is:

```text
data/
  train/<class_name>/*.jpg     # 2,400 images, 16 classes
  test/<class_name>/*.jpg
```

`prepare_data.py` prints per-class counts and warns if the total is not 2,400, so
an incomplete download is caught before any training time is spent.

### Running on a GPU (Colab)

The development laptop for this project has no CUDA GPU, so the full-resolution
runs were executed on a Colab T4 using
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), which clones this
repository, installs the requirements, prepares the data and calls the same
`train.py`. There is no separate Colab-only training path — the notebook is a thin
driver so that what runs on the GPU is exactly what is in this repository.

---

## 3. Repository layout

```text
configs/                 One YAML per experiment; best.yaml is the final recipe
  00_baseline_tnet.yaml    Reproduction of the provided starter
  01..08_*.yaml            The experiment ladder (see section 5)
  best.yaml                Final submitted recipe, with rationale in comments
  smoke.yaml               2-epoch CPU pipeline check, not a research result
src/cnn_challenge/
  config.py              YAML + DEFAULTS merge, validated --set overrides
  data.py                Stratified split, per-split transforms, DataLoaders
  transforms.py          none / basic / strong augmentation presets
  models.py              TNet, SmallCNN, 10 pretrained torchvision CNNs
  engine.py              Optimiser groups, schedules, mixup/cutmix, EMA, TTA
  utils.py               Seeding, environment capture, checkpoint I/O
train.py                 Training entry point
evaluate.py              Checkpoint evaluation + per-class failure report
predict.py               Inference on an unlabelled directory, writes a CSV
scripts/
  prepare_data.py        Unpack the dataset archive into data/ and verify counts
  check_models.py        Builds all model/resolution/freeze combinations
  make_dummy_data.py     Synthetic dataset for pipeline verification
  make_report_table.py   Renders the experiment table from results.csv
  pick_best_checkpoint.py  Ranks runs from history.jsonl, including unfinished ones
  plot_history.py        Training curves for the report
experiments/results/     One CSV row per run, written by train.py
experiments/run_logs/    Per-run config, epoch history and summary for every run
reports/REPORT.md        Source of the submitted two-page report
reports/REPORT.pdf       The submitted report
notebooks/               Colab driver notebook
```

---

## 4. The final recipe, and which parts mattered

Full settings are in [`configs/best.yaml`](configs/best.yaml), commented with the
reason each component is present. The short version:

- **ImageNet-pretrained ConvNeXt-Tiny, fully fine-tuned.** With 1,920 training
  images, pretrained initialisation is the single largest factor. Everything else
  is a refinement on top of it.
- **224×224 RGB input.** The starter's grayscale 64×64 preprocessing discards
  colour and spatial detail that scene categories depend on.
- **Strong augmentation** (RandomResizedCrop, horizontal flip, RandAugment,
  colour jitter, RandomErasing), because the dominant failure mode of a large
  pretrained CNN on 1,920 images is overfitting, not underfitting.
- **Discriminative learning rates.** The pretrained backbone trains at 1e-4 and
  the randomly initialised head at 1e-3. The head starts from noise and needs to
  move fast; the backbone already encodes useful features and needs protecting.
- **Cosine schedule with 3 warmup epochs.** Warmup stops the untrained head's
  large early gradients from damaging the backbone.
- **AdamW, weight decay 0.05, excluded from biases and norm parameters.**
- **Label smoothing 0.1**, **weight EMA**, and **horizontal-flip TTA** as
  low-risk variance reducers — individually small, and each one verified on
  validation rather than assumed.
- **Model selection on validation accuracy only.**

---

## 5. Experimental journey

Each config changes approximately one factor from the previous row, so a
difference can be attributed. Numbers come from `experiments/results/results.csv`.

Two numbers in a cell are two runs at the **same** seed, which is how the noise
floor above was measured.

| # | Experiment | What changed | Params | Val acc (%) | Observation |
|---|---|---|---:|---:|---|
| 00 | Starter TNet | — (reproduction) | 57.8K | 49.58 / 49.58 | Matches the handout's "under ~50%" |
| 01 | + colour, 128px | Preprocessing only, same architecture | 246.5K | 58.75 / 58.13 | +8.8 pp for free; preprocessing was the first bottleneck, not capacity |
| 03 | ResNet-18 linear probe | ImageNet weights frozen | **8.2K** | 91.04 / 90.83 | +32.4 pp training 8,208 parameters |
| 04 | ResNet-18 fine-tuned | Unfreeze backbone | 11.2M | 94.58 / 93.96 | +3.4 pp for 1,360× the trainable parameters |
| 02 | Modern CNN from scratch | No ImageNet weights, 60 epochs | 4.7M | 89.38 | Below the 8.2K frozen probe, with 572× the parameters |
| 05 | ResNet-50 fine-tuned | Capacity only | 23.5M | 95.21 | Test 94.25% |
| 06 | + strong augmentation | `augment.preset` only, 40 epochs | 23.5M | 94.38 | −0.83 pp; the diagnosis behind it was wrong (see failure analysis) |
| 07 | + mixup / cutmix | Label-space regularisation | 23.5M | _not run_ | Killed at epoch 0 by a session interrupt; excluded rather than reported |
| 08 | ConvNeXt-Tiny | Architecture family | 27.8M | (97.29) | Peaked higher but never completed and no checkpoint survived |
| — | **Final: ConvNeXt-Tiny** | + strong aug, EMA, flip TTA, 40 epochs | 27.8M | **95.63** | Selected. **Test 96.75%** |

**The architecture comparison is not actually resolved.** ResNet-18 94.58% →
ResNet-50 95.21% → ConvNeXt-Tiny 95.63% is a consistent ordering, but every step is
*inside* the 0.62 pp noise floor. ConvNeXt won selection because the rule takes the
highest validation number, not because this data demonstrates it is the better
architecture. Separating them would need several seeds per architecture, which did not
fit the compute budget.

**Runs excluded, and why.** ConvNeXt-Tiny's first run peaked at 97.29% validation but
was killed before writing a summary and its checkpoint did not survive; two further
seeds of the final recipe were cut short at epochs 3 and 8. Quoting an unfinished run's
peak epoch is the best-of-N cherry-picking the noise floor argues against, so these are
recorded and not claimed. `scripts/pick_best_checkpoint.py` ranks runs from
`history.jsonl` so that runs missing from `results.csv` stay visible to a human — that
script is what found the submitted checkpoint.

**Where the remaining error is.** Nine of 16 test classes are perfect; 13 errors total,
all between adjacent scenes: Mountain 84% (→ OpenCountry), then Forest, Industrial and
Kitchen at 92%, and LivingRoom, OpenCountry and Store at 96%. The two evaluated models
fail *differently* — the ResNet-50's worst class was Kitchen at 76%, which ConvNeXt
lifts to 92%, while Mountain drops from 88% to 84% — which suggests an ensemble rather
than a larger model. `evaluate.py --report` regenerates the full per-class breakdown and
confusion matrix from the checkpoint.

Regenerate this table from the recorded runs at any time:

```bash
python scripts/make_report_table.py
python scripts/make_report_table.py --seed-summary
```

Reproduce any row with:

```bash
python train.py --config configs/05_resnet50_finetune.yaml
```

Single-factor ablations without editing files:

```bash
python train.py --config configs/best.yaml --set augment.preset=basic
python train.py --config configs/best.yaml --set optim.label_smoothing=0.0
```

---

## 6. Failure analysis

**Strong augmentation did not help, and the diagnosis behind it was wrong.**

Experiment 05 showed validation accuracy peaking at epoch 2 and then declining
while training loss kept falling — the classic signature of overfitting, which on
1,920 images against 23.5M parameters was entirely expected. Experiment 06 added
RandAugment, colour jitter and RandomErasing and extended training to 40 epochs.

It scored 94.38% against 94.58% for a plain ResNet-18 with basic augmentation:
no improvement, from a model with twice the parameters and a third more epochs.

What that early peak actually reflects is that pretrained features are already
nearly optimal for these classes, so the useful adaptation finishes within a
couple of epochs; afterwards the head fits residual noise and the validation
metric wanders inside its own noise floor. Pixel-level augmentation cannot help
because input diversity was never the limitation. The train/validation **loss**
curves show this where the accuracy trace does not — training loss had flattened
at the label-smoothing floor, so the model was not straining against a hard
objective at all.

The deeper mistake was measuring before establishing the noise floor. The
"decline" from 93.96% looked like signal until the same configuration, run twice
at the same seed, produced a 0.62 pp spread. Establishing that first would have
redirected the effort to per-class error analysis, where the remaining ~6%
actually lives.

Reproduce the evidence:

```bash
python scripts/plot_history.py --runs runs/05_resnet50_finetune_seed0 runs/06_resnet50_strong_aug_seed0 \
    --out reports/augmentation_effect.png
```

---

## 7. Reproducibility

- Seed is set for Python, NumPy and Torch, and DataLoader workers are seeded
  individually (`utils.seed_worker`).
- `--deterministic` additionally pins cuDNN algorithm choice. It is off by default
  because it slows convolutions noticeably; run-to-run variance is better
  characterised by repeating a run with `--seed 1 --seed 2` than by forcing
  bitwise determinism.
- Every checkpoint embeds its config, class ordering, and an environment snapshot
  (Python, OS, torch/torchvision versions, CUDA version, GPU name, git commit).
- Verified environment: Python 3.14.2, torch 2.14.0, torchvision 0.29.0.

Pipeline verification without the real dataset:

```bash
python scripts/make_dummy_data.py --out data_dummy
python train.py --config configs/smoke.yaml --data-root data_dummy --output-root runs_smoke
```

This trains on a learnable synthetic 16-class set and exercises the mixup, EMA and
TTA code paths, so pipeline bugs surface before GPU time is spent.

---

## 8. AI usage

See [`AI_USAGE.md`](AI_USAGE.md).

## 9. Model restrictions

The primary classifier is a convolutional network throughout, as required. No
CLIP, DINO/DINOv2, Vision Transformer, or vision-language model is used anywhere
in the pipeline. The only pretrained weights are ImageNet-1k classification
weights distributed with torchvision; what was pretrained and which parameters
were fine-tuned is stated in section 4.
