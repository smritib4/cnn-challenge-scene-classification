# CNN Challenge — 16-Class Scene Recognition from 2,400 Images

ITCS 6169/8169 Computer Vision, Assignment 1.
Author: Smriti Bhemireddy

Task: classify images into 16 scene categories, training on only 2,400 labelled
images. The provided starter is a single-convolution network on grayscale 64×64
inputs that reaches under ~50% validation accuracy. This repository builds a
substantially stronger CNN system through controlled experiments.

> **Status:** experiments in progress. The result tables below are filled in from
> `experiments/results/results.csv`, which is written automatically by `train.py`.
> Any row marked `pending` has not been run yet.

---

## 1. Headline result

| Metric | Value |
|---|---|
| **Test accuracy (final model)** | _pending_ |
| Validation accuracy (selected checkpoint) | _pending_ |
| Architecture | ConvNeXt-Tiny, ImageNet-1k pretrained, fully fine-tuned |
| Input resolution | 224 × 224 RGB |
| Validation strategy | Stratified 80/20 split of the 2,400 training images (seed 0), 30 validation images per class |
| Config | [`configs/best.yaml`](configs/best.yaml) |
| Checkpoint | _pending_ |

The test set is used **once**, to score the single checkpoint already selected on
validation accuracy. `train.py` does not touch the test set unless
`--evaluate-test` is passed explicitly.

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
scripts/prepare_data.py  Unpack the dataset archive into data/
scripts/make_dummy_data.py  Synthetic dataset for pipeline verification
experiments/results/     One CSV row per run, written by train.py
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

| # | Experiment | What changed | Hypothesis | Val acc | Observation |
|---|---|---|---|---:|---|
| 00 | Starter TNet | — (reproduction) | Reference point | _pending_ | _pending_ |
| 01 | + colour, 128px, stratified split | Preprocessing only, same weak architecture | Grayscale 64×64 throws away usable signal | _pending_ | _pending_ |
| 02 | Modern CNN from scratch | 8-conv net, BatchNorm, GAP | Depth/normalisation beat a one-layer net; also the no-pretraining control | _pending_ | _pending_ |
| 03 | ResNet-18 linear probe | ImageNet weights frozen | Generic features may already suffice at this data scale | _pending_ | _pending_ |
| 04 | ResNet-18 fine-tuned | Unfreeze backbone | Scene classes differ from ImageNet objects enough to need adaptation | _pending_ | _pending_ |
| 05 | ResNet-50 fine-tuned | Capacity only | More capacity helps — or overfits 1,920 images | _pending_ | _pending_ |
| 06 | + strong augmentation | `augment.preset` only | If the train/val gap is the bottleneck, this closes it | _pending_ | _pending_ |
| 07 | + mixup / cutmix | Label-space regularisation | Helps most when data are scarce | _pending_ | _pending_ |
| 08 | ConvNeXt-Tiny | Architecture family | Better ImageNet recipe transfers better at similar cost | _pending_ | _pending_ |
| — | **Final (best.yaml)** | + EMA + flip TTA | Variance reduction | _pending_ | _pending_ |

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

_To be completed once the experiment ladder has run._ Candidate write-ups will be
based on whichever of these actually fails: mixup/cutmix at a 40-epoch budget,
capacity scaling to ResNet-50, or resolution increases beyond 224.

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
