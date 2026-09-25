# The CNN Challenge: 16-Class Scene Recognition from 2,400 Images

**Smriti Bhemireddy** · ITCS 6169/8169 Computer Vision, Assignment 1
**Repository:** https://github.com/smritib4/cnn-challenge-scene-classification

## 1. Final Result

| | |
|---|---|
| **Test accuracy** | **94.25%** (377/400; 25 per class), top-5 **100%** |
| Validation accuracy of selected checkpoint | **95.21%** (epoch 20 of 30) |
| Architecture | ResNet-50, ImageNet-1k pretrained, all 23.5M parameters fine-tuned |
| Input / cost | 224 × 224 RGB; 352 s on one Tesla T4 |
| Reproduce | `python train.py --config configs/best.yaml` → `runs/best_seed0/best.pt` |

The provided starter reaches 49.58% under the same protocol, so this is **+44.7 pp**.

**Validation strategy.** The 2,400 training images were split 80/20 into 1,920 train
and 480 validation images with a **stratified** split at a fixed seed, giving exactly
30 validation images per class. Every architecture and hyperparameter decision was
made on validation accuracy alone; the test set was scored **once**, on the
already-selected checkpoint. Stratification was not cosmetic: reproducing the
starter's unstratified `random_split` gives per-class train counts from 115 to 124,
and on 480 validation images a 1% difference is five images — so an unbalanced split
injects noise as large as the effects being measured, and because it is fixed by the
seed it biases every experiment the same way instead of averaging out.

## 2. Secret Recipe

ResNet-50 with ImageNet-1k weights, fully fine-tuned, 224px RGB, basic augmentation
(RandomResizedCrop 0.65–1.0 + horizontal flip), AdamW at backbone LR 3e-4 with the
head at 10×, weight decay 0.05 excluded from biases and norm parameters, per-batch
cosine schedule with 2 warmup epochs, label smoothing 0.1, 30 epochs.

**What actually mattered, ranked by measured effect:**

1. **Pretrained features: ~32 pp, and nothing else is close.** A *frozen* ResNet-18
   with only a 16-way linear head — **8,208** trainable parameters — reaches
   **91.04%**. That is the result of this assignment: at this data scale the task is
   almost entirely a question of whether you start from good features.
2. **Resolution and colour: 8.8 pp, before changing the architecture.** Giving the
   starter's own one-layer network colour at 128px instead of grayscale at 64px took
   it from 49.58% to 58.75%. Its preprocessing, not its capacity, was the first
   bottleneck.
3. **Fine-tuning the backbone: 3.4 pp.** Unfreezing takes ResNet-18 from 91.04% to
   94.58% — 1,360× more trainable parameters for a tenth of what the pretrained
   initialisation already provided.
4. **Capacity: 0.63 pp, barely measurable.** ResNet-50's 95.21% over ResNet-18's
   94.58% is only just outside the noise floor below.

**What did not matter:** strong augmentation (§4), and the variance reducers. Weight
EMA and flip TTA are implemented and were exercised, but the selected model uses
neither — where EMA was enabled the plain weights won selection anyway. I report them
as available and unhelpful here rather than including them for appearance.

## 3. Experimental Journey

Each row changes roughly one factor. Two numbers = the same configuration run twice
at the **same seed**, which is how the noise floor was measured.

| # | Experiment | Change | Params | Val acc (%) | Observation |
|---|---|---|---:|---:|---|
| 00 | Starter TNet, gray 64px | reference | 57.8K | 49.58 / 49.58 | Matches the stated "<50%" baseline |
| 01 | + colour, 128px | preprocessing only | 246.5K | 58.75 / 58.13 | +8.8 pp for free |
| 03 | ResNet-18 **linear probe** | ImageNet weights frozen | **8.2K** | 91.04 / 90.83 | +32.4 pp training 8,208 parameters |
| 04 | ResNet-18 fine-tuned | unfreeze backbone | 11.2M | 94.58 / 93.96 | +3.4 pp for 1,360× the parameters |
| 02 | Modern CNN **from scratch** | no ImageNet weights, 60 ep | 4.7M | 89.38 | **Pretraining control:** lands 1.7 pp *below* the 8.2K frozen probe, using 572× the parameters |
| 05 | **ResNet-50 fine-tuned** | capacity only | 23.5M | **95.21** | Selected. Best completed run |
| 06 | + strong augmentation | preset only, 40 ep | 23.5M | 94.38 | −0.83 pp; the reasoning was wrong (§4) |
| 08 | ConvNeXt-Tiny | architecture family | 27.8M | (97.29) | Run never completed → not eligible |

**The noise floor governs every conclusion here.** Repeating a configuration at a
fixed seed gave 94.58% and 93.96% — a **0.62 pp** spread, 3 of 480 images. Identical
seeds do not reproduce bitwise on GPU: cuDNN benchmarking picks algorithms by runtime
timing and reduction orders differ between them. So any single-run claim below ~1 pp
is unsupported, which is why this table reports repeats rather than best-of.
Accuracy also plainly **saturates near 95%**: experiment 06 sat at 0.93–0.94 from
epoch 13 to 40 while train loss flattened at 0.568, essentially the label-smoothing
floor.

**A selection decision to be explicit about.** ConvNeXt-Tiny reached 97.29%
validation — higher than what I am submitting. It is excluded because the run never
finished, and my selection rule, fixed in advance, was best validation accuracy among
*completed* runs. Quoting an unfinished run's peak epoch is exactly the best-of-N
cherry-picking the noise-floor analysis argues against. It is recorded as the most
promising direction, not as a claimed result.

**Where the remaining 5.75% lives.** Seven of 16 test classes are perfect. Every
error is a confusion between adjacent scenes: Kitchen 76% (→ Bedroom), Industrial 84%
(→ LivingRoom), Mountain 88% (→ OpenCountry), then Bedroom, InsideCity, LivingRoom,
OpenCountry at 92%. Indoor rooms are 12 of the 23 errors — categories sharing objects
and layout, where a *scene* label is genuinely ambiguous. More capacity or generic
augmentation is not the missing ingredient.

## 4. Failure Analysis

**Strong augmentation failed, and my diagnosis was wrong.**

*What I tried.* Experiment 06 added RandAugment (2 ops, magnitude 9), colour jitter
and RandomErasing to the ResNet-50 recipe, extending training to 40 epochs.

*Why I expected it to work.* 1,920 images against 23.5M parameters is a textbook
overfitting setup, and experiment 05 appeared to confirm it: validation accuracy
peaked early and then declined while train loss kept falling.

*What happened.* 94.38% against 95.21% for the same network with basic augmentation
and ten fewer epochs — it cost 0.83 pp and a third more compute.

*What I learned.* Early-peak-then-decline is not evidence of the kind of overfitting
augmentation fixes. Pretrained features are already near-optimal for these classes,
so useful adaptation finishes in a few epochs; after that the head fits residual
noise and validation wanders inside its own noise floor. Pixel augmentation cannot
help when input diversity was never the constraint. The **loss** curves showed this
where the accuracy trace hid it: train loss had already flattened at the
label-smoothing floor. The broader lesson is ordering — I measured before
establishing the noise floor, so a 0.62 pp wobble read as signal. Establishing it
first would have redirected the run toward per-class error analysis, where the
remaining error actually is.

**A second failure, worse because it did not crash.** The results writer took its
column order from the row being written while the header on disk came from an earlier
schema, so after I added `epochs_completed` and `interrupted`, every new row landed
two fields off. The reported table showed a validation accuracy of 2,354,081,600% — a
parameter count read as an accuracy — and "pick the best run" selected the largest
garbage number. It happened to choose the correct model, which is the uncomfortable
part: a defensible answer for an indefensible reason. The writer now compares against
the on-disk header and migrates the file when columns differ.

## 5. AI + Human

**Where AI helped.** Cursor wrote the config system, argparse plumbing, the
per-family classifier-head replacement across torchvision backbones, and the plotting
utilities. It also produced a throwaway script that zlib-decompressed the assignment
PDF's object streams to recover the dataset link, which plain text extraction missed.

**Where my judgement was required.** Asked to fetch the dataset, it started a
`gdown --folder` crawl pulling images one HTTP request at a time. The code *worked* —
and would have taken two hours with rate-limiting risk. I killed it and fetched the
folder as one archive, keeping reproducibility in `scripts/prepare_data.py`, which
unpacks whatever archive you have and then *verifies* it: per-class counts, class
names matching across splits, and a warning if the training total is not 2,400.
Verification is worth more than a scripted download that is slow and fragile. I also
rejected the `random_split` that the starter uses and that generated code reproduces
by default, for the measurement reasons in §1. The recurring pattern: an assistant
optimises for "the code runs", which is not "this is the right approach."

Full detail, including three further cases where suggestions were wrong, is in
`AI_USAGE.md`.

---

*References:* He et al., *Deep Residual Learning*, CVPR 2016 · Liu et al., *A ConvNet
for the 2020s*, CVPR 2022 · Cubuk et al., *RandAugment*, NeurIPS 2020 · Loshchilov &
Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
