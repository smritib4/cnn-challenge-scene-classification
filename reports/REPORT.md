# The CNN Challenge: 16-Class Scene Recognition from 2,400 Images

**Smriti Bhemireddy** · ITCS 6169/8169 Computer Vision, Assignment 1
**Repository:** https://github.com/smritib4/cnn-challenge-scene-classification

## 1. Final Result

| | |
|---|---|
| **Test accuracy** | **96.75%** (387/400; 25 per class), top-5 **100%** |
| Validation accuracy of selected checkpoint | **95.63%** (epoch 31 of 40) |
| Architecture | ConvNeXt-Tiny, ImageNet-1k pretrained, all 27.8M parameters fine-tuned |
| Input / cost | 224 × 224 RGB; ~14 min on one Tesla T4 |
| Reproduce | `python train.py --config configs/best.yaml` → `runs/best_seed1/best.pt` |

The provided starter reaches 49.58% under the same protocol, so this is **+47.2 pp**.
Nine of 16 classes are classified perfectly and no image's true class falls outside
the top five.

**Validation strategy.** The 2,400 training images were split 80/20 into 1,920 train
and 480 validation images with a **stratified** split at a fixed seed, giving exactly
30 validation images per class. Every decision was made on validation accuracy alone,
under a rule fixed in advance: highest validation accuracy among runs that *completed*.
Stratification was not cosmetic — the starter's unstratified `random_split` gives
per-class train counts from 115 to 124, and on 480 validation images 1% is five images,
so an unbalanced split injects noise as large as the effects being measured, and being
fixed by the seed it biases every experiment the same way instead of averaging out.

**Test-set discipline.** The test set was scored twice, on two already-selected
checkpoints, and both are reported: the fine-tuned ResNet-50 scored **94.25%** and the
ConvNeXt-Tiny above scored **96.75%**. The ResNet-50 went first because a fault in my
results logging (§4) hid the ConvNeXt run from the selection step. Neither number
influenced any hyperparameter choice.

## 2. Secret Recipe

ConvNeXt-Tiny with ImageNet-1k weights, fully fine-tuned, 224px RGB, strong
augmentation, AdamW at backbone LR 1e-4 with the head at 10×, weight decay 0.05
excluded from biases and norm parameters, per-batch cosine schedule with 3 warmup
epochs, label smoothing 0.1, 40 epochs.

**What actually mattered, ranked by measured effect:**

1. **Pretrained features: ~32 pp, and nothing else is close.** A *frozen* ResNet-18
   with only a 16-way linear head — **8,208** trainable parameters — reaches
   **91.04%**. The same class of architecture trained from scratch for 60 epochs
   reaches only 89.38%, with 572× the trainable parameters. At this data scale the
   task is overwhelmingly a question of whether you start from good features.
2. **Resolution and colour: 8.8 pp, before changing the architecture.** Giving the
   starter's own one-layer network colour at 128px instead of grayscale at 64px took
   it from 49.58% to 58.75%. Its preprocessing, not its capacity, was the first
   bottleneck.
3. **Fine-tuning the backbone: 3.4 pp.** Unfreezing takes ResNet-18 from 91.04% to
   94.58% — 1,360× more trainable parameters for a tenth of what the pretrained
   initialisation already provided.
4. **Architecture and capacity: ~1 pp in total, at the edge of measurability.**
   ResNet-18 94.58% → ResNet-50 95.21% → ConvNeXt-Tiny 95.63%. Each step is *inside*
   the 0.62 pp noise floor established in §3, so while the ordering is consistent with
   the ImageNet literature I cannot claim from this data that ConvNeXt-Tiny is better
   than ResNet-50. It won selection because the rule picks the highest validation
   number, not because the difference is demonstrated.

**What I cannot take credit for.** The submitted model uses strong augmentation, yet
the one controlled test of augmentation (experiment 6) showed it *costing* 0.83 pp on
ResNet-50. Its presence here follows from which run was selected, not from evidence
that it helped. Weight EMA was enabled and lost: the plain weights scored higher at
the best epoch, so the submitted weights are not the EMA weights. Reporting these
honestly matters more than a tidy story in which every component is justified.

## 3. Experimental Journey

Each row changes roughly one factor. Two numbers = the same configuration run twice
at the **same seed**, which is how the noise floor was measured.

| # | Experiment | Change | Params | Val acc (%) | Observation |
|---|---|---|---:|---:|---|
| 00 | Starter TNet, gray 64px | reference | 57.8K | 49.58 / 49.58 | Matches the stated "<50%" baseline |
| 01 | + colour, 128px | preprocessing only | 246.5K | 58.75 / 58.13 | +8.8 pp for free |
| 03 | ResNet-18 **linear probe** | ImageNet weights frozen | **8.2K** | 91.04 / 90.83 | +32.4 pp training 8,208 parameters |
| 04 | ResNet-18 fine-tuned | unfreeze backbone | 11.2M | 94.58 / 93.96 | +3.4 pp for 1,360× the parameters |
| 02 | Modern CNN **from scratch** | no ImageNet weights, 60 ep | 4.7M | 89.38 | **Pretraining control:** 1.7 pp *below* the 8.2K frozen probe |
| 05 | ResNet-50 fine-tuned | capacity only | 23.5M | 95.21 | Test 94.25% |
| 06 | + strong augmentation | preset only, 40 ep | 23.5M | 94.38 | −0.83 pp; the reasoning was wrong (§4) |
| 08 | ConvNeXt-Tiny | architecture family | 27.8M | (97.29) | Peaked higher but never completed; no checkpoint survived |
| — | **Final: ConvNeXt-Tiny** | + strong aug, EMA, flip TTA, 40 ep | 27.8M | **95.63** | Selected. **Test 96.75%** |

**The noise floor governs every conclusion here.** Repeating a configuration at a
fixed seed gave 94.58% and 93.96% — a **0.62 pp** spread, 3 of 480 images. Identical
seeds do not reproduce bitwise on GPU: cuDNN benchmarking picks algorithms by runtime
timing and reduction orders differ between them. Any single-run claim below ~1 pp is
therefore unsupported, which is why this table reports repeats rather than best-of,
and why §2 declines to claim the architecture comparison. Accuracy also plainly
**saturates**: experiment 06 sat at 0.93–0.94 from epoch 13 to 40 while train loss
flattened at 0.568, essentially the label-smoothing floor.

**Runs I excluded, and why.** ConvNeXt-Tiny's first run reached 97.29% validation but
was killed before writing a summary and its checkpoint did not survive; two further
seeds of the final recipe were cut short at epochs 3 and 8. Quoting an unfinished
run's peak epoch would be exactly the best-of-N cherry-picking the paragraph above
argues against, so all three are recorded and none is claimed.

**Where the remaining 3.25% lives.** Nine of 16 test classes are perfect. Every error
is a confusion between adjacent scenes: Mountain 84% (→ OpenCountry), then Forest,
Industrial and Kitchen at 92%, and LivingRoom, OpenCountry and Store at 96% — 13
errors in total. Notably the ResNet-50's worst class was Kitchen at 76%, which
ConvNeXt lifts to 92%, while Mountain gets *worse* (88% → 84%). The two models fail
differently on semantically adjacent categories, which is where a single *scene* label
is genuinely ambiguous, and suggests an ensemble rather than a bigger model.

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
augmentation fixes. Pretrained features are already near-optimal for these classes, so
useful adaptation finishes within a few epochs; after that the head fits residual noise
and validation wanders inside its own noise floor. Pixel augmentation cannot help when
input diversity was never the constraint. The **loss** curves showed this where the
accuracy trace hid it: train loss had already flattened at the label-smoothing floor.
The broader lesson is ordering — I measured before establishing the noise floor, so a
0.62 pp wobble read as signal.

**A second failure, worse because it never crashed.** The results writer took its
column order from the row being written while the header on disk came from an earlier
schema, so once I added `epochs_completed` and `interrupted` every new row landed two
fields off. The table reported a validation accuracy of 2,354,081,600% — a parameter
count read as an accuracy — and the automatic "best run" step selected the largest
garbage number. That is how the ResNet-50 came to be evaluated on test while a better
model sat unnoticed on disk, and it is why §1 reports two test evaluations instead of
one. The writer now compares against the on-disk header and migrates the file when the
columns differ, and `scripts/pick_best_checkpoint.py` ranks runs from `history.jsonl`
so that runs missing from `results.csv` are still visible to a human.

## 5. AI + Human

**Where AI helped.** Cursor wrote the config system, argparse plumbing, the per-family
classifier-head replacement across torchvision backbones, and the plotting utilities.
It also produced a throwaway script that zlib-decompressed the assignment PDF's object
streams to recover the dataset link, which plain text extraction missed.

**Where my judgement was required.** Asked to fetch the dataset, it started a
`gdown --folder` crawl pulling images one request at a time. The code *worked* — and
would have taken two hours with rate-limiting risk. I killed it, fetched the folder as
one archive, and put reproducibility in `scripts/prepare_data.py`, which unpacks
whatever archive you have and then *verifies* it: per-class counts, class names matching
across splits, a warning if the training total is not 2,400. I also rejected the
`random_split` the starter uses, and that generated code reproduces by default, for the
measurement reasons in §1. The recurring pattern: an assistant optimises for "the code
runs", which is not "this is the right approach."

Full detail, including three further cases where suggestions were wrong, is in
`AI_USAGE.md`. *References:* He et al. CVPR 2016 · Liu et al. CVPR 2022 · Cubuk et al.
NeurIPS 2020 · Loshchilov & Hutter ICLR 2019.
