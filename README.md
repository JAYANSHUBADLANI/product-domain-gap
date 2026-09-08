# Does a product image classifier survive leaving the studio

I trained a classifier on clean, studio-lit product photographs and asked one
question before running anything: how much accuracy does the gap between a
studio photo and a real customer's photo of the same product actually cost.

> A standard ImageNet-pretrained classifier supervised only with clean
> Adaptiope Product images will have lower class-balanced top-1 accuracy on
> held-out Adaptiope Real Life customer review photographs of the same
> categories than on held-out Product photographs.

That claim was fixed in `config/protocol.yaml` before I trained or scored
anything. I treated a zero or negative gap as a finding I would report as
plainly as a large one.

## The headline result

| | class-balanced top-1 accuracy | 95% CI (2000 bootstrap resamples) |
|---|---|---|
| Clean (Product) test | 0.8275 | [0.7925, 0.86] |
| Shifted (Real Life) test | 0.4820 | [0.453, 0.512] |
| **Gap** | **0.3455** | intervals do not overlap |

The claim holds, clearly. A classifier that never sees anything but clean
product photography loses about 35 points of accuracy the moment it meets a
real customer's photo of the same object, and the two intervals are nowhere
near touching, so this is not a result that could plausibly have gone the
other way on a slightly different held-out sample.

## Data

[Adaptiope](https://gitlab.com/tringwald/adaptiope) (Ringwald and Otte, WACV
2021), a domain adaptation benchmark whose Product and Real Life domains
share the same category system: Product images come from Amazon listings,
Real Life images come from Amazon customer reviews of the same product
categories. I picked it specifically because that shared category system
avoids a manual cross-dataset category mapping, which would have introduced
its own source of error into the comparison.

The full archive is 6,650,336,447 bytes, 36,900 images across Product, Real
Life, and a Synthetic domain I did not use. I selected a 20-category subset
of consumer electronics and peripherals before looking at any result:
calculator, computer, computer mouse, cordless fixed phone, game controller,
hard-wired fixed phone, in-ear headphones, keyboard, laptop, mixing console,
monitor, network switch, over-ear headphones, power strip, printer,
projector, speakers, usb stick, vr goggles, webcam. The category list is
recorded in `config/protocol.yaml` and was fixed before any model result
existed; the only edit I made to it before results existed was replacing a
subjective cellphone-versus-smartphone pair with power strip and mixing
console.

I found no license file or stated license for this dataset anywhere in the
archive, the paper, or the official repository. It is publicly downloadable
for research, not openly licensed; I have not packaged or redistributed the
images, and the fetch step downloads them fresh from the original source.

**Split**, fixed before any result existed: 70 train, 10 validation, 20
clean test images per category from Product; 50 adaptation-pool, 50
shifted-test images per category from Real Life. Every category landed at
exactly its target count; the extracted set had zero exact-duplicate images
(checked by content hash, not filename), so the duplicate-atomic-split logic
that exists to keep an exact duplicate from ever landing on both sides of a
held-out boundary was never actually exercised against real data on this
run, only against a fabricated duplicate group in the test suite.

## Method

1. **`src/studio_shift/extract.py`** pulls only the 20 locked categories
   from the local archive (a full local copy, not a selectively-streamed
   remote read, see "Getting the data" below for why), with a per-file
   sha256 inventory and an expected-count check.
2. **`src/studio_shift/splits.py`** builds the splits above, seeded from one
   root value (`20260908`) so a rerun reproduces the identical manifests.
3. **`src/studio_shift/model.py` / `train.py`** fine-tune only a fresh
   20-way classifier head on a frozen, ImageNet-pretrained
   `google/mobilenet_v2_1.0_224` backbone (pinned to revision
   `75e607b00aeae1297cc89d026a118bce012f5c5a`), transfer learning
   deliberately, not a novel technique: the contribution here is the
   measured gap and its recovery curve, not the model. 15 epochs, best
   validation accuracy 0.880, 234.6 seconds on an Apple Silicon `mps`
   device. The resulting checkpoint's own sha256 is locked in
   `artifacts/checkpoints/clean_classifier.lock.json` so later stages can
   assert they evaluated the exact weights this run produced.
4. **`src/studio_shift/evaluate.py`** scores class-balanced top-1 accuracy
   (each category's own accuracy averaged, not overall micro accuracy, so
   one large category's performance cannot mask a smaller category doing
   badly) plus a bootstrap confidence interval and a confidence-gap analysis
   between correct and incorrect shifted predictions.
5. **`src/studio_shift/recovery.py`** measures how much labelled Real Life
   data closes the gap, at budgets of 0, 1, 2, 4, 8, 16, 32, and 50 images
   per category, 3 nested-sampling runs per budget (each larger budget's
   sample is a strict superset of the same run's smaller budgets), every
   budget restarting fine-tuning from the same locked clean checkpoint
   rather than continuing from the previous budget's weights.
6. **`src/studio_shift/saliency.py`** runs Grad-CAM on `mobilenet_v2.conv_1x1`
   (the last convolutional feature map before the classifier) for a handful
   of shifted-domain misses, illustrative evidence, not a quantified result.

`run_pipeline.py` runs all of it end to end, phase by phase, skipping a
phase whose output already exists.

## What kind of mistake it makes on the shifted domain

Per-category shifted accuracy is not spread evenly around the 0.482 mean,
it is bimodal:

| holds up well | collapses |
|---|---|
| computer mouse: 0.80 | usb stick: 0.04 |
| keyboard: 0.74 | network switch: 0.06 |
| game controller: 0.68 | computer: 0.14 |

My reading of this, from the categories and from looking directly at the
saliency examples below, not just the numbers: the categories that survive
are the ones whose whole shape stays recognizable regardless of scale,
angle, or what else is in frame. A mouse or a controller is usually shown
close to full frame as itself in both domains. A usb stick or a network
switch in a real customer's photo is often small, partially occluded, or
one component in a larger scene the studio photograph never has to contend
with.

**Confidence.** Mean confidence on correct shifted predictions is 0.695,
against 0.478 on incorrect ones. 0.478 is well above the 0.05 a uniform
guess across 20 classes would carry, so a meaningful share of the shifted
misses are the model being confidently wrong, not uncertain and wrong. A
downstream system built on this classifier cannot safely treat "high
confidence" as a proxy for "probably correct" on real-world input the way
it more reasonably could on studio input.

## How much labelled real-world data recovers the gap

| labels per category | mean class-balanced accuracy | range across 3 runs |
|---|---|---|
| 0 (no adaptation) | 0.482 | [0.482, 0.482] |
| 1 | 0.482 | [0.419, 0.577] |
| 2 | 0.558 | [0.540, 0.576] |
| 4 | 0.611 | [0.607, 0.617] |
| 8 | 0.642 | [0.635, 0.648] |
| 16 | 0.609 | [0.574, 0.667] |
| 32 | 0.686 | [0.649, 0.712] |
| 50 | 0.646 | [0.598, 0.714] |

This is not a clean, monotonically flattening curve, and I am reporting it
as it actually came out rather than smoothing it into one. From 0 to 8
labels per category the climb is steep and consistent across all 3 runs,
recovering close to half of the 0.3455 gap, 0.482 to 0.642, with a genuinely
small amount of labelled data: about 8 real customer photos per category.
Past 8 labels, the curve stops behaving monotonically. 16 dips below 8's
mean. 32 is the single highest point measured. 50, the largest budget
tried, comes in below 32. The run-to-run range widens at the same point:
at 50 labels per category the 3 runs span 0.598 to 0.714, a wider spread
than the entire climb from 0 to 8 labels covers. 3 nested-sampling runs is
not enough to average out that much variance. I would not claim a true peak
at 32 from this data, only that recovery is fast and reliable while data is
scarce and noisy, without a clear further trend, once a few dozen labels
per category are already in hand.

## Saliency

Grad-CAM overlays for one shifted-domain miss per true category are in
`artifacts/figures/saliency/`. Looked directly at one before trusting the
method: a real "computer" (a desktop tower) misclassified as "laptop," the
CAM heat concentrated on the case's actual front panel and drive bay, a
real region of the object, not scattered over the background. That is the
check that matters for trusting an illustrative method like this one: a
saliency map that looked uniform or background-focused would have meant a
mathematically defined but meaningless map, not a genuinely informative
one.

## Getting the data

The archive is served through Google Drive's large-file confirm-token
redirect. A ZIP's central directory sits at the end of the file, so listing
or selectively fetching members from it needs either a server that honors
HTTP range requests cleanly or the whole file locally; I did not want to
rely on the former against this specific hosting, so `extract.py` reads a
full local copy rather than streaming selected members remotely, at the
cost of downloading more than this project's 20-category subset needs.

The download itself was the least reliable part of building this: `gdown`
against this file stalled twice mid-download (the process stayed alive,
its connection stayed open, but zero bytes moved for several minutes
straight). `gdown.download` has a `resume` parameter that picks a stalled
download back up from its partial file's byte offset instead of restarting
from zero; I did not pass it the first time and lost about 5.4 GB of an
already 84%-complete download to a full restart before checking. Passed it
on the second stall and confirmed the restart resumed from the existing
partial file rather than zero. The fetch script accordingly always passes
`resume=True`, and re-running it after an interruption is safe rather than
wasteful.

## Running it

Requires the source archive; `run_pipeline.py` fetches it if
`data/raw/Adaptiope.zip` is not already present.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers huggingface_hub pillow numpy \
    scikit-learn pyyaml pytest gdown

python run_pipeline.py all
# or one phase at a time: extract, split, train, eval, recovery, saliency
```

Run `pytest` from the project root for the test suite: 33 pass, covering
the split partition and duplicate-atomicity properties, the extraction
logic against a small in-memory ZIP built for the test, the class-balanced
accuracy and bootstrap-CI math against fabricated prediction lists, and the
nested-sampling and resume properties from `recovery.py`. No test loads the
real dataset or a trained model, so the suite runs in seconds regardless of
whether the archive or a checkpoint exists on the machine it runs on; it
validates the logic these results depend on, not the results themselves.

## The sharpest ways this could be wrong

**The shifted sample may not represent real deployment conditions.** Real
Life images are Amazon customer review photographs specifically, a
particular style and motivation for taking a photo (documenting a purchase,
often for a complaint or a review), not a representative sample of every
way a real product ends up photographed in the wild. A different real-world
source, a different platform's user uploads, a different country's typical
phone camera and lighting conditions, could show a smaller or larger gap.

**Category overlap between domains may be doing some of the work that looks
like generalisation.** Both domains are Amazon-sourced. Product listing
photos and customer review photos of a purchase made through the same
platform could share more incidental visual similarity, a particular
background, a particular typical framing, than two genuinely independent
sources would, which could make the measured gap smaller than a harder,
more independent pairing would show. I have not tested against a second,
independently-sourced real-world domain to check this.

**The recovery curve's noise past 8 labels per category is under-sampled,
not resolved.** 3 nested-sampling runs cannot distinguish a genuine
plateau, a genuine peak at 32, or pure sampling noise from each other at
the budget sizes where the range is widest. More runs, not a different
method, is what this would need to actually answer.

**The saliency examples are illustrative, one image per category, not a
systematic study.** I looked at one closely enough to trust the map is
real and not degenerate; I have not characterised what fraction of misses
show the same "attending to the right object, wrong specific identity"
pattern versus attending somewhere clearly wrong.

**This is transfer learning on a small, frozen-backbone classifier head,
not a claim about what a fully fine-tuned or larger model would show.**
A bigger backbone, or unfreezing more of it, could plausibly show a smaller
gap, a different recovery curve, or both; that is a different, larger
experiment than this one.
