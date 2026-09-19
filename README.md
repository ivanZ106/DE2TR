# DE2TR: Dual Evidence Detection Transformer for Video Temporal Grounding

[Homepage](https://ivanz106.github.io/DE2TR-page/)

<!-- The rest of this README is still being written up.

Official implementation of **DE2TR**, a DETR-based framework for video temporal
grounding (VTG) — joint moment retrieval (MR) and highlight detection (HD).

DE2TR targets the boundary-misalignment problem of query-based VTG methods. Unlike
objects in images, video moments have no physical edges: their boundaries are blurred
by a semantic gradient. DE2TR therefore decodes two kinds of evidence separately:

- a **dual-branch decoder (DEFormer)** that splits each moment query channel-wise into
  a semantic branch and two boundary branches (start / end), each with its own
  cross-attention, so global relevance and fine-grained boundary cues are learned
  independently instead of conflicting in one representation;
- a **prior-guided refinement** mechanism that derives a semantic prior map and a
  boundary prior map from the fused video representation and injects them into the
  semantic queries and the decoder memory respectively;
- a **boundary augmentation strategy (BAS)** that splices the ground-truth moment
  features into a background taken from an unrelated video, providing an unambiguous
  boundary signal during training.

## Repository layout

```
de2tr/
  model.py                 DE2TR model + SetCriterion (training losses)
  transformer.py           DEFormer: dual-branch transformer decoder, prior heads
  start_end_dataset.py     QVHighlights dataset, Gaussian boundary labels, BAS
  train.py                 training loop
  inference.py             evaluation / inference (MR + HD)
  config.py, matcher.py, span_utils.py, position_encoding.py, attention.py
  interaction/test_CQA.py  modal fusion & interaction module
  loss_fun/                video-text and video-clip consistency losses
  scripts/                 train.sh, inference.sh
data/                      QVHighlights annotations
standalone_eval/           official QVHighlights evaluation script
utils/                     shared utilities
results/                   released run configs (opt.json) + validation metrics
```

## Setup

**1. Install dependencies.** Tested with Python 3.8 / PyTorch 1.9.0 / CUDA 11.1 on a
single RTX 3090.

```bash
pip install -r requirements.txt
```

**2. Prepare features.** Download the QVHighlights features from
[Moment-DETR](https://github.com/jayleicn/moment_detr) and unpack them so that the
following directories exist relative to the repository root:

```
../features/qvhighlight/
  slowfast_features/     # 2s clips, 2304-d
  clip_b32_vid_k4/       # 2s clips, 3072-d
  clip_b32_txt_k4/       # query features, 2048-d
```

Adjust `feat_root` in [de2tr/scripts/train.sh](de2tr/scripts/train.sh) if you keep the
features elsewhere.

## Training

Each released run corresponds to one command:

```bash
# DE2TR  (matches results/hl-video_tef-test_data-2026_02_03_22_47_15)
bash de2tr/scripts/train.sh --seed 2018

# DE2TR + BAS  (matches results/hl-video_tef-test_data-2026_01_29_22_32_50)
bash de2tr/scripts/train.sh --seed 2018 --use_synthetic_data
```

Runs are written to `results/hl-video_tef-<exp_id>-<timestamp>/`. The best checkpoint
(selected on validation `MR-full-mAP`) is `model_best.ckpt`, and the script launches
validation inference automatically once training finishes.

`--use_synthetic_data` appends a copy of the training set in which each sample keeps its
ground-truth moment but has its background clips replaced by clips from an unrelated
video, so the moment boundaries become unambiguous (`--synth_prob`, default 1.0,
controls the probability of splicing a given sample).

`--use_synthetic_data` selects the older code layout as well as the augmentation. Both
released runs come from two revisions of this codebase that differ in two numerically
inert ways, but those differences change the **order in which gradients are
accumulated**, which is enough to make two 200-epoch runs diverge by ~1 mAP:

| | newer layout (plain) | older layout (`--use_synthetic_data`) |
|---|---|---|
| decoder `saliency_cls_film` projections | instantiated | omitted |
| where the saliency scores are computed | inside `Transformer.forward` | in `model.py` |
| `decoder.saliency_proj1/2` aliases | registered | not registered |

Each run's `opt.json` records the resolved `sal_priori_film` and `saliency_in_transformer`
values. Reproducing a released run requires matching its layout, which is why the two
commands above are the only supported way to retrain the published numbers.

## Inference

Download the checkpoint for the variant you want (see the table below) and drop it into
its run directory under `results/`, next to the `opt.json` that ships with the repo:

```
results/hl-video_tef-test_data-2026_02_03_22_47_15/
  opt.json                      # provided
  best_hl_val_preds_metrics.json# provided
  model_best.ckpt               # <- put the downloaded checkpoint here
```

`opt.json` records the feature paths and dimensions used at training time;
`inference.py` reads it from the checkpoint's own directory and will not start without
it.

```bash
bash de2tr/scripts/inference.sh results/{run_dir}/model_best.ckpt 'val'
bash de2tr/scripts/inference.sh results/{run_dir}/model_best.ckpt 'test'
```

This writes `hl_val_submission.jsonl` / `hl_test_submission.jsonl`, which are the files
expected by the Codalab server for the test split. See
[standalone_eval/README.md](standalone_eval/README.md) for the submission format.

## Released models

| Run directory | Variant | MR R1@0.5 | MR R1@0.7 | mAP@0.5 | mAP@0.75 | Avg. mAP | HIT@1 | Download |
|---|---|---|---|---|---|---|---|---|
| `hl-video_tef-test_data-2026_02_03_22_47_15` | DE2TR | 68.13 | 53.81 | 68.89 | 51.61 | 50.67 | 66.65 | _link to be added_ |
| `hl-video_tef-test_data-2026_01_29_22_32_50` | DE2TR + BAS | 69.81 | 55.29 | 70.52 | 53.43 | 52.12 | 64.90 | _link to be added_ |

Validation-split numbers, taken from each run's `best_hl_val_preds_metrics.json`, which
is included under `results/` for both runs.

Both checkpoints load into this code unchanged. The **+ BAS** checkpoint was trained from
an earlier revision that did not yet contain the (unused, zero-initialised)
`decoder.layers.*.saliency_cls_film` FiLM, nor the `decoder.saliency_proj1/2` aliases, so
it has 614 parameter entries where the newer one has 634. Loading reports the difference
as a warning; it does not affect the forward pass, and either layout loads into either
build. `--use_synthetic_data` selects the 614-parameter layout, `--use_sal_priori`
selects the 634 one.

## Configuration notes

- `--use_bnd_priori` (default **on**, disable with `--no_bnd_priori`) controls whether
  the boundary prior map is injected into the decoder memory.
- `--use_sal_priori` (default **off**) controls whether the saliency prior is injected
  into the semantic queries.
- `--use_synthetic_data` (default **off**) turns on BAS and selects the older code layout;
  `--synth_prob` (default 1.0) sets the splice probability.
- `sal_priori_film` and `saliency_in_transformer` are not command-line flags: they are
  derived in `config.py` from `--use_synthetic_data` (forced on by `--use_sal_priori`).
  See the Training section for why they exist.
- These defaults are the settings both released checkpoints were trained with, and the
  `opt.json` files stored by those runs predate the flags.

The training entry point supports an audio modality via `--a_feat_dir`, but only the
video-only path is used by the released models.

## Acknowledgements

This codebase is built on [QD-DETR](https://github.com/wjun0830/QDDETR), which in turn
derives from [Moment-DETR](https://github.com/jayleicn/moment_detr). The QVHighlights
annotations and the evaluation script under `standalone_eval/` come from Moment-DETR.
We thank the authors for making their code and data available.

## License

MIT. See [LICENSE](LICENSE).
-->
