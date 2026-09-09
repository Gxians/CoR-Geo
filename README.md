# CoR-Geo: Weakly Ordered Directional Representation for Limited-FoV Cross-View Geo-Localization

[[arXiv](https://arxiv.org/abs/XXXX.XXXXX)]

Official method implementation of CoR-Geo.

## Introduction

CoR-Geo addresses limited-FoV cross-view geo-localization under unknown relative headings. It organizes ground patch columns and satellite rays into direction-level weakly ordered representations, avoiding dense vertical--radial alignment. FoV-masked cyclic matching enables one model and one pre-encoded satellite gallery to support multiple FoVs without heading supervision.

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="CoR-Geo architecture">
</p>

## Installation

The code has been tested with:

- Linux
- Python 3.10
- PyTorch 2.3.1
- CUDA 12.1

Install CoR-Geo and its required environment with:

```bash
git clone https://github.com/Gxians/CoR-Geo.git
cd CoR-Geo

conda create -n cor_geo python=3.10 -y
conda activate cor_geo
pip install -e .
```

### DINOv2 Backbone

CoR-Geo uses the official [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 implementation and pretrained weights. Run:

```bash
bash scripts/setup_dinov2.sh
```

The script downloads a pinned DINOv2 source revision and the ViT-B/14 pretrained checkpoint.

## Dataset Preparation

Request CVACT and CVUSA from their respective providers and place them under `data/`. The expected directory structure and dataset access links are provided in [`data/README.md`](data/README.md).

Build the manifests and resized-image caches with:

```bash
python -m cor_geo.datasets --dataset cvact
```

Replace `cvact` with `cvusa` to prepare CVUSA.

## Usage

### Training

Train on CVACT using one GPU:

```bash
python -m cor_geo.train --dataset cvact
```

Use two GPUs for the setting reported in the paper:

```bash
python -m cor_geo.train --dataset cvact --devices 0,1
```

Replace `cvact` with `cvusa` to train on CVUSA.

Training evaluates the Val split every 8 epochs at 360°, 180°, 90°, and 70°. One random crop schedule is created per run and reused by every checkpoint. The checkpoint with the highest four-FoV Macro R@1 is saved as `checkpoints/best.ckpt`, with its metrics in `evaluations/val_random/best_summary.json`.

To continue an interrupted run, append `--resume <checkpoint>`.

### Evaluation

Evaluate a checkpoint under random FoV crops:

```bash
python -m cor_geo.evaluate --dataset cvact
```

By default, evaluation loads `outputs/cvact/cor_geo_cvact/checkpoints/best.ckpt`. Replace `cvact` with `cvusa` to evaluate the CVUSA model, or append `--checkpoint epoch_008` to evaluate another retained checkpoint. Add `--devices 0,1` to use two GPUs. Val evaluation reuses the run's recorded random crop schedule. Use `--crop-schedule <file.parquet>` to replay another recorded schedule.

## Results

Validation Macro R@1 averaged over 360°, 180°, 90°, and 70° random crops:

| Dataset | Checkpoint | Macro R@1 (%) |
|:--|:--:|--:|
| CVACT | Coming soon | 75.4 |
| CVUSA | Coming soon | 79.8 |

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[DINOv2](https://github.com/facebookresearch/dinov2)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{author2026corgeo,
  title   = {CoR-Geo: Weakly Ordered Directional Representation for Limited-FoV Cross-View Geo-Localization},
  author  = {...},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
