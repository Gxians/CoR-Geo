# CoR-Geo

Official implementation of **CoR-Geo: Weakly Ordered Directional Representation for Limited-FoV Cross-View Geo-Localization**.

> The paper and pretrained models will be released with the preprint.

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

Install CoR-Geo with:

```bash
git clone https://github.com/Gxians/CoR-Geo.git
cd CoR-Geo

conda create -n cor_geo python=3.10 -y
conda activate cor_geo
pip install -e . --no-build-isolation
```

### DINOv2 Backbone

Download the DINOv2 source code and ViT-B/14 pretrained weights:

```bash
bash scripts/setup_dinov2.sh
```

## Dataset Preparation

Request CVACT and CVUSA from their respective providers and place them under `data/`. The expected directory structure and dataset access links are provided in [`data/README.md`](data/README.md).

Build the manifests and resized-image caches with:

```bash
python -m cor_geo.datasets --dataset cvact
python -m cor_geo.datasets --dataset cvusa
```

## Usage

### Training

Train on CVACT or CVUSA using one GPU:

```bash
python -m cor_geo.train --dataset cvact
python -m cor_geo.train --dataset cvusa
```

Use two GPUs for the setting reported in the paper:

```bash
python -m cor_geo.train --dataset cvact --devices 0,1
python -m cor_geo.train --dataset cvusa --devices 0,1
```

To continue an interrupted run, append `--resume <checkpoint>`.

### Evaluation

Evaluate a checkpoint under random FoV crops:

```bash
python -m cor_geo.evaluate \
  --run-dir outputs/cvact/cor_geo_cvact \
  --checkpoint epoch_064
```

Replace `cvact` with `cvusa` to evaluate the CVUSA model. Add `--devices 0,1` to use two GPUs. By default, evaluation draws a new random crop schedule. Use `--crop-schedule <file.parquet>` to replay a recorded schedule.

## Pretrained Models and Results

Pretrained checkpoints will be released with the preprint.

Validation Macro R@1 averaged over 360°, 180°, 90°, and 70° random crops:

| Dataset | Checkpoint | Macro R@1 (%) |
|:--|:--:|--:|
| CVACT | Coming soon | 75.4 |
| CVUSA | Coming soon | 79.8 |

## Citation

The BibTeX entry will be added when the preprint becomes available.

## Acknowledgements

CoR-Geo uses the [DINOv2](https://github.com/facebookresearch/dinov2) backbone. We thank its authors and the providers of CVACT and CVUSA for making their research resources available.

## License

The CoR-Geo source code is released under the [MIT License](LICENSE). Third-party code, pretrained weights, and datasets remain subject to their respective licenses.
