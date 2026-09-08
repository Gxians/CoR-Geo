# CoR-Geo

Official implementation of **CoR-Geo: Weakly Ordered Directional
Representation for Limited-FoV Cross-View Geo-Localization**.

> Paper and pretrained checkpoints will be released with the preprint.

CoR-Geo represents a ground view with bottom-to-top columns and a satellite
view with center-to-boundary rays. A shared Content--Order encoder preserves
directional evidence, while FoV-masked cyclic matching handles unknown
headings. One model and one pre-encoded satellite gallery support multiple
fields of view without heading supervision.

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="CoR-Geo architecture">
</p>

## Installation

The experiments use Python 3.10, PyTorch 2.3.1, CUDA 12.1, and DINOv2-B/14.

```bash
conda create -n cor_geo python=3.10 -y
conda activate cor_geo
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .

mkdir -p third_party checkpoints/dinov2
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
wget -O checkpoints/dinov2/dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

## Data

Request CVACT and CVUSA from their providers, then place them directly under
`data/`. The expected directory layout and split sizes are listed in
[`data/README.md`](data/README.md). Prepare manifests and resized caches with:

```bash
python -m cor_geo.datasets --dataset cvact
python -m cor_geo.datasets --dataset cvusa
```

All default paths are repository-relative: datasets are stored in `data/`,
caches in `.cache/cor_geo/`, and experiments in `outputs/`.

## Training

Training keeps the reported global batch size of 64 under both supported
topologies. Single-GPU training is the default; `--devices 0,1` selects the
reported two-GPU topology. The launcher automatically uses 64 samples on one
GPU or 32 per GPU on two GPUs, while learning rates and global FoV quotas remain
unchanged. The executions implement the same batch protocol, although floating-
point reduction order means their checkpoints are not expected to be bitwise
identical. Resume a run with the same GPU count that created it.

```bash
# Single GPU (default): 1 x 64 = 64
python -m cor_geo.train --dataset cvact
python -m cor_geo.train --dataset cvusa

# Two GPUs (paper setting): 2 x 32 = 64
python -m cor_geo.train --dataset cvact --devices 0,1
python -m cor_geo.train --dataset cvusa --devices 0,1
```

Use `--run-name` to select another output directory and `--resume` to continue
from a checkpoint.

## Evaluation

Evaluation supports one or more GPUs and draws an independent random heading
for every query--FoV pair. The sampled schedule is saved with the results.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 -m cor_geo.evaluate \
  --run-dir outputs/cvact/cor_geo_cvact \
  --checkpoint epoch_064
```

Pass `--crop-schedule <file.parquet>` to replay a recorded draw. To evaluate
additional FoVs, use `--fovs ... --allow-unseen-fovs`.

## Results

Validation Macro R@1 over 360°, 180°, 90°, and 70° random crops:

| Dataset | Macro R@1 |
|:--|--:|
| CVACT | 75.4 |
| CVUSA | 79.8 |

These values come from one trained checkpoint and one recorded random-crop
evaluation. Download links will be added after the checkpoint release.

## Tests

The unit tests require neither datasets nor pretrained weights.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Citation

The BibTeX entry will be added when the preprint is available.

## License

The source code is released under the [MIT License](LICENSE). CoR-Geo uses
[DINOv2](https://github.com/facebookresearch/dinov2) as an external dependency;
its code and pretrained weights remain subject to the upstream license.
