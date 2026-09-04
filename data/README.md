# Dataset placement

Download the datasets from their official providers and extract them directly
under this directory. The default configuration expects the following layout:

| Dataset | Official access and terms | Paper citation |
|---|---|---|
| CVACT | [ACT dataset instructions](https://github.com/Liumouliu/OriCNN#act-dataset); request access from the dataset author and do not redistribute the data | Liu and Li, *Lending Orientation to Neural Networks for Cross-View Geo-Localization*, CVPR 2019 |
| CVUSA | [MVRL CVUSA request page](https://mvrl.cse.wustl.edu/datasets/cvusa/) | Zhai et al., *Predicting Ground-Level Scene Layout from Aerial Imagery*, CVPR 2017 |

CoR-Geo does not redistribute either dataset. Access, use, and redistribution
remain governed by the respective providers.

```text
data/
├── CVACT/
│   ├── ACT_data.mat
│   ├── ANU_data_small/
│   │   ├── streetview/
│   │   └── satview_polish/
│   └── ANU_data_test/
│       ├── streetview/
│       └── satview_polish/
└── CVUSA/
    ├── splits/
    │   ├── train-19zl.csv
    │   └── val-19zl.csv
    ├── split_locations/
    │   ├── all.csv
    │   ├── train.csv
    │   └── test.csv
    ├── streetview/
    ├── bingmap/
    │   └── 19/
```

Dataset files are intentionally ignored by Git. Do not commit images,
annotations, generated manifests, or caches.

## Expected protocol counts

After manifest preparation and verification, the expected numbers of paired
ground/satellite records are:

| Dataset | Split | Annotated | Used by CoR-Geo |
|---|---|---:|---:|
| CVACT | train | 35,532 | 35,531 |
| CVACT | validation | 8,884 | 8,884 |
| CVACT | test | 92,802 | 92,802 |
| CVUSA | train | 35,532 | 35,532 |
| CVUSA | validation | 8,884 | 8,884 |

The excluded CVACT training pair and its provenance are recorded in
`../configs/cvact_exclusions.yaml`; it must not be silently deleted from the
source annotations. CVUSA uses the official `19zl` split CSV row order.

## Integrity verification

From the repository root, run:

```bash
python tools/prepare_cvact_manifests.py \
  --dataset-config configs/cvact.yaml
python tools/verify_cvact_manifests.py \
  --dataset-config configs/cvact.yaml

python tools/prepare_cvusa_manifests.py
python tools/verify_cvusa_manifests.py
```

Verification checks expected counts, one-to-one IDs, source file inventories,
paths, dimensions, and manifest/provenance hashes. Generated manifests use
repository-relative paths and remain valid when the complete repository is
moved without changing the documented `data/` layout.
