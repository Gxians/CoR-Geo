# Data preparation

CoR-Geo does not redistribute CVACT or CVUSA. Request each dataset from its
provider and extract it into this directory:

- [CVACT access instructions](https://github.com/Liumouliu/OriCNN#act-dataset)
- [CVUSA request page](https://mvrl.cse.wustl.edu/datasets/cvusa/)

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
    ├── streetview/
    │   ├── panos/
    │   └── annotations/
    └── bingmap/
        ├── 18/                 # optional; not used by the 19zl protocol
        ├── 19/                 # required satellite images
        └── 20/                 # optional; not used by the 19zl protocol
```

The official `19zl` split CSVs reference images under `streetview/panos/`,
annotations under `streetview/annotations/`, and satellites under
`bingmap/19/`. `split_locations/` is retained as part of the distributed
dataset but is not read by the CoR-Geo training pipeline. The `bingmap/18/`
and `bingmap/20/` image levels and downloaded archive files may remain in the
dataset directory, but the default configuration neither scans nor caches
them.

Expected paired records are 35,531/8,884/92,802 for CVACT
train/validation/test and 35,532/8,884 for CVUSA train/validation. The single
unavailable CVACT training pair is declared in `configs/cvact.yaml`.

From the repository root, build portable manifests and resized image caches:

```bash
python -m cor_geo.datasets --dataset cvact
python -m cor_geo.datasets --dataset cvusa
```

Preparation verifies IDs, file inventories, image dimensions, and expected
split sizes. Generated manifests use repository-relative paths, so the whole
repository can be moved without editing configuration files.
