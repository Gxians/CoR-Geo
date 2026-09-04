#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

dataset=${1:-}
if [[ "${dataset}" != "cvact" && "${dataset}" != "cvusa" ]]; then
  echo "usage: bash scripts/train.sh {cvact|cvusa}" >&2
  exit 2
fi

dataset_config="configs/${dataset}.yaml"
train_config=$(train_config_for "${dataset}")
run_name=$(run_name_for "${dataset}")
run_dir="${output_root}/${dataset}/${run_name}"
visible_devices=${COR_GEO_GPUS:-0,1}
master_port=${COR_GEO_MASTER_PORT:-29580}

if [[ ! -f "data_manifests/${dataset}/train.parquet" || ! -f "data_manifests/${dataset}/val.parquet" ]]; then
  "${python_bin}" "tools/prepare_${dataset}_manifests.py" \
    --dataset-config "${dataset_config}" \
    --paths "${paths_config}"
fi
"${python_bin}" "tools/verify_${dataset}_manifests.py" \
  --dataset-config "${dataset_config}" \
  --paths "${paths_config}"

"${python_bin}" tools/build_dataset_cache.py \
  --dataset-config "${dataset_config}" \
  --train-config "${train_config}" \
  --model-config configs/model.yaml \
  --paths "${paths_config}" \
  --splits train val \
  --workers 8 \
  --batch-size 16

"${python_bin}" tools/verify_dataset_cache.py \
  --dataset-config "${dataset_config}" \
  --train-config "${train_config}" \
  --model-config configs/model.yaml \
  --paths "${paths_config}" \
  --splits train val \
  --samples-per-split 8

final_checkpoint="${run_dir}/checkpoints/epoch_064.ckpt"
if [[ -f "${final_checkpoint}" ]]; then
  echo "CoR-Geo ${dataset^^} training is already complete: ${final_checkpoint}"
  exit 0
fi

train_args=(
  --dataset-config "${dataset_config}"
  --model-config configs/model.yaml
  --train-config "${train_config}"
  --eval-config configs/evaluation.yaml
  --paths "${paths_config}"
  --run-name "${run_name}"
)
last_checkpoint="${run_dir}/checkpoints/last.ckpt"
if [[ -f "${last_checkpoint}" ]]; then
  train_args+=(--resume "${last_checkpoint}")
fi

CUDA_VISIBLE_DEVICES="${visible_devices}" "${torchrun_bin}" \
  --nproc_per_node=2 \
  --master_port="${master_port}" \
  tools/train.py \
  "${train_args[@]}"

echo "CoR-Geo ${dataset^^} training completed: ${final_checkpoint}"
