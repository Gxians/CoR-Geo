#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

dataset=${1:-}
split=${2:-val}
checkpoint=${3:-epoch_064}
if [[ "${dataset}" != "cvact" && "${dataset}" != "cvusa" ]]; then
  echo "usage: bash scripts/evaluate.sh {cvact|cvusa} [val|test] [epoch_NNN]" >&2
  exit 2
fi
if [[ "${split}" != "val" && "${split}" != "test" ]]; then
  echo "split must be val or test" >&2
  exit 2
fi
if [[ "${dataset}" == "cvusa" && "${split}" != "val" ]]; then
  echo "CVUSA configuration exposes only the val evaluation split" >&2
  exit 2
fi
if [[ ! "${checkpoint}" =~ ^epoch_[0-9]{3}$ ]]; then
  echo "checkpoint must have the form epoch_NNN" >&2
  exit 2
fi
run_name=$(run_name_for "${dataset}")
run_dir="${output_root}/${dataset}/${run_name}"
visible_devices=${COR_GEO_GPUS:-0,1}
master_port=${COR_GEO_MASTER_PORT:-29590}
runtime_config=${COR_GEO_EVAL_RUNTIME_CONFIG:-configs/evaluation_runtime.yaml}
if [[ ! -f "${runtime_config}" ]]; then
  echo "Missing evaluation runtime config: ${runtime_config}" >&2
  exit 2
fi
nproc_per_node=${COR_GEO_EVAL_PROCESSES:-}
if [[ -z "${nproc_per_node}" ]]; then
  nproc_per_node=$("${python_bin}" -c '
import sys
devices = [value.strip() for value in sys.argv[1].split(",") if value.strip()]
if not devices:
    raise SystemExit("COR_GEO_GPUS must expose at least one GPU")
print(len(devices))
' "${visible_devices}")
fi
bank_dir="${run_dir}/evaluations/shared_satellite_bank"
fovs=(360 180 90 70)
evaluation_args=(
  --run-dir "${run_dir}"
  --checkpoint "${checkpoint}"
  --split "${split}"
  --fovs "${fovs[@]}"
  --runtime-config "${runtime_config}"
  --paths "${paths_config}"
  --satellite-bank-dir "${bank_dir}"
)
if [[ -n "${COR_GEO_CROP_SCHEDULE:-}" ]]; then
  evaluation_args+=(--crop-schedule "${COR_GEO_CROP_SCHEDULE}")
fi

CUDA_VISIBLE_DEVICES="${visible_devices}" "${torchrun_bin}" \
  --nproc_per_node="${nproc_per_node}" \
  --master_port="${master_port}" \
  tools/evaluate.py \
  "${evaluation_args[@]}"
