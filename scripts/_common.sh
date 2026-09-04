#!/usr/bin/env bash

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/.." && pwd)
cd "${project_root}"

paths_config=${COR_GEO_PATHS_CONFIG:-configs/paths.yaml}
python_bin=${PYTHON_BIN:-python}
torchrun_bin=${TORCHRUN_BIN:-torchrun}

if [[ ! -f "${paths_config}" ]]; then
  echo "Missing ${paths_config}. The repository should include the default configs/paths.yaml." >&2
  exit 2
fi

output_root=$("${python_bin}" -c '
import pathlib, sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
output = pathlib.Path(value["output_root"]).expanduser()
if output.is_absolute() or ".." in output.parts:
    raise SystemExit("output_root must be repository-relative")
print(output.as_posix())
' "${paths_config}")

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

run_name_for() {
  local dataset=$1
  if [[ -n "${COR_GEO_RUN_NAME:-}" ]]; then
    printf '%s\n' "${COR_GEO_RUN_NAME}"
  else
    printf 'cor_geo_%s\n' "${dataset}"
  fi
}

train_config_for() {
  local dataset=$1
  if [[ "${dataset}" == "cvact" ]]; then
    printf '%s\n' configs/train_cvact.yaml
  else
    printf '%s\n' configs/train_cvusa.yaml
  fi
}
