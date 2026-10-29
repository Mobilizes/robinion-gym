#!/usr/bin/env bash
# Wrapper for running the Isaac Lab CLI with the Isaac Sim physics backend.
#
# Isaac Sim and the pinned usd-exchange package can both ship a namespace `pxr`
# package without a top-level __init__.py, so Python may merge them. `pxr.Tf` can
# then load from one OpenUSD runtime while `pxr.UsdLux`, `pxr.UsdPhysics`, ... load
# from the other. The mixed runtime breaks `omni.physx` startup with errors such as:
#   RuntimeError: extension class wrapper ... has not been created yet
#   TypeError: No to_python (by-value) converter found for C++ type: std::vector<SdfPath>
#
# This wrapper puts Isaac Sim's bundled pxr (and its native libraries) ahead of the
# standalone usd-exchange pxr so only one USD runtime is used. Remove once the pinned
# USD provider matches Isaac Sim's bundled OpenUSD version.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTS_DIR="$(ls -d "$PROJECT_ROOT"/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.usd.libs-*/ 2>/dev/null | head -n1)"

if [[ -z "${EXTS_DIR}" ]]; then
  echo "error: Isaac Sim 'omni.usd.libs' extscache not found under .venv." >&2
  echo "       Run 'uv sync --extra isaacsim' first." >&2
  exit 1
fi

# Drop ROS 2 and system CUDA toolkit paths inherited from the calling shell.
# A system CUDA toolkit (e.g. CUDA 13 under /usr/local/cuda) shadows the CUDA 12
# runtime bundled with Isaac Lab's PyTorch and breaks `import torch` with:
#   ImportError: libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion
_filter_path() {
  local IFS=':'
  local p cleaned=""
  for p in $1; do
    case "$p" in
      ""|*"/opt/ros"*|*"ros2-ws"*|*"/usr/local/cuda"*) continue ;;
      *) cleaned="${cleaned}${cleaned:+:}$p" ;;
    esac
  done
  printf '%s' "$cleaned"
}

_clean_ld="$( _filter_path "${LD_LIBRARY_PATH:-}" )"
_clean_py="$( _filter_path "${PYTHONPATH:-}" )"

export PYTHONPATH="${EXTS_DIR}${_clean_py:+:${_clean_py}}"
export LD_LIBRARY_PATH="${EXTS_DIR}bin${_clean_ld:+:${_clean_ld}}"

exec uv run --extra isaacsim isaaclab "$@"
