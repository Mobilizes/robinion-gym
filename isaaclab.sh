#!/usr/bin/env bash
# Wrapper for running the Isaac Lab CLI with the Isaac Sim physics backend.
#
# Workaround for an Isaac Lab 3.0.0 / Isaac Sim 6.1.0.0 USD runtime conflict:
#   * isaacsim 6.1.0.0 bundles OpenUSD 25.11 (pxrInternal_v0_25_11)
#   * the pinned usd-exchange 2.3.0 vendors OpenUSD 25.05 (pxrInternal_v0_25_5)
# Both ship a namespace `pxr` package without a top-level __init__.py, so Python
# merges them. `pxr.Tf` then loads from usd-exchange (25.05) while `pxr.UsdLux`,
# `pxr.UsdPhysics`, ... load from Isaac Sim's extscache (25.11). The mixed runtime
# breaks `omni.physx` startup with:
#   RuntimeError: extension class wrapper ... has not been created yet
#   TypeError: No to_python (by-value) converter found for C++ type: std::vector<SdfPath>
#
# This wrapper puts Isaac Sim's bundled pxr (and its native libraries) ahead of the
# standalone usd-exchange pxr so only one USD runtime is used. Remove once Isaac Lab
# pins a usd provider that matches Isaac Sim's bundled OpenUSD version.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTS_DIR="$(ls -d "$PROJECT_ROOT"/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.usd.libs-*/ 2>/dev/null | head -n1)"

if [[ -z "${EXTS_DIR}" ]]; then
  echo "error: Isaac Sim 'omni.usd.libs' extscache not found under .venv." >&2
  echo "       Run 'uv sync --extra isaacsim' first." >&2
  exit 1
fi

export PYTHONPATH="${EXTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${EXTS_DIR}bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec uv run --extra isaacsim isaaclab "$@"
