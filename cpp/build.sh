#!/usr/bin/env bash
# build.sh — configure and compile all C++ extension modules
#
# Usage:
#   ./build.sh              # Release
#   ./build.sh Debug        # Debug + AddressSanitizer
#   ./build.sh Release stubs  # Release + generate .pyi stubs
#
# Python resolution order (first wins):
#   1. $PYTHON env var — set this explicitly if you need a specific interpreter
#   2. Active virtualenv  ($VIRTUAL_ENV/bin/python)
#   3. Poetry venv        (poetry run python)
#   4. uv venv            (.venv/bin/python  relative to the project root)
#   5. System python3 / python
#
# pybind11 resolution order (CMake, first wins):
#   1. Found via the resolved Python's pybind11.get_cmake_dir()
#   2. System cmake config  (find_package)
#   3. git submodule at cpp/third_party/pybind11  (offline / CI fallback)
#
# One-time setup (if pybind11 is not yet in your Python env):
#   ./setup.sh

set -euo pipefail

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"
EXTRA_TARGET="${2:-}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Resolve Python interpreter
# ─────────────────────────────────────────────────────────────────────────────

find_python() {
    # Explicit override always wins
    if [[ -n "${PYTHON:-}" ]]; then
        echo "${PYTHON}"; return
    fi

    # Active venv (pip install / venv / conda)
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"; return
    fi

    # Poetry venv — resolve without spawning a subshell each time
    if command -v poetry &>/dev/null; then
        local p
        p="$(poetry env info --executable 2>/dev/null || true)"
        if [[ -n "$p" && -x "$p" ]]; then
            echo "$p"; return
        fi
    fi

    # uv project venv (look relative to THIS script's parent = cpp/)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root
    project_root="$(cd "${script_dir}/.." && pwd)"

    for candidate in \
        "${project_root}/.venv/bin/python" \
        "${script_dir}/../.venv/bin/python"
    do
        if [[ -x "$candidate" ]]; then
            echo "$(realpath "$candidate")"; return
        fi
    done

    # Fall back to whatever python3/python is on PATH
    if command -v python3 &>/dev/null; then
        echo "$(command -v python3)"; return
    fi
    echo "$(command -v python)"
}

PYTHON="$(find_python)"

# Sanity-check the interpreter exists
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: could not find a Python interpreter."
    echo "  Set the PYTHON env var:  PYTHON=/path/to/python ./build.sh"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify pybind11 is importable from this interpreter
# ─────────────────────────────────────────────────────────────────────────────

PB11_VER="$("$PYTHON" -c 'import pybind11; print(pybind11.__version__)' 2>/dev/null || true)"

echo "=================================================="
echo "  breast C++ build  (${BUILD_TYPE})"
echo "=================================================="
echo "Python:    $PYTHON  ($(${PYTHON} --version 2>&1))"

if [[ -z "$PB11_VER" ]]; then
    echo ""
    echo "WARNING: pybind11 not importable from this interpreter."
    echo "  CMake will fall back to the git submodule in third_party/pybind11."
    echo "  To avoid this, install pybind11 in your env:"
    echo "    pip install pybind11       # pip / venv"
    echo "    poetry add --group dev pybind11  # poetry"
    echo "    uv add --dev pybind11      # uv"
    echo ""
else
    echo "pybind11:  ${PB11_VER}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Initialize pybind11 submodule if not present (offline: already there)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE_DIR="${SCRIPT_DIR}/third_party/pybind11"

if [[ ! -f "${SUBMODULE_DIR}/CMakeLists.txt" ]]; then
    echo "Initializing pybind11 git submodule..."
    git -C "${SCRIPT_DIR}" submodule update --init --recursive \
        third_party/pybind11 2>/dev/null \
    || echo "  (git submodule init failed — network may be required on first run)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. CMake configure + build
# ─────────────────────────────────────────────────────────────────────────────

cmake \
    -S "${SCRIPT_DIR}" \
    -B "${SCRIPT_DIR}/${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPython_EXECUTABLE="${PYTHON}" \
    -DBREAST_PKG_DIR="${SCRIPT_DIR}/../src/breast"

JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
cmake --build "${SCRIPT_DIR}/${BUILD_DIR}" \
    --config "${BUILD_TYPE}" \
    --parallel "${JOBS}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional stub generation
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${EXTRA_TARGET}" == "stubs" ]]; then
    echo ""
    echo "Generating .pyi stubs..."
    cmake --build "${SCRIPT_DIR}/${BUILD_DIR}" --target stubs \
    || echo "WARNING: stub generation failed (pip install pybind11-stubgen to enable)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Report
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "Build complete!"
echo ""
echo "Extension modules:"
find "${SCRIPT_DIR}/${BUILD_DIR}/lib" \
    \( -name '_*.so' -o -name '_*.pyd' \) 2>/dev/null | sort

echo ""
echo "Usage — change the import in sim.py:"
echo "  - from breast.solver_numpy import UltraStableSolver"
echo "  + from breast.solver_cpp   import UltraStableSolver"
