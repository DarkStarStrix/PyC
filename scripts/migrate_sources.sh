#!/usr/bin/env bash
# migrate_sources.sh
#
# Copies source files from the three original repositories into the
# unified PyC_unified directory structure.
#
# Usage:
#   NEXA_VORTEX_DIR=~/Nexa_Vortex \
#   NEXA_INFERENCE_DIR=~/Nexa_Inference \
#   PYC_DIR=~/PyC \
#   bash scripts/migrate_sources.sh
#
# After running this script, review the diff and resolve any conflicts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_DIR="$(dirname "$SCRIPT_DIR")"

VORTEX_DIR="${NEXA_VORTEX_DIR:-$HOME/Nexa_Vortex}"
INFERENCE_DIR="${NEXA_INFERENCE_DIR:-$HOME/Nexa_Inference}"
PYC_SRC_DIR="${PYC_DIR:-$HOME/PyC}"

echo "=== PyC Unified Source Migration ==="
echo "Source: Nexa_Vortex  → $VORTEX_DIR"
echo "Source: Nexa_Inference → $INFERENCE_DIR"
echo "Source: PyC           → $PYC_SRC_DIR"
echo "Target: $UNIFIED_DIR"
echo ""

# ----------------------------------------------------------------
# 1. PyC compiler sources → src/compiler/
# ----------------------------------------------------------------
echo "[1/3] Migrating PyC compiler sources..."

# IR
cp -v "$PYC_SRC_DIR/src/compiler/ir/ir.c"          "$UNIFIED_DIR/src/compiler/ir/ir.c"
cp -v "$PYC_SRC_DIR/include/pyc/ir.h"              "$UNIFIED_DIR/include/pyc/ir.h"

# Pass manager
cp -v "$PYC_SRC_DIR/src/compiler/passes/pass_manager.c" "$UNIFIED_DIR/src/compiler/passes/pass_manager.c"

# Runtime (kernel registry, allocator, control)
cp -v "$PYC_SRC_DIR/src/compiler/runtime/kernel_registry.c"  "$UNIFIED_DIR/src/compiler/runtime/kernel_registry.c"
cp -v "$PYC_SRC_DIR/src/compiler/runtime/runtime_allocator.c" "$UNIFIED_DIR/src/compiler/runtime/runtime_allocator.c"
cp -v "$PYC_SRC_DIR/src/compiler/runtime/runtime_control.c"   "$UNIFIED_DIR/src/compiler/runtime/runtime_control.c"
cp -v "$PYC_SRC_DIR/src/compiler/runtime/cuda_backend.c"      "$UNIFIED_DIR/src/compiler/runtime/cuda_backend.cu"

# Compiler API and AI bridge
cp -v "$PYC_SRC_DIR/src/compiler/compiler_api.c"     "$UNIFIED_DIR/src/compiler/compiler_api.c"
cp -v "$PYC_SRC_DIR/src/compiler/ai/ai_bridge.c"     "$UNIFIED_DIR/src/compiler/ai/ai_bridge.c"

# Public headers
cp -v "$PYC_SRC_DIR/include/pyc/compiler_api.h"      "$UNIFIED_DIR/include/pyc/compiler_api.h"
cp -v "$PYC_SRC_DIR/include/pyc/kernel_registry.h"   "$UNIFIED_DIR/include/pyc/kernel_registry.h"
cp -v "$PYC_SRC_DIR/include/pyc/runtime_allocator.h" "$UNIFIED_DIR/include/pyc/runtime_allocator.h"
cp -v "$PYC_SRC_DIR/include/pyc/optimizer_policy.h"  "$UNIFIED_DIR/include/pyc/optimizer_policy.h"
cp -v "$PYC_SRC_DIR/include/pyc/ai_bridge.h"         "$UNIFIED_DIR/include/pyc/ai_bridge.h"

# Tests
cp -rv "$PYC_SRC_DIR/tests/compiler_next/"* "$UNIFIED_DIR/tests/compiler_next/"

# ----------------------------------------------------------------
# 2. Nexa_Vortex runtime sources → src/runtime/vortex_core/src/
# ----------------------------------------------------------------
echo "[2/3] Migrating Nexa_Vortex runtime sources..."

VORTEX_SRC="$VORTEX_DIR/rust/vortex_core/src"
UNIFIED_RT="$UNIFIED_DIR/src/runtime/vortex_core/src"

# NOTE: hw_profile, allocator, cpu_dispatch, telemetry, errors have been
# rewritten in this unified repo. Only migrate integrations/ (Mesocarp).
cp -rv "$VORTEX_SRC/integrations/"* "$UNIFIED_RT/integrations/"

# Python control plane and telemetry manager
cp -v "$VORTEX_DIR/python/nexa_vortex/core/controlplane/control_plane.py" \
      "$UNIFIED_DIR/python/pyc/runtime/control_plane.py"
cp -v "$VORTEX_DIR/python/nexa_vortex/core/telemetry/telemetry_manager.py" \
      "$UNIFIED_DIR/python/pyc/runtime/telemetry_manager.py"

# ----------------------------------------------------------------
# 3. Nexa_Inference app sources → apps/inference_api/
# ----------------------------------------------------------------
echo "[3/3] Migrating Nexa_Inference app sources..."

cp -v "$INFERENCE_DIR/src/main.py"       "$UNIFIED_DIR/apps/inference_api/src/main.py"
cp -v "$INFERENCE_DIR/src/inference.py"  "$UNIFIED_DIR/apps/inference_api/src/inference.py"
cp -v "$INFERENCE_DIR/src/engines.py"    "$UNIFIED_DIR/apps/inference_api/src/engines.py"
cp -v "$INFERENCE_DIR/src/Pipelines.py"  "$UNIFIED_DIR/apps/inference_api/src/pipelines.py"
cp -v "$INFERENCE_DIR/src/models.py"     "$UNIFIED_DIR/apps/inference_api/models/schemas.py"
cp -v "$INFERENCE_DIR/src/auth.py"       "$UNIFIED_DIR/apps/inference_api/src/auth.py"
cp -v "$INFERENCE_DIR/src/Config.py"     "$UNIFIED_DIR/apps/inference_api/src/config.py"

echo ""
echo "=== Migration complete. Review changes with: git diff ==="
echo "Next steps:"
echo "  1. Update import paths in migrated Python files"
echo "  2. Run: cmake -B build -DPYC_BUILD_CUDA=ON && cmake --build build"
echo "  3. Run: maturin develop --features python_ext"
echo "  4. Run: pytest tests/"
