#!/usr/bin/env bash
# Rebuild monty_demo and source the workspace. Needed when you change code/launch/config/urdf.
# Run before start_ros2_bringup.sh (terminal 2). Not required before start_isaac (terminal 1).
#
# Usage (source so env applies to current shell):
#   source scripts/rebuild_and_source.sh            # incremental build
#   source scripts/rebuild_and_source.sh --clean     # full clean build
# Then in terminal 2: ./scripts/start_ros2_bringup.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PACKAGES=(monty_demo)
CLEAN=false
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=true ;;
    *) echo "Unknown option: $arg"; return 1 2>/dev/null || exit 1 ;;
  esac
done

# ── Setuptools compatibility guard ──────────────────────────────────────────
# colcon uses /usr/bin/python3 (system Python) regardless of the active venv.
# System setuptools >= 72 removed `setup.py develop --uninstall`, which colcon
# calls when stale develop-mode artifacts exist in the build dir.  Detect this
# and auto-clean so the build doesn't fail with "option --uninstall not recognized".
_sys_setuptools_ver=$(/usr/bin/python3 -c "import setuptools; print(setuptools.__version__)" 2>/dev/null || echo "0")
_sys_setuptools_major=${_sys_setuptools_ver%%.*}

_needs_develop_cleanup() {
  local pkg="$1"
  local build_dir="$REPO_ROOT/build/$pkg"
  local egg_info="$build_dir/${pkg//-/_}.egg-info"
  [[ -d "$egg_info" && -L "$build_dir/setup.py" ]]
}

for pkg in "${PACKAGES[@]}"; do
  if _needs_develop_cleanup "$pkg"; then
    if [[ "$_sys_setuptools_major" -ge 72 ]]; then
      echo "⚠  Stale develop artifacts in build/$pkg (system setuptools $_sys_setuptools_ver >= 72)."
      echo "   Auto-cleaning build/$pkg to avoid --uninstall error."
      rm -rf "build/$pkg" "install/$pkg" "log/$pkg"
    fi
  fi
done

# ── Optional full clean ────────────────────────────────────────────────────
if $CLEAN; then
  echo "=== Cleaning build artifacts ==="
  for pkg in "${PACKAGES[@]}"; do
    rm -rf "build/$pkg" "install/$pkg" "log/$pkg"
  done
fi

# ── Build ───────────────────────────────────────────────────────────────────
echo "=== Rebuilding ${PACKAGES[*]} ==="
source /opt/ros/jazzy/setup.bash
colcon build --packages-select "${PACKAGES[@]}"

# ── Source ──────────────────────────────────────────────────────────────────
echo "=== Sourcing workspace ==="
source "$REPO_ROOT/install/setup.bash"

echo "Done. Workspace is sourced in this shell."
echo "Run: ./scripts/run_x3plus_isaac_ros2.sh"
