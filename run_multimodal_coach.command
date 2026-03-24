#!/usr/bin/env bash
# run_multimodal_coach.command
# ----------------------------
# One-click macOS launcher for the Multimodal Coach app.
# Double-click this file in Finder (or run it from Terminal).
#
# Prerequisites (one-time):
#   conda activate dslcv2
#   pip install -r requirements.txt
#
# macOS permissions required on first run:
#   System Settings → Privacy & Security → Camera  → allow Terminal / your IDE
#   System Settings → Privacy & Security → Microphone → allow Terminal / your IDE

# ── Locate this script's directory regardless of where it is double-clicked ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }

# ── Activate conda env ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || \
  source "$CONDA_BASE/bin/activate" 2>/dev/null || true

conda activate dslcv2 2>/dev/null || true

# ── Verify mediapipe solutions API is present ────────────────────────────────
PYTHON="$CONDA_BASE/envs/dslcv2/bin/python"
if [ ! -f "$PYTHON" ]; then
  PYTHON=python3
fi

"$PYTHON" - <<'EOF' 2>/dev/null
import mediapipe as mp
assert hasattr(mp, 'solutions'), "mediapipe solutions API missing"
EOF
if [ $? -ne 0 ]; then
  echo ""
  echo "⚠️  mediapipe version incompatible — installing 0.10.14 ..."
  "$PYTHON" -m pip install "mediapipe==0.10.14" --quiet
fi

# ── Launch app ───────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Multimodal Coach  —  starting up ..."
echo "  Press Q inside the camera window to quit."
echo "============================================"
echo ""

PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" "$SCRIPT_DIR/apps/run_multimodal_coach.py"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "App closed cleanly."
else
  echo "App exited with code $EXIT_CODE."
fi

# Keep Terminal window open so the user can read any error messages
echo ""
echo "Press any key to close this window ..."
read -r -n 1
