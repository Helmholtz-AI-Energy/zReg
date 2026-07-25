#!/usr/bin/env bash
# Sets up an isolated Jupyter kernel for the zReg tutorials.
#
# Run from the repo root:
#   bash docs/tutorials/setup_tutorial_env.sh
#
# After this script completes, open JupyterLab and select the
# "zreg-tutorials" kernel when opening a tutorial notebook.
# To remove the environment: rm -rf .tutorial-venv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.tutorial-venv"
KERNEL_NAME="zreg-tutorials"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Installing zreg with visualization extras"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -e "$REPO_ROOT[viz]"

echo "==> Installing Jupyter kernel"
"$VENV_DIR/bin/pip" install --quiet ipykernel
"$VENV_DIR/bin/python" -m ipykernel install --user --name="$KERNEL_NAME" --display-name="Python (zreg-tutorials)"

echo ""
echo "Done. Open JupyterLab and select the '$KERNEL_NAME' kernel."
