#!/bin/bash

# Exit on error
set -e

VENV_NAME="asr_venv"

echo "================================================"
echo "Setting up Omnilingual ASR Dashboard Environment"
echo "================================================"

# 1. Create Virtual Environment
if [ ! -f "$VENV_NAME/bin/python" ]; then
    echo "Creating or repairing virtual environment: $VENV_NAME..."
    # Clean up if directory exists but is broken
    if [ -d "$VENV_NAME" ]; then
        echo "Found broken venv directory, removing and recreating..."
        rm -rf "$VENV_NAME"
    fi
    python3.10 -m venv "$VENV_NAME" || python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment $VENV_NAME already exists and is valid."
fi

# 2. Upgrade Pip
echo "Upgrading pip..."
./$VENV_NAME/bin/python -m pip install --upgrade pip

# 3. Install PyTorch and core ML libraries (Version 2.6.0 as required)
echo "Installing PyTorch 2.6.0 + Audio..."
./$VENV_NAME/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install Fairseq2 (Specific Meta version)
echo "Installing Fairseq2..."
./$VENV_NAME/bin/python -m pip install fairseq2==0.6 fairseq2n==0.6 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124

# 5. Install Dashboard Requirements
if [ -f "requirements-dashboard.txt" ]; then
    echo "Installing dashboard requirements..."
    ./$VENV_NAME/bin/python -m pip install -r requirements-dashboard.txt
fi

# 6. Install omnilingual-asr package
# Use last version pulled from PyPI.
# We use --no-deps to avoid re-triggering dependency conflicts with torch.
echo "Installing omnilingual-asr>=0.2.0 from PyPI..."
./$VENV_NAME/bin/python -m pip install "omnilingual-asr>=0.2.0" --no-deps
# If you want to develop locally, uncomment the lines below and comment the line above:
# if [ -f "pyproject.toml" ]; then
#     echo "Installing omnilingual-asr package from local directory..."
#     ./$VENV_NAME/bin/python -m pip install -e .
# fi

# 7. Install other requirements (training related) if present
if [ -f "requirements.txt" ]; then
    echo "Installing remaining requirements from requirements.txt..."
    # Avoiding conflict if possible, but letting pip handle it
    ./$VENV_NAME/bin/python -m pip install -r requirements.txt
fi

# 8. Setup Local Model Card Configuration
echo "Setting up local model card configuration..."
FAIRSEQ2_CONFIG_DIR=~/.config/fairseq2/assets/cards/models
mkdir -p "$FAIRSEQ2_CONFIG_DIR"
# Symlink the repo's config file to the fairseq2 config dir
# We use absolute path via $(pwd)
if [ -f "src/omnilingual_asr/cards/models/omniasr_local.yaml" ]; then
    ln -sf "$(pwd)/src/omnilingual_asr/cards/models/omniasr_local.yaml" "$FAIRSEQ2_CONFIG_DIR/omniasr_local.yaml"
    echo "Linked omniasr_local.yaml to $FAIRSEQ2_CONFIG_DIR"
else
    echo "Warning: src/omnilingual_asr/cards/models/omniasr_local.yaml not found. Model cards might be missing."
fi

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo "Environment: $VENV_NAME"
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To run the dashboard:"
echo "  ./asr_venv/bin/python app.py"
echo "================================================"
