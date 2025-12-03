#!/bin/bash
# Run Jupyter Lab with the isletscope mamba environment

echo "Starting Jupyter Lab with isletscope environment..."
echo "=========================================="

# Activate environment and start Jupyter
/home/smith6jt/miniconda3/condabin/mamba run -n isletscope jupyter lab --notebook-dir=/home/smith6jt/isletscope/notebooks
