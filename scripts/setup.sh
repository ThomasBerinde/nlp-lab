#!/usr/bin/env bash
set -e

echo "Running template setup..."

# 1. Run the Python init script (renames project, updates pyproject)
python scripts/init_template.py -d

# 2. Install dependencies now that project name is correct
poetry install --with dev

# 3. Install pre-commit now that it's available
poetry run pre-commit install

echo "Setup complete!"
