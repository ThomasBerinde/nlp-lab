Write-Host "Running template setup..."
# 1. Run the Python init script
python scripts/init_template.py -d

# 2. Install dependencies
poetry install --with dev

# 3. Install pre-commit hooks
poetry run pre-commit install

Write-Host "Setup complete!"