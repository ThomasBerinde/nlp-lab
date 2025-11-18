# 🧩 My Template Project

A clean and modern Python project template with **Poetry**, **Ruff**, **Mypy**, **Pytest**, and **Pre-commit** configured out of the box.

---

## 🚀 Features

- 🧱 Standardized project layout (`src/` + `tests/`)
- 🔄 Automatic first-run project initialization  
  (renames `project_name/` → `<your-folder-name>/`, updates `pyproject.toml`, installs the project and pre-commit hooks)
- 🧹 Auto linting & formatting with [Ruff](https://github.com/astral-sh/ruff)
- 🔍 Static type checking via [Mypy](https://mypy.readthedocs.io/)
- 🧪 Testing setup with [Pytest](https://pytest.org/)
- 🪝 Git hooks via [Pre-commit](https://pre-commit.com/)
- ⚙️ Dependency management using [Poetry](https://python-poetry.org/)

---

## 📦 Project Structure

```text
.
├── scripts/
│   ├── init_template.py
│   ├── setup.ps1
│   └── setup.sh
├── src/
│   └── project_name/
│       └── __init__.py
├── tests/
│   └── test_sample.py
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── pyproject.toml
```

## 🧰 Setup Instructions

### 1️⃣ Initialize and Install Dependencies

Run the setup script for your platform:

**On Linux/macOS or Windows (Git Bash):**

```bash
./scripts/setup.sh
```

**On Windows (PowerShell):**

```powershell
./scripts/setup.ps1
```
