#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

# Options
parser = argparse.ArgumentParser(description="Initialize the template project.")
parser.add_argument(
    "-c", "--current-name",
    default="project_name",
    help="The current name of the project. (default: project_name)"
)
parser.add_argument(
    "-d", "--delete",
    action="store_true",
    help="Delete this script when its execution finishes."
)
args = parser.parse_args()

# Paths
script_path = Path(__file__).resolve()
root = script_path.parent.parent       # 👈 one level up from scripts/
curr_name = args.current_name
new_name = re.sub("-", "_", root.name)

pyproject = root / "pyproject.toml"
src_old = root / "src" / curr_name
src_new = root / "src" / new_name

print(f"🔧 Setting up project: {new_name}")

# 1️. Replace project name in pyproject.toml
if pyproject.exists():
    text = pyproject.read_text(encoding="utf-8")
    if curr_name in text:
        pyproject.write_text(text.replace(curr_name, new_name), encoding="utf-8")
        print("✅ Updated pyproject.toml")

# 2️. Rename src/<old_name> → src/<new_name>
if src_old.exists():
    src_old.rename(src_new)
    print(f"📁 Renamed folder: src/{curr_name} → src/{new_name}")

# 3. Optionally delete this script
if args.delete:
    try:
        script_path.unlink()
        print(f"🧹 Removed setup script: {script_path.name}")
    except Exception as e:
        print(f"⚠️ Could not delete {script_path.name}: {e}")

print(f"\n✨ Done! Project initialized as '{new_name}'.")
