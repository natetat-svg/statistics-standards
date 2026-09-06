# test_venv.py
import sys
import os

print("=== Virtual Environment Check ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set!')}")
print(f"sys.prefix: {sys.prefix}")

# Check if in virtual environment
in_venv = hasattr(sys, 'real_prefix') or (
    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
)
print(f"In virtual environment: {in_venv}")

# Check site-packages
import site
print(f"Site packages: {site.getsitepackages()}")
