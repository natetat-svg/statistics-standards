import sys
import os
import subprocess
import site

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_python_path():
    print_header("Python Path Check")
    print(f"Python executable: {sys.executable}")
    
    # Check if path contains 'venv'
    if 'venv' in sys.executable or 'virtualenv' in sys.executable:
        print("✅ Python is running from virtual environment")
        return True
    else:
        print("❌ Python is NOT running from virtual environment")
        print(f"   Expected: /path/to/project/venv/bin/python")
        print(f"   Got: {sys.executable}")
        return False

def check_venv_activation():
    print_header("Activation Check")
    
    # Method 1: Check VIRTUAL_ENV environment variable
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print(f"✅ VIRTUAL_ENV is set: {venv_path}")
    else:
        print("❌ VIRTUAL_ENV is not set (environment not activated)")
        print("   Run: source venv/bin/activate  (macOS/Linux)")
        print("   Run: venv\\Scripts\\activate     (Windows)")
        return False
    
    # Method 2: Check sys.prefix
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"✅ sys.prefix shows virtual environment: {sys.prefix}")
    else:
        print("❌ sys.prefix does NOT indicate virtual environment")
    
    return in_venv

def check_pip_location():
    print_header("Pip Location Check")
    try:
        result = subprocess.run(['which', 'pip'], 
                              capture_output=True, text=True)
        pip_path = result.stdout.strip()
        
        if 'venv' in pip_path:
            print(f"✅ Pip is from venv: {pip_path}")
            return True
        else:
            print(f"❌ Pip is NOT from venv: {pip_path}")
            print("   Try: pip install --upgrade pip")
            return False
    except:
        print("❌ Could not find 'which' command")
        return False

def check_site_packages():
    print_header("Site Packages Check")
    site_packages = site.getsitepackages()
    
    for path in site_packages:
        if 'venv' in path:
            print(f"✅ Site-packages includes venv: {path}")
        else:
            print(f"   System site-packages: {path}")
    
    # Check if any packages are installed
    try:
        result = subprocess.run(['pip', 'list', '--format=freeze'], 
                              capture_output=True, text=True)
        packages = [line for line in result.stdout.split('\n') if line]
        print(f"\nInstalled packages ({len(packages)}):")
        for pkg in packages[:5]:  # Show first 5
            print(f"  - {pkg}")
        if len(packages) > 5:
            print(f"  ... and {len(packages)-5} more")
    except:
        print("❌ Could not list packages")

def test_import():
    print_header("Import Test (try importing common packages)")
    test_packages = ['os', 'sys', 'json']  # Should always work
    
    for pkg in test_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} imported successfully")
        except ImportError:
            print(f"❌ Could not import {pkg}")
    
    # Try to import scipy (if installed)
    try:
        import scipy
        print(f"✅ scipy imported from: {scipy.__file__}")
    except ImportError:
        print("ℹ️ scipy not installed (this is OK, run 'pip install scipy')")

def main():
    print_header("VIRTUAL ENVIRONMENT VERIFICATION")
    print(f"Current directory: {os.getcwd()}")
    
    checks = [
        check_python_path(),
        check_venv_activation(),
        check_pip_location(),
    ]
    
    check_site_packages()
    test_import()
    
    print_header("VERIFICATION SUMMARY")
    if all(checks):
        print("✅✅✅ Your virtual environment is properly set up! 🎉")
        print("\nYou can now install packages:")
        print("  pip install -r requirements.txt")
    else:
        print("❌❌❌ Issues detected with your virtual environment")
        print("\nTry these steps:")
        print("1. Deactivate: deactivate")
        print("2. Delete venv: rm -rf venv")
        print("3. Recreate: python -m venv venv")
        print("4. Activate: source venv/bin/activate")
        print("5. Run this script again")

if __name__ == "__main__":
    main()
