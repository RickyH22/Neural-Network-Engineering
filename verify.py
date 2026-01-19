"""
Quick verification script to check if all components are ready.
"""

def verify_files():
    """Check if all required files exist."""
    import os
    
    required_files = [
        'layers.py',
        'train.py',
        'metrics.md',
        'repro.md',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    print("=" * 60)
    print("Project File Verification")
    print("=" * 60)
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    print("=" * 60)
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} file(s)")
        return False
    else:
        print("\n✓ All required files present!")
        return True


def check_imports():
    """Check if required packages can be imported."""
    print("\n" + "=" * 60)
    print("Package Import Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"✗ torch - NOT INSTALLED")
        print(f"  Error: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ torchvision - NOT INSTALLED")
        print(f"  Error: {e}")
        return False
    
    print("=" * 60)
    print("\n✓ All packages installed correctly!")
    return True


if __name__ == "__main__":
    files_ok = verify_files()
    
    if files_ok:
        try:
            imports_ok = check_imports()
            
            if imports_ok:
                print("\n" + "=" * 60)
                print("🎉 Project is ready!")
                print("=" * 60)
                print("\nNext steps:")
                print("1. Run: python layers.py")
                print("   (Tests the custom LearnedAffine layer)")
                print("\n2. Run: python train.py")
                print("   (Trains the model on FashionMNIST)")
                print("\n3. Check metrics.md and repro.md for documentation")
                print("=" * 60)
        except Exception as e:
            print(f"\nError during import check: {e}")
            print("You may need to run: pip install -r requirements.txt")
