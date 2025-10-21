"""
Environment Setup Script for ResNet and Big Data Computer Vision Simulations
This script verifies that all required dependencies are installed and working correctly.
"""

import sys
import subprocess
import importlib
import warnings

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8 or higher")
        return False

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally check version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Check version if specified
        if min_version and hasattr(module, '__version__'):
            installed_version = module.__version__
            print(f"✅ {package_name}: {installed_version}")
            
            # Simple version comparison (works for most cases)
            if installed_version.split('.')[0] >= min_version.split('.')[0]:
                return True
            else:
                print(f"⚠️  {package_name}: Version {installed_version} found, {min_version} recommended")
                return True
        else:
            print(f"✅ {package_name}: Installed")
            return True
            
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_gpu_availability():
    """Check if GPU is available for TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU Available: {len(gpus)} device(s) found")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("💻 No GPU found, will use CPU")
            return False
    except:
        print("❓ Could not check GPU availability")
        return False

def test_tensorflow():
    """Test basic TensorFlow functionality"""
    print("\\n🧪 Testing TensorFlow...")
    try:
        import tensorflow as tf
        
        # Create a simple computation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        
        print(f"✅ TensorFlow test successful")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   Test computation result shape: {c.shape}")
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_keras():
    """Test Keras functionality"""
    print("\\n🧪 Testing Keras...")
    try:
        from tensorflow import keras
        
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        print("✅ Keras test successful")
        print(f"   Model created with {model.count_params()} parameters")
        return True
    except Exception as e:
        print(f"❌ Keras test failed: {e}")
        return False

def test_visualization():
    """Test visualization libraries"""
    print("\\n🧪 Testing Visualization Libraries...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create a simple plot (don't show it)
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        plt.close(fig)
        
        print("✅ Visualization libraries test successful")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def create_test_directories():
    """Create necessary directories for the project"""
    print("\\n📁 Creating project directories...")
    import os
    
    directories = [
        "results",
        "data", 
        "results/simulation4",
        "results/simulation5"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("ResNet and Big Data Computer Vision - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Required packages with their import names and minimum versions
    required_packages = [
        ("tensorflow", "tensorflow", "2.10.0"),
        ("numpy", "numpy", "1.21.0"),
        ("matplotlib", "matplotlib", "3.5.0"),
        ("seaborn", "seaborn", "0.11.0"),
        ("scikit-learn", "sklearn", "1.1.0"),
        ("Pillow", "PIL", "9.0.0"),
        ("opencv-python", "cv2", None),
        ("pandas", "pandas", "1.4.0")
    ]
    
    print("\\n📦 Checking required packages...")
    missing_packages = []
    
    for package_name, import_name, min_version in required_packages:
        if not check_package(package_name, import_name, min_version):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\\n📥 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    
    # Test GPU availability
    print("\\n🔍 Checking hardware...")
    check_gpu_availability()
    
    # Run functionality tests
    tests_passed = 0
    total_tests = 3
    
    if test_tensorflow():
        tests_passed += 1
    
    if test_keras():
        tests_passed += 1
        
    if test_visualization():
        tests_passed += 1
    
    # Create directories
    create_test_directories()
    
    # Summary
    print("\\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("🎉 Environment setup completed successfully!")
        print("✅ All dependencies are working correctly")
        print("✅ Project directories created")
        print("\\n🚀 You're ready to run the simulations!")
        print("\\nNext steps:")
        print("1. Run individual simulations:")
        print("   python simulations/simulation4_residual_blocks.py")
        print("   python simulations/simulation5_transfer_learning.py")
        print("\\n2. Or run all simulations:")
        print("   python run_all_simulations.py")
        return True
    else:
        print(f"⚠️  Setup completed with warnings")
        print(f"📊 Tests passed: {tests_passed}/{total_tests}")
        print("\\nSome functionality may not work correctly.")
        print("Please check the error messages above and install missing packages.")
        return False

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings during setup
        success = main()
        sys.exit(0 if success else 1)