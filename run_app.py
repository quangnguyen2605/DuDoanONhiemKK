#!/usr/bin/env python3
"""
Launch script for Hanoi Air Pollution Prediction System
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('streamlit', 'streamlit'), 
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('plotly', 'plotly'), 
        ('joblib', 'joblib')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main launch function"""
    print("🌫️ Hanoi Air Pollution Prediction System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ Error: main.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit application...")
    print("📱 The application will open in your web browser at: http://localhost:8501")
    print("🔄 Use Ctrl+C to stop the application")
    print("\n" + "=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'main.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
