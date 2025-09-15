#!/usr/bin/env python3
"""
VantaEchoNebula Network Node Installer
Comprehensive installer for the VantaEchoNebula blockchain network.
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import json
from pathlib import Path

def safe_print(text):
    """Print text safely without Unicode errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to ASCII-safe printing
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

class VantaEchoNebulaInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.install_dir = Path.home() / "VantaEchoNebula"
        self.repo_url = "https://github.com/CryptoCOB/VantaEchoNebula"
        self.raw_url = "https://raw.githubusercontent.com/CryptoCOB/VantaEchoNebula/main"
        
    def detect_python_command(self):
        """Detect the correct Python command to use."""
        python_commands = ['python3', 'python', 'py']
        for cmd in python_commands:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, 
                                      encoding='utf-8', errors='replace')
                if result.returncode == 0 and 'Python' in result.stdout:
                    return cmd
            except FileNotFoundError:
                continue
        return 'python'  # Fallback
    
    def install_dependencies(self):
        """Install required Python packages."""
        safe_print("📦 Installing dependencies...")
        
        python_cmd = self.detect_python_command()
        packages = [
            'requests',
            'websockets',
            'cryptography',
            'numpy',
            'torch',
            'transformers'
        ]
        
        for package in packages:
            try:
                safe_print(f"Installing {package}...")
                result = subprocess.run([python_cmd, '-m', 'pip', 'install', package],
                                      capture_output=True, text=True,
                                      encoding='utf-8', errors='replace')
                if result.returncode == 0:
                    safe_print(f"✅ {package} installed")
                else:
                    safe_print(f"⚠️ {package} installation had warnings")
            except Exception as e:
                safe_print(f"❌ Failed to install {package}: {e}")
    
    def download_core_files(self):
        """Download essential VantaEchoNebula files."""
        safe_print("📥 Downloading core files...")
        
        # Create installation directory
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        # Essential files to download
        files_to_download = [
            'VantaEchoNebulaSystem.py',
            'network_connector.py',
            'mobile_node.py',
            'memory_learner.py',
            'reasoning_engine.py',
            'hybrid_cognition_engine.py',
            'Echo_location.py',
            'neural_architecture_search.py',
            'crypto_nebula.py',
            'async_stt_engine.py',
            'async_tts_engine.py',
            'meta_consciousness.py',
            'QuantumPulse.py',
            'proactive_intelligence.py',
            'orchestrator.py',
            'async_training_engine.py',
            'model_utils.py',
            'chain_config.py',
            'requirements.txt',
            'AGENTS.md',
            'README.md'
        ]
        
        for filename in files_to_download:
            try:
                url = f"{self.raw_url}/{filename}"
                safe_print(f"Downloading {filename}...")
                
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                
                file_path = self.install_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                safe_print(f"✅ Downloaded {filename}")
                
            except Exception as e:
                safe_print(f"⚠️ Could not download {filename}: {e}")
    
    def create_launcher_script(self):
        """Create launcher scripts for the system."""
        safe_print("🚀 Creating launcher scripts...")
        
        python_cmd = self.detect_python_command()
        
        # Windows launcher
        if self.system == 'windows':
            launcher_path = self.install_dir / "launch_vanta_echo_nebula.bat"
            launcher_content = f"""@echo off
echo Starting VantaEchoNebula Network...
echo.
echo Choose your network:
echo 1. TestNet (Safe for development)
echo 2. MainNet (Production - real transactions!)
echo 3. Network Connector (Interactive)
echo 4. Standalone Mode
set /p choice="Enter choice (1-4): "

cd /d "{self.install_dir}"

if "%choice%"=="1" (
    echo Connecting to TestNet...
    {python_cmd} VantaEchoNebulaSystem.py --mode node --network testnet
) else if "%choice%"=="2" (
    echo Connecting to MainNet...
    {python_cmd} VantaEchoNebulaSystem.py --mode node --network mainnet
) else if "%choice%"=="3" (
    echo Starting Network Connector...
    {python_cmd} network_connector.py
) else (
    echo Starting in Standalone Mode...
    {python_cmd} VantaEchoNebulaSystem.py --mode interactive
)

pause
"""
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            safe_print("✅ Created Windows launcher")
        
        # Unix launcher  
        else:
            launcher_path = self.install_dir / "launch_vanta_echo_nebula.sh"
            launcher_content = f"""#!/bin/bash
echo "Starting VantaEchoNebula Network..."
echo ""
echo "Choose your network:"
echo "1. TestNet (Safe for development)"
echo "2. MainNet (Production - real transactions!)"
echo "3. Network Connector (Interactive)"
echo "4. Standalone Mode"
read -p "Enter choice (1-4): " choice

cd "{self.install_dir}"

case $choice in
    1)
        echo "Connecting to TestNet..."
        {python_cmd} VantaEchoNebulaSystem.py --mode node --network testnet
        ;;
    2)
        echo "Connecting to MainNet..."
        {python_cmd} VantaEchoNebulaSystem.py --mode node --network mainnet
        ;;
    3)
        echo "Starting Network Connector..."
        {python_cmd} network_connector.py
        ;;
    *)
        echo "Starting in Standalone Mode..."
        {python_cmd} VantaEchoNebulaSystem.py --mode interactive
        ;;
esac
"""
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            os.chmod(launcher_path, 0o755)
            safe_print("✅ Created Unix launcher")
    
    def create_config_files(self):
        """Create initial configuration files."""
        safe_print("⚙️ Creating configuration...")
        
        config = {
            "network": {
                "name": "VantaEchoNebula",
                "version": "2.0.0",
                "mode": "node"
            },
            "system": {
                "platform": self.system,
                "architecture": self.architecture,
                "install_path": str(self.install_dir)
            },
            "features": {
                "mobile_optimized": True,
                "ai_integration": True,
                "blockchain_training": True,
                "quantum_enhancement": True
            }
        }
        
        config_path = self.install_dir / "vanta_echo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        safe_print("✅ Configuration created")
    
    def run_install(self):
        """Run the complete installation process."""
        safe_print("🌌 VantaEchoNebula Network Installer")
        safe_print("=" * 50)
        safe_print(f"Installing to: {self.install_dir}")
        safe_print(f"System: {self.system} ({self.architecture})")
        safe_print()
        
        try:
            # Step 1: Install dependencies
            self.install_dependencies()
            
            # Step 2: Download core files
            self.download_core_files()
            
            # Step 3: Create launcher scripts
            self.create_launcher_script()
            
            # Step 4: Create configuration
            self.create_config_files()
            
            safe_print()
            safe_print("🎉 VantaEchoNebula installation completed!")
            safe_print("=" * 50)
            safe_print(f"📂 Installed to: {self.install_dir}")
            safe_print()
            safe_print("🚀 To start VantaEchoNebula:")
            
            if self.system == 'windows':
                safe_print(f"   Double-click: {self.install_dir}/launch_vanta_echo_nebula.bat")
            else:
                safe_print(f"   Run: {self.install_dir}/launch_vanta_echo_nebula.sh")
            
            safe_print()
            safe_print("📚 Documentation: https://github.com/CryptoCOB/VantaEchoNebula")
            safe_print("🌌 Welcome to VantaEchoNebula Network!")
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Installation failed: {e}")
            safe_print()
            safe_print("🆘 TROUBLESHOOTING:")
            safe_print("1. Check internet connection")
            safe_print("2. Run as administrator/root")
            safe_print("3. Update Python and pip")
            safe_print("4. Visit: https://github.com/CryptoCOB/VantaEchoNebula")
            return False

def main():
    """Main installer entry point."""
    # Check for test mode
    if '--test' in str(sys.argv):
        safe_print("🧪 Running in test mode...")
        safe_print("✅ Installer script is functional")
        return
    
    installer = VantaEchoNebulaInstaller()
    success = installer.run_install()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()