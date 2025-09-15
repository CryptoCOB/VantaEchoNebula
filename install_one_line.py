#!/usr/bin/env python3
"""
VantaEchoNebula Network - Simple One-Line Installer
The easiest way to get started with VantaEchoNebula.
"""

import urllib.request
import os
import sys
import platform

def install_vanta_echo_nebula():
    """Install VantaEchoNebula Network."""
    print("🌌 VantaEchoNebula Network - One-Line Installer")
    print("=" * 50)
    print("🚀 Starting installation...")
    
    try:
        # Download the main installer
        print("📥 Downloading VantaEchoNebula installer...")
        url = "https://raw.githubusercontent.com/CryptoCOB/VantaEchoNebula/main/vanta_echo_nebula_node_installer.py"
        
        with urllib.request.urlopen(url) as response:
            installer_code = response.read().decode('utf-8')
        
        print("✅ Installer downloaded successfully")
        print("🔧 Running installation...")
        
        # Execute the installer
        exec(installer_code)
        
    except urllib.error.URLError:
        print("❌ Could not download installer. Please check your internet connection.")
        print()
        print("🔄 Alternative installation methods:")
        print("1. Manual download: https://github.com/CryptoCOB/VantaEchoNebula/releases")
        print("2. Clone repository: git clone https://github.com/CryptoCOB/VantaEchoNebula.git")
        print("3. Visit: https://github.com/CryptoCOB/VantaEchoNebula for help")
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        print()
        print("🆘 Troubleshooting:")
        print("1. Ensure Python 3.7+ is installed")
        print("2. Check internet connection")
        print("3. Try running as administrator/root")
        print("4. Visit: https://github.com/CryptoCOB/VantaEchoNebula for support")

if __name__ == "__main__":
    install_vanta_echo_nebula()