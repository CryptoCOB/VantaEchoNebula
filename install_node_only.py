#!/usr/bin/env python3
"""
VantaEchoNebula Node-Only Installer
Installs only what's needed to run a blockchain node (no heavy AI models)
"""

import urllib.request
import os
import sys

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def install_node_only():
    """Install minimal VantaEchoNebula node."""
    safe_print("🌌 VantaEchoNebula Node-Only Installer")
    safe_print("=" * 50)
    safe_print("📦 Installing minimal blockchain node...")
    safe_print("🚫 No AI models or heavy processing")
    safe_print("⚡ Just pure blockchain participation")
    
    try:
        # Download basic node runner
        safe_print("\n📥 Downloading basic node...")
        url = "https://raw.githubusercontent.com/CryptoCOB/VantaEchoNebula/main/basic_node.py"
        
        with urllib.request.urlopen(url) as response:
            node_code = response.read().decode('utf-8')
        
        # Save to file
        with open('basic_node.py', 'w', encoding='utf-8') as f:
            f.write(node_code)
        
        safe_print("✅ Basic node downloaded")
        
        # Download chain config
        safe_print("📥 Downloading blockchain configuration...")
        config_url = "https://raw.githubusercontent.com/CryptoCOB/VantaEchoNebula/main/chain_config.py"
        
        try:
            with urllib.request.urlopen(config_url) as response:
                config_code = response.read().decode('utf-8')
            
            with open('chain_config.py', 'w', encoding='utf-8') as f:
                f.write(config_code)
            
            safe_print("✅ Blockchain config downloaded")
        except:
            safe_print("⚠️ Config download failed - node will run in basic mode")
        
        safe_print("\n🎉 VantaEchoNebula Node Installation Complete!")
        safe_print("=" * 50)
        safe_print("🚀 To start your node:")
        safe_print("")
        safe_print("  TestNet (Safe):")
        safe_print("  python basic_node.py")
        safe_print("")
        safe_print("  MainNet (Production):")
        safe_print("  python basic_node.py --mainnet")
        safe_print("")
        safe_print("🌐 Your node will:")
        safe_print("  ✓ Sync with blockchain")
        safe_print("  ✓ Participate in mining")
        safe_print("  ✓ Process transactions")
        safe_print("  ✓ Earn VEN tokens")
        safe_print("")
        safe_print("💡 No AI models needed - just blockchain!")
        
    except urllib.error.URLError:
        safe_print("❌ Could not download node files.")
        safe_print("🔄 Please check your internet connection.")
        safe_print("🌐 Visit: https://github.com/CryptoCOB/VantaEchoNebula")
        
    except Exception as e:
        safe_print(f"❌ Installation error: {e}")
        safe_print("🆘 Try running as administrator/root")
        safe_print("🌐 Visit: https://github.com/CryptoCOB/VantaEchoNebula")

if __name__ == "__main__":
    install_node_only()