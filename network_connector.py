#!/usr/bin/env python3
"""
VantaEchoNebula Network Connector
Easy connection to MainNet or TestNet blockchain networks
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from chain_config import ChainConfigManager, ChainType, use_main_chain, use_test_chain, get_current_chain
    from mobile_node import MobileNode
    from crypto_nebula import main as crypto_main
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("📥 Please ensure all VantaEchoNebula modules are installed")
    sys.exit(1)

def safe_print(text):
    """Print text safely without Unicode errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

class VantaEchoNebulaConnector:
    """Manages connections to VantaEchoNebula MainNet and TestNet"""
    
    def __init__(self):
        self.current_config = None
        self.mobile_node = None
        
    def display_network_info(self, config):
        """Display network configuration information"""
        safe_print(f"\n🌌 VantaEchoNebula Network Information")
        safe_print("=" * 50)
        safe_print(f"Network Name: {config.network_name}")
        safe_print(f"Chain ID: {config.chain_id}")
        safe_print(f"Chain Type: {config.chain_type.value.upper()}")
        safe_print(f"Block Time: {config.block_time_target}s")
        safe_print(f"Mining Reward: {config.mining_reward}")
        safe_print(f"Difficulty: {config.difficulty}")
        safe_print(f"Validators: {len(config.validators)}")
        safe_print("=" * 50)
        safe_print(f"🔗 Network Endpoints:")
        safe_print(f"  P2P Port: {config.p2p_port}")
        safe_print(f"  RPC Port: {config.rpc_port}")
        safe_print(f"  API Port: {config.api_port}")
        safe_print(f"  Max Peers: {config.max_peers}")
        safe_print("=" * 50)
        safe_print(f"💾 Storage:")
        safe_print(f"  Data Directory: {config.data_dir}")
        safe_print(f"  Blockchain File: {config.blockchain_file}")
        safe_print(f"  Task Cache: {config.task_cache_dir}")
        safe_print("=" * 50)
        if config.dvpn_enabled:
            safe_print(f"🔒 dVPN: ENABLED")
            safe_print(f"  Bandwidth Proof Interval: {config.bandwidth_proof_interval}s")
            safe_print(f"  Tunnel Timeout: {config.tunnel_timeout}s")
        else:
            safe_print(f"🔒 dVPN: DISABLED")
        safe_print("=" * 50)
    
    def connect_to_mainnet(self):
        """Connect to VantaEchoNebula MainNet"""
        safe_print("🔗 Connecting to VantaEchoNebula MainNet...")
        config = use_main_chain()
        self.current_config = config
        self.display_network_info(config)
        
        safe_print("✅ Connected to MainNet")
        safe_print("⚠️ MainNet is for production use - transactions are real!")
        return config
    
    def connect_to_testnet(self):
        """Connect to VantaEchoNebula TestNet"""
        safe_print("🧪 Connecting to VantaEchoNebula TestNet...")
        config = use_test_chain()
        self.current_config = config
        self.display_network_info(config)
        
        safe_print("✅ Connected to TestNet")
        safe_print("🛡️ TestNet is safe for development and testing")
        return config
    
    async def start_mobile_node(self):
        """Start mobile node on current network"""
        if not self.current_config:
            safe_print("❌ No network connection. Please connect to MainNet or TestNet first.")
            return False
        
        try:
            safe_print(f"📱 Starting Mobile Node on {self.current_config.network_name}...")
            
            # Initialize mobile node with current config
            self.mobile_node = MobileNode(self.current_config)
            await self.mobile_node.start()
            
            safe_print("✅ Mobile Node started successfully")
            return True
            
        except Exception as e:
            safe_print(f"❌ Failed to start Mobile Node: {e}")
            return False
    
    async def start_crypto_training(self):
        """Start crypto training on current network"""
        if not self.current_config:
            safe_print("❌ No network connection. Please connect to MainNet or TestNet first.")
            return False
        
        try:
            safe_print(f"🧠 Starting Crypto Training on {self.current_config.network_name}...")
            
            # Set environment variables for the current chain
            os.environ['NEBULA_CHAIN'] = self.current_config.chain_type.value
            
            # Start crypto training
            await crypto_main()
            
            safe_print("✅ Crypto Training started successfully")
            return True
            
        except Exception as e:
            safe_print(f"❌ Failed to start Crypto Training: {e}")
            return False
    
    def show_connection_status(self):
        """Show current connection status"""
        if self.current_config:
            safe_print(f"\n📊 Connection Status:")
            safe_print(f"  Connected to: {self.current_config.network_name}")
            safe_print(f"  Chain Type: {self.current_config.chain_type.value.upper()}")
            safe_print(f"  API Endpoint: http://localhost:{self.current_config.api_port}")
            safe_print(f"  RPC Endpoint: http://localhost:{self.current_config.rpc_port}")
        else:
            safe_print("\n📊 Connection Status: Not connected to any network")
    
    async def interactive_mode(self):
        """Run interactive network selection mode"""
        safe_print("🌌 VantaEchoNebula Network Connector")
        safe_print("=" * 50)
        safe_print("Choose your network:")
        safe_print("1. 🌍 MainNet (Production)")
        safe_print("2. 🧪 TestNet (Development/Testing)")
        safe_print("3. 📊 Show current status")
        safe_print("4. 🚪 Exit")
        
        while True:
            try:
                choice = input("\nVantaEcho> ").strip()
                
                if choice == '1':
                    self.connect_to_mainnet()
                    await self.ask_start_services()
                elif choice == '2':
                    self.connect_to_testnet()
                    await self.ask_start_services()
                elif choice == '3':
                    self.show_connection_status()
                elif choice == '4':
                    safe_print("👋 Goodbye!")
                    break
                else:
                    safe_print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                safe_print("\n👋 Goodbye!")
                break
            except EOFError:
                break
    
    async def ask_start_services(self):
        """Ask user which services to start"""
        safe_print("\n🚀 Available services:")
        safe_print("1. 📱 Start Mobile Node")
        safe_print("2. 🧠 Start Crypto Training")
        safe_print("3. ⚡ Start Both")
        safe_print("4. 🏠 Back to main menu")
        
        try:
            choice = input("Start services> ").strip()
            
            if choice == '1':
                await self.start_mobile_node()
            elif choice == '2':
                await self.start_crypto_training()
            elif choice == '3':
                await self.start_mobile_node()
                await self.start_crypto_training()
            elif choice == '4':
                return
            else:
                safe_print("❌ Invalid choice.")
                
        except (KeyboardInterrupt, EOFError):
            return

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VantaEchoNebula Network Connector')
    parser.add_argument('--network', choices=['mainnet', 'testnet'], 
                       help='Connect to specific network')
    parser.add_argument('--start-node', action='store_true', 
                       help='Start mobile node after connection')
    parser.add_argument('--start-training', action='store_true', 
                       help='Start crypto training after connection')
    parser.add_argument('--status', action='store_true', 
                       help='Show current network status')
    
    args = parser.parse_args()
    
    connector = VantaEchoNebulaConnector()
    
    # Handle command line arguments
    if args.status:
        current = get_current_chain()
        connector.current_config = current
        connector.show_connection_status()
        return
    
    async def run_connector():
        if args.network == 'mainnet':
            connector.connect_to_mainnet()
        elif args.network == 'testnet':
            connector.connect_to_testnet()
        
        if args.start_node and connector.current_config:
            await connector.start_mobile_node()
        
        if args.start_training and connector.current_config:
            await connector.start_crypto_training()
        
        if not args.network:
            await connector.interactive_mode()
    
    # Display banner
    safe_print("🌌 VantaEchoNebula Network Connector v2.0")
    safe_print("🔗 Easy connection to MainNet and TestNet")
    safe_print("=" * 50)
    
    try:
        asyncio.run(run_connector())
    except KeyboardInterrupt:
        safe_print("\n🛑 VantaEchoNebula Connector terminated")
    except Exception as e:
        safe_print(f"❌ Connector error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())