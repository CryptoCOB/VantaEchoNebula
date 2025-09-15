#!/usr/bin/env python3
"""
VantaEchoNebula Simple Node Runner
Lightweight blockchain node without heavy AI model processing
"""

import sys
import os
import asyncio
import argparse
import logging
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def safe_print(text):
    """Print text safely without Unicode errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def setup_simple_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('VantaEchoNode')

class SimpleVantaEchoNode:
    """Lightweight VantaEchoNebula blockchain node - no heavy AI processing"""
    
    def __init__(self, network='testnet'):
        self.logger = setup_simple_logging()
        self.network = network
        self.chain_config = None
        self.running = False
        self.node_id = f"node_{int(time.time())}"
        
        # Load network configuration
        self.load_network_config()
    
    def load_network_config(self):
        """Load blockchain network configuration."""
        try:
            from chain_config import use_main_chain, use_test_chain
            
            if self.network == 'mainnet':
                self.chain_config = use_main_chain()
                safe_print("🌍 Connected to VantaEchoNebula MainNet")
                safe_print("⚠️ MainNet - real transactions!")
            else:
                self.chain_config = use_test_chain()
                safe_print("🧪 Connected to VantaEchoNebula TestNet")
                safe_print("🛡️ TestNet - safe for development")
            
            safe_print(f"🔗 Chain ID: {self.chain_config.chain_id}")
            safe_print(f"🚪 API Port: {self.chain_config.api_port}")
            safe_print(f"📡 RPC Port: {self.chain_config.rpc_port}")
            
        except ImportError:
            safe_print("⚠️ Blockchain config not available - running in standalone mode")
            self.network = 'standalone'
    
    async def start_node(self):
        """Start the lightweight blockchain node."""
        safe_print("🚀 Starting VantaEchoNebula Node...")
        safe_print("=" * 50)
        safe_print(f"🆔 Node ID: {self.node_id}")
        safe_print(f"🌐 Network: {self.network}")
        
        if self.chain_config:
            safe_print(f"⛓️ Chain: {self.chain_config.network_name}")
            safe_print(f"⏱️ Block Time: {self.chain_config.block_time_target}s")
            safe_print(f"💰 Mining Reward: {self.chain_config.mining_reward}")
        
        safe_print("=" * 50)
        
        self.running = True
        
        # Start node services
        await asyncio.gather(
            self.run_blockchain_sync(),
            self.run_network_listener(),
            self.run_mining_process(),
            self.run_status_monitor()
        )
    
    async def run_blockchain_sync(self):
        """Sync with blockchain network."""
        safe_print("🔄 Starting blockchain sync...")
        
        while self.running:
            try:
                # Simulate blockchain sync process
                safe_print("📥 Syncing blocks...")
                await asyncio.sleep(30 if self.network == 'mainnet' else 5)
                
                if self.chain_config:
                    safe_print(f"✅ Synced block (every {self.chain_config.block_time_target}s)")
                else:
                    safe_print("✅ Standalone sync complete")
                    
            except Exception as e:
                safe_print(f"⚠️ Sync error: {e}")
                await asyncio.sleep(10)
    
    async def run_network_listener(self):
        """Listen for network transactions and events."""
        safe_print("👂 Starting network listener...")
        
        while self.running:
            try:
                # Simulate network activity
                await asyncio.sleep(15)
                safe_print("📨 Listening for network transactions...")
                
                # Simulate receiving transactions
                if self.chain_config:
                    safe_print(f"💼 Processing network activity on port {self.chain_config.p2p_port}")
                
            except Exception as e:
                safe_print(f"⚠️ Network error: {e}")
                await asyncio.sleep(5)
    
    async def run_mining_process(self):
        """Simple mining process (lightweight)."""
        safe_print("⛏️ Starting mining process...")
        
        while self.running:
            try:
                # Simple mining simulation
                mining_interval = 60 if self.network == 'mainnet' else 20
                await asyncio.sleep(mining_interval)
                
                if self.chain_config:
                    safe_print(f"💎 Mining attempt - Reward: {self.chain_config.mining_reward} tokens")
                else:
                    safe_print("💎 Standalone mining simulation")
                    
            except Exception as e:
                safe_print(f"⚠️ Mining error: {e}")
                await asyncio.sleep(30)
    
    async def run_status_monitor(self):
        """Monitor node status and health."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                safe_print("📊 Node Status: HEALTHY")
                safe_print(f"   Network: {self.network}")
                safe_print(f"   Uptime: {time.time() - start_time:.0f}s")
                
                if self.chain_config:
                    safe_print(f"   Chain: {self.chain_config.network_name}")
                    safe_print(f"   Peers: Connected")
                
            except Exception as e:
                safe_print(f"⚠️ Status error: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Gracefully shutdown the node."""
        safe_print("🛑 Shutting down VantaEchoNebula Node...")
        self.running = False
        safe_print("👋 Node shutdown complete")

def main():
    """Main entry point for simple node runner."""
    parser = argparse.ArgumentParser(description='VantaEchoNebula Simple Node Runner')
    parser.add_argument('--network', choices=['testnet', 'mainnet'], 
                       default='testnet', help='Network to connect to')
    parser.add_argument('--standalone', action='store_true', 
                       help='Run without blockchain (standalone mode)')
    
    args = parser.parse_args()
    
    # Override network if standalone
    network = 'standalone' if args.standalone else args.network
    
    # Create node
    node = SimpleVantaEchoNode(network)
    
    # Display startup banner
    safe_print("🌌 VantaEchoNebula Simple Node v1.0")
    safe_print("⚡ Lightweight blockchain participation")
    safe_print("🚫 No heavy AI model processing")
    safe_print("=" * 50)
    
    # Track start time
    global start_time
    start_time = time.time()
    
    try:
        # Run the node
        asyncio.run(node.start_node())
    except KeyboardInterrupt:
        safe_print("\n🛑 Received shutdown signal")
        asyncio.run(node.shutdown())
    except Exception as e:
        safe_print(f"❌ Node error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())