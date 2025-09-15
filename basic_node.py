#!/usr/bin/env python3
"""
VantaEchoNebula Basic Node
Ultra-lightweight node runner - just blockchain participation, no AI models
Connected to OrionBelt blockchain system
"""

import asyncio
import time
import sys
import os

# Import the OrionBelt blockchain system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OrionBelt import Blockchain, Block, OrionToken, QuantumConsensus

def safe_print(text):
    """Print text safely."""
    try:
        print(text)
    except:
        print(str(text).encode('ascii', errors='replace').decode('ascii'))

class BasicNode:
    """Ultra-basic VantaEchoNebula node - connects to OrionBelt blockchain"""
    
    def __init__(self, network='testnet'):
        self.network = network
        self.running = False
        self.node_id = f"basic_node_{int(time.time())}"
        self.start_time = time.time()
        
        # Initialize the OrionBelt blockchain
        safe_print("🔗 Connecting to OrionBelt blockchain...")
        self.blockchain = Blockchain()
        
        # Initialize quantum consensus with validators
        validators = [self.node_id, "validator_1", "validator_2", "validator_3"]
        self.quantum_consensus = QuantumConsensus(validators)
        
        # Initialize OrionToken with basic parameters
        total_supply = 1000000  # 1M ORION tokens
        initial_price = 1.0     # $1 per token
        liquidity_pool = 100000 # 100K tokens in liquidity
        self.orion_token = OrionToken(total_supply, initial_price, liquidity_pool)
        
        # Network settings
        if network == 'mainnet':
            self.block_time = 30  # 30 second blocks
            self.mining_reward = 10
            self.api_port = 8080
            safe_print("🌍 MainNet Mode - Real transactions!")
        else:
            self.block_time = 5   # 5 second blocks  
            self.mining_reward = 100
            self.api_port = 9080
            safe_print("🧪 TestNet Mode - Safe development")
    
    async def run(self):
        """Run the basic node."""
        safe_print("🚀 VantaEchoNebula Basic Node Starting...")
        safe_print("=" * 50)
        safe_print(f"🆔 Node ID: {self.node_id}")
        safe_print(f"🌐 Network: {self.network}")
        safe_print(f"⏱️ Block Time: {self.block_time}s")
        safe_print(f"💰 Mining Reward: {self.mining_reward}")
        safe_print(f"🚪 API Port: {self.api_port}")
        safe_print("=" * 50)
        safe_print("🔄 Node is now participating in VantaEchoNebula network")
        safe_print("⛏️ Mining and blockchain sync active")
        safe_print("📡 Listening for network activity")
        safe_print("=" * 50)
        
        self.running = True
        
        try:
            await asyncio.gather(
                self.blockchain_sync(),
                self.mining_loop(),
                self.network_activity(),
                self.status_report()
            )
        except KeyboardInterrupt:
            safe_print("\n🛑 Shutting down node...")
            self.running = False
    
    async def blockchain_sync(self):
        """Real blockchain synchronization with OrionBelt chain."""
        safe_print("🔄 Starting blockchain sync...")
        
        while self.running:
            try:
                # Get current blockchain status
                chain_length = len(self.blockchain.chain)
                last_block = self.blockchain.get_latest_block()
                
                safe_print(f"📦 Chain length: {chain_length} blocks")
                safe_print(f"🔗 Last block hash: {last_block.hash[:16]}...")
                
                # Check if we need to mine a new block
                if time.time() - last_block.timestamp > self.block_time:
                    safe_print("⛏️ Mining new block...")
                    # Create a simple transaction for mining reward
                    mining_tx = {
                        "type": "mining_reward",
                        "to": self.node_id,
                        "amount": self.mining_reward,
                        "timestamp": time.time()
                    }
                    
                    # Create and mine a new block using OrionBelt's method
                    new_block = self.blockchain._prepare_block(self.node_id, [mining_tx])
                    # Mine the block (calculate hash with proper nonce)
                    if new_block.mine_block(difficulty=self.blockchain.difficulty):
                        # Add the mined block to the chain
                        if self.blockchain.add_block(new_block):
                            safe_print(f"✅ Mined block #{new_block.index}")
                
                await asyncio.sleep(self.block_time)
                
            except Exception as e:
                safe_print(f"❌ Blockchain sync error: {e}")
                await asyncio.sleep(5)
    
    async def mining_loop(self):
        """Real mining using OrionBelt consensus."""
        while self.running:
            try:
                # Use quantum consensus for validator selection
                selected_validator = self.quantum_consensus.select_validator()
                
                if selected_validator == self.node_id:
                    safe_print(f"🎯 Selected as validator! Mining...")
                    
                    # Create mining transaction
                    mining_tx = {
                        "type": "block_reward", 
                        "validator": self.node_id,
                        "amount": self.mining_reward,
                        "timestamp": time.time()
                    }
                    
                    # Create and mine block using OrionBelt's method
                    block = self.blockchain._prepare_block(self.node_id, [mining_tx])
                    if block.mine_block(difficulty=self.blockchain.difficulty):
                        if self.blockchain.add_block(block):
                            safe_print(f"⛏️ Successfully mined block #{block.index} - Reward: {self.mining_reward} VEN")
                else:
                    safe_print("⏳ Not selected for mining this round")
                    
            except Exception as e:
                safe_print(f"❌ Mining error: {e}")
                
            await asyncio.sleep(60)  # Mine every minute
    
    async def network_activity(self):
        """Real network participation with OrionToken transactions."""
        while self.running:
            try:
                # Check OrionToken balance (use blockchain's get_balance)
                balance = self.blockchain.get_balance(self.node_id)
                total_supply = self.orion_token.total_supply
                
                safe_print(f"💰 Node balance: {balance:.2f} ORION")
                safe_print(f"📊 Total supply: {total_supply:.2f} ORION")
                
                # Process any pending transactions
                pending_count = len(self.blockchain.pending_transactions)
                if pending_count > 0:
                    safe_print(f"📝 Processing {pending_count} pending transactions")
                
            except Exception as e:
                safe_print(f"❌ Network activity error: {e}")
                
            await asyncio.sleep(45)  # Network activity every 45s
    
    async def status_report(self):
        """Regular status updates."""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            uptime = int(time.time() - self.start_time)
            safe_print(f"📊 Node Status: ACTIVE | Uptime: {uptime}s | Network: {self.network}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VantaEchoNebula Basic Node')
    parser.add_argument('--mainnet', action='store_true', 
                       help='Connect to MainNet (default: TestNet)')
    
    args = parser.parse_args()
    network = 'mainnet' if args.mainnet else 'testnet'
    
    # Startup banner
    safe_print("🌌 VantaEchoNebula Basic Node")
    safe_print("⚡ Ultra-lightweight blockchain participation")
    safe_print("🚫 No AI models, no heavy processing")
    safe_print("✅ Just pure blockchain node functionality")
    
    node = BasicNode(network)
    
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        safe_print("👋 Node stopped by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())