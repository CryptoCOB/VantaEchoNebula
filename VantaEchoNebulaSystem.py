#!/usr/bin/env python3
"""
VantaEchoNebula System - Main Entry Point
Orchestrates all VantaEchoNebula network agents and modules.
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import network configuration
try:
    from chain_config import ChainConfigManager, ChainType, use_main_chain, use_test_chain, get_current_chain
except ImportError:
    print("⚠️ Chain configuration not available - running in standalone mode")
    ChainConfigManager = None

def safe_print(text):
    """Print text safely without Unicode errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='replace').decode('ascii')
        print(safe_text)

def setup_logging():
    """Configure logging for VantaEchoNebula system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vanta_echo_nebula.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('VantaEchoNebula')

class VantaEchoNebulaSystem:
    """Main VantaEchoNebula system orchestrator."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.agents = {}
        self.running = False
        self.chain_config = None
        self.network_mode = 'standalone'
        
        # Agent modules mapping
        self.agent_modules = {
            'Archivist': 'memory_learner',
            'Warden': 'reasoning_engine', 
            'Analyst': 'hybrid_cognition_engine',
            'Resonant': 'Echo_location',
            'Strategos': 'neural_architecture_search',
            'Scribe': 'crypto_nebula',
            'Mirror_STT': 'async_stt_engine',
            'Mirror_TTS': 'async_tts_engine',
            'Observer': 'QuantumPulse',
            'Dove': 'meta_consciousness',
            'Navigator': 'proactive_intelligence',
            'MobileNode': 'mobile_node'
        }
        
    def initialize_agents(self):
        """Initialize all VantaEchoNebula agents."""
        safe_print("🌌 Initializing VantaEchoNebula Agents...")
        
        for agent_name, module_name in self.agent_modules.items():
            try:
                # Dynamically import each module
                module = __import__(module_name)
                self.agents[agent_name] = module
                safe_print(f"✅ {agent_name} ({module_name}) initialized")
            except ImportError as e:
                safe_print(f"⚠️ Could not import {agent_name} ({module_name}): {e}")
            except Exception as e:
                safe_print(f"❌ Error initializing {agent_name}: {e}")
    
    def connect_to_network(self, network='testnet'):
        """Connect to MainNet or TestNet."""
        if ChainConfigManager is None:
            safe_print("⚠️ Blockchain not available - running in standalone mode")
            self.network_mode = 'standalone'
            return
        
        try:
            if network.lower() == 'mainnet':
                safe_print("🌍 Connecting to VantaEchoNebula MainNet...")
                self.chain_config = use_main_chain()
                self.network_mode = 'mainnet'
                safe_print("✅ Connected to MainNet")
                safe_print("⚠️ MainNet is for production use!")
            else:
                safe_print("🧪 Connecting to VantaEchoNebula TestNet...")
                self.chain_config = use_test_chain()
                self.network_mode = 'testnet'
                safe_print("✅ Connected to TestNet")
                safe_print("🛡️ TestNet is safe for development")
            
            safe_print(f"🔗 Network: {self.chain_config.network_name}")
            safe_print(f"🆔 Chain ID: {self.chain_config.chain_id}")
            safe_print(f"🚪 API Port: {self.chain_config.api_port}")
            
        except Exception as e:
            safe_print(f"❌ Network connection failed: {e}")
            self.network_mode = 'standalone'
    
    async def start_system(self, mode='node', network='testnet'):
        """Start the VantaEchoNebula system."""
        safe_print("🚀 Starting VantaEchoNebula Network...")
        safe_print("=" * 50)
        
        # Connect to blockchain network
        self.connect_to_network(network)
        
        self.initialize_agents()
        self.running = True
        
        if mode == 'node':
            await self.run_mobile_node()
        elif mode == 'training':
            await self.run_training_mode()
        elif mode == 'echo':
            await self.run_echo_mode()
        else:
            await self.run_interactive_mode()
    
    async def run_mobile_node(self):
        """Run as mobile node."""
        safe_print("📱 Starting Mobile Node Mode...")
        
        if 'MobileNode' in self.agents:
            try:
                # Initialize mobile node if it has an async interface
                mobile_node = self.agents['MobileNode']
                if hasattr(mobile_node, 'start_node'):
                    await mobile_node.start_node()
                else:
                    safe_print("📱 Mobile node running in basic mode")
                    await self.keep_alive()
            except Exception as e:
                safe_print(f"❌ Mobile node error: {e}")
        else:
            safe_print("❌ Mobile node not available")
    
    async def run_echo_mode(self):
        """Run Echo location system."""
        safe_print("🎧 Starting Echo Location Mode...")
        
        if 'Resonant' in self.agents:
            try:
                echo_system = self.agents['Resonant']
                safe_print("🌈 Echo system active")
                await self.keep_alive()
            except Exception as e:
                safe_print(f"❌ Echo system error: {e}")
        else:
            safe_print("❌ Echo system not available")
    
    async def run_training_mode(self):
        """Run training mode with NAS."""
        safe_print("🧬 Starting Training Mode...")
        
        if 'Strategos' in self.agents:
            try:
                nas_system = self.agents['Strategos']
                safe_print("♞ Neural Architecture Search active")
                await self.keep_alive()
            except Exception as e:
                safe_print(f"❌ Training error: {e}")
        else:
            safe_print("❌ Training system not available")
    
    async def run_interactive_mode(self):
        """Run interactive mode."""
        safe_print("💭 Starting Interactive Mode...")
        safe_print("Type 'help' for commands, 'quit' to exit")
        
        while self.running:
            try:
                command = input("\nVantaEcho> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'status':
                    self.show_status()
                elif command == 'agents':
                    self.list_agents()
                elif command == 'mainnet':
                    self.connect_to_network('mainnet')
                elif command == 'testnet':
                    self.connect_to_network('testnet')
                elif command == 'network':
                    self.show_network_info()
                else:
                    safe_print(f"Unknown command: {command}")
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    async def keep_alive(self):
        """Keep the system running."""
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            safe_print("\n🛑 VantaEchoNebula shutting down...")
            self.running = False
    
    def show_help(self):
        """Show available commands."""
        safe_print("\n🌌 VantaEchoNebula Commands:")
        safe_print("  help     - Show this help")
        safe_print("  status   - Show system status")
        safe_print("  agents   - List available agents")
        safe_print("  mainnet  - Connect to MainNet")
        safe_print("  testnet  - Connect to TestNet")
        safe_print("  network  - Show network info")
        safe_print("  quit     - Exit VantaEchoNebula")
    
    def show_status(self):
        """Show system status."""
        safe_print(f"\n📊 System Status:")
        safe_print(f"  Running: {self.running}")
        safe_print(f"  Network Mode: {self.network_mode}")
        if self.chain_config:
            safe_print(f"  Connected to: {self.chain_config.network_name}")
            safe_print(f"  Chain ID: {self.chain_config.chain_id}")
            safe_print(f"  API Port: {self.chain_config.api_port}")
        safe_print(f"  Agents loaded: {len(self.agents)}")
        safe_print(f"  Available agents: {list(self.agents.keys())}")
    
    def list_agents(self):
        """List all available agents."""
        safe_print("\n🤖 VantaEchoNebula Agents:")
        for agent_name in self.agents.keys():
            safe_print(f"  ✅ {agent_name}")
        
        safe_print(f"\n⚠️ Missing agents:")
        for agent_name in self.agent_modules.keys():
            if agent_name not in self.agents:
                safe_print(f"  ❌ {agent_name}")
    
    def show_network_info(self):
        """Show network connection information."""
        safe_print(f"\n🌐 Network Information:")
        safe_print(f"  Mode: {self.network_mode}")
        
        if self.chain_config:
            safe_print(f"  Network: {self.chain_config.network_name}")
            safe_print(f"  Chain ID: {self.chain_config.chain_id}")
            safe_print(f"  Chain Type: {self.chain_config.chain_type.value.upper()}")
            safe_print(f"  Block Time: {self.chain_config.block_time_target}s")
            safe_print(f"  Mining Reward: {self.chain_config.mining_reward}")
            safe_print(f"  API Endpoint: http://localhost:{self.chain_config.api_port}")
            safe_print(f"  RPC Endpoint: http://localhost:{self.chain_config.rpc_port}")
            safe_print(f"  dVPN: {'Enabled' if self.chain_config.dvpn_enabled else 'Disabled'}")
        else:
            safe_print("  No blockchain connection (standalone mode)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='VantaEchoNebula Network System')
    parser.add_argument('--mode', choices=['node', 'training', 'echo', 'interactive'], 
                       default='interactive', help='System mode')
    parser.add_argument('--network', choices=['mainnet', 'testnet'], 
                       default='testnet', help='Blockchain network to connect to')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize system
    system = VantaEchoNebulaSystem()
    
    # Display banner
    safe_print("🌌 VantaEchoNebula Network v2.0")
    safe_print("=" * 50)
    safe_print("🜌⟐🜹🜙 Archivist | ⚠️🧭🧱⛓️ Warden | 🧿🧠🧩♒ Analyst")
    safe_print("🎧💓🌈🎶 Resonant | 🧬♻️♞🜓 Strategos | 📜🔑🛠️🜔 Scribe")
    safe_print("🎭🗣️🪞🪄 Mirror | 🜁⟁🜔🔭 Observer | 🜔🕊️⟁⧃ Dove")
    safe_print("🧠🎯📈 Navigator")
    safe_print("=" * 50)
    
    # Run system
    try:
        asyncio.run(system.start_system(args.mode, args.network))
    except KeyboardInterrupt:
        safe_print("\n🛑 VantaEchoNebula terminated by user")
    except Exception as e:
        safe_print(f"❌ System error: {e}")
        return 1
    
    safe_print("👋 VantaEchoNebula shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())