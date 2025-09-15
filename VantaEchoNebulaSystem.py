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
    
    async def start_system(self, mode='node'):
        """Start the VantaEchoNebula system."""
        safe_print("🚀 Starting VantaEchoNebula Network...")
        safe_print("=" * 50)
        
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
        safe_print("  help    - Show this help")
        safe_print("  status  - Show system status")
        safe_print("  agents  - List available agents")
        safe_print("  quit    - Exit VantaEchoNebula")
    
    def show_status(self):
        """Show system status."""
        safe_print(f"\n📊 System Status:")
        safe_print(f"  Running: {self.running}")
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='VantaEchoNebula Network System')
    parser.add_argument('--mode', choices=['node', 'training', 'echo', 'interactive'], 
                       default='interactive', help='System mode')
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
        asyncio.run(system.start_system(args.mode))
    except KeyboardInterrupt:
        safe_print("\n🛑 VantaEchoNebula terminated by user")
    except Exception as e:
        safe_print(f"❌ System error: {e}")
        return 1
    
    safe_print("👋 VantaEchoNebula shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())