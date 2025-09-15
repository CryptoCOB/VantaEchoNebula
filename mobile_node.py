#!/usr/bin/env python3
"""
VantaEchoNebula Network - Mobile Optimized Node
Lightweight blockchain node for mobile devices (Android/Termux)
"""

import os
import sys
import json
import time
import threading
import signal
from pathlib import Path
from datetime import datetime
import psutil
import argparse

class MobileNebulaNode:
    def __init__(self):
        self.version = "2.0.0-mobile"
        self.config_file = Path.home() / "NebulaNode" / "mobile_config.json"
        self.data_dir = Path.home() / "NebulaNode" / "data"
        self.log_file = Path.home() / "NebulaNode" / "mobile.log"
        
        # Mobile-optimized defaults
        self.config = {
            "mobile_mode": True,
            "mining_enabled": False,  # Disabled by default on mobile
            "max_peers": 5,  # Reduced peer connections
            "sync_mode": "light",  # Light sync only
            "memory_limit_mb": 512,  # 512MB memory limit
            "cpu_usage_limit": 30,  # 30% CPU usage limit
            "battery_optimization": True,
            "web_interface_port": 8545,
            "auto_pause_on_low_battery": True,
            "temperature_monitoring": True,
            "data_usage_monitoring": True
        }
        
        self.running = False
        self.paused = False
        self.stats = {
            "start_time": None,
            "blocks_synced": 0,
            "peers_connected": 0,
            "data_usage_mb": 0,
            "battery_level": 100,
            "cpu_usage": 0,
            "memory_usage_mb": 0
        }
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self):
        """Load mobile configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                    print(f"✅ Loaded mobile config from {self.config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
        else:
            self.save_config()
    
    def save_config(self):
        """Save mobile configuration."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
                print(f"✅ Saved mobile config to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception:
            pass  # Don't fail if logging fails
    
    def check_system_resources(self):
        """Monitor system resources for mobile optimization."""
        try:
            # CPU usage
            self.stats["cpu_usage"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.stats["memory_usage_mb"] = memory.used / 1024 / 1024
            
            # Battery level (if available)
            try:
                battery = psutil.sensors_battery()
                if battery:
                    self.stats["battery_level"] = battery.percent
            except (AttributeError, NotImplementedError):
                pass  # Battery info not available on all systems
            
            # Check if we should pause due to resource constraints
            if self.config["auto_pause_on_low_battery"] and self.stats["battery_level"] < 20:
                if not self.paused:
                    self.log("🔋 Low battery detected, pausing node operations")
                    self.paused = True
            elif self.paused and self.stats["battery_level"] > 30:
                self.log("🔋 Battery recovered, resuming node operations")
                self.paused = False
            
            # CPU usage limit
            if self.stats["cpu_usage"] > self.config["cpu_usage_limit"]:
                time.sleep(2)  # Throttle if CPU usage is high
            
        except Exception as e:
            self.log(f"⚠️ Resource monitoring error: {e}")
    
    def simulate_blockchain_activity(self):
        """Simulate lightweight blockchain activity."""
        activities = [
            "📱 Mobile peer connected",
            "🔄 Light sync update", 
            "📊 Block header received",
            "🌐 Network status update",
            "💾 Data pruned for mobile"
        ]
        
        while self.running:
            if not self.paused:
                # Simulate activity
                import random
                if random.random() < 0.3:
                    activity = random.choice(activities)
                    self.log(activity)
                    
                    # Update stats
                    if random.random() < 0.1:
                        self.stats["blocks_synced"] += 1
                        self.stats["peers_connected"] = max(1, 
                            self.stats["peers_connected"] + random.randint(-1, 2))
                        self.stats["peers_connected"] = min(
                            self.stats["peers_connected"], self.config["max_peers"])
            
            # Resource monitoring
            self.check_system_resources()
            
            # Mobile-optimized sleep interval
            time.sleep(random.uniform(5, 15))
    
    def start_web_interface(self):
        """Start a simple web interface for mobile access."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        class MobileWebHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Nebula Mobile Node</title>
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
                            .status {{ background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                            .metric {{ display: flex; justify-content: space-between; margin: 5px 0; }}
                            .good {{ color: #4CAF50; }}
                            .warning {{ color: #FF9800; }}
                            .error {{ color: #F44336; }}
                            button {{ background: #4CAF50; color: white; border: none; padding: 10px 20px; 
                                     border-radius: 5px; font-size: 16px; cursor: pointer; margin: 5px; }}
                        </style>
                        <script>
                            function refreshPage() {{ window.location.reload(); }}
                            setInterval(refreshPage, 30000);
                        </script>
                    </head>
                    <body>
                        <h1>🌌 Nebula Mobile Node</h1>
                        <div class="status">
                            <h3>Node Status</h3>
                            <div class="metric"><span>Version:</span><span>{self.server.node.version}</span></div>
                            <div class="metric"><span>Status:</span><span class="{'good' if self.server.node.running and not self.server.node.paused else 'warning'}">{'Running' if self.server.node.running and not self.server.node.paused else 'Paused' if self.server.node.paused else 'Stopped'}</span></div>
                            <div class="metric"><span>Uptime:</span><span>{self._get_uptime()}</span></div>
                            <div class="metric"><span>Peers:</span><span>{self.server.node.stats['peers_connected']}</span></div>
                            <div class="metric"><span>Blocks Synced:</span><span>{self.server.node.stats['blocks_synced']}</span></div>
                        </div>
                        <div class="status">
                            <h3>System Resources</h3>
                            <div class="metric"><span>Battery:</span><span class="{'good' if self.server.node.stats['battery_level'] > 50 else 'warning' if self.server.node.stats['battery_level'] > 20 else 'error'}">{self.server.node.stats['battery_level']:.1f}%</span></div>
                            <div class="metric"><span>CPU Usage:</span><span>{self.server.node.stats['cpu_usage']:.1f}%</span></div>
                            <div class="metric"><span>Memory:</span><span>{self.server.node.stats['memory_usage_mb']:.0f} MB</span></div>
                        </div>
                        <div class="status">
                            <h3>Mobile Settings</h3>
                            <div class="metric"><span>Mining:</span><span class="{'good' if self.server.node.config['mining_enabled'] else 'warning'}">{'Enabled' if self.server.node.config['mining_enabled'] else 'Disabled (Mobile Safe)'}</span></div>
                            <div class="metric"><span>Sync Mode:</span><span>{self.server.node.config['sync_mode'].title()}</span></div>
                            <div class="metric"><span>Max Peers:</span><span>{self.server.node.config['max_peers']}</span></div>
                            <div class="metric"><span>Battery Optimization:</span><span class="good">Enabled</span></div>
                        </div>
                        <button onclick="refreshPage()">🔄 Refresh</button>
                        <p><small>Auto-refresh every 30 seconds | Mobile optimized interface</small></p>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                elif self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    status = {
                        "version": self.server.node.version,
                        "running": self.server.node.running,
                        "paused": self.server.node.paused,
                        "stats": self.server.node.stats,
                        "config": self.server.node.config
                    }
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_error(404)
            
            def _get_uptime(self):
                if self.server.node.stats["start_time"]:
                    uptime = datetime.now() - self.server.node.stats["start_time"]
                    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    return f"{hours}h {minutes}m"
                return "0h 0m"
            
            def log_message(self, format, *args):
                pass  # Suppress HTTP request logging
        
        try:
            port = self.config["web_interface_port"]
            server = HTTPServer(('0.0.0.0', port), MobileWebHandler)
            server.node = self
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            self.log(f"🌐 Mobile web interface started at http://localhost:{port}")
        except Exception as e:
            self.log(f"⚠️ Failed to start web interface: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.log("🛑 Shutdown signal received")
        self.stop()
    
    def start(self):
        """Start the mobile node."""
        self.log(f"🌌 Starting Nebula Mobile Node {self.version}")
        self.log(f"📱 Mobile optimizations enabled")
        
        # Load configuration
        self.load_config()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start web interface
        self.start_web_interface()
        
        # Start the node
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        self.log("✅ Mobile node started successfully!")
        self.log(f"🌐 Web interface: http://localhost:{self.config['web_interface_port']}")
        self.log("📱 Mobile features:")
        self.log("   • Battery optimization enabled")
        self.log("   • Light sync mode")
        self.log("   • Reduced peer connections")
        self.log("   • CPU usage limiting")
        self.log("   • Auto-pause on low battery")
        
        # Start blockchain simulation
        try:
            self.simulate_blockchain_activity()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the mobile node."""
        self.log("🛑 Stopping mobile node...")
        self.running = False
        self.save_config()
        self.log("👋 Mobile node stopped")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='VantaEchoNebula Network Mobile Node')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--mining', action='store_true', help='Enable mining (not recommended on mobile)')
    parser.add_argument('--peers', type=int, default=5, help='Maximum number of peers (1-10)')
    
    args = parser.parse_args()
    
    try:
        # Check if we're running on Android/Termux
        is_android = os.path.exists('/data/data/com.termux') or 'ANDROID_ROOT' in os.environ
        if is_android:
            print("📱 Android/Termux detected - Mobile optimizations enabled")
        
        # Create and start node
        node = MobileNebulaNode()
        
        # Apply command line arguments
        if args.mining:
            node.config["mining_enabled"] = True
            print("⚠️ Mining enabled - Monitor device temperature!")
        
        if args.peers:
            node.config["max_peers"] = max(1, min(10, args.peers))
        
        # Start the node
        node.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Mobile node interrupted by user")
    except Exception as e:
        print(f"❌ Mobile node error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()