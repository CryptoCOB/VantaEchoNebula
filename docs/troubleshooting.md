# 🛠️ VantaEchoNebula Troubleshooting Guide

## Common Issues and Solutions

### System Startup Issues

#### Issue: "Module not found" errors
**Symptoms:** Import errors when starting system or agents
**Solution:**
```bash
# Install all dependencies
uv pip install -r requirements.txt

# For node-only operation
uv pip install -r requirements_node_only.txt

# Verify Python environment
uv python --version
```

#### Issue: Port already in use
**Symptoms:** "Address already in use" error on ports 8080/9080
**Solution:**
```bash
# Find process using port
netstat -ano | findstr :8080
netstat -ano | findstr :9080

# Kill the process (replace PID with actual process ID)
taskkill /f /pid <PID>

# Or use different ports in configuration
```

#### Issue: GPU not detected
**Symptoms:** CUDA errors or falling back to CPU
**Solution:**
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Install CUDA-compatible PyTorch if needed
# Visit: https://pytorch.org/get-started/locally/
```

### Agent Issues

#### Issue: Agent not responding
**Symptoms:** Agent tasks timeout or fail
**Diagnosis:**
```bash
# Check agent status via API
curl http://localhost:8080/api/v1/agents/Archivist
```

**Solution:**
```python
# Restart specific agent
system.restart_agent("Archivist")

# Or restart entire system
system.restart_system()
```

#### Issue: Memory leaks in agents
**Symptoms:** Increasing RAM usage over time
**Solution:**
```python
# Enable periodic memory cleanup
class AgentMemoryManager:
    def __init__(self, cleanup_interval=3600):  # 1 hour
        self.cleanup_interval = cleanup_interval
        
    def cleanup_agent_memory(self, agent_name):
        agent = self.system.agents[agent_name]
        
        # Clear old task results
        agent.clear_old_results(max_age=86400)  # 24 hours
        
        # Garbage collect
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Blockchain Issues

#### Issue: Blockchain sync problems
**Symptoms:** Node not syncing with network
**Diagnosis:**
```python
# Check sync status
sync_status = node.get_sync_status()
print(f"Current height: {sync_status['current_height']}")
print(f"Network height: {sync_status['network_height']}")
print(f"Sync progress: {sync_status['progress']}%")
```

**Solution:**
```python
# Force resync from genesis
node.resync_from_genesis()

# Or resync from specific block
node.resync_from_block(block_height=15000)

# Check peer connections
peer_count = len(node.get_peers())
if peer_count < 3:
    node.discover_more_peers()
```

#### Issue: Transaction not confirming
**Symptoms:** Pending transactions not included in blocks
**Diagnosis:**
```python
# Check transaction status
tx_status = node.get_transaction_status(tx_hash)
print(f"Status: {tx_status['status']}")
print(f"Gas price: {tx_status['gas_price']}")
print(f"Network gas price: {node.get_recommended_gas_price()}")
```

**Solution:**
```python
# Increase gas price and resubmit
if tx_status['gas_price'] < node.get_recommended_gas_price():
    new_tx = tx_status.copy()
    new_tx['gas_price'] = node.get_recommended_gas_price() * 1.2
    new_tx_hash = node.submit_transaction(new_tx)
```

### Network Issues

#### Issue: High network latency
**Symptoms:** Slow block propagation, delayed transactions
**Diagnosis:**
```python
# Measure network latency to peers
for peer in node.get_peers():
    latency = node.ping_peer(peer['id'])
    print(f"Peer {peer['id']}: {latency}ms")
```

**Solution:**
```python
# Optimize peer selection
class NetworkOptimizer:
    def optimize_peer_connections(self, node):
        peers = node.get_peers()
        
        # Sort by latency and connection quality
        sorted_peers = sorted(peers, 
            key=lambda p: (p['latency'], -p['connection_quality']))
        
        # Keep best peers, disconnect worst
        good_peers = sorted_peers[:node.config.max_good_peers]
        bad_peers = sorted_peers[node.config.max_peers:]
        
        for peer in bad_peers:
            node.disconnect_peer(peer['id'])
        
        # Discover new peers to replace bad ones
        node.discover_peers(count=len(bad_peers))
```

#### Issue: Peer discovery failures
**Symptoms:** Low peer count, isolation from network
**Solution:**
```python
# Use bootstrap nodes
bootstrap_nodes = [
    "bootstrap1.vantaecho.network:8080",
    "bootstrap2.vantaecho.network:8080", 
    "bootstrap3.vantaecho.network:8080"
]

for bootstrap in bootstrap_nodes:
    try:
        node.connect_to_peer(bootstrap)
    except Exception as e:
        print(f"Failed to connect to {bootstrap}: {e}")

# Enable peer discovery protocols
node.enable_upnp_discovery()
node.enable_mdns_discovery()
```

### Performance Issues

#### Issue: High CPU usage
**Symptoms:** System sluggish, high CPU utilization
**Diagnosis:**
```python
import psutil
import time

def profile_cpu_usage():
    for i in range(10):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        print(f"CPU: {cpu_percent}%, RAM: {memory_percent}%")
        time.sleep(1)
```

**Solution:**
```python
# Reduce agent processing frequency
class PerformanceOptimizer:
    def reduce_processing_load(self, system):
        # Increase processing intervals
        for agent in system.agents.values():
            if hasattr(agent, 'processing_interval'):
                agent.processing_interval *= 1.5
        
        # Disable non-critical agents temporarily
        non_critical_agents = ['Observer', 'Resonant']
        for agent_name in non_critical_agents:
            if system.get_system_load() > 0.8:
                system.agents[agent_name].set_status('idle')
```

#### Issue: Memory leaks
**Symptoms:** Continuously increasing RAM usage
**Solution:**
```python
# Implement memory monitoring and cleanup
class MemoryMonitor:
    def __init__(self, threshold_mb=8000):
        self.threshold_mb = threshold_mb
        
    def monitor_memory(self):
        import psutil
        import gc
        
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        
        if memory_mb > self.threshold_mb:
            # Force garbage collection
            gc.collect()
            
            # Clear agent caches
            for agent in system.agents.values():
                if hasattr(agent, 'clear_cache'):
                    agent.clear_cache()
            
            # Clear blockchain cache
            node.clear_block_cache(keep_recent=1000)
```

### Storage Issues

#### Issue: Disk space running low
**Symptoms:** Write failures, system crashes
**Solution:**
```python
class StorageManager:
    def cleanup_storage(self, node):
        # Archive old blocks
        current_height = node.get_blockchain_height()
        archive_before = current_height - 10000  # Keep last 10k blocks
        
        node.archive_blocks_before(archive_before)
        
        # Compress log files
        node.compress_old_logs(older_than_days=7)
        
        # Clean temporary files
        node.clean_temp_files()
        
    def monitor_disk_usage(self):
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_percent = free / total * 100
        
        if free_percent < 10:  # Less than 10% free
            self.cleanup_storage()
            
        return free_percent
```

## Debugging Tools

### System Diagnostics

```python
class SystemDiagnostics:
    def run_full_diagnostic(self):
        report = {
            "timestamp": time.time(),
            "system_health": self.check_system_health(),
            "agent_status": self.check_all_agents(),
            "blockchain_status": self.check_blockchain(),
            "network_status": self.check_network(),
            "performance_metrics": self.collect_performance_metrics(),
            "error_analysis": self.analyze_recent_errors()
        }
        
        return report
    
    def check_system_health(self):
        return {
            "uptime": time.time() - system.start_time,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_processes": len([p for p in psutil.process_iter() 
                                   if 'vantaecho' in p.name().lower()])
        }
```

### Log Analysis

```python
class LogAnalyzer:
    def analyze_error_patterns(self, log_file="system.log"):
        error_patterns = {}
        warning_patterns = {}
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'ERROR' in line:
                    pattern = self.extract_error_pattern(line)
                    error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
                elif 'WARNING' in line:
                    pattern = self.extract_warning_pattern(line)
                    warning_patterns[pattern] = warning_patterns.get(pattern, 0) + 1
        
        return {
            "errors": sorted(error_patterns.items(), key=lambda x: x[1], reverse=True),
            "warnings": sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)
        }
    
    def get_recent_errors(self, hours=24):
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = []
        
        with open("system.log", 'r') as f:
            for line in f:
                if self.extract_timestamp(line) > cutoff_time and 'ERROR' in line:
                    recent_errors.append(line.strip())
        
        return recent_errors
```

### Network Diagnostics

```python
class NetworkDiagnostics:
    def test_connectivity(self, node):
        results = {}
        
        # Test blockchain network connectivity
        results['blockchain_connectivity'] = self.test_blockchain_connectivity(node)
        
        # Test peer connectivity  
        results['peer_connectivity'] = self.test_peer_connectivity(node)
        
        # Test API connectivity
        results['api_connectivity'] = self.test_api_connectivity()
        
        return results
    
    def test_peer_connectivity(self, node):
        connectivity_results = []
        
        for peer in node.get_peers():
            try:
                response_time = node.ping_peer(peer['id'])
                connectivity_results.append({
                    'peer_id': peer['id'],
                    'status': 'connected',
                    'response_time': response_time
                })
            except Exception as e:
                connectivity_results.append({
                    'peer_id': peer['id'], 
                    'status': 'failed',
                    'error': str(e)
                })
        
        return connectivity_results
```

## Recovery Procedures

### System Recovery

```python
class SystemRecovery:
    def emergency_recovery(self):
        """Emergency recovery procedure for system failures."""
        
        print("Starting emergency recovery...")
        
        # Step 1: Save current state
        self.save_emergency_state()
        
        # Step 2: Stop all agents gracefully
        self.stop_all_agents()
        
        # Step 3: Clear locks and temporary files
        self.clear_system_locks()
        
        # Step 4: Restart core services
        self.restart_core_services()
        
        # Step 5: Restart agents one by one
        self.restart_agents_sequentially()
        
        # Step 6: Verify system integrity
        if self.verify_system_integrity():
            print("Emergency recovery completed successfully")
            return True
        else:
            print("Emergency recovery failed - manual intervention required")
            return False
```

### Blockchain Recovery

```python
class BlockchainRecovery:
    def recover_corrupted_blockchain(self, node):
        """Recover from blockchain corruption."""
        
        print("Starting blockchain recovery...")
        
        # Step 1: Backup current state
        backup_path = self.create_blockchain_backup(node)
        
        # Step 2: Verify corruption extent
        corruption_report = self.analyze_blockchain_corruption(node)
        
        # Step 3: Choose recovery method
        if corruption_report['blocks_affected'] < 100:
            # Minor corruption - rebuild affected blocks
            self.rebuild_corrupted_blocks(node, corruption_report['blocks_affected'])
        else:
            # Major corruption - resync from network
            self.resync_blockchain_from_network(node)
        
        # Step 4: Verify blockchain integrity
        if self.verify_blockchain_integrity(node):
            print("Blockchain recovery completed successfully")
            return True
        else:
            print("Blockchain recovery failed - restoring from backup")
            self.restore_from_backup(backup_path)
            return False
```

## Monitoring and Alerting

### Health Monitoring

```python
class HealthMonitor:
    def __init__(self):
        self.health_thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85, 
            'disk_usage': 90,
            'agent_failure_rate': 5,
            'network_latency': 1000
        }
    
    def continuous_monitoring(self):
        while True:
            health_report = self.collect_health_metrics()
            
            alerts = self.check_thresholds(health_report)
            
            if alerts:
                self.send_alerts(alerts)
            
            time.sleep(60)  # Check every minute
    
    def check_thresholds(self, metrics):
        alerts = []
        
        for metric, value in metrics.items():
            if metric in self.health_thresholds:
                if value > self.health_thresholds[metric]:
                    alerts.append({
                        'type': 'threshold_exceeded',
                        'metric': metric,
                        'value': value,
                        'threshold': self.health_thresholds[metric],
                        'severity': self.get_severity(metric, value)
                    })
        
        return alerts
```

---

*This troubleshooting guide covers common issues and their solutions. For complex problems not covered here, enable debug logging and contact the development team with detailed system diagnostics.*