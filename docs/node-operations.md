# 🚀 VantaEchoNebula Node Operations

## Node Operation Modes

VantaEchoNebula supports multiple node operation modes to suit different requirements and resource constraints:

### 1. Full System Node (VantaEchoNebulaSystem.py)
- **Purpose**: Complete AI system with all 12 agents
- **Resources**: High CPU, GPU recommended, significant RAM
- **Use Case**: Full AI capabilities, research, development

### 2. Basic Node (basic_node.py)
- **Purpose**: Pure blockchain participation without AI processing
- **Resources**: Minimal CPU/RAM, no GPU required
- **Use Case**: Network participation, mining, transaction validation

### 3. Simple Node (simple_node.py)
- **Purpose**: Enhanced blockchain node with some intelligent features
- **Resources**: Moderate CPU/RAM, no GPU required
- **Use Case**: Enhanced network participation with basic intelligence

## Basic Node Operation

### Quick Start
```bash
# Install minimal dependencies
pip install -r requirements_node_only.txt

# Run basic node
python basic_node.py
```

### Configuration
```python
# Basic node configuration in basic_node.py
NODE_CONFIG = {
    "network": "testnet",  # or "mainnet"
    "mining_enabled": True,
    "sync_interval": 5,    # seconds
    "max_peers": 10,
    "storage_limit": "1GB"
}
```

### Operation Flow
```
┌─────────────────┐
│   Node Start    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Network Connect │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Blockchain Sync │◄───┤  Mining Loop    │◄───┤ Network Monitor │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Block Processing                             │
└─────────────────────────────────────────────────────────────────┘
```

## Simple Node Operation

### Enhanced Features
```python
class SimpleNode:
    def __init__(self):
        # Basic blockchain functionality
        self.basic_node = BasicNode()
        
        # Enhanced features
        self.transaction_analyzer = TransactionAnalyzer()
        self.network_optimizer = NetworkOptimizer()
        self.basic_ai = BasicAI()
    
    async def enhanced_processing(self):
        """Process blocks with basic intelligence."""
        while self.running:
            # Get new blocks
            new_blocks = await self.basic_node.sync_blockchain()
            
            # Analyze transactions
            for block in new_blocks:
                analysis = self.transaction_analyzer.analyze_block(block)
                if analysis.anomaly_detected:
                    self.alert_network(analysis)
            
            # Optimize network connections
            await self.network_optimizer.optimize_connections()
            
            await asyncio.sleep(5)
```

## Full System Connection

### Connecting Nodes to Main System

Basic and simple nodes can connect to the full system for enhanced capabilities:

```python
class NodeSystemInterface:
    def __init__(self, node, system_endpoint=None):
        self.node = node
        self.system_endpoint = system_endpoint
        self.connection = None
    
    async def connect_to_main_system(self):
        """Connect to full VantaEchoNebula system."""
        if not self.system_endpoint:
            self.system_endpoint = self.discover_main_system()
        
        try:
            self.connection = await self.establish_connection()
            await self.register_node()
            return True
        except Exception as e:
            print(f"Failed to connect to main system: {e}")
            return False
    
    async def request_ai_processing(self, data):
        """Request AI processing from main system."""
        if not self.connection:
            return None
        
        request = {
            "type": "ai_processing",
            "data": data,
            "node_id": self.node.node_id,
            "timestamp": time.time()
        }
        
        response = await self.connection.send_request(request)
        return response.get("result")
```

### System Discovery
```python
def discover_main_system(self):
    """Discover main VantaEchoNebula system on network."""
    
    # Check for local system first
    local_endpoints = [
        "http://localhost:8080",
        "http://localhost:9080",
        "http://127.0.0.1:8080"
    ]
    
    for endpoint in local_endpoints:
        if self.test_system_endpoint(endpoint):
            return endpoint
    
    # Broadcast discovery on network
    discovery_response = self.broadcast_discovery_request()
    if discovery_response:
        return discovery_response.get("system_endpoint")
    
    return None

def test_system_endpoint(self, endpoint):
    """Test if endpoint is running VantaEchoNebula system."""
    try:
        response = requests.get(f"{endpoint}/api/v1/system/status", timeout=5)
        return response.json().get("system") == "VantaEchoNebula"
    except:
        return False
```

## Node Monitoring and Management

### Health Monitoring
```python
class NodeMonitor:
    def __init__(self, node):
        self.node = node
        self.metrics = {}
        self.alerts = []
    
    def collect_metrics(self):
        """Collect node performance metrics."""
        return {
            "uptime": time.time() - self.node.start_time,
            "blocks_processed": self.node.blocks_processed,
            "transactions_processed": self.node.transactions_processed,
            "peer_count": len(self.node.peers),
            "memory_usage": self.get_memory_usage(),
            "cpu_usage": self.get_cpu_usage(),
            "network_latency": self.measure_network_latency(),
            "blockchain_height": self.node.get_blockchain_height(),
            "sync_status": self.node.get_sync_status()
        }
    
    def generate_health_report(self):
        """Generate comprehensive health report."""
        metrics = self.collect_metrics()
        
        health_score = self.calculate_health_score(metrics)
        recommendations = self.generate_recommendations(metrics)
        
        return {
            "timestamp": time.time(),
            "health_score": health_score,
            "metrics": metrics,
            "alerts": self.alerts,
            "recommendations": recommendations
        }
```

### Performance Optimization
```python
class NodeOptimizer:
    def optimize_node_performance(self, node, metrics):
        """Optimize node performance based on metrics."""
        optimizations = []
        
        # Memory optimization
        if metrics["memory_usage"] > 0.8:
            optimizations.append({
                "type": "memory_cleanup",
                "action": "clear_old_blocks",
                "params": {"keep_recent": 1000}
            })
        
        # Network optimization
        if metrics["network_latency"] > 100:  # ms
            optimizations.append({
                "type": "network_optimization",
                "action": "optimize_peer_selection",
                "params": {"target_latency": 50}
            })
        
        # Storage optimization
        storage_usage = self.check_storage_usage()
        if storage_usage > 0.9:
            optimizations.append({
                "type": "storage_cleanup",
                "action": "archive_old_data",
                "params": {"archive_threshold": "30d"}
            })
        
        return optimizations
```

## Advanced Node Operations

### Multi-Network Operation
```python
class MultiNetworkNode:
    def __init__(self):
        self.networks = {}
        self.active_chains = []
    
    def add_network(self, chain_config):
        """Add support for additional blockchain network."""
        network_id = chain_config.chain_id
        
        if network_id not in self.networks:
            self.networks[network_id] = BasicNode(chain_config)
            self.active_chains.append(network_id)
    
    async def run_multi_network(self):
        """Run node across multiple networks."""
        tasks = []
        
        for network_id in self.active_chains:
            node = self.networks[network_id]
            task = asyncio.create_task(node.run())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
```

### Consensus Participation
```python
class ConsensusNode:
    def __init__(self, node):
        self.node = node
        self.validator_key = None
        self.stake_amount = 0
    
    def become_validator(self, stake_amount):
        """Become a validator node."""
        self.validator_key = self.generate_validator_key()
        self.stake_amount = stake_amount
        
        # Submit validator registration transaction
        validator_tx = {
            "type": "validator_registration",
            "validator_key": self.validator_key,
            "stake_amount": stake_amount,
            "node_id": self.node.node_id
        }
        
        return self.node.submit_transaction(validator_tx)
    
    async def participate_in_consensus(self):
        """Participate in blockchain consensus."""
        if not self.validator_key:
            return
        
        while self.node.running:
            # Wait for consensus round
            consensus_round = await self.wait_for_consensus_round()
            
            # Validate proposed block
            proposed_block = consensus_round.proposed_block
            is_valid = self.validate_block(proposed_block)
            
            # Cast vote
            vote = {
                "validator": self.validator_key,
                "block_hash": proposed_block.hash,
                "vote": "approve" if is_valid else "reject",
                "round": consensus_round.round_number
            }
            
            await self.submit_vote(vote)
```

## Node Security

### Security Best Practices
```python
class NodeSecurity:
    def __init__(self, node):
        self.node = node
        self.security_config = self.load_security_config()
    
    def secure_node(self):
        """Apply security measures to node."""
        
        # Enable firewall rules
        self.configure_firewall()
        
        # Set up SSL/TLS for network communication
        self.setup_ssl_certificates()
        
        # Configure access controls
        self.setup_access_controls()
        
        # Enable logging and monitoring
        self.setup_security_monitoring()
    
    def validate_peer_connection(self, peer):
        """Validate peer connections for security."""
        security_checks = [
            self.check_peer_reputation(peer),
            self.verify_peer_certificates(peer),
            self.validate_peer_version(peer),
            self.check_connection_limits(peer)
        ]
        
        return all(security_checks)
    
    def detect_suspicious_activity(self, activity_data):
        """Detect suspicious network activity."""
        alerts = []
        
        # Check for unusual transaction patterns
        if self.detect_unusual_transactions(activity_data):
            alerts.append("unusual_transaction_pattern")
        
        # Check for potential attacks
        if self.detect_potential_attack(activity_data):
            alerts.append("potential_attack_detected")
        
        # Check for resource abuse
        if self.detect_resource_abuse(activity_data):
            alerts.append("resource_abuse_detected")
        
        return alerts
```

## Deployment Strategies

### Docker Deployment
```dockerfile
# Dockerfile for basic node
FROM python:3.9-slim

WORKDIR /app

COPY requirements_node_only.txt .
RUN pip install -r requirements_node_only.txt

COPY basic_node.py .
COPY chain_config.py .

EXPOSE 8080 9080

CMD ["python", "basic_node.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vantaecho-node
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vantaecho-node
  template:
    metadata:
      labels:
        app: vantaecho-node
    spec:
      containers:
      - name: node
        image: vantaecho/node:latest
        ports:
        - containerPort: 8080
        - containerPort: 9080
        env:
        - name: NETWORK_MODE
          value: "testnet"
        - name: MINING_ENABLED
          value: "true"
```

### Cloud Deployment
```python
class CloudNodeDeployment:
    def deploy_to_aws(self, config):
        """Deploy node to AWS."""
        ec2_config = {
            "instance_type": "t3.micro",  # For basic node
            "ami_id": "ami-0abcdef1234567890",
            "security_groups": ["vantaecho-node-sg"],
            "user_data": self.generate_user_data_script()
        }
        
        return self.create_ec2_instance(ec2_config)
    
    def deploy_to_azure(self, config):
        """Deploy node to Azure."""
        # Azure deployment configuration
        pass
    
    def deploy_to_gcp(self, config):
        """Deploy node to Google Cloud."""
        # GCP deployment configuration
        pass
```

---

*Node operations provide flexible deployment options for different resource requirements and use cases, from minimal blockchain participation to full AI system capabilities.*