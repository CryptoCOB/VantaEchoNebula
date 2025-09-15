# 🔗 VantaEchoNebula Blockchain Integration

## Network Architecture

VantaEchoNebula operates on a dual-chain architecture that separates development from production environments while maintaining full interoperability.

## Network Comparison

| Feature | TestNet | MainNet | Standalone |
|---------|---------|---------|------------|
| **Purpose** | Development & Testing | Production Operations | Local Development |
| **Block Time** | 5 seconds | 30 seconds | N/A |
| **Mining Reward** | 100 VEN | 10 VEN | N/A |
| **Validators** | 5 (test) | 10 (production) | N/A |
| **Consensus Threshold** | 60% | 67% | N/A |
| **Economic Impact** | None (test tokens) | Real (VEN tokens) | None |
| **Data Directory** | `./data/testchain` | `./data/mainchain` | `./data/standalone` |
| **API Port** | 9080 | 8080 | 8090 |
| **RPC Port** | 36657 | 26657 | 26658 |
| **P2P Port** | 36656 | 26656 | N/A |

## Network Configuration (`chain_config.py`)

### TestNet Configuration
```python
TestNet = ChainConfig(
    chain_type=ChainType.TEST,
    chain_id="vanta-echo-nebula-test-1",
    network_name="Nebula TestNet",
    difficulty=1,                    # Easy mining
    mining_reward=100,               # High rewards for testing
    max_reorg_depth=10,             # Shallow reorganization
    block_time_target=5.0,          # Fast blocks
    validators=[
        "TestValidator01", "TestValidator02", 
        "TestValidator03", "TestValidator04", "TestValidator05"
    ],
    min_validators=3,
    consensus_threshold=0.6,         # 60% consensus
    # ... additional config
)
```

### MainNet Configuration  
```python
MainNet = ChainConfig(
    chain_type=ChainType.MAIN,
    chain_id="vanta-echo-nebula-main-1", 
    network_name="Nebula MainNet",
    difficulty=4,                    # Higher security
    mining_reward=10,                # Conservative rewards
    max_reorg_depth=100,            # Deep reorganization protection
    block_time_target=30.0,         # Stable blocks
    validators=[
        "MainValidator01", "MainValidator02", "MainValidator03",
        "MainValidator04", "MainValidator05", "MainValidator06",
        "MainValidator07", "MainValidator08", "MainValidator09",
        "MainValidator10"
    ],
    min_validators=7,
    consensus_threshold=0.67,        # 67% consensus (Byzantine fault tolerance)
    # ... additional config
)
```

## Connection Methods

### 1. Direct System Connection
```bash
# Connect to TestNet (recommended for development)
python VantaEchoNebulaSystem.py --network testnet --mode node

# Connect to MainNet (production use)
python VantaEchoNebulaSystem.py --network mainnet --mode node

# Standalone mode (no blockchain)
python VantaEchoNebulaSystem.py --mode interactive
```

### 2. Interactive Network Connector
```bash
# Launch interactive network selection
python network_connector.py

# Command line options
python network_connector.py --network testnet --start-node
python network_connector.py --network mainnet --start-training
```

### 3. Basic Node Operations
```bash
# Lightweight node participation
python basic_node.py              # TestNet by default
python basic_node.py --mainnet    # MainNet participation
```

## Blockchain Services Integration

### Smart Contract Integration (`crypto_nebula.py`)
```python
class CryptoNebula:
    def __init__(self, chain_config):
        self.chain_config = chain_config
        self.web3 = self.connect_to_chain()
    
    def connect_to_chain(self):
        """Connect to the appropriate blockchain network."""
        if self.chain_config.chain_type == ChainType.MAIN:
            # MainNet connection
            endpoint = f"http://localhost:{self.chain_config.rpc_port}"
        else:
            # TestNet connection  
            endpoint = f"http://localhost:{self.chain_config.rpc_port}"
        
        return Web3(Web3.HTTPProvider(endpoint))
    
    def execute_agent_task(self, agent_name, task_data):
        """Record agent task execution on blockchain."""
        transaction = {
            'agent': agent_name,
            'task': task_data,
            'timestamp': time.time(),
            'network': self.chain_config.chain_id
        }
        return self.submit_transaction(transaction)
```

### Agent-Blockchain Integration
Each agent can interact with the blockchain through the Scribe agent:

```python
# Agent records decision on blockchain
class ReasoningEngine:
    def make_ethical_decision(self, scenario):
        decision = self.analyze_scenario(scenario)
        
        # Record on blockchain for transparency
        blockchain_record = {
            'agent': 'Warden',
            'decision': decision,
            'reasoning': self.get_reasoning_trace(),
            'confidence': self.confidence_score
        }
        
        self.scribe_agent.record_decision(blockchain_record)
        return decision
```

## Economics and Incentives

### Mining Rewards
- **TestNet**: 100 VEN per block (for testing)
- **MainNet**: 10 VEN per block (production value)

### Task-Based Rewards
```python
# Different tasks have different reward multipliers
TASK_REWARDS = {
    'memory_storage': 1.0,      # Base reward
    'reasoning_task': 2.0,      # 2x multiplier  
    'neural_optimization': 3.0, # 3x multiplier
    'consensus_validation': 1.5 # 1.5x multiplier
}
```

### Slashing Penalties
- **TestNet**: 5% penalty for misbehavior
- **MainNet**: 10% penalty for misbehavior

## Network Synchronization

### Block Synchronization Process
```python
async def sync_blockchain(self, config):
    """Synchronize with the blockchain network."""
    
    # Get current block height
    local_height = self.get_local_block_height()
    network_height = self.get_network_block_height(config)
    
    # Sync missing blocks
    while local_height < network_height:
        block = self.fetch_block(local_height + 1, config)
        self.validate_and_add_block(block)
        local_height += 1
        
        # Respect block time
        await asyncio.sleep(config.block_time_target)
```

### Transaction Pool Management
```python
class TransactionPool:
    def __init__(self, config):
        self.config = config
        self.pending_transactions = []
        self.max_pool_size = 10000 if config.chain_type == ChainType.MAIN else 1000
    
    def add_agent_transaction(self, agent_tx):
        """Add agent-generated transaction to pool."""
        if self.validate_transaction(agent_tx):
            self.pending_transactions.append(agent_tx)
            return True
        return False
```

## dVPN Integration

### Decentralized VPN Features
```python
class DVPNService:
    def __init__(self, chain_config):
        self.config = chain_config
        self.bandwidth_proof_interval = chain_config.bandwidth_proof_interval
        self.tunnel_timeout = chain_config.tunnel_timeout
    
    async def start_dvpn_service(self):
        """Start decentralized VPN service."""
        if self.config.dvpn_enabled:
            await asyncio.gather(
                self.bandwidth_monitoring(),
                self.tunnel_management(),
                self.proof_of_bandwidth()
            )
```

### Bandwidth Economics
- **Bandwidth Rate**: 0.001 VEN per MB (MainNet), 0.01 VEN per MB (TestNet)
- **Proof Intervals**: 60s (MainNet), 10s (TestNet)
- **Quality Incentives**: Higher rewards for reliable connections

## Security Model

### Network Security
- **Cryptographic Signatures**: All transactions signed with private keys
- **Consensus Mechanisms**: Byzantine fault tolerance with 67% (MainNet) / 60% (TestNet) thresholds
- **Network Isolation**: Complete separation between TestNet and MainNet

### Agent Security on Blockchain
- **Action Verification**: All agent actions verified before blockchain recording
- **Ethical Oversight**: Warden agent reviews all blockchain transactions
- **Resource Limits**: Gas-like system prevents resource abuse

## API Endpoints

### TestNet API (Port 9080)
```
GET  /api/v1/status          # Network status
GET  /api/v1/agents          # Active agents  
POST /api/v1/agent/execute   # Execute agent task
GET  /api/v1/blocks/latest   # Latest block info
POST /api/v1/transaction     # Submit transaction
```

### MainNet API (Port 8080)
```
# Same endpoints as TestNet, but production data
GET  /api/v1/status
GET  /api/v1/agents
POST /api/v1/agent/execute
GET  /api/v1/blocks/latest
POST /api/v1/transaction
```

## Monitoring and Analytics

### Network Health Metrics
- Block production rate
- Transaction throughput  
- Network latency
- Validator participation
- Agent task completion rates

### Economic Metrics
- Token circulation
- Mining rewards distribution
- Task-based reward allocation
- dVPN bandwidth utilization

## Troubleshooting

### Common Connection Issues

**Problem**: Cannot connect to network
```bash
# Check network configuration
python -c "from chain_config import get_current_chain; print(get_current_chain())"

# Verify ports are available
netstat -an | findstr :9080    # TestNet API
netstat -an | findstr :8080    # MainNet API
```

**Problem**: Agent tasks not recording on blockchain
```python
# Verify Scribe agent is initialized
if 'Scribe' not in system.agents:
    print("Blockchain integration not available")
```

**Problem**: Mining not working
- Check validator configuration
- Verify wallet has sufficient balance for gas
- Ensure network connectivity

---

*The blockchain integration provides the foundation for decentralized operation, economic incentives, and transparent agent coordination in the VantaEchoNebula network.*