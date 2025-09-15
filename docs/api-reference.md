# 🌐 VantaEchoNebula API Reference

## API Overview

VantaEchoNebula provides comprehensive APIs for interaction with the system, agents, blockchain, and network services. All APIs support both REST and WebSocket protocols where applicable.

## Base URLs

- **TestNet API**: `http://localhost:9080/api/v1`
- **MainNet API**: `http://localhost:8080/api/v1`
- **System API**: `http://localhost:8000/api/v1`

## Authentication

All API endpoints support Bearer token authentication:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8080/api/v1/system/status
```

## System APIs

### System Status

#### GET /system/status
Get current system status and health information.

**Response:**
```json
{
  "system": "VantaEchoNebula",
  "version": "2.0.0",
  "status": "active",
  "uptime": 86400,
  "active_agents": 12,
  "blockchain_height": 15420,
  "network_peers": 25,
  "health_score": 0.95
}
```

#### GET /system/info
Get detailed system information and capabilities.

**Response:**
```json
{
  "system_info": {
    "name": "VantaEchoNebula",
    "version": "2.0.0",
    "build": "20241220-stable",
    "capabilities": ["ai_processing", "blockchain", "p2p_networking"],
    "supported_networks": ["mainnet", "testnet"],
    "agent_count": 12,
    "api_version": "v1"
  },
  "resource_usage": {
    "cpu_usage": 0.45,
    "memory_usage": 0.67,
    "gpu_usage": 0.23,
    "storage_usage": 0.34
  },
  "configuration": {
    "max_peers": 50,
    "consensus_threshold": 0.67,
    "mining_enabled": true,
    "ai_processing_enabled": true
  }
}
```

### Configuration Management

#### GET /system/config
Get current system configuration.

#### PUT /system/config
Update system configuration.

**Request Body:**
```json
{
  "consensus_threshold": 0.75,
  "max_peers": 75,
  "mining_enabled": false,
  "log_level": "INFO"
}
```

## Agent APIs

### Agent Management

#### GET /agents
List all available agents and their status.

**Response:**
```json
{
  "agents": [
    {
      "name": "Archivist",
      "module": "memory_learner.py",
      "status": "active",
      "capabilities": ["memory_storage", "learning", "retrieval"],
      "load": 0.34,
      "last_activity": "2024-12-20T10:30:00Z"
    },
    {
      "name": "Warden", 
      "module": "reasoning_engine.py",
      "status": "active",
      "capabilities": ["ethical_reasoning", "decision_validation"],
      "load": 0.67,
      "last_activity": "2024-12-20T10:29:45Z"
    }
  ],
  "total_agents": 12,
  "active_agents": 11,
  "inactive_agents": 1
}
```

#### GET /agents/{agent_name}
Get detailed information about a specific agent.

**Response:**
```json
{
  "name": "Archivist",
  "module": "memory_learner.py", 
  "sigil": "🜌⟐🜹🜙",
  "status": "active",
  "uptime": 3600,
  "capabilities": [
    "long_term_memory_integration",
    "modalities_processing", 
    "adaptive_embedding",
    "uncertainty_estimation"
  ],
  "current_tasks": [
    "processing_memory_consolidation",
    "updating_knowledge_graph"
  ],
  "performance_metrics": {
    "tasks_completed": 1247,
    "average_response_time": 0.045,
    "success_rate": 0.987,
    "memory_usage": 0.23
  }
}
```

### Agent Execution

#### POST /agents/{agent_name}/execute
Execute a task with a specific agent.

**Request Body:**
```json
{
  "task": "analyze_memory_patterns",
  "parameters": {
    "time_range": "24h",
    "pattern_type": "semantic_clusters",
    "include_metadata": true
  },
  "priority": "normal",
  "async": false
}
```

**Response:**
```json
{
  "task_id": "task_12345",
  "agent": "Archivist",
  "status": "completed",
  "result": {
    "patterns_found": 15,
    "clusters_identified": 3,
    "confidence_score": 0.89,
    "processing_time": 2.34,
    "metadata": {
      "memory_entries_analyzed": 1234,
      "time_span_covered": "24h",
      "cluster_details": [...]
    }
  },
  "execution_time": 2.34,
  "timestamp": "2024-12-20T10:30:15Z"
}
```

#### GET /agents/{agent_name}/tasks
Get current and recent tasks for an agent.

#### GET /agents/{agent_name}/tasks/{task_id}
Get detailed information about a specific task.

### Agent Communication

#### POST /agents/{agent_name}/message
Send a message to an agent.

**Request Body:**
```json
{
  "message": "Retrieve memories related to blockchain consensus",
  "context": {
    "conversation_id": "conv_123",
    "user_id": "user_456",
    "session_id": "session_789"
  },
  "response_format": "structured"
}
```

#### GET /agents/{agent_name}/conversations
Get conversation history for an agent.

## Blockchain APIs

### Blockchain Status

#### GET /blockchain/status
Get current blockchain status.

**Response:**
```json
{
  "network": "testnet",
  "chain_id": "vantaecho_testnet_v1",
  "current_height": 15420,
  "latest_block_hash": "0x1234567890abcdef...",
  "latest_block_time": "2024-12-20T10:30:00Z",
  "difficulty": 1000000,
  "hash_rate": "1.2 TH/s",
  "pending_transactions": 45,
  "validator_count": 21,
  "consensus_status": "healthy"
}
```

#### GET /blockchain/info
Get detailed blockchain information.

### Block Operations

#### GET /blockchain/blocks
Get list of recent blocks.

**Query Parameters:**
- `limit`: Number of blocks to return (default: 20, max: 100)
- `offset`: Starting block height
- `order`: `asc` or `desc` (default: desc)

**Response:**
```json
{
  "blocks": [
    {
      "height": 15420,
      "hash": "0x1234567890abcdef...",
      "previous_hash": "0x0987654321fedcba...",
      "timestamp": "2024-12-20T10:30:00Z",
      "transaction_count": 12,
      "validator": "validator_001",
      "size": 2048,
      "difficulty": 1000000
    }
  ],
  "total_blocks": 15420,
  "page_info": {
    "has_next_page": true,
    "has_previous_page": true,
    "current_page": 1,
    "total_pages": 771
  }
}
```

#### GET /blockchain/blocks/{height_or_hash}
Get a specific block by height or hash.

#### POST /blockchain/blocks/validate
Validate a block.

### Transaction Operations

#### GET /blockchain/transactions
Get list of transactions.

**Query Parameters:**
- `status`: `pending`, `confirmed`, `failed`
- `type`: Transaction type filter
- `limit`: Number of transactions (default: 50, max: 200)

#### GET /blockchain/transactions/{tx_hash}
Get a specific transaction by hash.

**Response:**
```json
{
  "hash": "0xabcdef1234567890...",
  "block_height": 15420,
  "block_hash": "0x1234567890abcdef...", 
  "from": "0x742d35Cc6634C0532925a3b8D401...",
  "to": "0x8ba1f109551bD432803012645Hac...",
  "value": "1000000000000000000",
  "gas_price": "20000000000",
  "gas_limit": 21000,
  "gas_used": 21000,
  "status": "confirmed",
  "timestamp": "2024-12-20T10:29:30Z",
  "confirmations": 15,
  "transaction_type": "transfer",
  "data": {
    "agent_action": "memory_store",
    "metadata": {...}
  }
}
```

#### POST /blockchain/transactions
Submit a new transaction.

**Request Body:**
```json
{
  "to": "0x8ba1f109551bD432803012645Hac...",
  "value": "1000000000000000000",
  "gas_price": "20000000000",
  "gas_limit": 21000,
  "data": {
    "type": "agent_action",
    "agent": "Archivist", 
    "action": "store_memory",
    "payload": {...}
  }
}
```

### Mining Operations

#### GET /blockchain/mining/status
Get current mining status.

**Response:**
```json
{
  "mining_enabled": true,
  "hash_rate": "125.5 MH/s",
  "blocks_mined": 23,
  "last_block_mined": 15418,
  "mining_difficulty": 1000000,
  "estimated_next_block": "2024-12-20T10:35:00Z",
  "mining_reward": "50000000000000000000",
  "pool_participation": false
}
```

#### POST /blockchain/mining/start
Start mining operations.

#### POST /blockchain/mining/stop
Stop mining operations.

## Network APIs

### Network Status

#### GET /network/status
Get current network status.

**Response:**
```json
{
  "peer_count": 25,
  "max_peers": 50,
  "network_id": "vantaecho_net",
  "protocol_version": "v1.0",
  "sync_status": "synchronized",
  "bandwidth_usage": {
    "inbound": "1.2 MB/s",
    "outbound": "0.8 MB/s"
  },
  "connection_quality": 0.94
}
```

#### GET /network/peers
Get list of connected peers.

**Response:**
```json
{
  "peers": [
    {
      "id": "peer_001",
      "address": "192.168.1.100:8080",
      "version": "v1.0.0",
      "latency": 45,
      "last_seen": "2024-12-20T10:30:00Z",
      "connection_quality": 0.98,
      "blocks_synced": 15420,
      "peer_type": "full_node"
    }
  ],
  "total_peers": 25,
  "peer_types": {
    "full_node": 20,
    "basic_node": 3,
    "validator": 2
  }
}
```

### Peer Management

#### POST /network/peers/connect
Connect to a new peer.

**Request Body:**
```json
{
  "address": "192.168.1.200:8080",
  "peer_type": "full_node"
}
```

#### DELETE /network/peers/{peer_id}
Disconnect from a peer.

## Data APIs

### Storage Operations

#### GET /data/stats
Get storage statistics.

**Response:**
```json
{
  "storage_usage": {
    "total_size": "10.5 GB",
    "blockchain_data": "8.2 GB",
    "agent_data": "1.8 GB",
    "system_data": "0.5 GB"
  },
  "data_counts": {
    "blocks": 15420,
    "transactions": 234567,
    "agent_memories": 12345,
    "system_logs": 98765
  },
  "growth_rate": {
    "daily": "125 MB",
    "weekly": "875 MB",
    "monthly": "3.5 GB"
  }
}
```

#### POST /data/backup
Create a backup of system data.

#### POST /data/restore
Restore system data from backup.

### Query Operations

#### POST /data/query
Query system data with advanced filters.

**Request Body:**
```json
{
  "query_type": "blockchain_search",
  "filters": {
    "date_range": {
      "start": "2024-12-19T00:00:00Z",
      "end": "2024-12-20T23:59:59Z"
    },
    "transaction_type": "agent_action",
    "agent_names": ["Archivist", "Warden"]
  },
  "sort": {
    "field": "timestamp",
    "order": "desc"
  },
  "limit": 100
}
```

## WebSocket APIs

### Real-time Event Stream

#### WS /events
Connect to real-time event stream.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/events');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Event:', data);
};
```

**Event Types:**
```json
{
  "type": "new_block",
  "data": {
    "height": 15421,
    "hash": "0x1234...",
    "timestamp": "2024-12-20T10:35:00Z"
  }
}

{
  "type": "agent_task_completed",
  "data": {
    "agent": "Archivist",
    "task_id": "task_12345",
    "status": "completed"
  }
}

{
  "type": "network_peer_connected",
  "data": {
    "peer_id": "peer_026",
    "address": "192.168.1.150:8080"
  }
}
```

### Agent Communication WebSocket

#### WS /agents/{agent_name}/chat
Real-time chat with an agent.

```javascript
const ws = new WebSocket('ws://localhost:8080/agents/Archivist/chat');

// Send message to agent
ws.send(JSON.stringify({
    "message": "What memories do you have about recent blockchain consensus?",
    "context": {"session_id": "session_123"}
}));

// Receive agent responses
ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Agent response:', response);
};
```

## Error Handling

### Standard Error Response

All APIs return errors in a consistent format:

```json
{
  "error": {
    "code": "AGENT_NOT_FOUND",
    "message": "The specified agent 'InvalidAgent' was not found",
    "details": {
      "available_agents": ["Archivist", "Warden", "Analyst", "..."],
      "suggestion": "Use GET /agents to see available agents"
    },
    "timestamp": "2024-12-20T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `AGENT_NOT_FOUND` | Specified agent does not exist |
| `AGENT_UNAVAILABLE` | Agent is not currently active |
| `INVALID_PARAMETERS` | Request parameters are invalid |
| `BLOCKCHAIN_ERROR` | Blockchain operation failed |
| `NETWORK_ERROR` | Network operation failed |
| `AUTHENTICATION_REQUIRED` | Valid authentication token required |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `SYSTEM_OVERLOAD` | System is currently overloaded |

## Rate Limiting

APIs are rate limited to prevent abuse:

- **System APIs**: 100 requests/minute
- **Agent APIs**: 500 requests/minute  
- **Blockchain APIs**: 200 requests/minute
- **Network APIs**: 50 requests/minute

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640000000
```

## SDK and Client Libraries

### Python SDK

```python
from vantaecho import VantaEchoClient

client = VantaEchoClient(
    api_url="http://localhost:8080/api/v1",
    auth_token="your_token_here"
)

# Get system status
status = client.system.get_status()

# Execute agent task
result = client.agents.execute_task(
    agent_name="Archivist",
    task="retrieve_memories",
    parameters={"query": "blockchain consensus"}
)

# Submit blockchain transaction
tx_hash = client.blockchain.submit_transaction({
    "to": "0x1234...",
    "value": "1000000000000000000",
    "data": {"type": "agent_action"}
})
```

### JavaScript SDK

```javascript
import { VantaEchoClient } from 'vantaecho-js';

const client = new VantaEchoClient({
    apiUrl: 'http://localhost:8080/api/v1',
    authToken: 'your_token_here'
});

// Get system status
const status = await client.system.getStatus();

// Execute agent task
const result = await client.agents.executeTask('Archivist', {
    task: 'retrieve_memories',
    parameters: { query: 'blockchain consensus' }
});
```

---

*The VantaEchoNebula API provides comprehensive access to all system capabilities, enabling integration with external applications and services.*