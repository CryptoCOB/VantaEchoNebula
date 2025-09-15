# 🌌 Nebula Dual-Chain Configuration
# Separates MainChain (production) from TestChain (development)

from typing import List
from dataclasses import dataclass
from enum import Enum
import os

class ChainType(Enum):
    MAIN = "main"
    TEST = "test"

@dataclass
class ChainConfig:
    """Configuration for Nebula blockchain instances"""
    chain_type: ChainType
    chain_id: str
    network_name: str
    
    # Blockchain parameters
    difficulty: int
    mining_reward: int
    max_reorg_depth: int
    block_time_target: float  # seconds
    
    # Consensus settings
    validators: List[str]
    min_validators: int
    consensus_threshold: float
    
    # Task Fabric settings
    min_task_complexity: int
    max_task_complexity: int
    task_verification_timeout: float
    min_redundancy: int
    
    # Economic parameters
    base_reward: int
    compute_reward_multiplier: float
    bandwidth_reward_rate: float
    slashing_penalty_rate: float
    
    # Network settings
    p2p_port: int
    rpc_port: int
    api_port: int
    max_peers: int
    
    # Storage paths
    data_dir: str
    blockchain_file: str
    task_cache_dir: str
    
    # dVPN settings
    dvpn_enabled: bool
    bandwidth_proof_interval: float
    tunnel_timeout: float
    
    def __post_init__(self):
        """Ensure directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.task_cache_dir, exist_ok=True)

class ChainConfigManager:
    """Manages MainChain and TestChain configurations"""
    
    @staticmethod
    def get_main_config() -> ChainConfig:
        """Production MainChain configuration"""
        return ChainConfig(
            chain_type=ChainType.MAIN,
            chain_id="vanta-echo-nebula-main-1",
            network_name="Nebula MainNet",
            
            # Conservative production settings
            difficulty=4,
            mining_reward=10,
            max_reorg_depth=100,
            block_time_target=30.0,  # 30 second blocks
            
            # Production consensus
            validators=[
                "MainValidator01", "MainValidator02", "MainValidator03",
                "MainValidator04", "MainValidator05", "MainValidator06",
                "MainValidator07", "MainValidator08", "MainValidator09",
                "MainValidator10"
            ],
            min_validators=7,
            consensus_threshold=0.67,
            
            # Task Fabric - production limits
            min_task_complexity=1,
            max_task_complexity=1000,
            task_verification_timeout=300.0,  # 5 minutes
            min_redundancy=2,
            
            # Economic parameters
            base_reward=10,
            compute_reward_multiplier=2.0,
            bandwidth_reward_rate=0.001,  # per MB
            slashing_penalty_rate=0.1,    # 10% penalty
            
            # Network - production ports
            p2p_port=26656,
            rpc_port=26657,
            api_port=8080,
            max_peers=50,
            
            # Storage
            data_dir="./data/mainchain",
            blockchain_file="./data/mainchain/blockchain_main.json",
            task_cache_dir="./data/mainchain/tasks",
            
            # dVPN
            dvpn_enabled=True,
            bandwidth_proof_interval=60.0,  # 1 minute intervals
            tunnel_timeout=3600.0,          # 1 hour timeout
        )
    
    @staticmethod
    def get_test_config() -> ChainConfig:
        """Development TestChain configuration"""
        return ChainConfig(
            chain_type=ChainType.TEST,
            chain_id="vanta-echo-nebula-test-1",
            network_name="Nebula TestNet",
            
            # Fast development settings
            difficulty=1,
            mining_reward=100,  # Higher for testing
            max_reorg_depth=10,
            block_time_target=5.0,   # 5 second blocks for fast testing
            
            # Test consensus - fewer validators
            validators=[
                "TestValidator01", "TestValidator02", "TestValidator03",
                "TestValidator04", "TestValidator05"
            ],
            min_validators=3,
            consensus_threshold=0.6,
            
            # Task Fabric - relaxed for testing
            min_task_complexity=1,
            max_task_complexity=100,
            task_verification_timeout=30.0,   # 30 seconds
            min_redundancy=1,                 # Single result OK for testing
            
            # Economic parameters - accelerated
            base_reward=50,
            compute_reward_multiplier=5.0,
            bandwidth_reward_rate=0.01,       # 10x higher for testing
            slashing_penalty_rate=0.05,       # 5% penalty (gentler)
            
            # Network - test ports (avoid conflicts)
            p2p_port=36656,
            rpc_port=36657,
            api_port=9080,
            max_peers=10,
            
            # Storage - separate test data
            data_dir="./data/testchain",
            blockchain_file="./data/testchain/blockchain_test.json",
            task_cache_dir="./data/testchain/tasks",
            
            # dVPN - faster intervals for testing
            dvpn_enabled=True,
            bandwidth_proof_interval=10.0,    # 10 second intervals
            tunnel_timeout=300.0,             # 5 minute timeout
        )
    
    @staticmethod
    def get_config(chain_type: ChainType = None, from_env: bool = True) -> ChainConfig:
        """Get configuration based on environment or explicit type"""
        
        if from_env:
            # Check environment variable
            env_chain = os.getenv('NEBULA_CHAIN', 'test').lower()
            if env_chain == 'main':
                chain_type = ChainType.MAIN
            else:
                chain_type = ChainType.TEST
        
        if chain_type is None:
            chain_type = ChainType.TEST  # Default to test for safety
            
        if chain_type == ChainType.MAIN:
            return ChainConfigManager.get_main_config()
        else:
            return ChainConfigManager.get_test_config()
    
    @staticmethod
    def set_chain_environment(chain_type: ChainType):
        """Set environment variable for chain selection"""
        os.environ['NEBULA_CHAIN'] = chain_type.value
        print(f"🔗 Switched to {chain_type.value.upper()} chain environment")

class ChainValidator:
    """Validates chain configurations and prevents cross-chain contamination"""
    
    @staticmethod
    def validate_config(config: ChainConfig) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if config.difficulty < 1:
            issues.append("Difficulty must be >= 1")
            
        if config.mining_reward <= 0:
            issues.append("Mining reward must be > 0")
            
        if len(config.validators) < config.min_validators:
            issues.append(f"Need at least {config.min_validators} validators")
            
        if config.consensus_threshold <= 0.5 or config.consensus_threshold > 1.0:
            issues.append("Consensus threshold must be > 0.5 and <= 1.0")
            
        if config.p2p_port == config.rpc_port or config.p2p_port == config.api_port:
            issues.append("Network ports must be unique")
            
        return issues
    
    @staticmethod
    def ensure_chain_separation(main_config: ChainConfig, test_config: ChainConfig):
        """Ensure MainChain and TestChain don't interfere with each other"""
        
        # Check port conflicts
        main_ports = {main_config.p2p_port, main_config.rpc_port, main_config.api_port}
        test_ports = {test_config.p2p_port, test_config.rpc_port, test_config.api_port}
        
        if main_ports & test_ports:
            raise ValueError("Port conflict between MainChain and TestChain")
            
        # Check data directory separation  
        if main_config.data_dir == test_config.data_dir:
            raise ValueError("MainChain and TestChain must use separate data directories")
            
        # Check chain ID uniqueness
        if main_config.chain_id == test_config.chain_id:
            raise ValueError("MainChain and TestChain must have unique chain IDs")

# Environment helpers
def use_main_chain():
    """Switch to MainChain for production operations"""
    ChainConfigManager.set_chain_environment(ChainType.MAIN)
    return ChainConfigManager.get_main_config()

def use_test_chain():
    """Switch to TestChain for development/testing"""
    ChainConfigManager.set_chain_environment(ChainType.TEST)
    return ChainConfigManager.get_test_config()

def get_current_chain() -> ChainConfig:
    """Get currently active chain configuration"""
    return ChainConfigManager.get_config()

def is_main_chain() -> bool:
    """Check if currently using MainChain"""
    return get_current_chain().chain_type == ChainType.MAIN

def is_test_chain() -> bool:
    """Check if currently using TestChain"""
    return get_current_chain().chain_type == ChainType.TEST

# Validation on import
if __name__ == "__main__":
    # Validate both configurations
    main_config = ChainConfigManager.get_main_config()
    test_config = ChainConfigManager.get_test_config()
    
    validator = ChainValidator()
    
    main_issues = validator.validate_config(main_config)
    test_issues = validator.validate_config(test_config)
    
    if main_issues:
        print("⚠️ MainChain configuration issues:")
        for issue in main_issues:
            print(f"  - {issue}")
    else:
        print("✅ MainChain configuration valid")
        
    if test_issues:
        print("⚠️ TestChain configuration issues:")
        for issue in test_issues:
            print(f"  - {issue}")
    else:
        print("✅ TestChain configuration valid")
    
    try:
        validator.ensure_chain_separation(main_config, test_config)
        print("✅ Chain separation validated")
    except ValueError as e:
        print(f"❌ Chain separation issue: {e}")
    
    # Display current configuration
    current = get_current_chain()
    print(f"\n🔗 Current active chain: {current.network_name} ({current.chain_id})")
    print(f"   API Port: {current.api_port}")
    print(f"   Data Directory: {current.data_dir}")
    print(f"   Block Time: {current.block_time_target}s")
    print(f"   Mining Reward: {current.mining_reward}")