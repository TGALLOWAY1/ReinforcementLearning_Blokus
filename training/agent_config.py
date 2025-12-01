"""
Agent configuration loader and validator.

This module handles loading and validating agent hyperparameter configurations
from YAML/JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class AgentConfig:
    """
    Agent hyperparameter configuration.
    
    This class represents a complete agent configuration including:
    - Agent identification (id, name, algorithm, version)
    - Hyperparameters (learning rate, gamma, network architecture, etc.)
    - Algorithm-specific parameters
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize agent config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        # Required fields
        self.agent_id = config_dict.get("agent_id", "ppo_agent")
        self.name = config_dict.get("name", self.agent_id)
        self.algorithm = config_dict.get("algorithm", "MaskablePPO")
        self.version = config_dict.get("version", 1)
        
        # Optional fields
        self.sweep_variant = config_dict.get("sweep_variant")
        self.description = config_dict.get("description", "")
        
        # Core hyperparameters
        self.learning_rate = config_dict.get("learning_rate", 3e-4)
        self.gamma = config_dict.get("gamma", 0.99)
        self.n_steps = config_dict.get("n_steps", 2048)
        self.batch_size = config_dict.get("batch_size", 64)
        self.n_epochs = config_dict.get("n_epochs", 10)
        
        # Network architecture
        network_config = config_dict.get("network", {})
        self.policy = network_config.get("policy", "MlpPolicy")
        self.net_arch = network_config.get("net_arch", [256, 256])
        self.activation_fn = network_config.get("activation_fn", "tanh")
        
        # PPO-specific parameters
        ppo_config = config_dict.get("ppo", {})
        self.clip_range = ppo_config.get("clip_range", 0.2)
        self.ent_coef = ppo_config.get("ent_coef", 0.01)
        self.vf_coef = ppo_config.get("vf_coef", 0.5)
        self.max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
        
        # Exploration (for future use with other algorithms)
        exploration_config = config_dict.get("exploration", {})
        self.exploration_type = exploration_config.get("type", "none")
        
        # Store raw config for reference
        self.raw_config = config_dict
    
    @classmethod
    def from_file(cls, config_path: Path) -> "AgentConfig":
        """
        Load agent config from YAML or JSON file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            AgentConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "PyYAML is required for YAML config files. "
                        "Install with: pip install pyyaml"
                    )
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        if not isinstance(config_dict, dict):
            raise ValueError("Config file must contain a dictionary/object")
        
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "algorithm": self.algorithm,
            "version": self.version,
            "sweep_variant": self.sweep_variant,
            "description": self.description,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "network": {
                "policy": self.policy,
                "net_arch": self.net_arch,
                "activation_fn": self.activation_fn
            },
            "ppo": {
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm
            },
            "exploration": {
                "type": self.exploration_type
            }
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """
        Get policy_kwargs for Stable-Baselines3.
        
        Returns:
            Dictionary of policy keyword arguments
        """
        # Map activation function string to actual function
        activation_map = {
            "tanh": "tanh",
            "relu": "relu",
            "elu": "elu"
        }
        activation = activation_map.get(self.activation_fn, "tanh")
        
        return {
            "net_arch": [dict(pi=self.net_arch, vf=self.net_arch)],
            "activation_fn": activation
        }
    
    def get_config_name(self) -> str:
        """Get a human-readable config name."""
        if self.sweep_variant:
            return f"{self.name} ({self.sweep_variant})"
        return self.name
    
    def get_config_short_name(self) -> str:
        """Get a short identifier for the config."""
        if self.sweep_variant:
            return f"{self.agent_id}_{self.sweep_variant}"
        return f"{self.agent_id}_v{self.version}"
    
    def log_config(self, logger: logging.Logger):
        """Log the agent configuration."""
        logger.info("=" * 80)
        logger.info("Agent Configuration")
        logger.info("=" * 80)
        logger.info(f"Agent ID: {self.agent_id}")
        logger.info(f"Name: {self.name}")
        logger.info(f"Algorithm: {self.algorithm}")
        logger.info(f"Version: {self.version}")
        if self.sweep_variant:
            logger.info(f"Sweep Variant: {self.sweep_variant}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Gamma: {self.gamma}")
        logger.info(f"N Steps: {self.n_steps}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"N Epochs: {self.n_epochs}")
        logger.info(f"Network Architecture: {self.net_arch}")
        logger.info(f"PPO Clip Range: {self.clip_range}")
        logger.info(f"PPO Entropy Coef: {self.ent_coef}")
        if self.description:
            logger.info(f"Description: {self.description}")
        logger.info("=" * 80)


def load_agent_config(config_path: Optional[Path]) -> Optional[AgentConfig]:
    """
    Load agent config from file, or return None if not provided.
    
    Args:
        config_path: Path to agent config file, or None
        
    Returns:
        AgentConfig instance or None
    """
    if config_path is None:
        return None
    
    try:
        return AgentConfig.from_file(config_path)
    except Exception as e:
        logger.error(f"Failed to load agent config from {config_path}: {e}")
        raise


def find_agent_configs(pattern: str) -> List[Path]:
    """
    Find agent config files matching a pattern.
    
    Args:
        pattern: Glob pattern (e.g., "config/agents/ppo_agent_sweep_*.yaml")
        
    Returns:
        List of matching config file paths
    """
    pattern_path = Path(pattern)
    
    # If pattern is a directory, search for all config files
    if pattern_path.is_dir():
        configs = []
        for ext in ["*.yaml", "*.yml", "*.json"]:
            configs.extend(pattern_path.glob(ext))
        return sorted(configs)
    
    # Otherwise, use glob pattern
    return sorted(Path(".").glob(pattern))

