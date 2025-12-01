"""
Training configuration system for Blokus RL.

Supports both smoke-test mode (for quick verification) and full training mode.
Configuration can be provided via CLI arguments or YAML/JSON config files.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from typing import Literal
except ImportError:
    # Python < 3.8 compatibility
    try:
        from typing_extensions import Literal
    except ImportError:
        # Fallback for very old Python
        Literal = str

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TrainingConfig:
    """
    Structured training configuration.
    
    Attributes:
        mode: Training mode - "smoke" for quick tests, "full" for production training
        max_episodes: Maximum number of episodes to run (None = unlimited, use total_timesteps)
        max_steps_per_episode: Maximum steps per episode before truncation
        total_timesteps: Total timesteps for training (used by SB3)
        n_steps: Number of steps to collect per update (SB3 parameter)
        learning_rate: Learning rate for the RL algorithm
        batch_size: Batch size for training
        logging_verbosity: Logging level (0=ERROR, 1=INFO, 2=DEBUG)
        random_seed: Random seed for reproducibility (None = no seed)
        env_seed: Optional separate seed for environment
        agent_seed: Optional separate seed for agent
        checkpoint_dir: Directory to save model checkpoints
        tensorboard_log_dir: Directory for TensorBoard logs
        checkpoint_interval: Save checkpoint every N episodes (None = only at end) [DEPRECATED: use checkpoint_interval_episodes]
        checkpoint_interval_episodes: Save checkpoint every N episodes (default: 50, None = only at end)
        keep_last_n_checkpoints: Number of recent checkpoints to keep (default: 3)
        resume_from_checkpoint: Path to checkpoint file to resume from (None = start fresh)
        agent_config_path: Path to agent hyperparameter config file (None = use defaults)
        enable_sanity_checks: Enable sanity checks (NaN/Inf detection, etc.)
        log_action_details: Log detailed action information (only in smoke mode or if explicitly enabled)
        disable_distribution_validation: Disable PyTorch distribution validation (needed for large action spaces)
    """
    
    mode: Literal["smoke", "full"] = "full"
    max_episodes: Optional[int] = None
    max_steps_per_episode: int = 1000
    total_timesteps: int = 100000
    n_steps: int = 2048
    learning_rate: float = 3e-4
    batch_size: int = 64
    logging_verbosity: int = 1
    random_seed: Optional[int] = None
    env_seed: Optional[int] = None
    agent_seed: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    tensorboard_log_dir: str = "./logs"
    checkpoint_interval: Optional[int] = None  # Deprecated, use checkpoint_interval_episodes
    checkpoint_interval_episodes: Optional[int] = 50
    keep_last_n_checkpoints: int = 3
    resume_from_checkpoint: Optional[str] = None
    agent_config_path: Optional[str] = None
    enable_sanity_checks: bool = True
    log_action_details: bool = False
    disable_distribution_validation: bool = True  # Default True for large action space (36400)
    
    def __post_init__(self):
        """Apply smoke-test mode defaults if mode is 'smoke'."""
        # Handle deprecated checkpoint_interval
        if self.checkpoint_interval is not None and self.checkpoint_interval_episodes == 50:
            # If checkpoint_interval is set but checkpoint_interval_episodes is default, use the old value
            self.checkpoint_interval_episodes = self.checkpoint_interval
        
        if self.mode == "smoke":
            # Override defaults for smoke-test mode
            if self.max_episodes is None:
                self.max_episodes = 5
            if self.max_steps_per_episode > 100:
                self.max_steps_per_episode = 100
            if self.total_timesteps > 10000:
                self.total_timesteps = 10000
            if self.n_steps > 512:
                self.n_steps = 512
            if self.logging_verbosity < 2:
                self.logging_verbosity = 2  # DEBUG level
            if not self.log_action_details:
                self.log_action_details = True
            if not self.enable_sanity_checks:
                self.enable_sanity_checks = True
            # Disable periodic checkpointing in smoke mode (too short)
            if self.checkpoint_interval_episodes is not None:
                self.checkpoint_interval_episodes = None
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        # Filter out None values and unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "TrainingConfig":
        """Load config from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
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
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_file(self, config_path: Path):
        """Save config to YAML or JSON file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError(
                        "PyYAML is required for YAML config files. "
                        "Install with: pip install pyyaml"
                    )
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2, sort_keys=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def log_config(self, logger: logging.Logger):
        """Log the effective configuration."""
        logger.info("=" * 80)
        logger.info("Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Max Episodes: {self.max_episodes if self.max_episodes else 'Unlimited (using total_timesteps)'}")
        logger.info(f"Max Steps per Episode: {self.max_steps_per_episode}")
        logger.info(f"Total Timesteps: {self.total_timesteps}")
        logger.info(f"N Steps (per update): {self.n_steps}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Logging Verbosity: {self.logging_verbosity}")
        logger.info(f"Random Seed: {self.random_seed}")
        logger.info(f"Env Seed: {self.env_seed if self.env_seed is not None else 'Same as random_seed'}")
        logger.info(f"Agent Seed: {self.agent_seed if self.agent_seed is not None else 'Same as random_seed'}")
        logger.info(f"Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"TensorBoard Log Dir: {self.tensorboard_log_dir}")
        logger.info(f"Checkpoint Interval: {self.checkpoint_interval_episodes if self.checkpoint_interval_episodes else 'End of training only'}")
        logger.info(f"Keep Last N Checkpoints: {self.keep_last_n_checkpoints}")
        logger.info(f"Resume From Checkpoint: {self.resume_from_checkpoint if self.resume_from_checkpoint else 'None (start fresh)'}")
        logger.info(f"Agent Config: {self.agent_config_path if self.agent_config_path else 'None (use defaults)'}")
        logger.info(f"Sanity Checks: {self.enable_sanity_checks}")
        logger.info(f"Log Action Details: {self.log_action_details}")
        logger.info("=" * 80)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO agent on Blokus environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode and basic training parameters
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "full"],
        default="full",
        help="Training mode: 'smoke' for quick verification, 'full' for production training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON config file (overrides CLI args)"
    )
    
    # Episode and step limits
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes (None = unlimited, use total_timesteps). In smoke mode, defaults to 5."
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=1000,
        help="Maximum steps per episode before truncation. In smoke mode, defaults to 100."
    )
    
    # SB3 training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total number of timesteps to train. In smoke mode, defaults to 10000."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to collect per update. In smoke mode, defaults to 512."
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=3e-4,
        dest="learning_rate",
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    
    # Logging and debugging
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=ERROR, 1=INFO, 2=DEBUG"
    )
    parser.add_argument(
        "--log-action-details",
        action="store_true",
        help="Log detailed action information (auto-enabled in smoke mode)"
    )
    
    # Seeds
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (sets all seeds if env_seed/agent_seed not specified)"
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Separate seed for environment (overrides --seed for env only)"
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        default=None,
        help="Separate seed for agent (overrides --seed for agent only)"
    )
    
    # Checkpoints and logging
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        dest="checkpoint_interval_episodes",
        help="[DEPRECATED] Use --checkpoint-interval-episodes. Save checkpoint every N episodes (None = only at end)"
    )
    parser.add_argument(
        "--checkpoint-interval-episodes",
        type=int,
        default=50,
        help="Save checkpoint every N episodes (default: 50, None = only at end)"
    )
    parser.add_argument(
        "--keep-last-n-checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep (default: 3)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default=None,
        dest="agent_config_path",
        help="Path to agent hyperparameter config file (YAML/JSON)"
    )
    
    # Sanity checks
    parser.add_argument(
        "--disable-sanity-checks",
        action="store_true",
        help="Disable sanity checks (NaN/Inf detection, etc.)"
    )
    
    return parser


def parse_args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert parsed arguments to TrainingConfig."""
    # If config file is provided, load it first
    if args.config:
        config = TrainingConfig.from_file(Path(args.config))
    else:
        config = TrainingConfig()
    
    # Override with CLI arguments (only if explicitly provided)
    if args.mode:
        config.mode = args.mode
    
    if args.max_episodes is not None:
        config.max_episodes = args.max_episodes
    if args.max_steps_per_episode is not None:
        config.max_steps_per_episode = args.max_steps_per_episode
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps
    if args.n_steps is not None:
        config.n_steps = args.n_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    config.logging_verbosity = args.verbosity
    if args.log_action_details:
        config.log_action_details = True
    
    if args.seed is not None:
        config.random_seed = args.seed
        # If env_seed/agent_seed not explicitly set, use random_seed
        if config.env_seed is None:
            config.env_seed = args.seed
        if config.agent_seed is None:
            config.agent_seed = args.seed
    
    if args.env_seed is not None:
        config.env_seed = args.env_seed
    if args.agent_seed is not None:
        config.agent_seed = args.agent_seed
    
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.tensorboard_log_dir:
        config.tensorboard_log_dir = args.tensorboard_log_dir
    if args.checkpoint_interval_episodes is not None:
        config.checkpoint_interval_episodes = args.checkpoint_interval_episodes
    if args.keep_last_n_checkpoints is not None:
        config.keep_last_n_checkpoints = args.keep_last_n_checkpoints
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.agent_config_path:
        config.agent_config_path = args.agent_config_path
    
    if args.disable_sanity_checks:
        config.enable_sanity_checks = False
    
    # Apply smoke-test mode defaults after all overrides
    config.__post_init__()
    
    return config

