from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Base training configuration."""
    model_name: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048


class BaseTrainer(ABC):
    """Abstract interface for all trainers (SFT, DPO, ORPO, PPO...)."""

    @abstractmethod
    def setup(self, config: TrainingConfig) -> None:
        """Load model, tokenizer, and dataset."""
        ...

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training loop. Returns metrics dict."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save trained model/adapter."""
        ...

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on held-out set."""
        ...
