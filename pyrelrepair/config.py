"""Configuration and constants for PyRelRepair."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # LLM settings
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 1.0

    # BaseRepair
    base_num_patches: int = 1

    # Validation
    test_timeout: int = 300  # seconds per pytest run
    repair_timeout: int = 18000  # 5 hours total per bug

    # Paths
    condefects_dir: Path = field(default_factory=lambda: Path("data/condefects"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
