"""Configuration and constants for PyRelRepair."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # LLM settings
    ollama_model: str = "qwen2.5-coder:32b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 1.0
    llm_timeout: int = 300  # seconds per LLM request

    # BaseRepair
    base_num_patches: int = 1

    # SigRepair
    sig_num_iterations: int = 20
    sig_num_patches_per_iter: int = 1
    sig_top_k: int = 25
    sig_num_root_causes: int = 2
    sig_num_candidate_functions: int = 5
    embed_model: str = "nomic-embed-text"

    # SnipRepair
    snip_num_patches_per_snippet: int = 10
    snip_top_k_intra: int = 15
    snip_top_k_inter: int = 15
    snip_top_similar_files: int = 5
    snip_alpha: float = 0.5
    snip_beta: float = 0.5

    # Validation
    test_timeout: int = 300  # seconds per pytest run
    repair_timeout: int = 18000  # 5 hours total per bug

    # Paths
    condefects_dir: Path = field(default_factory=lambda: Path("data/condefects"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
