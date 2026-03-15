"""Configuration for the Intelliton Spectrum Analyzer.

Lattice Gauge Theory framework:
  - Discrete propagator mass from pole extraction (not just exponential fit)
  - Lattice dispersion relation E(k) = 2*sinh^{-1}(m_lat/2) matching
  - RoPE as U(1) gauge field with helicity ablation
  - Non-abelian Wilson loops and Polyakov loops
  - EFT renormalization flow across layers

Supported models (auto-configured via config.json):
  - Qwen3-4B-Base, Qwen3-8B-Base
  - Qwen3-4B (Instruct), Qwen3-8B (Instruct)
  - Mistral-7B-v0.3
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
import os


@dataclass
class IntellitonConfig:
    # Model (overridden by --model-path; auto_configure_from_model reads config.json)
    model_path: str = ""
    device_map: str = "cuda:0"
    dtype: str = "bfloat16"

    # Architecture constants (populated by auto_configure_from_model)
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    vocab_size: int = 32768

    # Full-attention layer list (rebuilt by auto_configure_from_model)
    FULL_ATTN_LAYERS: List[int] = field(
        default_factory=lambda: list(range(32))
    )

    # RoPE parameters (populated by auto_configure_from_model)
    rope_theta: float = 1000000.0
    rope_dim: int = 128
    rope_scaling_type: str = ""  # auto-detected from config.json

    # Analysis parameters
    analysis_seq_len: int = 128
    n_top_modes: int = 20
    max_prompts_per_category: Optional[int] = None
    sv_threshold_ratio: float = 0.02
    merge_cosine_threshold: float = 0.97
    mass_fit_skip_layers: int = 5

    # Lattice gauge theory parameters (v5 new)
    lattice_spacing: float = 1.0  # a=1 (in token units)
    propagator_fit_method: str = "pole"  # "pole" or "exponential"
    wilson_loop_max_area: int = 4  # max spatial extent for Wilson loops
    polyakov_loop_layers: List[int] = field(
        default_factory=lambda: [0, 9, 18, 27, 35]
    )

    # RoPE ablation parameters (v5 new)
    rope_ablation_n_dims: int = 8  # number of RoPE dim groups to ablate
    rope_ablation_group_size: int = 16  # dims per ablation group

    # EFT renormalization parameters (v5 new)
    eft_window_size: int = 4  # rolling window for RG flow
    eft_n_effective_params: int = 2  # per Raju & Netrapalli 2026

    # Steering / intervention
    steering_alpha: float = 10.0
    steering_layer: int = 18  # middle of 36-layer model
    max_species_for_intervention: Optional[int] = None
    n_generate_tokens: int = 50

    # Vacuum analysis (v5 new)
    vacuum_enabled: bool = True

    # Fusion tracking (v5 new)
    fusion_tracking_enabled: bool = True

    # Gauge covariance testing (v5 new)
    gauge_covariance_enabled: bool = True
    gauge_covariance_layers: List[int] = field(
        default_factory=lambda: [0, 9, 18, 27, 35]
    )
    gauge_covariance_n_prompts: int = 2

    # Topological charge (v5 new)
    topological_charge_enabled: bool = True
    topological_quantization_threshold: float = 0.25

    # Resonance width (Lorentzian fitting) (v5 new)
    resonance_width_enabled: bool = True
    resonance_sharpness_threshold: float = 0.5  # max Gamma/omega_0 for "well-defined"

    # Differential spectroscopy (v5 experimental)
    differential_spectroscopy_enabled: bool = True
    diffspec_max_prompts_per_category: Optional[int] = 8
    diffspec_category_weight: float = 0.60
    diffspec_global_weight: float = 0.25
    diffspec_vacuum_weight: float = 0.15
    diffspec_peak_std_threshold: float = 1.5
    diffspec_min_recurrence: float = 0.30
    diffspec_min_supporting_prompts: int = 2
    diffspec_cluster_bins: int = 3

    # Hallucination diagnostics (v5 new)
    hallucination_diagnostic_enabled: bool = True

    # Output
    output_dir: str = "./results"
    figure_dpi: int = 150
    figure_format: str = "png"
    save_tensors: bool = True
    figure_style: str = "journal"
    figure_width_single: float = 3.4
    figure_width_double: float = 7.0
    figure_height_base: float = 2.6

    @property
    def model_name(self) -> str:
        """Short human-readable model name derived from model_path."""
        return os.path.basename(self.model_path.rstrip("/")) if self.model_path else "unknown"

    @property
    def total_field_layers(self) -> int:
        """Number of field layers: embedding + decoder layers."""
        return self.num_layers + 1

    @property
    def rope_pairs(self) -> int:
        """Number of RoPE frequency pairs per head."""
        return self.rope_dim // 2

    def auto_configure_from_model(self):
        """Read model config.json and update architecture parameters.

        Works for any HuggingFace-compatible model directory.  Falls back
        to sensible defaults when a field is missing (e.g. Mistral does
        not expose ``head_dim`` explicitly).
        """
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as f:
            mc = json.load(f)
        self.num_layers = mc.get("num_hidden_layers", self.num_layers)
        self.hidden_size = mc.get("hidden_size", self.hidden_size)
        self.num_attention_heads = mc.get("num_attention_heads", self.num_attention_heads)
        self.num_kv_heads = mc.get("num_key_value_heads", self.num_kv_heads)
        self.intermediate_size = mc.get("intermediate_size", self.intermediate_size)
        self.vocab_size = mc.get("vocab_size", self.vocab_size)
        self.rope_theta = mc.get("rope_theta", self.rope_theta)

        # head_dim: some models (e.g. Mistral) omit it; compute from hidden_size
        explicit_head_dim = mc.get("head_dim")
        if explicit_head_dim is not None:
            self.head_dim = explicit_head_dim
        else:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # rope_dim defaults to head_dim
        self.rope_dim = self.head_dim

        # Detect rope scaling type from config
        rope_scaling = mc.get("rope_scaling")
        if rope_scaling and isinstance(rope_scaling, dict):
            self.rope_scaling_type = rope_scaling.get("type", "")
        else:
            self.rope_scaling_type = ""

        # Update dependent fields
        self.FULL_ATTN_LAYERS = list(range(self.num_layers))
        self.steering_layer = self.num_layers // 2
        self.polyakov_loop_layers = [
            0,
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers - 1,
        ]
        self.gauge_covariance_layers = self.polyakov_loop_layers
