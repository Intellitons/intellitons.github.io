"""Effective Field Theory (EFT) Renormalization Module.

Based on Raju & Netrapalli (Jan 2026) "A model of errors in transformers"
which proved that LLM behavior collapses into ~2 macroscopic EFT parameters.

This module:
1. Tracks the renormalization group (RG) flow of Intelliton masses across layers
2. Identifies bare vs renormalized (effective) mass
3. Detects phase transitions (grokking = mass gap opening)
4. Computes the 2 EFT parameters that govern the model's error behavior
5. Maps the RG flow to identify stable vs unstable Intelliton resonances
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from src.config import IntellitonConfig
from src.lattice_field import LatticeField, SpinDecomposition, PropagatorResult

logger = logging.getLogger(__name__)


@dataclass
class RenormalizationFlow:
    """RG flow of an Intelliton across layers (energy scales)."""
    # Running coupling / mass at each layer
    running_mass: torch.Tensor        # [L+1, n_modes]
    # Beta function: dm/dl
    beta_function: torch.Tensor       # [L, n_modes]
    # Fixed points (layers where beta ~ 0)
    fixed_point_layers: torch.Tensor  # [n_modes]
    # Anomalous dimensions
    anomalous_dimension: torch.Tensor # [n_modes]
    # Stability at fixed point: eigenvalue of linearized RG
    stability: torch.Tensor           # [n_modes] -- negative = stable (IR attractive)
    # Classification
    fixed_point_type: List[str]       # "UV" / "IR" / "crossover" / "none"


@dataclass
class EFTParameters:
    """The 2 effective parameters governing the model per Raju & Netrapalli.

    In their EFT framework:
      - alpha: overall error scale (analogous to coupling constant)
      - beta: error-type ratio (analogous to mixing angle)

    These govern how bare Intelliton masses renormalize to physical masses.
    """
    alpha: float                       # Overall error/signal scale
    beta: float                        # Error type ratio
    # Per-mode: bare mass vs renormalized mass
    bare_mass: torch.Tensor           # [n_modes] -- mass at early layers
    renormalized_mass: torch.Tensor   # [n_modes] -- mass at late layers
    mass_shift: torch.Tensor          # [n_modes] -- renormalized - bare
    # Wavefunction renormalization Z
    Z: torch.Tensor                   # [n_modes] -- sigma_late / sigma_early


@dataclass
class PhaseTransition:
    """Detected phase transition (potential grokking signature)."""
    layer: int                         # layer where transition occurs
    order_parameter_jump: float        # magnitude of the jump
    mass_gap_before: float            # mass gap before transition
    mass_gap_after: float             # mass gap after transition
    transition_type: str              # "mass_gap_opening" / "symmetry_breaking" / "crossover"


class EFTRenormalization:
    """EFT renormalization analysis of Intelliton spectrum."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg

    def compute_rg_flow(
        self,
        decomp: SpinDecomposition,
        propagator: PropagatorResult,
    ) -> RenormalizationFlow:
        """Compute the RG flow of Intelliton masses across layers.

        The layer dimension acts as an RG scale (UV -> IR):
        - Early layers = high energy (UV) = bare parameters
        - Late layers = low energy (IR) = renormalized/physical parameters

        Running mass m(l) computed from local exponential decay rate.
        Beta function beta(l) = dm/dl.
        Fixed points where beta = 0 correspond to stable Intellitons.
        """
        sigma = decomp.sigma.float()  # [L+1, n_modes]
        n_layers = sigma.shape[0]
        n_modes = sigma.shape[1]
        w = self.cfg.eft_window_size
        eps = 1e-10

        # Signed running mass: m = -d(log sigma)/dl
        # Positive = decaying (normal mass), Negative = growing (tachyonic)
        running_mass = torch.zeros(n_layers, n_modes)
        for l in range(n_layers):
            l_s = max(0, l - w // 2)
            l_e = min(n_layers, l + w // 2 + 1)
            if l_e - l_s < 2:
                continue
            for i in range(n_modes):
                window = sigma[l_s:l_e, i].clamp(min=eps)
                log_w = torch.log(window)
                x = torch.arange(l_e - l_s, dtype=torch.float32)
                xm = x.mean()
                ym = log_w.mean()
                denom = ((x - xm) ** 2).sum().clamp(min=eps)
                slope = ((x - xm) * (log_w - ym)).sum() / denom
                running_mass[l, i] = -slope.item()

        # Smooth the running mass for cleaner beta function
        for i in range(n_modes):
            m_np = running_mass[:, i].numpy()
            if len(m_np) > 7:
                m_smooth = savgol_filter(m_np, min(7, len(m_np) // 2 * 2 + 1), 2)
                running_mass[:, i] = torch.from_numpy(m_smooth)

        # Beta function
        beta = running_mass[1:] - running_mass[:-1]

        # Fixed points: minimize |beta| in the IR region (last 2/3 of layers)
        fp_start = n_layers // 3
        fixed_points = torch.zeros(n_modes, dtype=torch.long)
        for i in range(n_modes):
            abs_beta = beta[fp_start:, i].abs()
            fixed_points[i] = fp_start + abs_beta.argmin().item()

        # Anomalous dimension: gamma = d(log sigma)/d(log l)
        anomalous = torch.zeros(n_modes)
        mid_s = n_layers // 3
        mid_e = 2 * n_layers // 3
        for i in range(n_modes):
            log_sigma = torch.log(sigma[mid_s:mid_e, i].clamp(min=eps))
            log_l = torch.log(torch.arange(mid_s, mid_e, dtype=torch.float32) + 1)
            x, y = log_l, log_sigma
            xm, ym = x.mean(), y.mean()
            slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=eps)
            anomalous[i] = slope.item()

        # Stability: eigenvalue of linearized RG around fixed point
        # Approximate by slope of beta at the fixed point
        stability = torch.zeros(n_modes)
        for i in range(n_modes):
            fp = fixed_points[i].item()
            if 1 <= fp < len(beta) - 1:
                # d(beta)/dl at fixed point
                db = beta[fp + 1, i] - beta[fp - 1, i]
                stability[i] = db.item() / 2.0
            else:
                stability[i] = 0.0

        # Classify fixed point type
        fp_types = []
        for i in range(n_modes):
            s = stability[i].item()
            fp = fixed_points[i].item()
            beta_val = abs(beta[min(fp, max(0, len(beta) - 1)), i].item())
            
            # True fixed point requires beta(m) ~ 0
            if beta_val > 0.01:
                fp_types.append("none")
            elif s < -0.01:
                fp_types.append("IR")     # stable in IR = physical Intelliton
            elif s > 0.01:
                fp_types.append("UV")     # stable in UV = high-energy resonance
            else:
                fp_types.append("crossover")

        return RenormalizationFlow(
            running_mass=running_mass,
            beta_function=beta,
            fixed_point_layers=fixed_points,
            anomalous_dimension=anomalous,
            stability=stability,
            fixed_point_type=fp_types,
        )

    def compute_eft_parameters(
        self,
        decomp: SpinDecomposition,
        propagator: PropagatorResult,
    ) -> EFTParameters:
        """Extract the 2 EFT parameters from the Intelliton spectrum.

        Following Raju & Netrapalli (2026):
        - alpha = overall signal-to-noise ratio (spectral gap / total variance)
        - beta = ratio of systematic to random error modes

        In Intelliton language:
        - alpha ~ mean(log(sigma_top / sigma_bottom)) = spectral hierarchy
        - beta ~ mass_gap / mean_mass = how distinct the Intelliton masses are

        Also compute bare vs renormalized mass and wavefunction renormalization Z.
        """
        sigma = decomp.sigma.float()
        n_modes = sigma.shape[1]
        n_layers = sigma.shape[0]

        # Trace mode indices across layers properly
        matched = getattr(decomp, 'matched_indices', None)
        matched_sigma = torch.zeros_like(sigma)
        matched_sigma[0] = sigma[0]
        current_idx = torch.arange(n_modes)
        for l in range(n_layers - 1):
            if matched is not None and l < len(matched):
                # Bounds check to prevent dimension mismatch
                valid_mask = current_idx < matched[l].shape[0]
                # Default mapping (identity) handles any out of bounds
                next_idx = current_idx.clone()
                next_idx[valid_mask] = matched[l][current_idx[valid_mask]]
                
                # Check target bounds
                valid_target = next_idx < sigma.shape[1]
                next_idx[~valid_target] = current_idx[~valid_target] # Fallback to identity
                current_idx = next_idx

            for i in range(n_modes):
                if current_idx[i] < sigma.shape[1]:
                    matched_sigma[l + 1, i] = sigma[l + 1, current_idx[i]]
                else:
                    matched_sigma[l + 1, i] = sigma[l + 1, min(i, sigma.shape[1]-1)]
                    
        # Override sigma to use the properly traced layer-modes
        sigma = matched_sigma
        eps = 1e-10

        # alpha: spectral hierarchy
        # Ratio of dominant to subdominant mode amplitudes, averaged
        sigma_ratio = sigma[:, 0] / (sigma[:, min(1, n_modes - 1)].clamp(min=eps))
        alpha = torch.log(sigma_ratio.mean()).item()

        # beta: mass gap ratio
        masses = propagator.mass
        mass_sorted = masses.sort().values
        if len(mass_sorted) >= 2:
            mass_gap = (mass_sorted[1] - mass_sorted[0]).item()
            mean_mass = masses.mean().item() + eps
            beta_param = mass_gap / mean_mass
        else:
            beta_param = 0.0

        # Bare mass (early layers) vs renormalized mass (late layers)
        early_end = n_layers // 4
        late_start = 3 * n_layers // 4

        bare_mass = torch.zeros(n_modes)
        renorm_mass = torch.zeros(n_modes)
        Z = torch.zeros(n_modes)

        for i in range(n_modes):
            # Bare mass from early layers: m = -slope (signed)
            early_sigma = sigma[:early_end, i].clamp(min=eps)
            if len(early_sigma) >= 2:
                log_s = torch.log(early_sigma)
                t_e = torch.arange(len(early_sigma), dtype=torch.float32)
                slope_e = self._linear_slope(t_e, log_s)
                bare_mass[i] = -slope_e

            # Renormalized mass from late layers: m = -slope (signed)
            late_sigma = sigma[late_start:, i].clamp(min=eps)
            if len(late_sigma) >= 2:
                log_s_l = torch.log(late_sigma)
                t_l = torch.arange(len(late_sigma), dtype=torch.float32)
                slope_l = self._linear_slope(t_l, log_s_l)
                renorm_mass[i] = -slope_l

            # Wavefunction renormalization: Z = sigma_late / sigma_early
            Z[i] = sigma[late_start:, i].mean() / (sigma[:early_end, i].mean() + eps)

        return EFTParameters(
            alpha=alpha,
            beta=beta_param,
            bare_mass=bare_mass,
            renormalized_mass=renorm_mass,
            mass_shift=renorm_mass - bare_mass,
            Z=Z,
        )

    def detect_phase_transitions(
        self,
        decomp: SpinDecomposition,
    ) -> List[PhaseTransition]:
        """Detect phase transitions across layers.

        Phase transitions manifest as:
        1. Sudden changes in the mass gap between modes
        2. Symmetry breaking: one mode suddenly becoming dominant
        3. Order parameter jumps in the singular value spectrum

        These correspond to "grokking" events where the network
        suddenly learns a structured representation.
        """
        sigma = decomp.sigma.float()
        n_layers = sigma.shape[0]
        n_modes = min(sigma.shape[1], 5)  # focus on top modes
        eps = 1e-10
        transitions = []

        # Order parameter: mass gap between mode 0 and mode 1
        mass_gap = torch.zeros(n_layers)
        for l in range(n_layers):
            if n_modes >= 2:
                mass_gap[l] = (sigma[l, 0] - sigma[l, 1]).item()

        # Detect jumps in mass gap
        if n_layers > 5:
            mg_np = mass_gap.numpy()
            mg_smooth = savgol_filter(mg_np, min(5, len(mg_np)), 1)
            mg_deriv = np.gradient(mg_smooth)

            # Find layers where |d(mass_gap)/dl| is large
            threshold = np.std(mg_deriv) * 2.0
            for l in range(2, n_layers - 2):
                if abs(mg_deriv[l]) > threshold:
                    transitions.append(PhaseTransition(
                        layer=l,
                        order_parameter_jump=abs(mg_deriv[l]),
                        mass_gap_before=float(mg_smooth[max(0, l - 2)]),
                        mass_gap_after=float(mg_smooth[min(n_layers - 1, l + 2)]),
                        transition_type=self._classify_transition(
                            mg_smooth, l, mg_deriv[l]
                        ),
                    ))

        return transitions

    def _classify_transition(
        self, mass_gap: np.ndarray, layer: int, derivative: float
    ) -> str:
        """Classify a phase transition type."""
        if derivative > 0:
            return "mass_gap_opening"
        elif derivative < 0:
            return "symmetry_breaking"
        else:
            return "crossover"

    def _linear_slope(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Simple linear regression slope."""
        xm = x.mean()
        ym = y.mean()
        denom = ((x - xm) ** 2).sum().clamp(min=1e-10)
        return float(((x - xm) * (y - ym)).sum() / denom)
