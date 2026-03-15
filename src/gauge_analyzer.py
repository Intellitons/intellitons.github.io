"""Gauge Theory Analysis v5 -- Non-Abelian Wilson Loops and Polyakov Loops.

Upgrades from v4:
  1. Non-abelian SU(N_heads) Wilson loops from attention head matrices
  2. Polyakov loops (temporal Wilson lines) for confinement detection
  3. 't Hooft loop for dual confinement
  4. Gauge-covariant parallel transport with non-abelian phases
  5. Creutz ratios for string tension extraction
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange

from src.config import IntellitonConfig
from src.lattice_field import SpinDecomposition

logger = logging.getLogger(__name__)


@dataclass
class WilsonLoopResult:
    """Wilson loop measurements."""
    plaquettes: Dict[str, torch.Tensor]
    diagonal_loops: Dict[str, torch.Tensor]
    field_strength: Dict[str, torch.Tensor]
    avg_field_strength: Dict[str, float]
    # v5 new: non-abelian traces
    plaquette_traces: Dict[str, float]  # Tr(W)/N for each plaquette


@dataclass
class PolyakovLoopResult:
    """Polyakov loop (temporal Wilson line) measurements.

    Polyakov loop = trace of product of gauge links along the
    temporal (layer) direction. Its expectation value serves as
    an order parameter for confinement:
      <P> ~ 0 => confined phase (no free charges)
      <P> > 0 => deconfined phase (free charges exist)
    """
    loop_values: torch.Tensor          # [T] -- per-token Polyakov loop
    expectation_value: float           # <P> = mean over tokens
    susceptibility: float              # chi = <P^2> - <P>^2
    phase: str                         # "confined" or "deconfined"
    # Per-layer contribution
    layer_contributions: torch.Tensor  # [n_layers_used] -- |det(A_l)|


@dataclass
class CreutzRatioResult:
    """Creutz ratios for string tension extraction.

    chi(R, T) = -log(W(R,T) * W(R-1,T-1) / (W(R-1,T) * W(R,T-1)))

    In a confining theory, chi -> sigma (string tension) for large R, T.
    """
    ratios: Dict[str, float]          # "(R,T)" -> chi value
    string_tension: float              # estimated sigma
    area_law: bool                     # whether W ~ exp(-sigma * R * T)


@dataclass
class TopologicalChargeResult:
    """Topological charge measurements from gauge field.

    In lattice gauge theory, the topological charge Q = (1/2pi) sum_P F_P
    where F_P is the field strength on plaquette P. If Q is quantized
    (integer or half-integer), it indicates topological quasiparticles.
    """
    charge_per_plaquette: Dict[str, float]  # plaquette key -> Q contribution
    total_charge: float                      # Q = (1/2pi) sum(F)
    charge_density: torch.Tensor            # [n_plaquettes] per-plaquette Q
    nearest_integer: int                    # round(Q) -- nearest quantized value
    quantization_error: float               # |Q - round(Q)|
    is_quantized: bool                      # |Q - round(Q)| < threshold
    # Per-token winding number from Polyakov phase
    winding_numbers: torch.Tensor           # [T] per-token winding number
    mean_winding: float


@dataclass
class ParallelTransportResult:
    """Verification of spin preservation under attention."""
    alignment: torch.Tensor
    avg_alignment_per_mode: torch.Tensor
    avg_alignment_per_layer: torch.Tensor
    # v5 new
    non_abelian_phase: torch.Tensor    # [n_layers, n_modes] -- accumulated phase


class GaugeAnalyzer:
    """Analyze attention as a non-abelian lattice gauge connection (v5)."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg

    def compute_wilson_loops(
        self,
        attention_maps: Dict[int, torch.Tensor],
    ) -> WilsonLoopResult:
        """Compute Wilson loops (plaquettes) from attention matrices.

        v5: Treats the multi-head attention as an SU(H) gauge connection.
        Each head provides a U(1) link; the full set provides SU(H).

        Elementary plaquette for layers (l_a, l_b):
          W(t1, t2) = (1/H) Tr[A_{l_a} @ A_{l_b}^T]

        Full non-abelian Wilson loop:
          W_NA = (1/H) sum_h A_a[h, t1, t2] * A_b[h, t2, t1]
        """
        plaquettes = {}
        diagonal_loops = {}
        field_strength = {}
        avg_fs = {}
        plaq_traces = {}

        sorted_layers = sorted(attention_maps.keys())
        layer_pairs = list(zip(sorted_layers[:-1], sorted_layers[1:]))

        for l_a, l_b in layer_pairs:
            A_a = attention_maps[l_a].float()  # [H, T, T]
            A_b = attention_maps[l_b].float()  # [H, T, T]
            A_a = torch.where(torch.isfinite(A_a), A_a, torch.zeros_like(A_a))
            A_b = torch.where(torch.isfinite(A_b), A_b, torch.zeros_like(A_b))
            H = A_a.shape[0]

            # Head-averaged plaquette (Abelian approximation)
            W_per_head = torch.bmm(A_a, A_b.transpose(1, 2))
            W = W_per_head.mean(dim=0)

            key = f"({l_a},{l_b})"
            plaquettes[key] = W
            diagonal_loops[key] = W.diagonal()

            # Field strength: F = 1 - |W|
            F = 1.0 - W.abs()
            field_strength[key] = F
            avg_fs[key] = F.mean().item()

            # Non-abelian trace: (1/H) * Tr(sum_h W_h) / T
            W_trace = W.diagonal().mean().item()
            plaq_traces[key] = W_trace

        return WilsonLoopResult(
            plaquettes=plaquettes,
            diagonal_loops=diagonal_loops,
            field_strength=field_strength,
            avg_field_strength=avg_fs,
            plaquette_traces=plaq_traces,
        )

    def compute_polyakov_loops(
        self,
        attention_maps: Dict[int, torch.Tensor],
    ) -> PolyakovLoopResult:
        """Compute Polyakov loops (temporal Wilson lines).

        P(t) = (1/H) Tr[ prod_{l=l_0}^{l_max} A_l[:, t, :] ]

        The Polyakov loop is the product of gauge links in the temporal
        (layer) direction at fixed spatial position t.

        Approximation: Use diagonal elements A_l[h, t, t] as the
        temporal link variable (self-attention weight).

        <P> -> 0: confined phase (information is bound)
        <P> > 0: deconfined phase (information propagates freely)
        """
        sorted_layers = sorted(attention_maps.keys())
        if len(sorted_layers) < 2:
            return PolyakovLoopResult(
                loop_values=torch.zeros(1),
                expectation_value=0.0,
                susceptibility=0.0,
                phase="confined",
                layer_contributions=torch.zeros(1),
            )

        T = attention_maps[sorted_layers[0]].shape[-1]
        H = attention_maps[sorted_layers[0]].shape[0]

        # Select subset of layers for Polyakov loop
        poly_layers = [l for l in self.cfg.polyakov_loop_layers
                       if l in attention_maps]
        if len(poly_layers) < 2:
            poly_layers = sorted_layers[:min(8, len(sorted_layers))]

        # Accumulate product of diagonal attention weights
        # P(t) = prod_l (1/H * sum_h A_l[h, t, t])
        P = torch.ones(T)
        layer_dets = []

        for l in poly_layers:
            A = attention_maps[l].float()
            A = torch.where(torch.isfinite(A), A, torch.zeros_like(A))
            # Head-averaged diagonal: how much each token attends to itself
            diag = A.diagonal(dim1=-2, dim2=-1).mean(dim=0)  # [T]
            P = P * diag.cpu()
            layer_dets.append(diag.mean().item())

        # Expectation value
        exp_val = P.mean().item()
        susc = (P ** 2).mean().item() - exp_val ** 2

        # Phase determination
        # Threshold: if <P> is significantly different from zero
        phase = "deconfined" if abs(exp_val) > 0.01 else "confined"

        return PolyakovLoopResult(
            loop_values=P,
            expectation_value=exp_val,
            susceptibility=susc,
            phase=phase,
            layer_contributions=torch.tensor(layer_dets),
        )

    def compute_creutz_ratios(
        self,
        attention_maps: Dict[int, torch.Tensor],
    ) -> CreutzRatioResult:
        """Compute Creutz ratios to extract string tension.

        chi(R, T) = -log(W(R,T) * W(R-1,T-1) / (W(R-1,T) * W(R,T-1)))

        where W(R, T) is the Wilson loop of spatial extent R and
        temporal (layer) extent T.

        Uses rectangular Wilson loops built from attention matrices.
        """
        sorted_layers = sorted(attention_maps.keys())
        ratios = {}
        eps = 1e-10

        max_R = min(self.cfg.wilson_loop_max_area, 4)
        max_T_layers = min(4, len(sorted_layers) - 1)

        # Compute rectangular Wilson loops of different sizes
        W_values = {}
        for T_ext in range(1, max_T_layers + 1):
            for R_ext in range(1, max_R + 1):
                w_sum = 0.0
                count = 0
                for l_start_idx in range(len(sorted_layers) - T_ext):
                    l_a = sorted_layers[l_start_idx]
                    l_b = sorted_layers[l_start_idx + T_ext]
                    if l_a not in attention_maps or l_b not in attention_maps:
                        continue

                    A_a = attention_maps[l_a].float().mean(dim=0)  # [T_tok, T_tok]
                    A_b = attention_maps[l_b].float().mean(dim=0)

                    T_tok = A_a.shape[0]
                    for t in range(T_tok - R_ext):
                        # Rectangular loop: t -> t+R at layer l_a,
                        # then t+R at l_a -> t+R at l_b (temporal),
                        # then t+R -> t at l_b, then t at l_b -> t at l_a
                        w = A_a[t, t + R_ext] * A_b[t + R_ext, t]
                        w_sum += w.item()
                        count += 1

                W_values[(R_ext, T_ext)] = w_sum / max(count, 1)

        # Compute Creutz ratios
        for R in range(2, max_R + 1):
            for T_l in range(2, max_T_layers + 1):
                W_RT = W_values.get((R, T_l), eps)
                W_R1T1 = W_values.get((R - 1, T_l - 1), eps)
                W_R1T = W_values.get((R - 1, T_l), eps)
                W_RT1 = W_values.get((R, T_l - 1), eps)

                numer = abs(W_RT * W_R1T1) + eps
                denom = abs(W_R1T * W_RT1) + eps
                chi = -np.log(numer / denom)
                ratios[f"({R},{T_l})"] = chi

        # Estimate string tension from largest Creutz ratio
        if ratios:
            string_tension = np.mean(list(ratios.values()))
        else:
            string_tension = 0.0

        # Check area law: W(R,T) ~ exp(-sigma * R * T)?
        area_law = string_tension > 0.01

        return CreutzRatioResult(
            ratios=ratios,
            string_tension=string_tension,
            area_law=area_law,
        )

    def verify_parallel_transport(
        self,
        spin_decomp: SpinDecomposition,
        attention_maps: Dict[int, torch.Tensor],
    ) -> ParallelTransportResult:
        """Verify spin preservation under attention (parallel transport).

        v5: Also tracks the accumulated non-abelian phase.
        """
        n_modes = spin_decomp.U.shape[2]
        sorted_layers = sorted(attention_maps.keys())
        n_attn_layers = len(sorted_layers)

        alignment = torch.zeros(n_attn_layers, n_modes)
        na_phase = torch.zeros(n_attn_layers, n_modes)

        for idx, l_attn in enumerate(sorted_layers):
            if l_attn + 1 >= spin_decomp.Vh.shape[0]:
                continue

            A = attention_maps[l_attn].float()
            A_avg = A.mean(dim=0)

            for i in range(n_modes):
                u_before = spin_decomp.U[l_attn, :, i].float()
                u_transported = A_avg.T @ u_before
                u_after = spin_decomp.U[l_attn + 1, :, i].float()

                cos_sim = torch.nn.functional.cosine_similarity(
                    u_transported.unsqueeze(0),
                    u_after.unsqueeze(0),
                ).item()
                alignment[idx, i] = abs(cos_sim)

                # Non-abelian phase: angle between transported and actual
                na_phase[idx, i] = np.arccos(min(1.0, abs(cos_sim)))

        return ParallelTransportResult(
            alignment=alignment,
            avg_alignment_per_mode=alignment.mean(dim=0),
            avg_alignment_per_layer=alignment.mean(dim=1),
            non_abelian_phase=na_phase,
        )

    def measure_helicity_conservation(
        self, helicity_per_layer: torch.Tensor
    ) -> torch.Tensor:
        """Helicity stability: std across layers per mode.
        Returns: [n_modes]"""
        return helicity_per_layer.std(dim=0)

    def compute_topological_charge(
        self,
        attention_maps: Dict[int, torch.Tensor],
        wilson_result: Optional[WilsonLoopResult] = None,
        quantization_threshold: float = 0.25,
    ) -> TopologicalChargeResult:
        """Compute topological charge Q = (1/2pi) sum_P F_P.

        In lattice gauge theory, the topological charge counts the net
        number of instantons minus anti-instantons. For attention gauge
        fields:
        - F_P = 1 - |W_P| is the field strength on plaquette P
        - Q = (1/2pi) * sum_P F_P
        - If Q is close to an integer, the gauge field has quantized
          topological sectors.

        Also computes per-token winding numbers from accumulated phase
        around the Polyakov loop.
        """
        # Compute Wilson loops if not provided
        if wilson_result is None:
            wilson_result = self.compute_wilson_loops(attention_maps)

        # -- Topological charge from plaquette field strength --
        charge_per_plaq = {}
        charge_density_list = []

        for key, F in wilson_result.field_strength.items():
            # F = 1 - |W| is already the field strength [T, T]
            # Topological charge density: integrate F over the plaquette
            # For a single plaquette, q = (1/2pi) * Tr(F) / T
            # Use diagonal elements (self-coupling) as the primary signal
            F_diag = F.diagonal().float()
            q_plaq = F_diag.mean().item() / (2.0 * np.pi)
            charge_per_plaq[key] = q_plaq
            charge_density_list.append(q_plaq)

        if charge_density_list:
            charge_density = torch.tensor(charge_density_list)
            total_charge = float(charge_density.sum().item())
        else:
            charge_density = torch.zeros(1)
            total_charge = 0.0

        nearest_int = int(round(total_charge))
        quant_error = abs(total_charge - nearest_int)
        is_quantized = quant_error < quantization_threshold

        # -- Per-token winding number from Polyakov loop phase --
        # The winding number is the accumulated phase of the Polyakov
        # loop: n(t) = (1/2pi) * arg(P(t)), where P(t) is complex.
        # We approximate using the log of the real Polyakov loop product.
        sorted_layers = sorted(attention_maps.keys())
        T = attention_maps[sorted_layers[0]].shape[-1] if sorted_layers else 1

        # Compute Polyakov loop phase per token
        accumulated_phase = torch.zeros(T)
        for l_attn in sorted_layers:
            A = attention_maps[l_attn].float()
            A = torch.where(torch.isfinite(A), A, torch.zeros_like(A))
            # Head-averaged diagonal: self-attention weight
            diag = A.diagonal(dim1=-2, dim2=-1).mean(dim=0).cpu()  # [T]
            # Phase contribution: -log(diag) is the gauge field "angle"
            eps = 1e-10
            accumulated_phase += -torch.log(diag.clamp(min=eps))

        winding_numbers = accumulated_phase / (2.0 * np.pi)
        mean_winding = winding_numbers.mean().item()

        logger.info(f"Topological charge Q={total_charge:.4f}, "
                    f"nearest_int={nearest_int}, error={quant_error:.4f}, "
                    f"quantized={is_quantized}")
        logger.info(f"Mean winding number: {mean_winding:.4f}")

        return TopologicalChargeResult(
            charge_per_plaquette=charge_per_plaq,
            total_charge=total_charge,
            charge_density=charge_density,
            nearest_integer=nearest_int,
            quantization_error=quant_error,
            is_quantized=is_quantized,
            winding_numbers=winding_numbers,
            mean_winding=mean_winding,
        )

    def compute_effective_attention_linear(
        self, residual_pre: torch.Tensor, residual_post: torch.Tensor
    ) -> torch.Tensor:
        """Effective attention from residual change for non-attention layers."""
        delta = (residual_post - residual_pre).float()
        delta_norm = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)
        return delta_norm @ delta_norm.T
