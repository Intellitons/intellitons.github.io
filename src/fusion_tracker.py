"""Intelliton Fusion Tracking v5.

Detects when low-level SVD modes fuse into higher-level modes through
attention-mediated nonlinear coupling as network depth increases.

Key observables:
- Fusion events: two modes at layer l map to the same mode at layer l+1
- Mode-coupling matrix: M_{ij}(l) = |U_i^T A_avg U_j| at each attention layer
- Mode genealogy: trace ancestry of each final-layer mode back to layer 0
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.config import IntellitonConfig
from src.lattice_field import SpinDecomposition

logger = logging.getLogger(__name__)


@dataclass
class FusionEvent:
    layer: int                    # layer where fusion occurs (the l+1 side)
    source_modes: List[int]       # mode indices at layer l that fuse
    target_mode: int              # mode index at layer l+1
    coupling_strength: float      # max off-diagonal M[l] among source pairs
    source_sigmas: List[float]    # sigma values of source modes at layer l
    target_sigma: float           # sigma of target mode at layer l+1


@dataclass
class FusionResult:
    fusion_events: List[FusionEvent]
    coupling_matrix: torch.Tensor       # [n_attn_layers, n_modes, n_modes]
    coupling_layers: List[int]          # which layers the coupling matrix rows correspond to
    mode_genealogy: Dict[int, List[List[int]]]
        # key = final mode index, value = list of ancestor mode sets per layer
    fusion_rate: torch.Tensor           # [L] -- number of fusion events per layer transition
    n_active_modes_per_layer: torch.Tensor  # [L+1] -- unique target count


class IntellitonFusionTracker:
    """Track Intelliton mode fusion across layers."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg

    def detect_fusion_events(
        self, decomp: SpinDecomposition
    ) -> Tuple[List[FusionEvent], torch.Tensor]:
        """Detect fusion events from matched_indices.

        A fusion event occurs when two or more modes at layer l map to the
        same mode at layer l+1.

        Returns:
            events: list of FusionEvent
            fusion_rate: [L] tensor of fusion event counts per transition
        """
        if decomp.matched_indices is None or len(decomp.matched_indices) == 0:
            return [], torch.zeros(0)

        events = []
        n_transitions = len(decomp.matched_indices)
        fusion_rate = torch.zeros(n_transitions)

        for l in range(n_transitions):
            target_to_sources = defaultdict(list)
            for src_i, target_j in enumerate(decomp.matched_indices[l].tolist()):
                target_to_sources[int(target_j)].append(src_i)

            for target_j, sources in target_to_sources.items():
                if len(sources) >= 2:
                    events.append(FusionEvent(
                        layer=l + 1,
                        source_modes=sources,
                        target_mode=target_j,
                        coupling_strength=0.0,  # filled later
                        source_sigmas=[decomp.sigma[l, s].item() for s in sources],
                        target_sigma=decomp.sigma[l + 1, target_j].item(),
                    ))
                    fusion_rate[l] += 1

        return events, fusion_rate

    def compute_mode_coupling_matrix(
        self,
        decomp: SpinDecomposition,
        attention_maps: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute attention-mediated mode-coupling matrix.

        At each attention layer l:
            M_{ij}(l) = |U_i(l)^T @ A_avg(l) @ U_j(l)|

        Off-diagonal entries reveal cross-mode coupling that drives fusion.

        Returns:
            coupling_matrix: [n_attn_layers, n_modes, n_modes]
            coupling_layers: list of layer indices
        """
        sorted_layers = sorted(attention_maps.keys())
        n_modes = min(decomp.U.shape[2], self.cfg.n_top_modes)
        M = torch.zeros(len(sorted_layers), n_modes, n_modes)

        for idx, l_attn in enumerate(sorted_layers):
            if l_attn >= decomp.U.shape[0]:
                continue

            A = attention_maps[l_attn].float()      # [H, T, T]
            A_avg = A.mean(dim=0)                    # [T, T]
            U_l = decomp.U[l_attn, :, :n_modes].float()  # [T, n_modes]

            # M = |U^T A U| -- two matmuls
            M_l = U_l.T @ A_avg @ U_l               # [n_modes, n_modes]
            M[idx] = M_l.abs()

        return M, sorted_layers

    def build_mode_genealogy(
        self, decomp: SpinDecomposition
    ) -> Dict[int, List[List[int]]]:
        """Build backward-tracing ancestry tree for each final-layer mode.

        For each mode at the last layer, traces through matched_indices
        to find all ancestor modes at every layer.

        Returns:
            dict mapping final_mode_index -> list of ancestor sets per layer
        """
        if decomp.matched_indices is None or len(decomp.matched_indices) == 0:
            return {}

        n_modes = decomp.sigma.shape[1]
        n_layers = decomp.sigma.shape[0]
        genealogy = {}

        for final_mode in range(n_modes):
            ancestors_per_layer = [[] for _ in range(n_layers)]
            ancestors_per_layer[-1] = [final_mode]

            current_targets = {final_mode}
            for l in range(len(decomp.matched_indices) - 1, -1, -1):
                sources = set()
                for src_i in range(n_modes):
                    if decomp.matched_indices[l][src_i].item() in current_targets:
                        sources.add(src_i)
                ancestors_per_layer[l] = sorted(sources)
                current_targets = sources

            genealogy[final_mode] = ancestors_per_layer

        return genealogy

    def _compute_n_active_modes(
        self, decomp: SpinDecomposition
    ) -> torch.Tensor:
        """Count effectively unique modes at each layer.

        At layer 0, all n_modes are distinct. After fusion, some modes
        share the same target, reducing the effective count.
        """
        if decomp.matched_indices is None or len(decomp.matched_indices) == 0:
            n = decomp.sigma.shape[1]
            return torch.full((decomp.sigma.shape[0],), float(n))

        n_modes = decomp.sigma.shape[1]
        n_layers = decomp.sigma.shape[0]
        active = torch.zeros(n_layers)
        active[0] = n_modes

        for l in range(len(decomp.matched_indices)):
            unique_targets = len(set(decomp.matched_indices[l].tolist()))
            active[l + 1] = unique_targets

        return active

    def track_fusion(
        self,
        decomp: SpinDecomposition,
        attention_maps: Dict[int, torch.Tensor],
    ) -> FusionResult:
        """Full fusion analysis.

        Detects fusion events, computes coupling matrix, builds genealogy.
        """
        logger.info("Detecting fusion events...")
        events, fusion_rate = self.detect_fusion_events(decomp)
        logger.info(f"Found {len(events)} fusion events across {len(fusion_rate)} transitions")

        logger.info("Computing mode-coupling matrix...")
        coupling_matrix, coupling_layers = self.compute_mode_coupling_matrix(
            decomp, attention_maps
        )

        # Fill coupling_strength on fusion events from the coupling matrix
        # Map layer index to coupling matrix index
        layer_to_idx = {l: i for i, l in enumerate(coupling_layers)}
        for event in events:
            # The coupling that drives fusion is at layer l (before the fusion at l+1)
            source_layer = event.layer - 1
            # Find the closest attention layer <= source_layer
            best_attn_idx = None
            for cl in coupling_layers:
                if cl <= source_layer:
                    best_attn_idx = layer_to_idx[cl]
            if best_attn_idx is not None:
                # Max off-diagonal coupling among source mode pairs
                max_coupling = 0.0
                for i, si in enumerate(event.source_modes):
                    for j, sj in enumerate(event.source_modes):
                        if i < j and si < coupling_matrix.shape[1] and sj < coupling_matrix.shape[2]:
                            c = coupling_matrix[best_attn_idx, si, sj].item()
                            max_coupling = max(max_coupling, c)
                event.coupling_strength = max_coupling

        logger.info("Building mode genealogy...")
        genealogy = self.build_mode_genealogy(decomp)

        n_active = self._compute_n_active_modes(decomp)

        return FusionResult(
            fusion_events=events,
            coupling_matrix=coupling_matrix,
            coupling_layers=coupling_layers,
            mode_genealogy=genealogy,
            fusion_rate=fusion_rate,
            n_active_modes_per_layer=n_active,
        )
