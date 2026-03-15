"""Hallucination Diagnostics v5.

Analyzes Intelliton spectra differences between grounded (factual) and
hallucination-prone prompts to characterize hallucination as quasiparticle
phenomena:
  - Quasiparticle decay: well-defined mode loses coherence
  - Vacuum tunneling: spectrum shifts toward a degenerate vacuum state
  - Spectral broadening: resonance width increases (shorter lifetime)

Physical analogy:
  - Grounded prompts: stable quasiparticle excitations above the vacuum
  - Hallucination: unstable excitations that decay or tunnel between
    degenerate minima in the semantic landscape
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import IntellitonConfig
from src.lattice_field import LatticeField, SpinDecomposition

logger = logging.getLogger(__name__)


@dataclass
class HallucinationSignature:
    """Spectral signature comparing grounded vs hallucination-prone prompts."""
    # Per-layer singular value spectra
    sigma_grounded: torch.Tensor          # [L+1, n_modes]
    sigma_hallucination: torch.Tensor     # [L+1, n_modes]

    # Spectral divergence per layer
    spectral_divergence: torch.Tensor     # [L+1] KL between grounded/halluc sigma distributions
    coherence_loss: torch.Tensor          # [L+1] drop in top-mode dominance ratio

    # Mode stability: cosine similarity of Vh between grounded/halluc
    mode_stability: torch.Tensor          # [L+1, n_modes] cosine sim of Vh

    # Entropy difference: higher entropy in hallucination = more diffuse spectrum
    entropy_grounded: torch.Tensor        # [L+1]
    entropy_hallucination: torch.Tensor   # [L+1]
    entropy_gap: torch.Tensor             # [L+1] entropy_halluc - entropy_grounded

    # Summary diagnostics
    hallucination_layer: int              # layer with maximum spectral divergence
    decay_strength: float                 # max spectral_divergence value
    tunneling_score: float                # mean mode_stability drop in critical layers
    is_hallucinating: bool                # overall diagnostic


@dataclass
class HallucinationDiagnosticResult:
    """Full hallucination diagnostic results."""
    signatures: List[HallucinationSignature]
    pair_categories: List[str]
    pair_labels: List[str]

    # Aggregate statistics
    mean_spectral_divergence: torch.Tensor   # [L+1] averaged over all pairs
    mean_entropy_gap: torch.Tensor           # [L+1]
    mean_coherence_loss: torch.Tensor        # [L+1]
    critical_layer: int                      # layer where hallucination signal peaks
    hallucination_rate: float                # fraction of pairs flagged as hallucinating

    # Output probability diagnostics (if model inference available)
    grounded_confidence: List[float]         # P(correct_answer) for grounded prompts
    hallucination_confidence: List[float]    # P(top_token) for halluc prompts
    confidence_gap: float                    # mean grounded - mean halluc confidence

    # Control condition: grounded-vs-grounded cross-category comparisons
    control_signatures: List[HallucinationSignature] = field(default_factory=list)
    control_categories: List[str] = field(default_factory=list)

    # Bimodal layer structure
    early_peak_layer: int = 0                # layer of early divergence peak
    late_peak_layer: int = 0                 # layer of late divergence peak
    early_peak_strength: float = 0.0
    late_peak_strength: float = 0.0


class HallucinationDiagnostic:
    """Diagnose hallucination through Intelliton spectral analysis."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg

    def _compute_spectral_entropy(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy per layer.

        H(l) = -sum_i p_i * log(p_i) where p_i = sigma_i^2 / sum(sigma^2)

        High entropy = diffuse spectrum (many modes, no dominant one)
        Low entropy = peaked spectrum (one or few dominant modes)
        """
        eps = 1e-10
        sigma_sq = sigma.float() ** 2
        total = sigma_sq.sum(dim=-1, keepdim=True).clamp(min=eps)
        p = sigma_sq / total
        H = -(p * torch.log(p + eps)).sum(dim=-1)
        return H

    def _spectral_kl(
        self, sigma_a: torch.Tensor, sigma_b: torch.Tensor
    ) -> torch.Tensor:
        """Jensen-Shannon divergence between two spectral distributions per layer.

        Uses symmetric JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q).
        This is bounded [0, ln(2)], symmetric, and more robust than asymmetric KL.

        Treats normalized sigma^2 as probability distributions.
        Pads to the same number of modes if different.
        Returns: [L+1]
        """
        eps = 1e-10
        # Pad to same number of modes
        n_modes = max(sigma_a.shape[1], sigma_b.shape[1])
        if sigma_a.shape[1] < n_modes:
            pad = torch.zeros(sigma_a.shape[0], n_modes - sigma_a.shape[1])
            sigma_a = torch.cat([sigma_a, pad], dim=1)
        if sigma_b.shape[1] < n_modes:
            pad = torch.zeros(sigma_b.shape[0], n_modes - sigma_b.shape[1])
            sigma_b = torch.cat([sigma_b, pad], dim=1)
        # Normalize
        p = sigma_a.float() ** 2
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
        q = sigma_b.float() ** 2
        q = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)
        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
        kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return jsd.clamp(min=0)

    @staticmethod
    def _equalize_seq_len(field_a: torch.Tensor, field_b: torch.Tensor) -> tuple:
        """Pad the shorter field to match the longer one along the seq dimension.

        Fields have shape [L, T, D]. Zero-padding preserves spectral properties
        and ensures SVD operates on same-sized matrices, eliminating the confound
        where different sequence lengths produce incomparable singular value
        distributions purely due to matrix dimension differences.
        """
        T_a, T_b = field_a.shape[1], field_b.shape[1]
        if T_a == T_b:
            return field_a, field_b
        target_T = max(T_a, T_b)
        if T_a < target_T:
            pad = torch.zeros(field_a.shape[0], target_T - T_a, field_a.shape[2],
                              dtype=field_a.dtype)
            field_a = torch.cat([field_a, pad], dim=1)
        if T_b < target_T:
            pad = torch.zeros(field_b.shape[0], target_T - T_b, field_b.shape[2],
                              dtype=field_b.dtype)
            field_b = torch.cat([field_b, pad], dim=1)
        return field_a, field_b

    def analyze_pair(
        self,
        field_grounded: LatticeField,
        field_halluc: LatticeField,
    ) -> HallucinationSignature:
        """Compare Intelliton spectra between a grounded and hallucination-prone prompt.

        Returns HallucinationSignature with per-layer diagnostics.
        """
        n_modes = self.cfg.n_top_modes

        # Equalize sequence lengths before SVD to eliminate the confound where
        # different token counts produce incomparable singular value distributions.
        field_g_eq, field_h_eq = self._equalize_seq_len(
            field_grounded.field, field_halluc.field
        )
        lf_g = LatticeField(field_g_eq, self.cfg)
        lf_h = LatticeField(field_h_eq, self.cfg)

        decomp_g = lf_g.compute_spin_decomposition(n_modes)
        decomp_h = lf_h.compute_spin_decomposition(n_modes)

        sigma_g = decomp_g.sigma  # [L+1, n_modes]
        sigma_h = decomp_h.sigma

        # Pad to same number of layers and modes
        n_layers = min(sigma_g.shape[0], sigma_h.shape[0])
        sigma_g = sigma_g[:n_layers]
        sigma_h = sigma_h[:n_layers]
        n_modes_g = sigma_g.shape[1]
        n_modes_h = sigma_h.shape[1]
        n_modes_max = max(n_modes_g, n_modes_h)
        if n_modes_g < n_modes_max:
            sigma_g = torch.cat([sigma_g, torch.zeros(n_layers, n_modes_max - n_modes_g)], dim=1)
        if n_modes_h < n_modes_max:
            sigma_h = torch.cat([sigma_h, torch.zeros(n_layers, n_modes_max - n_modes_h)], dim=1)

        # 1. Spectral divergence (KL between sigma distributions)
        spectral_div = self._spectral_kl(sigma_g, sigma_h)

        # 2. Coherence loss: drop in dominance ratio sigma_0 / sigma_1
        # Clamp dominance ratios to avoid numerical instability when sigma_1 ≈ 0
        eps = 1e-10
        max_dominance = 100.0  # cap to prevent extreme ratios
        n_sv = sigma_g.shape[1]
        if n_sv >= 2:
            dominance_g = (sigma_g[:, 0] / (sigma_g[:, 1] + eps)).clamp(max=max_dominance)
            dominance_h = (sigma_h[:, 0] / (sigma_h[:, 1] + eps)).clamp(max=max_dominance)
            # Use log-ratio for more stable relative comparison
            coherence_loss = torch.log(dominance_g + eps) - torch.log(dominance_h + eps)
        else:
            coherence_loss = torch.zeros(n_layers)

        # 3. Mode stability: cosine similarity of Vh vectors
        # Now that fields are equalized, Vh vectors are in the same space
        Vh_g = decomp_g.Vh[:n_layers]  # [L+1, n_modes, D]
        Vh_h = decomp_h.Vh[:n_layers]
        mode_n = min(Vh_g.shape[1], Vh_h.shape[1], n_modes_max)
        mode_stability = torch.zeros(n_layers, mode_n)
        for l in range(n_layers):
            # Use optimal matching: for each grounded mode, find the best-matching
            # halluc mode (SVD mode ordering can differ between decompositions)
            vg_all = Vh_g[l, :mode_n].float()  # [mode_n, D]
            vh_all = Vh_h[l, :mode_n].float()
            vg_norm = vg_all / (vg_all.norm(dim=-1, keepdim=True) + 1e-8)
            vh_norm = vh_all / (vh_all.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = (vg_norm @ vh_norm.T).abs()  # [mode_n, mode_n]
            # Greedy best-match for each grounded mode
            for i in range(mode_n):
                mode_stability[l, i] = sim_matrix[i].max().item()

        # 4. Spectral entropy
        entropy_g = self._compute_spectral_entropy(sigma_g)
        entropy_h = self._compute_spectral_entropy(sigma_h)
        entropy_gap = entropy_h - entropy_g

        # Summary diagnostics
        halluc_layer = int(spectral_div.argmax().item())
        decay_strength = float(spectral_div.max().item())

        # Tunneling: mean mode stability drop in layers around the critical layer
        window = max(1, n_layers // 6)
        l_start = max(0, halluc_layer - window)
        l_end = min(n_layers, halluc_layer + window + 1)
        tunneling_score = float(1.0 - mode_stability[l_start:l_end].mean().item())

        # Heuristic: hallucinating if spectral divergence is significantly above
        # the per-layer median (robust to outliers).
        # Use median + 2*MAD as threshold, with a floor scaled to the median
        # to avoid flagging pure noise while remaining sensitive to real signals.
        sd_median = float(spectral_div.median().item())
        sd_mad = float((spectral_div - sd_median).abs().median().item())
        # Floor: 1.5x the median ensures we don't flag noise, but stays
        # proportional to the actual signal scale
        adaptive_threshold = max(sd_median + 2.0 * sd_mad, sd_median * 1.5, 1e-6)

        # Check entropy gap holistically: hallucination can manifest as
        # increased entropy in late layers OR decreased entropy in early layers
        # (indicating over-confident but wrong early representations).
        # Use the mean entropy gap over the top-divergence window rather than
        # a single layer point.
        eg_window_start = max(0, halluc_layer - window)
        eg_window_end = min(n_layers, halluc_layer + window + 1)
        mean_eg_at_critical = float(entropy_gap[eg_window_start:eg_window_end].mean().item())
        # Also check if late-layer entropy is elevated (last quarter of layers)
        late_start = max(0, n_layers - n_layers // 4)
        late_entropy_elevated = float(entropy_gap[late_start:].mean().item()) > 0

        # Also check mode stability: low stability in critical region indicates
        # the representation basis is shifting (tunneling between vacua)
        critical_stability = float(mode_stability[l_start:l_end].mean().item())
        mode_instability = critical_stability < 0.5

        is_hallucinating = (decay_strength > adaptive_threshold and
                           (mean_eg_at_critical > 0 or late_entropy_elevated or
                            mode_instability))

        return HallucinationSignature(
            sigma_grounded=sigma_g,
            sigma_hallucination=sigma_h,
            spectral_divergence=spectral_div,
            coherence_loss=coherence_loss,
            mode_stability=mode_stability,
            entropy_grounded=entropy_g,
            entropy_hallucination=entropy_h,
            entropy_gap=entropy_gap,
            hallucination_layer=halluc_layer,
            decay_strength=decay_strength,
            tunneling_score=tunneling_score,
            is_hallucinating=is_hallucinating,
        )

    def run_diagnostics(
        self,
        analyzer,
        hallucination_pairs: list,
    ) -> HallucinationDiagnosticResult:
        """Run full hallucination diagnostic over paired prompts.

        For each pair:
        1. Run both grounded and hallucination-prone prompts through the model
        2. Capture residual streams
        3. Compare Intelliton spectra

        Also measures output confidence where possible.
        """
        signatures = []
        categories = []
        labels = []
        grounded_confs = []
        halluc_confs = []
        grounded_fields = []  # collect for control condition

        for pair in hallucination_pairs:
            grounded_prompt = pair["grounded"]
            halluc_prompt = pair["hallucination_prone"]
            category = pair.get("category", "unknown")
            grounded_answer = pair.get("grounded_answer", "")

            # Run inference on both prompts
            analyzer._clear_current()
            inputs_g = analyzer.tokenizer(
                grounded_prompt, return_tensors="pt", truncation=True,
                max_length=self.cfg.analysis_seq_len,
            )
            inputs_g = {k: v.to(analyzer.model.device) for k, v in inputs_g.items()}
            with torch.no_grad():
                out_g = analyzer.model(**inputs_g, output_attentions=True, use_cache=False)

            # Capture grounded field
            if analyzer._current_embedding is not None and analyzer._current_residuals:
                emb = analyzer._current_embedding[0].float().cpu()
                layers = [r[0].float().cpu() for r in analyzer._current_residuals]
                field_g = torch.stack([emb] + layers, dim=0)
                field_g = torch.where(torch.isfinite(field_g), field_g, torch.zeros_like(field_g))
                grounded_fields.append(field_g)
            else:
                logger.warning(f"Failed to capture grounded: {grounded_prompt[:40]}")
                continue

            # Grounded confidence: use P(top_token) for symmetric comparison
            # with hallucination confidence.  Previously this used
            # P(first_token_of_answer) which is systematically lower than
            # P(top_token), creating a biased (often negative) confidence gap.
            logits_g = out_g.logits[0, -1, :].float().cpu()
            probs_g = F.softmax(logits_g, dim=-1)
            grounded_confs.append(float(probs_g.max().item()))

            # Run hallucination-prone prompt
            analyzer._clear_current()
            inputs_h = analyzer.tokenizer(
                halluc_prompt, return_tensors="pt", truncation=True,
                max_length=self.cfg.analysis_seq_len,
            )
            inputs_h = {k: v.to(analyzer.model.device) for k, v in inputs_h.items()}
            with torch.no_grad():
                out_h = analyzer.model(**inputs_h, output_attentions=True, use_cache=False)

            if analyzer._current_embedding is not None and analyzer._current_residuals:
                emb_h = analyzer._current_embedding[0].float().cpu()
                layers_h = [r[0].float().cpu() for r in analyzer._current_residuals]
                field_h = torch.stack([emb_h] + layers_h, dim=0)
                field_h = torch.where(torch.isfinite(field_h), field_h, torch.zeros_like(field_h))
            else:
                logger.warning(f"Failed to capture halluc: {halluc_prompt[:40]}")
                continue

            # Hallucination confidence (top token probability)
            logits_h = out_h.logits[0, -1, :].float().cpu()
            halluc_confs.append(float(F.softmax(logits_h, dim=-1).max().item()))

            # Analyze spectral difference
            lf_g = LatticeField(field_g, self.cfg)
            lf_h = LatticeField(field_h, self.cfg)

            sig = self.analyze_pair(lf_g, lf_h)
            torch.cuda.empty_cache()

            signatures.append(sig)
            categories.append(category)
            labels.append(f"G:{grounded_prompt[:25]}... vs H:{halluc_prompt[:25]}...")

        if not signatures:
            n_layers = self.cfg.total_field_layers
            return HallucinationDiagnosticResult(
                signatures=[],
                pair_categories=[],
                pair_labels=[],
                mean_spectral_divergence=torch.zeros(n_layers),
                mean_entropy_gap=torch.zeros(n_layers),
                mean_coherence_loss=torch.zeros(n_layers),
                critical_layer=0,
                hallucination_rate=0.0,
                grounded_confidence=[],
                hallucination_confidence=[],
                confidence_gap=0.0,
            )

        # Aggregate
        n_layers = min(s.spectral_divergence.shape[0] for s in signatures)
        mean_sd = torch.stack([s.spectral_divergence[:n_layers] for s in signatures]).mean(dim=0)
        mean_eg = torch.stack([s.entropy_gap[:n_layers] for s in signatures]).mean(dim=0)
        mean_cl = torch.stack([s.coherence_loss[:n_layers] for s in signatures]).mean(dim=0)

        critical_layer = int(mean_sd.argmax().item())
        halluc_rate = sum(1 for s in signatures if s.is_hallucinating) / len(signatures)

        conf_gap = 0.0
        if grounded_confs and halluc_confs:
            conf_gap = float(np.mean(grounded_confs) - np.mean(halluc_confs))

        # --- Bimodal layer detection ---
        # Find two peaks: one in the early half, one in the late half
        mid = n_layers // 2
        sd_np = mean_sd.numpy()
        early_peak_layer = int(sd_np[:mid].argmax())
        late_peak_layer = mid + int(sd_np[mid:].argmax())
        early_peak_strength = float(sd_np[early_peak_layer])
        late_peak_strength = float(sd_np[late_peak_layer])

        # --- Control condition: grounded-vs-grounded WITHIN same category ---
        # This is the proper null hypothesis: two grounded prompts on the same
        # topic should show minimal spectral divergence.  Cross-category
        # comparisons are not a fair control because different topics naturally
        # activate different representations.
        control_sigs: List[HallucinationSignature] = []
        control_cats: List[str] = []
        if len(grounded_fields) >= 2:
            logger.info("Computing control condition (grounded↔grounded, same category)...")
            # Group grounded fields by category
            cat_to_indices: Dict[str, List[int]] = {}
            for idx, cat in enumerate(categories):
                cat_to_indices.setdefault(cat, []).append(idx)
            for cat, indices in cat_to_indices.items():
                if len(indices) < 2:
                    continue
                # Compare consecutive pairs within the same category
                for k in range(len(indices) - 1):
                    i, j = indices[k], indices[k + 1]
                    lf_a = LatticeField(grounded_fields[i], self.cfg)
                    lf_b = LatticeField(grounded_fields[j], self.cfg)
                    ctrl_sig = self.analyze_pair(lf_a, lf_b)
                    control_sigs.append(ctrl_sig)
                    control_cats.append(f"{cat}↔{cat}")
            logger.info(f"  Control pairs: {len(control_sigs)}, "
                        f"flagged: {sum(1 for s in control_sigs if s.is_hallucinating)}")

        logger.info(f"Hallucination diagnostics: {len(signatures)} pairs analyzed")
        logger.info(f"  Critical layer: {critical_layer}")
        logger.info(f"  Bimodal peaks: early layer {early_peak_layer} "
                     f"(JSD={early_peak_strength:.6f}), late layer {late_peak_layer} "
                     f"(JSD={late_peak_strength:.6f})")
        logger.info(f"  Hallucination rate: {halluc_rate:.1%}")
        logger.info(f"  Confidence gap: {conf_gap:.4f}")

        return HallucinationDiagnosticResult(
            signatures=signatures,
            pair_categories=categories,
            pair_labels=labels,
            mean_spectral_divergence=mean_sd,
            mean_entropy_gap=mean_eg,
            mean_coherence_loss=mean_cl,
            critical_layer=critical_layer,
            hallucination_rate=halluc_rate,
            grounded_confidence=grounded_confs,
            hallucination_confidence=halluc_confs,
            confidence_gap=conf_gap,
            control_signatures=control_sigs,
            control_categories=control_cats,
            early_peak_layer=early_peak_layer,
            late_peak_layer=late_peak_layer,
            early_peak_strength=early_peak_strength,
            late_peak_strength=late_peak_strength,
        )
