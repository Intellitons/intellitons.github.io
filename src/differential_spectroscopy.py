"""Differential spectroscopy for isolating sharper, background-subtracted modes.

This module implements a "difference-spectrum particle physics" pass:
  1. Compute per-prompt layer-mode signatures from SVD spectra
  2. Subtract category / global / vacuum backgrounds
  3. Build differential spectra from detrended residual signatures
  4. Search for prompt-recurrent, high-contrast narrow peaks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, peak_widths

from src.config import IntellitonConfig
from src.lattice_field import LatticeField, SpinDecomposition


@dataclass
class DifferentialQuasiparticleCandidate:
    """Aggregated candidate from differential spectra."""

    category: str
    mode_index: int
    omega_0: float
    gamma: float
    contrast: float
    recurrence: float
    coherence: float
    score: float
    supporting_prompts: int


@dataclass
class DifferentialSpectroscopyResult:
    """Outputs of the differential spectroscopy pass."""

    omega: torch.Tensor
    category_total_spectrum: Dict[str, torch.Tensor]
    category_mode_spectrum: Dict[str, torch.Tensor]
    category_mode_scores: Dict[str, torch.Tensor]
    candidates: List[DifferentialQuasiparticleCandidate]
    candidate_df: pd.DataFrame


class DifferentialSpectroscopyAnalyzer:
    """Background-subtracted spectral analysis over prompts and categories."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg

    def _pad_modes(self, tensor: torch.Tensor, target_modes: int) -> torch.Tensor:
        if tensor.shape[1] >= target_modes:
            return tensor[:, :target_modes]
        pad = torch.zeros(tensor.shape[0], target_modes - tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)

    def _pad_centroid(self, centroid: torch.Tensor, target_modes: int) -> torch.Tensor:
        if centroid.shape[0] >= target_modes:
            return centroid[:target_modes]
        pad = torch.zeros(target_modes - centroid.shape[0], centroid.shape[1], device=centroid.device, dtype=centroid.dtype)
        return torch.cat([centroid, pad], dim=0)

    def _signature_from_decomp(self, decomp: SpinDecomposition) -> torch.Tensor:
        """Layer-mode signature used for differential spectral analysis."""
        sigma = torch.log1p(decomp.sigma.float())
        l0 = min(self.cfg.mass_fit_skip_layers, sigma.shape[0] - 1)
        sigma = sigma - sigma[l0:l0 + 1]
        sigma = sigma - sigma.mean(dim=0, keepdim=True)
        sigma = sigma / (sigma.std(dim=0, keepdim=True) + 1e-6)
        return self._pad_modes(sigma, self.cfg.n_top_modes)

    def _mode_centroid(self, decomp: SpinDecomposition) -> torch.Tensor:
        """Prompt-level mode centroid for cross-prompt coherence scoring."""
        centroid = decomp.Vh.float().mean(dim=0)
        centroid = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8)
        return self._pad_centroid(centroid, self.cfg.n_top_modes)

    def _detrend_curve(self, curve: torch.Tensor) -> torch.Tensor:
        """Remove linear trend so differential peaks are not dominated by drift."""
        x = torch.arange(curve.shape[0], dtype=curve.dtype, device=curve.device)
        x_mean = x.mean()
        y_mean = curve.mean()
        denom = ((x - x_mean) ** 2).sum().clamp(min=1e-8)
        slope = ((x - x_mean) * (curve - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        return curve - (slope * x + intercept)

    def _spectrum_from_signature(self, signature: torch.Tensor) -> torch.Tensor:
        """Compute per-mode differential spectrum from a background-subtracted signature."""
        n_layers, n_modes = signature.shape
        n_fft = max(64, n_layers * 4)
        window = torch.hann_window(n_layers, dtype=signature.dtype, device=signature.device)
        spectra = []
        for mode_idx in range(n_modes):
            curve = self._detrend_curve(signature[:, mode_idx])
            curve = curve * window
            spec = torch.fft.rfft(curve, n=n_fft).abs() ** 2
            spectra.append(spec)
        return torch.stack(spectra, dim=1)

    def _build_background(
        self,
        signature: torch.Tensor,
        category_signatures: List[torch.Tensor],
        global_mean: torch.Tensor,
        vacuum_signature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Blend category, global, and vacuum baselines into one background."""
        cat_sum = torch.stack(category_signatures).sum(dim=0)
        if len(category_signatures) > 1:
            category_mean = (cat_sum - signature) / (len(category_signatures) - 1)
        else:
            category_mean = signature.new_zeros(signature.shape)

        components = []
        weights = []

        if self.cfg.diffspec_category_weight > 0:
            components.append(category_mean)
            weights.append(self.cfg.diffspec_category_weight)
        if self.cfg.diffspec_global_weight > 0:
            components.append(global_mean)
            weights.append(self.cfg.diffspec_global_weight)
        if vacuum_signature is not None and self.cfg.diffspec_vacuum_weight > 0:
            components.append(vacuum_signature)
            weights.append(self.cfg.diffspec_vacuum_weight)

        if not components:
            return signature.new_zeros(signature.shape)

        weight_sum = sum(weights)
        background = sum(w * c for w, c in zip(weights, components)) / max(weight_sum, 1e-8)
        return background

    def _find_mode_peak(
        self,
        spectrum: torch.Tensor,
        coherence: float,
        omega: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Find the sharpest high-contrast differential peak in one mode."""
        rho = spectrum.detach().cpu().numpy().astype(np.float64)
        if np.allclose(rho, 0.0):
            return None

        kernel = np.ones(5, dtype=np.float64) / 5.0
        baseline = np.convolve(rho, kernel, mode="same")
        residual = np.maximum(rho - baseline, 0.0)
        peak_floor = residual.mean() + self.cfg.diffspec_peak_std_threshold * residual.std()
        if not np.isfinite(peak_floor) or peak_floor <= 0:
            peak_floor = max(residual.max() * 0.25, 1e-8)

        peaks, props = find_peaks(
            residual,
            height=peak_floor,
            prominence=max(peak_floor * 0.5, 1e-8),
            distance=2,
        )
        if len(peaks) == 0:
            return None

        peak_scores = []
        for peak in peaks:
            local_bg = max(baseline[peak], 1e-8)
            contrast = residual[peak] / local_bg
            width_res = peak_widths(residual, [peak], rel_height=0.5)
            width_bins = float(width_res[0][0]) if len(width_res[0]) else 1.0
            d_omega = float(omega[1].item() - omega[0].item()) if len(omega) > 1 else 1.0
            gamma = max(width_bins * d_omega, d_omega)
            omega_0 = float(omega[peak].item())
            width_ratio = gamma / max(omega_0, 1e-6)
            score = contrast * max(coherence, 0.0) / (1.0 + width_ratio)
            peak_scores.append((score, peak, omega_0, gamma, contrast))

        peak_scores.sort(key=lambda item: item[0], reverse=True)
        _, peak, omega_0, gamma, contrast = peak_scores[0]
        return {
            "peak_index": int(peak),
            "omega_0": float(omega_0),
            "gamma": float(gamma),
            "contrast": float(contrast),
        }

    def analyze(
        self,
        all_fields_by_cat: Dict[str, List[LatticeField]],
        vacuum_field: Optional[torch.Tensor] = None,
    ) -> DifferentialSpectroscopyResult:
        """Run differential spectroscopy over prompt ensembles."""
        limit = self.cfg.diffspec_max_prompts_per_category
        fields_by_cat = {
            cat: fields[:limit] if limit is not None else fields
            for cat, fields in all_fields_by_cat.items()
            if fields
        }

        decomp_by_cat: Dict[str, List[SpinDecomposition]] = {}
        signatures_by_cat: Dict[str, List[torch.Tensor]] = {}
        centroids_by_cat: Dict[str, List[torch.Tensor]] = {}

        for cat, fields in fields_by_cat.items():
            cat_decomps = []
            cat_signatures = []
            cat_centroids = []
            for field in fields:
                decomp = field.compute_spin_decomposition(n_modes=self.cfg.n_top_modes)
                cat_decomps.append(decomp)
                cat_signatures.append(self._signature_from_decomp(decomp))
                cat_centroids.append(self._mode_centroid(decomp))
            decomp_by_cat[cat] = cat_decomps
            signatures_by_cat[cat] = cat_signatures
            centroids_by_cat[cat] = cat_centroids

        all_signatures = [sig for sigs in signatures_by_cat.values() for sig in sigs]
        if not all_signatures:
            empty_df = pd.DataFrame(columns=[
                "category", "mode_index", "omega_0", "gamma", "contrast",
                "recurrence", "coherence", "score", "supporting_prompts",
            ])
            return DifferentialSpectroscopyResult(
                omega=torch.zeros(1),
                category_total_spectrum={},
                category_mode_spectrum={},
                category_mode_scores={},
                candidates=[],
                candidate_df=empty_df,
            )

        global_mean = torch.stack(all_signatures).mean(dim=0)

        vacuum_signature = None
        if vacuum_field is not None:
            vacuum_lattice = LatticeField(vacuum_field, self.cfg)
            vacuum_decomp = vacuum_lattice.compute_spin_decomposition(n_modes=self.cfg.n_top_modes)
            vacuum_signature = self._signature_from_decomp(vacuum_decomp)

        category_mode_reference: Dict[str, torch.Tensor] = {}
        for cat, centroids in centroids_by_cat.items():
            category_mode_reference[cat] = torch.stack(centroids).mean(dim=0)
            ref = category_mode_reference[cat]
            category_mode_reference[cat] = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)

        d_omega_holder = None
        omega = None
        category_mode_spectrum: Dict[str, torch.Tensor] = {}
        category_total_spectrum: Dict[str, torch.Tensor] = {}
        category_mode_scores: Dict[str, torch.Tensor] = {}
        raw_peak_rows: List[Dict[str, float]] = []

        for cat, signatures in signatures_by_cat.items():
            prompt_spectra = []
            n_prompts = len(signatures)
            mode_scores = torch.zeros(self.cfg.n_top_modes)

            for prompt_idx, signature in enumerate(signatures):
                background = self._build_background(
                    signature,
                    signatures,
                    global_mean,
                    vacuum_signature,
                )
                differential = signature - background
                differential = differential - differential.mean(dim=0, keepdim=True)
                spectrum = self._spectrum_from_signature(differential)
                prompt_spectra.append(spectrum)

                if omega is None:
                    omega = torch.linspace(0, torch.pi, spectrum.shape[0])
                    if len(omega) > 1:
                        d_omega_holder = float(omega[1].item() - omega[0].item())

                prompt_centroid = centroids_by_cat[cat][prompt_idx]
                ref_centroid = category_mode_reference[cat]
                coherence_vec = (prompt_centroid * ref_centroid).sum(dim=-1).clamp(-1.0, 1.0)

                for mode_idx in range(min(spectrum.shape[1], self.cfg.n_top_modes)):
                    peak = self._find_mode_peak(
                        spectrum[:, mode_idx],
                        float(coherence_vec[mode_idx].item()),
                        omega,
                    )
                    if peak is None:
                        continue
                    mode_scores[mode_idx] = max(
                        mode_scores[mode_idx],
                        peak["contrast"] * max(float(coherence_vec[mode_idx].item()), 0.0),
                    )
                    raw_peak_rows.append({
                        "category": cat,
                        "prompt_index": prompt_idx,
                        "mode_index": mode_idx,
                        "omega_0": peak["omega_0"],
                        "gamma": peak["gamma"],
                        "contrast": peak["contrast"],
                        "coherence": float(coherence_vec[mode_idx].item()),
                        "n_prompts": n_prompts,
                    })

            if prompt_spectra:
                mean_mode_spectrum = torch.stack(prompt_spectra).mean(dim=0)
                category_mode_spectrum[cat] = mean_mode_spectrum
                category_total_spectrum[cat] = mean_mode_spectrum.sum(dim=1)
            else:
                category_mode_spectrum[cat] = torch.zeros(1, self.cfg.n_top_modes)
                category_total_spectrum[cat] = torch.zeros(1)
            category_mode_scores[cat] = mode_scores

        candidates: List[DifferentialQuasiparticleCandidate] = []
        if raw_peak_rows and omega is not None:
            raw_df = pd.DataFrame(raw_peak_rows)
            bin_width = max(self.cfg.diffspec_cluster_bins * (d_omega_holder or 0.05), 1e-6)
            raw_df["cluster_bin"] = (raw_df["omega_0"] / bin_width).round().astype(int)

            grouped = raw_df.groupby(["category", "mode_index", "cluster_bin"], sort=False)
            for (cat, mode_idx, _), group in grouped:
                n_prompts = int(group["n_prompts"].iloc[0])
                support = int(group["prompt_index"].nunique())
                recurrence = support / max(n_prompts, 1)
                if support < max(2, self.cfg.diffspec_min_supporting_prompts):
                    continue
                if recurrence < self.cfg.diffspec_min_recurrence:
                    continue

                omega_0 = float(group["omega_0"].median())
                gamma = float(group["gamma"].median())
                contrast = float(group["contrast"].mean())
                coherence = float(group["coherence"].mean())
                width_ratio = gamma / max(omega_0, 1e-6)
                score = contrast * recurrence * max(coherence, 0.0) / (1.0 + width_ratio)

                candidates.append(DifferentialQuasiparticleCandidate(
                    category=cat,
                    mode_index=int(mode_idx),
                    omega_0=omega_0,
                    gamma=gamma,
                    contrast=contrast,
                    recurrence=recurrence,
                    coherence=coherence,
                    score=score,
                    supporting_prompts=support,
                ))

        candidates.sort(key=lambda cand: cand.score, reverse=True)
        candidate_df = pd.DataFrame([
            {
                "category": cand.category,
                "mode_index": cand.mode_index,
                "omega_0": cand.omega_0,
                "gamma": cand.gamma,
                "contrast": cand.contrast,
                "recurrence": cand.recurrence,
                "coherence": cand.coherence,
                "score": cand.score,
                "supporting_prompts": cand.supporting_prompts,
            }
            for cand in candidates
        ])

        if omega is None:
            omega = torch.zeros(1)

        return DifferentialSpectroscopyResult(
            omega=omega,
            category_total_spectrum=category_total_spectrum,
            category_mode_spectrum=category_mode_spectrum,
            category_mode_scores=category_mode_scores,
            candidates=candidates,
            candidate_df=candidate_df,
        )