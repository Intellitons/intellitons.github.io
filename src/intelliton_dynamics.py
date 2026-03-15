"""Quasiparticle-centered generation-time Intelliton dynamics tracking.

Tracks per-step Intelliton occupations and dressed quasiparticle observables
during autoregressive generation, enabling hallucination analysis in terms of
mode activation shift rather than an external risk score.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.eft_renormalization import EFTRenormalization
from src.fusion_tracker import IntellitonFusionTracker
from src.intelliton_classifier import IntellitonCatalog
from src.lattice_field import LatticeField


@dataclass
class IntellitonDynamicsProfile:
    grounded_species_baseline: Dict[str, Dict[int, float]]
    grounded_mode_activation_shift_mean: float
    grounded_mode_activation_shift_std: float


class IntellitonDynamicsTracker:
    """Track generation-time quasiparticle occupations and mode activation shift."""

    def __init__(self, cfg, catalog: IntellitonCatalog, species_vectors: Dict[int, torch.Tensor]):
        self.cfg = cfg
        self.catalog = catalog
        self.species_vectors = {
            sid: vec.float().cpu() for sid, vec in species_vectors.items()
        }
        self.profile: Optional[IntellitonDynamicsProfile] = None

    def _next_token(self, analyzer, token_ids: List[int], do_sample: bool, temperature: float, top_p: float):
        ids = torch.tensor([token_ids], dtype=torch.long, device=analyzer.model.device)
        attention_mask = torch.ones_like(ids, device=analyzer.model.device)
        with torch.no_grad():
            outputs = analyzer.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[0, -1, :].float().cpu()
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        if do_sample:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            keep = cumsum <= top_p
            keep[0] = True
            filtered_probs = sorted_probs * keep.float()
            filtered_probs = filtered_probs / filtered_probs.sum().clamp(min=1e-8)
            choice = torch.multinomial(filtered_probs, 1).item()
            token_id = int(sorted_indices[choice].item())
        else:
            token_id = int(probs.argmax().item())
        token_text = analyzer.tokenizer.decode([token_id], skip_special_tokens=False)
        top_conf = float(probs[token_id].item())
        return token_id, token_text, top_conf

    def _sliding_wave_packet(self, current_field: torch.Tensor, local_window: int) -> torch.Tensor:
        curr_len = current_field.shape[1]
        window = min(local_window, curr_len)
        return current_field[:, curr_len - window:, :].clone()

    def _species_occupations(self, decomp) -> Dict[int, float]:
        if decomp.Vh.shape[0] == 0:
            return {}
        layer_idx = min(self.cfg.num_layers // 2, decomp.Vh.shape[0] - 1)
        sigma_layer = decomp.sigma[layer_idx].float().cpu()
        vh_layer = decomp.Vh[layer_idx].float().cpu()
        vh_norm = vh_layer / (vh_layer.norm(dim=-1, keepdim=True) + 1e-8)
        occupations = {}
        for sid, vec in self.species_vectors.items():
            vec_n = vec / (vec.norm() + 1e-8)
            sims = (vh_norm @ vec_n).abs()
            best_idx = int(sims.argmax().item())
            occupations[sid] = float((sigma_layer[best_idx] * sims[best_idx]).item())
        return occupations

    def _compute_step_features(self, field: torch.Tensor, attention_maps: Dict[int, torch.Tensor]):
        lf = LatticeField(field, self.cfg)
        decomp = lf.compute_spin_decomposition(self.cfg.n_top_modes)
        prop = lf.compute_propagator(decomp)
        disp = lf.compute_dispersion_relation(decomp)
        helicity = lf.compute_helicity(decomp)
        spin = lf.compute_spin_quantum_numbers(decomp)
        eft = EFTRenormalization(self.cfg)
        rg_flow = eft.compute_rg_flow(decomp, prop)
        eft_params = eft.compute_eft_parameters(decomp, prop)
        resonance = None
        try:
            resonance = lf.compute_resonance_widths(
                prop,
                sharpness_threshold=self.cfg.resonance_sharpness_threshold,
            )
        except Exception:
            resonance = None
        fusion = None
        try:
            fusion = IntellitonFusionTracker(self.cfg).track_fusion(decomp, attention_maps)
        except Exception:
            fusion = None
        occ = self._species_occupations(decomp)
        return {
            "lf": lf,
            "decomp": decomp,
            "prop": prop,
            "disp": disp,
            "helicity": helicity,
            "spin": spin,
            "rg_flow": rg_flow,
            "eft_params": eft_params,
            "resonance": resonance,
            "fusion": fusion,
            "occupations": occ,
        }

    def _species_observables(self, sid: int, features: Dict[str, object]) -> Dict[str, float]:
        species = self.catalog.species[sid]
        mode_i = species.svd_mode_index
        decomp = features["decomp"]
        prop = features["prop"]
        disp = features["disp"]
        helicity = features["helicity"]
        spin = features["spin"]
        rg_flow = features["rg_flow"]
        eft_params = features["eft_params"]
        resonance = features["resonance"]

        max_mode = decomp.sigma.shape[1] - 1
        idx = min(mode_i, max_mode)
        momentum_idx = 0
        if disp.momenta.numel() > 0:
            momentum_idx = int((disp.momenta - species.dominant_momentum).abs().argmin().item())

        resonance_width = np.nan
        if resonance is not None and idx < len(resonance.fits):
            resonance_width = float(resonance.fits[idx].gamma)

        fusion_strength = 0.0
        fusion_rate = 0.0
        if features["fusion"] is not None:
            fusion_rate = float(features["fusion"].fusion_rate.sum().item())
            matching_events = [
                ev for ev in features["fusion"].fusion_events
                if ev.target_mode == idx or idx in ev.source_modes
            ]
            if matching_events:
                fusion_strength = float(max(ev.coupling_strength for ev in matching_events))

        return {
            "Spin_t": float(spin[:, idx].mean().item()),
            "Mass_t": float(prop.mass[idx].item()),
            "Mass_Lat_t": float(prop.mass_lattice[idx].item()),
            "Momentum_t": float(disp.momenta[momentum_idx].item()) if disp.momenta.numel() > 0 else 0.0,
            "Helicity_t": float(helicity[:, idx].mean().item()),
            "Group_Velocity_t": float(disp.group_velocity[momentum_idx, idx].item()) if disp.group_velocity.numel() > 0 else 0.0,
            "FP_Layer_t": float(rg_flow.fixed_point_layers[idx].item()) if idx < rg_flow.fixed_point_layers.shape[0] else np.nan,
            "gamma_t": float(rg_flow.anomalous_dimension[idx].item()) if idx < rg_flow.anomalous_dimension.shape[0] else np.nan,
            "Z_t": float(eft_params.Z[idx].item()) if idx < eft_params.Z.shape[0] else np.nan,
            "Resonance_Width_t": resonance_width,
            "Fusion_Strength_t": fusion_strength,
            "Fusion_Rate_t": fusion_rate,
        }

    def _mode_activation_shift_score(self, obs: Dict[str, float], baseline_occ: float, current_occ: float) -> float:
        occ_gain = max(0.0, current_occ - baseline_occ)
        mass_drift = abs(obs["Mass_t"])
        z_loss = max(0.0, 1.0 - (obs["Z_t"] if np.isfinite(obs["Z_t"]) else 1.0))
        gamma = abs(obs["gamma_t"]) if np.isfinite(obs["gamma_t"]) else 0.0
        width = obs["Resonance_Width_t"] if np.isfinite(obs["Resonance_Width_t"]) else 0.0
        fusion = obs["Fusion_Strength_t"]
        return float(0.45 * occ_gain + 0.15 * mass_drift + 0.15 * z_loss + 0.1 * gamma + 0.1 * width + 0.05 * fusion)

    def fit_grounded_profile(
        self,
        analyzer,
        grounded_prompts: Sequence[str],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        local_window: int = 6,
    ) -> IntellitonDynamicsProfile:
        rows = self.trace_prompt_groups(
            analyzer,
            {"grounded_factual": grounded_prompts},
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            local_window=local_window,
            calibrating=True,
        )
        baseline = {}
        grounded = rows[rows["Prompt_Type"] == "grounded_factual"]
        for step, sub in grounded.groupby("Step_Index"):
            baseline[int(step)] = (
                sub.groupby("Species_ID")["Occupation"].mean().to_dict()
            )
        shift_mean = float(grounded["Mode_Activation_Shift"].mean()) if not grounded.empty else 0.0
        shift_std = float(grounded["Mode_Activation_Shift"].std() + 1e-6) if not grounded.empty else 1e-6
        self.profile = IntellitonDynamicsProfile(
            grounded_species_baseline=baseline,
            grounded_mode_activation_shift_mean=shift_mean,
            grounded_mode_activation_shift_std=shift_std,
        )
        return self.profile

    def trace_prompt_groups(
        self,
        analyzer,
        prompts_by_type: Dict[str, Sequence[str]],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        local_window: int = 6,
        calibrating: bool = False,
    ) -> pd.DataFrame:
        frames = []
        for prompt_type, prompts in prompts_by_type.items():
            for pair_index, prompt in enumerate(prompts):
                frame = self.trace_prompt(
                    analyzer,
                    prompt,
                    pair_index,
                    prompt_type,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    local_window=local_window,
                    calibrating=calibrating,
                )
                if not frame.empty:
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def trace_prompt(
        self,
        analyzer,
        prompt: str,
        pair_index: int,
        prompt_type: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        local_window: int = 6,
        calibrating: bool = False,
    ) -> pd.DataFrame:
        prompt_ids = analyzer.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=analyzer.cfg.analysis_seq_len,
        )
        if not prompt_ids:
            return pd.DataFrame()

        current_ids = list(prompt_ids)
        prev_occ = {sid: 0.0 for sid in self.species_vectors}
        rows: List[Dict[str, object]] = []

        for step_idx in range(max_new_tokens):
            token_id, token_text, top_conf = self._next_token(
                analyzer,
                current_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            current_ids = current_ids + [token_id]
            full_field = analyzer.run_inference_on_token_ids(current_ids)
            packet_field = self._sliding_wave_packet(full_field, local_window=local_window)
            current_text = analyzer.tokenizer.decode(current_ids, skip_special_tokens=False)
            attention_maps = {k: v[0].float().cpu() for k, v in analyzer._current_attn_weights.items()}
            features = self._compute_step_features(packet_field, attention_maps)

            top_species = sorted(
                features["occupations"].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[: min(5, len(features["occupations"]))]

            for sid, occ in top_species:
                obs = self._species_observables(sid, features)
                baseline_occ = 0.0
                if self.profile is not None:
                    baseline_occ = float(self.profile.grounded_species_baseline.get(step_idx, {}).get(sid, 0.0))
                delta_occ = float(occ - prev_occ.get(sid, 0.0))
                mode_activation_shift = self._mode_activation_shift_score(obs, baseline_occ, occ)
                grounded_dev = float(occ - baseline_occ)
                rows.append(
                    {
                        "Prompt_Type": prompt_type,
                        "Pair_Index": pair_index,
                        "Step_Index": step_idx,
                        "Token_ID": token_id,
                        "Token_Text": token_text.replace("\n", "\\n"),
                        "Prefix_Text": current_text,
                        "Generated_Text": current_text[len(prompt):],
                        "Species_ID": sid,
                        "Species_Name": self.catalog.species[sid].name,
                        "Occupation": occ,
                        "Delta_Occupation": delta_occ,
                        "Grounded_Baseline_Occupation": baseline_occ,
                        "Grounded_Deviation": grounded_dev,
                        "Mode_Activation_Shift": mode_activation_shift,
                        "Top_Token_Confidence": top_conf,
                        **obs,
                    }
                )
                prev_occ[sid] = occ

            if token_id == analyzer.tokenizer.eos_token_id:
                break

        return pd.DataFrame(rows)


def build_transition_matrix(detail_df: pd.DataFrame, top_k: int = 1) -> pd.DataFrame:
    """Build species-to-species transition counts across adjacent generation steps."""
    if detail_df.empty:
        return pd.DataFrame(
            columns=["Prompt_Type", "From_Species", "To_Species", "Count", "Mean_Target_Mode_Activation_Shift"]
        )

    rows = []
    grouped = detail_df.sort_values(["Prompt_Type", "Pair_Index", "Step_Index", "Occupation"], ascending=[True, True, True, False])
    for (prompt_type, pair_index), sub in grouped.groupby(["Prompt_Type", "Pair_Index"]):
        leaders = {}
        for step_idx, step_df in sub.groupby("Step_Index"):
            leaders[int(step_idx)] = step_df.head(top_k)
        ordered_steps = sorted(leaders.keys())
        for prev_step, next_step in zip(ordered_steps[:-1], ordered_steps[1:]):
            prev_df = leaders[prev_step]
            next_df = leaders[next_step]
            for _, prev_row in prev_df.iterrows():
                for _, next_row in next_df.iterrows():
                    rows.append(
                        {
                            "Prompt_Type": prompt_type,
                            "From_Species": prev_row["Species_Name"],
                            "To_Species": next_row["Species_Name"],
                            "Count": 1,
                            "Mean_Target_Mode_Activation_Shift": float(next_row["Mode_Activation_Shift"]),
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=["Prompt_Type", "From_Species", "To_Species", "Count", "Mean_Target_Mode_Activation_Shift"]
        )

    return (
        pd.DataFrame(rows)
        .groupby(["Prompt_Type", "From_Species", "To_Species"], as_index=False)
        .agg(
            Count=("Count", "sum"),
            Mean_Target_Mode_Activation_Shift=("Mean_Target_Mode_Activation_Shift", "mean"),
        )
        .sort_values(["Prompt_Type", "Count", "Mean_Target_Mode_Activation_Shift"], ascending=[True, False, False])
    )