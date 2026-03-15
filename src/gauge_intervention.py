"""Gauge transformation interventions v5.

v5 features:
- Standard amplify/suppress interventions per species
- Gauge-covariant interventions: parallel-transport species vectors
  through the attention gauge connection before applying at different layers
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import IntellitonConfig
from src.datasets import COREFERENCE_BENCHMARK, REASONING_BENCHMARK
from src.intelliton_classifier import IntellitonCatalog

logger = logging.getLogger(__name__)


@dataclass
class GaugeCovarianceResult:
    layers_tested: List[int]
    kl_raw: torch.Tensor                       # [n_species, n_layers_tested]
    kl_transported: torch.Tensor               # [n_species, n_layers_tested]
    gauge_violation: torch.Tensor              # kl_raw - kl_transported
    top_token_change_raw: torch.Tensor         # [n_species, n_layers_tested]
    top_token_change_transported: torch.Tensor
    transport_fidelity: torch.Tensor           # [n_species, n_layers_tested]
    species_names: List[str]


class GaugeIntervention:
    """Apply gauge transformations in Intelliton mode subspaces (v5)."""

    def __init__(self, cfg: IntellitonConfig, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def _get_language_model(self):
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "language_model"):
                return inner.language_model
            if hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
                return inner
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self.model, "layers") and hasattr(self.model, "embed_tokens"):
            return self.model
        raise AttributeError("Could not locate language model")

    def _get_logits(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.cfg.analysis_seq_len,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
        logits = outputs.logits[0, -1, :].float().cpu()
        return torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

    def _kl_divergence(self, a, b):
        p = F.softmax(a, dim=-1)
        return (p * (F.log_softmax(a, dim=-1) - F.log_softmax(b, dim=-1))).sum().item()

    def _cosine_sim(self, a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def _top_token_changed(self, a, b):
        return a.argmax().item() != b.argmax().item()

    def _top_k_overlap(self, a, b, k=10):
        return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / k

    def _token_prob(self, logits, text):
        ids = self.tokenizer.encode(" " + text, add_special_tokens=False)
        if not ids:
            return 0.0
        return float(F.softmax(logits, dim=-1)[ids[0]].item())

    def _evaluate_reasoning_accuracy(self, intervention=None, layer_idx=None):
        if not REASONING_BENCHMARK:
            return 0.0
        correct = 0
        for item in REASONING_BENCHMARK:
            logits = self._get_logits(str(item["prompt"])) if intervention is None \
                     else self._run_with_hook(str(item["prompt"]), intervention, layer_idx)
            scores = [self._token_prob(logits, c) for c in item["candidates"]]
            if list(item["candidates"])[int(np.argmax(scores))] == str(item["answer"]):
                correct += 1
        return correct / max(1, len(REASONING_BENCHMARK))

    def _evaluate_coref_score(self, intervention=None, layer_idx=None):
        if not COREFERENCE_BENCHMARK:
            return 0.0
        pronouns = ["he", "she", "it", "they"]
        margins = []
        for item in COREFERENCE_BENCHMARK:
            logits = self._get_logits(str(item["prompt"])) if intervention is None \
                     else self._run_with_hook(str(item["prompt"]), intervention, layer_idx)
            correct = str(item["correct"]).lower()
            p_correct = self._token_prob(logits, correct)
            wrong = [self._token_prob(logits, p) for p in pronouns if p != correct]
            margins.append(p_correct - max(wrong) if wrong else p_correct)
        return float(np.mean(margins))

    def _create_amplify_hook(self, v, alpha):
        def hook(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hs is None:
                return args, kwargs
            vec = v.to(hs.device, hs.dtype)
            vec = vec / (vec.norm() + 1e-8)
            proj = (hs @ vec).unsqueeze(-1) * vec
            hs_new = hs + alpha * proj
            if len(args) > 0:
                return (hs_new,) + args[1:], kwargs
            kwargs["hidden_states"] = hs_new
            return args, kwargs
        return hook

    def _create_suppress_hook(self, v):
        def hook(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hs is None:
                return args, kwargs
            vec = v.to(hs.device, hs.dtype)
            vec = vec / (vec.norm() + 1e-8)
            proj = (hs @ vec).unsqueeze(-1) * vec
            hs_new = hs - proj
            if len(args) > 0:
                return (hs_new,) + args[1:], kwargs
            kwargs["hidden_states"] = hs_new
            return args, kwargs
        return hook

    def _run_with_hook(self, prompt, hook_fn, layer_idx):
        lang = self._get_language_model()
        handle = lang.layers[layer_idx].register_forward_pre_hook(
            hook_fn, with_kwargs=True
        )
        try:
            return self._get_logits(prompt)
        finally:
            handle.remove()

    def run_intervention_experiment(self, catalog, mode_vectors, test_prompts):
        rows = []
        layer_idx = self.cfg.steering_layer
        alpha = self.cfg.steering_alpha
        baseline_cache = {}
        reasoning_base = self._evaluate_reasoning_accuracy()
        coref_base = self._evaluate_coref_score()

        for species in catalog.species:
            sp_id = species.species_id
            if sp_id not in mode_vectors:
                continue
            v = mode_vectors[sp_id]

            hook_amp = self._create_amplify_hook(v, alpha)
            hook_sup = self._create_suppress_hook(v)
            r_amp = self._evaluate_reasoning_accuracy(hook_amp, layer_idx)
            c_amp = self._evaluate_coref_score(hook_amp, layer_idx)
            r_sup = self._evaluate_reasoning_accuracy(hook_sup, layer_idx)
            c_sup = self._evaluate_coref_score(hook_sup, layer_idx)

            for cat, prompts in test_prompts.items():
                amp_kls, amp_coss, amp_ch, amp_ov = [], [], [], []
                sup_kls, sup_coss, sup_ch, sup_ov = [], [], [], []

                for p_idx, prompt in enumerate(prompts):
                    try:
                        key = (cat, p_idx, sp_id)
                        if key not in baseline_cache:
                            baseline_cache[key] = self._get_logits(prompt)
                        orig = baseline_cache[key]

                        h_a = self._create_amplify_hook(v, alpha)
                        la = self._run_with_hook(prompt, h_a, layer_idx)
                        h_s = self._create_suppress_hook(v)
                        ls = self._run_with_hook(prompt, h_s, layer_idx)

                        amp_kls.append(self._kl_divergence(orig, la))
                        amp_coss.append(self._cosine_sim(orig, la))
                        amp_ch.append(float(self._top_token_changed(orig, la)))
                        amp_ov.append(self._top_k_overlap(orig, la))

                        sup_kls.append(self._kl_divergence(orig, ls))
                        sup_coss.append(self._cosine_sim(orig, ls))
                        sup_ch.append(float(self._top_token_changed(orig, ls)))
                        sup_ov.append(self._top_k_overlap(orig, ls))
                    except Exception as e:
                        logger.warning(f"Intervention failed: {e}")

                if amp_kls:
                    rows.append({
                        "Species": species.name, "Category": cat,
                        "Intervention": "amplify",
                        "KL_Divergence": round(float(np.mean(amp_kls)), 4),
                        "KL_Std": round(float(np.std(amp_kls)), 4),
                        "Cosine_Sim": round(float(np.mean(amp_coss)), 4),
                        "CosSim_Std": round(float(np.std(amp_coss)), 4),
                        "Top_Token_Changed": round(float(np.mean(amp_ch)), 2),
                        "Top10_Overlap": round(float(np.mean(amp_ov)), 2),
                        "Top10_Std": round(float(np.std(amp_ov)), 2),
                        "N_Prompts": len(amp_kls),
                        "Reasoning_Acc_Base": round(reasoning_base, 4),
                        "Reasoning_Acc_Steered": round(r_amp, 4),
                        "Reasoning_Acc_Delta": round(r_amp - reasoning_base, 4),
                        "Coref_Score_Base": round(coref_base, 4),
                        "Coref_Score_Steered": round(c_amp, 4),
                        "Coref_Score_Delta": round(c_amp - coref_base, 4),
                    })
                if sup_kls:
                    rows.append({
                        "Species": species.name, "Category": cat,
                        "Intervention": "suppress",
                        "KL_Divergence": round(float(np.mean(sup_kls)), 4),
                        "KL_Std": round(float(np.std(sup_kls)), 4),
                        "Cosine_Sim": round(float(np.mean(sup_coss)), 4),
                        "CosSim_Std": round(float(np.std(sup_coss)), 4),
                        "Top_Token_Changed": round(float(np.mean(sup_ch)), 2),
                        "Top10_Overlap": round(float(np.mean(sup_ov)), 2),
                        "Top10_Std": round(float(np.std(sup_ov)), 2),
                        "N_Prompts": len(sup_kls),
                        "Reasoning_Acc_Base": round(reasoning_base, 4),
                        "Reasoning_Acc_Steered": round(r_sup, 4),
                        "Reasoning_Acc_Delta": round(r_sup - reasoning_base, 4),
                        "Coref_Score_Base": round(coref_base, 4),
                        "Coref_Score_Steered": round(c_sup, 4),
                        "Coref_Score_Delta": round(c_sup - coref_base, 4),
                    })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Gauge-Covariant Interventions (v5 NEW)
    # ------------------------------------------------------------------

    def _parallel_transport_vector(
        self,
        v: torch.Tensor,
        source_layer: int,
        target_layer: int,
        attention_maps: Dict[int, torch.Tensor],
        residual_field: torch.Tensor,
    ) -> torch.Tensor:
        """Transport a D-dimensional vector through the attention gauge connection.

        Algorithm:
        1. Project v into token-space using residual at source layer
        2. Transport through attention maps layer by layer
        3. Reconstruct D-dimensional vector at target layer

        Args:
            v: [D] species vector at source_layer
            source_layer: layer where v is defined
            target_layer: layer to transport to
            attention_maps: {layer_idx: [H, T, T]} attention weights
            residual_field: [L+1, T, D] residual stream

        Returns: [D] transported vector at target_layer
        """
        if source_layer == target_layer:
            return v.clone()

        v_norm = v.float() / (v.float().norm() + 1e-8)

        # Step 1: Project into token-space
        R_source = residual_field[source_layer].float()  # [T, D]
        c = R_source @ v_norm                             # [T]

        # Step 2: Transport through attention
        sorted_attn_layers = sorted(attention_maps.keys())

        if target_layer > source_layer:
            # Forward transport
            for l_attn in sorted_attn_layers:
                if l_attn < source_layer:
                    continue
                if l_attn >= target_layer:
                    break
                A = attention_maps[l_attn].float().mean(dim=0)  # [T, T]
                c = A.T @ c
        else:
            # Backward transport (adjoint)
            for l_attn in reversed(sorted_attn_layers):
                if l_attn >= source_layer:
                    continue
                if l_attn < target_layer:
                    break
                A = attention_maps[l_attn].float().mean(dim=0)  # [T, T]
                c = A @ c

        # Step 3: Reconstruct in D-space
        R_target = residual_field[target_layer].float()  # [T, D]
        v_transported = R_target.T @ c                    # [D]
        v_transported = v_transported / (v_transported.norm() + 1e-8)

        return v_transported

    def run_gauge_covariance_experiment(
        self,
        catalog: IntellitonCatalog,
        mode_vectors: Dict[int, torch.Tensor],
        attention_maps: Dict[int, torch.Tensor],
        residual_field: torch.Tensor,
        test_prompts: Dict[str, List[str]],
    ) -> GaugeCovarianceResult:
        """Compare gauge-fixed vs gauge-covariant interventions.

        For each species, at each test layer:
        1. Raw: amplify with original v at target_layer
        2. Transported: amplify with PT(v, ref->target) at target_layer
        3. Measure KL divergence and top-token change rate
        """
        ref_layer = self.cfg.steering_layer
        test_layers = self.cfg.gauge_covariance_layers
        alpha = self.cfg.steering_alpha
        n_prompts_per_cat = self.cfg.gauge_covariance_n_prompts

        species_list = catalog.species
        n_species = len(species_list)
        n_layers_test = len(test_layers)

        kl_raw = torch.zeros(n_species, n_layers_test)
        kl_transported = torch.zeros(n_species, n_layers_test)
        tc_raw = torch.zeros(n_species, n_layers_test)
        tc_transported = torch.zeros(n_species, n_layers_test)
        transport_fidelity = torch.zeros(n_species, n_layers_test)
        species_names = []

        # Flatten test prompts (small subset for efficiency)
        flat_prompts = []
        for cat, prompts in test_prompts.items():
            flat_prompts.extend(prompts[:n_prompts_per_cat])

        logger.info(f"Gauge covariance: {n_species} species x {n_layers_test} layers "
                    f"x {len(flat_prompts)} prompts")

        for sp_idx, species in enumerate(species_list):
            sp_id = species.species_id
            species_names.append(species.name)
            if sp_id not in mode_vectors:
                continue
            v_raw = mode_vectors[sp_id]

            for l_idx, target_layer in enumerate(test_layers):
                # Transport vector from ref to target
                v_transported = self._parallel_transport_vector(
                    v_raw, ref_layer, target_layer,
                    attention_maps, residual_field,
                )

                # Measure transport fidelity (cosine sim of transported vs raw)
                transport_fidelity[sp_idx, l_idx] = F.cosine_similarity(
                    v_raw.unsqueeze(0).float(),
                    v_transported.unsqueeze(0).float(),
                ).item()

                raw_kls, pt_kls = [], []
                raw_tcs, pt_tcs = [], []

                for prompt in flat_prompts:
                    try:
                        baseline = self._get_logits(prompt)

                        # Raw intervention at target layer
                        hook_r = self._create_amplify_hook(v_raw, alpha)
                        logits_r = self._run_with_hook(prompt, hook_r, target_layer)
                        raw_kls.append(self._kl_divergence(baseline, logits_r))
                        raw_tcs.append(float(self._top_token_changed(baseline, logits_r)))

                        # Transported intervention at target layer
                        hook_t = self._create_amplify_hook(v_transported, alpha)
                        logits_t = self._run_with_hook(prompt, hook_t, target_layer)
                        pt_kls.append(self._kl_divergence(baseline, logits_t))
                        pt_tcs.append(float(self._top_token_changed(baseline, logits_t)))
                    except Exception as e:
                        logger.warning(f"Gauge covariance failed: {e}")

                if raw_kls:
                    kl_raw[sp_idx, l_idx] = np.mean(raw_kls)
                    kl_transported[sp_idx, l_idx] = np.mean(pt_kls)
                    tc_raw[sp_idx, l_idx] = np.mean(raw_tcs)
                    tc_transported[sp_idx, l_idx] = np.mean(pt_tcs)

        gauge_violation = kl_raw - kl_transported

        logger.info(f"Gauge violation mean: {gauge_violation.abs().mean():.4f}")
        logger.info(f"Transport fidelity range: "
                    f"[{transport_fidelity.min():.3f}, {transport_fidelity.max():.3f}]")

        return GaugeCovarianceResult(
            layers_tested=test_layers,
            kl_raw=kl_raw,
            kl_transported=kl_transported,
            gauge_violation=gauge_violation,
            top_token_change_raw=tc_raw,
            top_token_change_transported=tc_transported,
            transport_fidelity=transport_fidelity,
            species_names=species_names,
        )
