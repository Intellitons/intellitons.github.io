"""Intelliton Analyzer v5 -- Main Orchestrator.

Lattice Gauge Theory upgrade with:
  - Discrete propagator + pole mass extraction
  - Lattice dispersion relation E(k)
  - Non-abelian Wilson loops + Polyakov loops
  - RoPE U(1) gauge analysis + ablation
  - EFT renormalization flow
  - Phase transition detection
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.config import IntellitonConfig
from src.datasets import get_prompts_by_category
from src.differential_spectroscopy import DifferentialSpectroscopyAnalyzer
from src.gauge_analyzer import GaugeAnalyzer
from src.gauge_intervention import GaugeIntervention
from src.intelliton_classifier import IntellitonCatalog, IntellitonClassifier
from src.lattice_field import LatticeField, SpinDecomposition
from src.eft_renormalization import EFTRenormalization
from src.rope_gauge import RoPEGaugeAnalyzer
from src.visualization import IntellitonVisualizer

logger = logging.getLogger(__name__)


class IntellitonAnalyzer:
    """Main orchestrator for Intelliton Spectrum Analyzer v5."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._hooks = []
        self._current_residuals = []
        self._current_embedding = None
        self._current_attn_weights = {}

    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {self.cfg.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=getattr(torch, self.cfg.dtype),
            device_map=self.cfg.device_map,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Model loaded: {n_params:.2f}B params")
        self._install_hooks()

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

    def _install_hooks(self):
        lang = self._get_language_model()

        # Embedding hook
        h = lang.embed_tokens.register_forward_hook(self._embedding_hook())
        self._hooks.append(h)

        # Decoder layer hooks
        for i, layer in enumerate(lang.layers):
            h = layer.register_forward_hook(self._residual_hook(i))
            self._hooks.append(h)

        # Attention weight hooks
        # Only capture a subset to save memory
        capture_layers = list(range(0, self.cfg.num_layers, 4))  # every 4th layer
        for i in capture_layers:
            if i < len(lang.layers) and hasattr(lang.layers[i], "self_attn"):
                h = lang.layers[i].self_attn.register_forward_hook(
                    self._attention_hook(i)
                )
                self._hooks.append(h)

        logger.info(f"Installed {len(self._hooks)} hooks "
                    f"(capturing attention at layers: {capture_layers})")

    def _embedding_hook(self):
        def hook(module, input, output):
            self._current_embedding = output.detach()
        return hook

    def _residual_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self._current_residuals.append(output.detach())
        return hook

    def _attention_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self._current_attn_weights[layer_idx] = attn_weights.detach()
        return hook

    def _clear_current(self):
        self._current_residuals = []
        self._current_embedding = None
        self._current_attn_weights = {}

    def run_inference(self, prompts):
        residual_streams = []
        attention_maps = []

        for prompt in tqdm(prompts, desc="Inference"):
            self._clear_current()
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.cfg.analysis_seq_len, padding=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                self.model(**inputs, output_attentions=True, use_cache=False)

            if self._current_embedding is not None and self._current_residuals:
                emb = self._current_embedding[0].float().cpu()
                layers = [r[0].float().cpu() for r in self._current_residuals]
                field = torch.stack([emb] + layers, dim=0)
                field = torch.where(torch.isfinite(field), field, torch.zeros_like(field))
                residual_streams.append(field)
            else:
                logger.warning(f"Failed capture for: {prompt[:50]}...")
                continue

            attn_map = {}
            for li, w in self._current_attn_weights.items():
                attn_map[li] = w[0].float().cpu()
            attention_maps.append(attn_map)

        logger.info(f"Captured {len(residual_streams)} residual streams")
        return residual_streams, attention_maps

    def run_inference_on_token_ids(self, input_ids: List[int]) -> torch.Tensor:
        """Capture a single residual field from an explicit token-id prefix."""
        self._clear_current()
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones_like(ids, device=self.model.device)

        with torch.no_grad():
            self.model(input_ids=ids, attention_mask=attention_mask, output_attentions=True, use_cache=False)

        if self._current_embedding is None or not self._current_residuals:
            raise RuntimeError("Failed to capture residual stream from explicit token ids")

        emb = self._current_embedding[0].float().cpu()
        layers = [r[0].float().cpu() for r in self._current_residuals]
        field = torch.stack([emb] + layers, dim=0)
        field = torch.where(torch.isfinite(field), field, torch.zeros_like(field))
        return field

    def run_full_pipeline(self, skip_intervention=False,
                          skip_vacuum=False, skip_fusion=False,
                          skip_gauge_covariance=False,
                          skip_topological=False,
                          skip_resonance=False,
                          skip_differential_spectroscopy=False,
                          skip_hallucination=False):
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # STEP 1: Load model
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 1: Loading model")
        logger.info("=" * 60)
        self.load_model()

        # ============================================================
        # STEP 2: Inference and activation capture
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Running inference")
        logger.info("=" * 60)
        prompts_by_cat = get_prompts_by_category()
        all_fields_by_cat = {}
        all_attn_maps_by_cat = {}

        for cat, prompts in prompts_by_cat.items():
            if self.cfg.max_prompts_per_category is not None:
                prompts = prompts[:self.cfg.max_prompts_per_category]
            logger.info(f"Category: {cat} ({len(prompts)} prompts)")
            streams, attn_maps = self.run_inference(prompts)
            all_fields_by_cat[cat] = [LatticeField(s, self.cfg) for s in streams]
            all_attn_maps_by_cat[cat] = attn_maps

        # ============================================================
        # STEP 2.5: Vacuum Expectation Value Analysis (v5 NEW)
        # ============================================================
        vacuum_result = None
        if self.cfg.vacuum_enabled and not skip_vacuum:
            logger.info("=" * 60)
            logger.info("STEP 2.5: Vacuum Analysis (Symmetry Breaking)")
            logger.info("=" * 60)
            from vacuum_analysis import VacuumAnalyzer
            vac_analyzer = VacuumAnalyzer(self.cfg)

            vacuum_field = vac_analyzer.run_vacuum_inference(self)
            vev, vev_norm = vac_analyzer.compute_vev(vacuum_field)
            logger.info(f"VEV norm range: [{vev_norm.min():.2f}, {vev_norm.max():.2f}]")

            vacuum_result = vac_analyzer.analyze_symmetry_breaking(
                {cat: fields for cat, fields in all_fields_by_cat.items()},
                vev,
            )
            vacuum_result.vacuum_field = vacuum_field
            logger.info(f"Symmetry breaking analyzed for {len(vacuum_result.order_parameter_by_cat)} categories")

        # ============================================================
        # STEP 3: Lattice Field Theory Analysis
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 3: Lattice Field Theory Analysis")
        logger.info("=" * 60)
        viz = IntellitonVisualizer(self.cfg)

        first_cat = list(all_fields_by_cat.keys())[0]
        rep_field = all_fields_by_cat[first_cat][0]

        # Momentum spectrum
        power = rep_field.momentum_power_spectrum()
        momenta = rep_field.brillouin_zone_momenta()
        logger.info(f"Momentum spectrum: {power.shape}")
        viz.plot_momentum_spectrum(power, momenta, title_suffix=f" ({first_cat})")

        # Spin decomposition
        decomp = rep_field.compute_spin_decomposition()
        logger.info(f"SVD: sigma {decomp.sigma.shape}")
        viz.plot_spin_spectrum(decomp.sigma, title_suffix=f" ({first_cat})")

        # Mass / propagator (v5: pole extraction)
        propagator = rep_field.compute_propagator(decomp)
        logger.info(f"Mass spectrum: {propagator.mass_category[:5]}")
        viz.plot_mass_spectrum(propagator, title_suffix=f" ({first_cat})")

        # Dispersion relation (v5 NEW)
        logger.info("Computing lattice dispersion relation...")
        dispersion = rep_field.compute_dispersion_relation(decomp)
        logger.info(f"Dispersion masses: {dispersion.fitted_mass[:5]}")
        viz.plot_dispersion_relation(dispersion)

        # Lorentzian resonance widths (v5 NEW)
        resonance_result = None
        if self.cfg.resonance_width_enabled and not skip_resonance:
            logger.info("Fitting Lorentzian resonance widths...")
            resonance_result = rep_field.compute_resonance_widths(
                propagator,
                sharpness_threshold=self.cfg.resonance_sharpness_threshold,
            )
            n_sharp = sum(1 for wd in resonance_result.is_well_defined if wd)
            n_total = len(resonance_result.fits)
            logger.info(f"Resonance widths: {n_sharp}/{n_total} well-defined quasiparticles")
            for i, fit in enumerate(resonance_result.fits[:5]):
                logger.info(f"  Mode {i}: omega_0={fit.omega_0:.3f}, "
                           f"Gamma={fit.gamma:.3f}, tau={fit.lifetime:.1f}, "
                           f"R^2={fit.fit_r_squared:.3f}")
            viz.plot_resonance_widths(resonance_result)

        # Helicity
        helicity = rep_field.compute_helicity(decomp)
        viz.plot_helicity_conservation(helicity)

        # ============================================================
        # STEP 3.5: Differential Spectroscopy (v5 EXPERIMENTAL)
        # ============================================================
        differential_result = None
        if self.cfg.differential_spectroscopy_enabled and not skip_differential_spectroscopy:
            logger.info("=" * 60)
            logger.info("STEP 3.5: Differential Spectroscopy")
            logger.info("=" * 60)
            try:
                diffspec = DifferentialSpectroscopyAnalyzer(self.cfg)
                vacuum_field = None
                if vacuum_result is not None and hasattr(vacuum_result, "vacuum_field"):
                    vacuum_field = vacuum_result.vacuum_field
                differential_result = diffspec.analyze(
                    all_fields_by_cat,
                    vacuum_field=vacuum_field,
                )
                if not differential_result.candidate_df.empty:
                    differential_result.candidate_df.to_csv(
                        output_dir / "differential_spectrum_candidates.csv",
                        index=False,
                    )
                    logger.info(
                        f"Differential spectroscopy: {len(differential_result.candidates)} recurrent candidates"
                    )
                    for row in differential_result.candidate_df.head(5).itertuples():
                        logger.info(
                            f"  {row.category} mode {row.mode_index}: "
                            f"omega={row.omega_0:.3f}, Gamma={row.gamma:.3f}, "
                            f"score={row.score:.3f}, recurrence={row.recurrence:.2f}"
                        )
                else:
                    logger.info("Differential spectroscopy: no recurrent narrow candidates found")
                viz.plot_differential_spectroscopy(differential_result)
            except Exception as e:
                logger.error(f"Differential spectroscopy failed: {e}", exc_info=True)

        # ============================================================
        # STEP 4: EFT Renormalization (v5 NEW)
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 4: EFT Renormalization Flow")
        logger.info("=" * 60)
        eft_engine = EFTRenormalization(self.cfg)

        rg_flow = eft_engine.compute_rg_flow(decomp, propagator)
        logger.info(f"RG fixed point types: {rg_flow.fixed_point_type[:5]}")
        viz.plot_rg_flow(rg_flow)

        eft_params = eft_engine.compute_eft_parameters(decomp, propagator)
        logger.info(f"EFT parameters: alpha={eft_params.alpha:.4f}, beta={eft_params.beta:.4f}")
        viz.plot_eft_parameters(eft_params)

        transitions = eft_engine.detect_phase_transitions(decomp)
        logger.info(f"Phase transitions: {len(transitions)} detected")
        viz.plot_phase_transitions(transitions, decomp.sigma)

        # ============================================================
        # STEP 5: Wigner Classification (v5 upgraded)
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 5: Wigner Classification")
        logger.info("=" * 60)
        classifier = IntellitonClassifier(self.cfg)
        catalog = classifier.classify(
            all_fields_by_cat,
            rg_flow=rg_flow,
            eft_params=eft_params,
            dispersion=dispersion,
        )
        self.species_vectors = dict(classifier.species_vectors)
        viz.plot_particle_table(catalog)

        df = catalog.build_dataframe()
        df.to_csv(output_dir / "intelliton_catalog.csv", index=False)
        logger.info(f"Catalog: {len(catalog.species)} species")

        # ============================================================
        # STEP 6: Gauge Field Analysis
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 6: Gauge Field Analysis")
        logger.info("=" * 60)
        gauge = GaugeAnalyzer(self.cfg)

        first_attn = all_attn_maps_by_cat[first_cat]
        if first_attn and first_attn[0]:
            rep_attn = first_attn[0]

            # Wilson loops
            wilson = gauge.compute_wilson_loops(rep_attn)
            logger.info(f"Wilson loops: {len(wilson.plaquettes)} plaquettes")
            logger.info(f"Plaquette traces: {wilson.plaquette_traces}")
            viz.plot_wilson_loops(wilson)

            # Polyakov loops (v5 NEW)
            polyakov = gauge.compute_polyakov_loops(rep_attn)
            logger.info(f"Polyakov loop: <P>={polyakov.expectation_value:.4f}, "
                       f"phase={polyakov.phase}")
            viz.plot_polyakov_loops(polyakov)

            # Creutz ratios (v5 NEW)
            creutz = gauge.compute_creutz_ratios(rep_attn)
            logger.info(f"String tension: sigma={creutz.string_tension:.4f}, "
                       f"area_law={creutz.area_law}")

            # Topological charge (v5 NEW)
            topo_result = None
            if self.cfg.topological_charge_enabled and not skip_topological:
                logger.info("Computing topological charge...")
                topo_result = gauge.compute_topological_charge(
                    rep_attn, wilson,
                    quantization_threshold=self.cfg.topological_quantization_threshold,
                )
                viz.plot_topological_charge(topo_result)

            # Parallel transport
            transport = gauge.verify_parallel_transport(decomp, rep_attn)
            logger.info(f"Parallel transport: avg={transport.avg_alignment_per_mode.mean():.3f}")
            viz.plot_parallel_transport(transport)

            # Helicity conservation
            h_stab = gauge.measure_helicity_conservation(helicity)
            logger.info(f"Helicity stability: {h_stab[:5]}")
        else:
            logger.warning("No attention weights; skipping gauge analysis")

        # ============================================================
        # STEP 6.5: Intelliton Fusion Tracking (v5 NEW)
        # ============================================================
        fusion_result = None
        if self.cfg.fusion_tracking_enabled and not skip_fusion:
            logger.info("=" * 60)
            logger.info("STEP 6.5: Intelliton Fusion Tracking")
            logger.info("=" * 60)
            try:
                from src.fusion_tracker import IntellitonFusionTracker
                tracker = IntellitonFusionTracker(self.cfg)

                first_attn_for_fusion = all_attn_maps_by_cat[first_cat]
                rep_attn_fusion = first_attn_for_fusion[0] if first_attn_for_fusion else {}

                fusion_result = tracker.track_fusion(decomp, rep_attn_fusion)
                logger.info(f"Fusion events: {len(fusion_result.fusion_events)}")
                logger.info(f"Coupling matrix shape: {fusion_result.coupling_matrix.shape}")

                for event in fusion_result.fusion_events[:5]:
                    logger.info(
                        f"  Layer {event.layer}: modes {event.source_modes} -> "
                        f"mode {event.target_mode} (strength={event.coupling_strength:.3f})"
                    )
                viz.plot_fusion_tree(fusion_result)
            except Exception as e:
                logger.error(f"Fusion tracking failed: {e}", exc_info=True)

        # ============================================================
        # STEP 7: RoPE Gauge Analysis (v5 NEW)
        # ============================================================
        logger.info("=" * 60)
        logger.info("STEP 7: RoPE U(1) Gauge Analysis")
        logger.info("=" * 60)
        rope_analyzer = RoPEGaugeAnalyzer(self.cfg)

        rope_struct = rope_analyzer.extract_rope_structure(rep_field.seq_len)
        logger.info(f"RoPE frequencies: {rope_struct.frequencies[:5]}")

        # Gauge connection
        links = rope_analyzer.compute_gauge_connection(rope_struct)
        logger.info(f"Gauge links shape: {links.shape}")

        # Field strength
        curvature = rope_analyzer.compute_field_strength(rope_struct)
        logger.info(f"RoPE curvature (should be ~0): max={curvature.abs().max():.6f}")

        # Helicity conservation under RoPE
        hc = rope_analyzer.verify_helicity_conservation(decomp, rope_struct)
        logger.info(f"Helicity conservation: avg={hc.avg_conservation.mean():.4f}")

        # RoPE ablation experiment
        logger.info("Running RoPE ablation experiment...")
        ablation = rope_analyzer.run_rope_ablation(
            self.model, self.tokenizer, decomp,
            list(prompts_by_cat.values())[0][:5],
        )
        logger.info(f"Ablation KL divergences: {ablation.kl_divergences}")
        logger.info(f"Circuit collapses: {ablation.circuit_collapsed}")
        viz.plot_rope_ablation(ablation)

        # ============================================================
        # STEP 7.5: Hallucination Diagnostics (v5 NEW)
        # ============================================================
        halluc_result = None
        if self.cfg.hallucination_diagnostic_enabled and not skip_hallucination:
            logger.info("=" * 60)
            logger.info("STEP 7.5: Hallucination Diagnostics")
            logger.info("=" * 60)
            try:
                from src.hallucination_diagnostic import HallucinationDiagnostic
                from src.datasets import HALLUCINATION_PAIRS
                halluc_diag = HallucinationDiagnostic(self.cfg)
                halluc_result = halluc_diag.run_diagnostics(self, HALLUCINATION_PAIRS)
                logger.info(f"Hallucination analysis: {len(halluc_result.signatures)} pairs")
                logger.info(f"  Critical layer: {halluc_result.critical_layer}")
                logger.info(f"  Hallucination rate: {halluc_result.hallucination_rate:.1%}")
                logger.info(f"  Confidence gap: {halluc_result.confidence_gap:.4f}")
                viz.plot_hallucination_diagnostics(halluc_result)
            except Exception as e:
                logger.error(f"Hallucination diagnostics failed: {e}", exc_info=True)

        # ============================================================
        # STEP 8: Intervention Experiments
        # ============================================================
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info("Removed capture hooks")

        if not skip_intervention and catalog.species:
            logger.info("=" * 60)
            logger.info("STEP 8: Gauge Intervention Experiments")
            logger.info("=" * 60)
            try:
                intervener = GaugeIntervention(self.cfg, self.model, self.tokenizer)
                mode_vectors = {}
                species_for_iv = catalog.species
                if self.cfg.max_species_for_intervention is not None:
                    species_for_iv = species_for_iv[:self.cfg.max_species_for_intervention]

                for sp in species_for_iv:
                    if sp.species_id in classifier.species_vectors:
                        mode_vectors[sp.species_id] = classifier.species_vectors[sp.species_id]

                test_prompts = prompts_by_cat
                if self.cfg.max_prompts_per_category is not None:
                    test_prompts = {c: ps[:self.cfg.max_prompts_per_category]
                                   for c, ps in test_prompts.items()}

                from src.intelliton_classifier import IntellitonCatalog as IC
                limited = IC(species_for_iv, catalog.eft_alpha, catalog.eft_beta)
                results_df = intervener.run_intervention_experiment(
                    limited, mode_vectors, test_prompts
                )
                if not results_df.empty:
                    results_df.to_csv(output_dir / "intervention_results.csv", index=False)
                    logger.info(f"Interventions: {len(results_df)} experiments")
                    viz.plot_intervention_results(results_df)

                # --- Gauge-Covariant Intervention Experiment (v5 NEW) ---
                if (self.cfg.gauge_covariance_enabled and not skip_gauge_covariance
                        and mode_vectors):
                    first_attn_gc = all_attn_maps_by_cat[first_cat]
                    rep_attn_gc = first_attn_gc[0] if first_attn_gc else {}
                    rep_residual = all_fields_by_cat[first_cat][0].field

                    if rep_attn_gc:
                        logger.info("Running gauge-covariance experiment...")
                        gc_result = intervener.run_gauge_covariance_experiment(
                            limited, mode_vectors, rep_attn_gc,
                            rep_residual, test_prompts,
                        )
                        viz.plot_gauge_covariance(gc_result)
                        torch.save({
                            "kl_raw": gc_result.kl_raw,
                            "kl_transported": gc_result.kl_transported,
                            "gauge_violation": gc_result.gauge_violation,
                            "transport_fidelity": gc_result.transport_fidelity,
                            "layers_tested": gc_result.layers_tested,
                        }, output_dir / "gauge_covariance.pt")

            except Exception as e:
                logger.error(f"Intervention failed: {e}", exc_info=True)

        # VEV visualization (deferred to here so it's near other plots)
        if vacuum_result is not None:
            viz.plot_vev(vacuum_result)

        # Summary
        viz.plot_summary(power, momenta, decomp.sigma, propagator, dispersion, rg_flow)

        # Save tensors
        if self.cfg.save_tensors:
            save_dict = {
                "momentum_power": power,
                "momenta": momenta,
                "sigma": decomp.sigma,
                "mass_pole": propagator.mass,
                "mass_lattice": propagator.mass_lattice,
                "spectral_function": propagator.spectral_function,
                "decay_curves": propagator.decay_curves,
                "helicity": helicity,
                "dispersion_energies": dispersion.energies,
                "dispersion_mass": dispersion.fitted_mass,
                "rg_running_mass": rg_flow.running_mass,
                "rg_beta": rg_flow.beta_function,
                "eft_alpha": eft_params.alpha,
                "eft_beta": eft_params.beta,
                "bare_mass": eft_params.bare_mass,
                "renorm_mass": eft_params.renormalized_mass,
                "Z": eft_params.Z,
            }
            if vacuum_result is not None:
                save_dict["vev"] = vacuum_result.vev
                save_dict["vev_norm"] = vacuum_result.vev_norm
            if fusion_result is not None:
                save_dict["fusion_coupling"] = fusion_result.coupling_matrix
                save_dict["fusion_rate"] = fusion_result.fusion_rate
            if resonance_result is not None:
                save_dict["resonance_omega"] = resonance_result.omega
                save_dict["resonance_gamma"] = torch.tensor(
                    [f.gamma for f in resonance_result.fits]
                )
                save_dict["resonance_lifetime"] = torch.tensor(
                    [f.lifetime for f in resonance_result.fits]
                )
            if differential_result is not None:
                save_dict["diffspec_omega"] = differential_result.omega
                if differential_result.category_total_spectrum:
                    cat_names = list(differential_result.category_total_spectrum.keys())
                    save_dict["diffspec_categories"] = cat_names
                    save_dict["diffspec_total_spectrum"] = torch.stack(
                        [differential_result.category_total_spectrum[cat] for cat in cat_names]
                    )
                    save_dict["diffspec_mode_scores"] = torch.stack(
                        [differential_result.category_mode_scores[cat] for cat in cat_names]
                    )
            if halluc_result is not None:
                save_dict["halluc_spectral_divergence"] = halluc_result.mean_spectral_divergence
                save_dict["halluc_entropy_gap"] = halluc_result.mean_entropy_gap
            torch.save(save_dict, output_dir / "analysis_tensors.pt")

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Results: {output_dir}")
        logger.info("=" * 60)

        return catalog
