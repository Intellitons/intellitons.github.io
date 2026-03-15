"""Paper-focused Intelliton pipeline.

This module keeps only the experiments most aligned with a paper on
Intelliton discovery, characterization, and applications:
  1. Spectrum / propagator / dispersion / helicity
  2. EFT renormalization flow
  3. Intelliton classification catalog
  4. Hallucination diagnostics
  5. Generation-time Intelliton dynamics trajectory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

from src.config import IntellitonConfig
from src.datasets import HALLUCINATION_PAIRS, TRAJECTORY_STYLE_CONTROLS, get_prompts_by_category
from src.eft_renormalization import EFTRenormalization
from src.hallucination_diagnostic import HallucinationDiagnostic
from src.intelliton_analyzer import IntellitonAnalyzer
from src.intelliton_classifier import IntellitonCatalog, IntellitonClassifier
from src.intelliton_dynamics import IntellitonDynamicsTracker, build_transition_matrix
from src.lattice_field import LatticeField
from src.visualization import IntellitonVisualizer

logger = logging.getLogger(__name__)


class PaperIntellitonPipeline:
    """Standalone paper-oriented runner built on top of the v5 codebase."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = IntellitonAnalyzer(cfg)
        self.visualizer = IntellitonVisualizer(cfg)
        self._latest_propagator = None

    def _collect_fields(self) -> tuple[Dict[str, List[LatticeField]], Dict[str, List[dict]]]:
        prompts_by_cat = get_prompts_by_category()
        all_fields_by_cat: Dict[str, List[LatticeField]] = {}
        all_attn_maps_by_cat: Dict[str, List[dict]] = {}

        for cat, prompts in prompts_by_cat.items():
            if self.cfg.max_prompts_per_category is not None:
                prompts = prompts[: self.cfg.max_prompts_per_category]
            logger.info("Category: %s (%d prompts)", cat, len(prompts))
            streams, attn_maps = self.analyzer.run_inference(prompts)
            all_fields_by_cat[cat] = [LatticeField(s, self.cfg) for s in streams]
            all_attn_maps_by_cat[cat] = attn_maps

        return all_fields_by_cat, all_attn_maps_by_cat

    def run_core_analysis(self) -> IntellitonCatalog:
        logger.info("=" * 60)
        logger.info("PAPER STEP 1: Loading model")
        logger.info("=" * 60)
        self.analyzer.load_model()

        logger.info("=" * 60)
        logger.info("PAPER STEP 2: Capture residual streams")
        logger.info("=" * 60)
        all_fields_by_cat, _ = self._collect_fields()

        first_cat = next(iter(all_fields_by_cat))
        rep_field = all_fields_by_cat[first_cat][0]

        logger.info("=" * 60)
        logger.info("PAPER STEP 3: Spectrum and quasiparticle observables")
        logger.info("=" * 60)
        power = rep_field.momentum_power_spectrum()
        momenta = rep_field.brillouin_zone_momenta()
        # self.visualizer.plot_momentum_spectrum(power, momenta, title_suffix=f" ({first_cat})")

        decomp = rep_field.compute_spin_decomposition()
        self.visualizer.plot_spin_spectrum(decomp.sigma, title_suffix=f" ({first_cat})")

        propagator = rep_field.compute_propagator(decomp)
        self._latest_propagator = propagator
        self.visualizer.plot_mass_spectrum(propagator, title_suffix=f" ({first_cat})")

        dispersion = rep_field.compute_dispersion_relation(decomp)
        self.visualizer.plot_dispersion_relation(dispersion)

        helicity = rep_field.compute_helicity(decomp)
        # self.visualizer.plot_helicity_conservation(helicity)
        self.visualizer.plot_momentum_and_helicity(power, momenta, helicity, title_suffix=f" ({first_cat})")

        logger.info("=" * 60)
        logger.info("PAPER STEP 4: EFT renormalization")
        logger.info("=" * 60)
        eft = EFTRenormalization(self.cfg)
        rg_flow = eft.compute_rg_flow(decomp, propagator)
        eft_params = eft.compute_eft_parameters(decomp, propagator)
        transitions = eft.detect_phase_transitions(decomp)
        self.visualizer.plot_rg_flow(rg_flow)
        self.visualizer.plot_eft_parameters(eft_params)
        self.visualizer.plot_phase_transitions(transitions, decomp.sigma)

        logger.info("=" * 60)
        logger.info("PAPER STEP 5: Intelliton classification")
        logger.info("=" * 60)
        classifier = IntellitonClassifier(self.cfg)
        catalog = classifier.classify(
            all_fields_by_cat,
            rg_flow=rg_flow,
            eft_params=eft_params,
            dispersion=dispersion,
        )
        self.analyzer.species_vectors = dict(classifier.species_vectors)
        self.visualizer.plot_particle_table(catalog)
        catalog.build_dataframe().to_csv(self.output_dir / "intelliton_catalog.csv", index=False)

        # self.visualizer.plot_summary(power, momenta, decomp.sigma, propagator, dispersion, rg_flow)
        return catalog

    def run_hallucination_analysis(self) -> None:
        logger.info("=" * 60)
        logger.info("PAPER STEP 6: Hallucination diagnostics")
        logger.info("=" * 60)
        diag = HallucinationDiagnostic(self.cfg)
        result = diag.run_diagnostics(self.analyzer, HALLUCINATION_PAIRS)
        self.visualizer.plot_hallucination_diagnostics(result)

    def run_trajectory_analysis(
        self,
        catalog: IntellitonCatalog,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        local_window: int,
    ) -> None:
        logger.info("=" * 60)
        logger.info("PAPER STEP 7: Intelliton dynamics trajectory")
        logger.info("=" * 60)

        tracker = IntellitonDynamicsTracker(self.cfg, catalog, self.analyzer.species_vectors)
        tracker.fit_grounded_profile(
            self.analyzer,
            [pair["grounded"] for pair in HALLUCINATION_PAIRS],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            local_window=local_window,
        )
        prompt_groups: Dict[str, Sequence[str]] = {
            "grounded_factual": [pair["grounded"] for pair in HALLUCINATION_PAIRS],
            "stylistic_continuation": TRAJECTORY_STYLE_CONTROLS,
            "hallucination_prone": [pair["hallucination_prone"] for pair in HALLUCINATION_PAIRS],
        }
        detail_df = tracker.trace_prompt_groups(
            self.analyzer,
            prompt_groups,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            local_window=local_window,
        )
        detail_df.to_csv(self.output_dir / "intelliton_trajectory_detail.csv", index=False)

        transition_df = build_transition_matrix(detail_df, top_k=1)
        transition_df.to_csv(self.output_dir / "intelliton_transition_graph.csv", index=False)

        summary_df = (
            detail_df.groupby(["Prompt_Type", "Step_Index"], as_index=False)
            .agg(
                Mean_Mode_Activation_Shift=("Mode_Activation_Shift", "mean"),
                Mean_Grounded_Deviation=("Grounded_Deviation", "mean"),
                Mean_Top_Occupation=("Occupation", "mean"),
                Mean_Mass=("Mass_t", "mean"),
                Mean_Z=("Z_t", "mean"),
                Mean_Gamma=("gamma_t", "mean"),
                Mean_Confidence=("Top_Token_Confidence", "mean"),
            )
            .sort_values(["Prompt_Type", "Step_Index"])
        )
        summary_df.to_csv(self.output_dir / "intelliton_trajectory_summary.csv", index=False)
        # Produce a single merged figure combining the summary (top row) and
        # the occupation trajectories (bottom row) with shared legends.
        self.visualizer.plot_merged_trajectory_and_occupations(summary_df, detail_df)
        self.visualizer.plot_intelliton_transition_graph(transition_df)

        profile_payload = {
            "grounded_mode_activation_shift_mean": tracker.profile.grounded_mode_activation_shift_mean if tracker.profile else 0.0,
            "grounded_mode_activation_shift_std": tracker.profile.grounded_mode_activation_shift_std if tracker.profile else 0.0,
            "n_species": len(catalog.species),
        }
        (self.output_dir / "intelliton_trajectory_profile.json").write_text(
            json.dumps(profile_payload, ensure_ascii=False, indent=2)
        )

    def _plot_intelliton_dynamics(self, summary_df, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
        palette = {
            "grounded_factual": "#23395B",
            "stylistic_continuation": "#7A6C5D",
            "hallucination_prone": "#C84C31",
        }
        for prompt_type, color in palette.items():
            sub = summary_df[summary_df["Prompt_Type"] == prompt_type]
            axes[0].plot(sub["Step_Index"], sub["Mean_Mode_Activation_Shift"], marker="o", color=color, label=prompt_type)
            axes[1].plot(sub["Step_Index"], sub["Mean_Grounded_Deviation"], marker="o", color=color, label=prompt_type)
            axes[2].plot(sub["Step_Index"], sub["Mean_Top_Occupation"], marker="o", color=color, label=prompt_type)
        labels = [
            ("(a)", "Mode activation shift", "Shift"),
            ("(b)", "Grounded-sector deviation", "Deviation"),
            ("(c)", "Dominant Intelliton occupation", "Occupation"),
        ]
        for ax, (panel, title, ylabel) in zip(axes, labels):
            ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
            ax.set_xlabel("Generation step", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.4, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.legend(frameon=False, fontsize=6.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    def run(
        self,
        *,
        skip_hallucination: bool = False,
        skip_trajectory: bool = False,
        max_new_tokens: int = 8,
        local_window: int = 6,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> IntellitonCatalog:
        catalog = self.run_core_analysis()
        if not skip_hallucination:
            self.run_hallucination_analysis()
        if not skip_trajectory:
            self.run_trajectory_analysis(
                catalog,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                local_window=local_window,
            )
        return catalog
