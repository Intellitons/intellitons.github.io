"""Publication-quality visualizations for Intelliton Spectrum Analyzer v5.

New plots:
  - Lattice dispersion relation E(k) with fitted mass curves
  - RG flow diagram (running mass vs layer)
  - Phase transition detection
  - Spectral function rho(omega) with poles
  - Polyakov loop values
  - Creutz ratio / string tension
  - RoPE ablation heatmap
  - Enhanced PDG particle table
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.config import IntellitonConfig
from src.gauge_analyzer import (
    ParallelTransportResult, WilsonLoopResult, PolyakovLoopResult,
    CreutzRatioResult, TopologicalChargeResult,
)
from src.intelliton_classifier import IntellitonCatalog
from src.lattice_field import PropagatorResult, SpinDecomposition, DispersionRelation
from src.eft_renormalization import (
    RenormalizationFlow, EFTParameters, PhaseTransition,
)
from src.differential_spectroscopy import DifferentialSpectroscopyResult
from src.rope_gauge import (
    RoPEGaugeStructure, HelicityConservationResult, RoPEAblationResult,
)

logger = logging.getLogger(__name__)


class IntellitonVisualizer:
    """Generate publication-quality plots (v5)."""

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(context="paper", style="whitegrid")
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "figure.dpi": cfg.figure_dpi,
            "savefig.dpi": cfg.figure_dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })
        self.main_palette = {
            "blue": "#1F77B4",
            "orange": "#E17C05",
            "green": "#2CA02C",
            "red": "#D62728",
            "purple": "#9467BD",
            "brown": "#8C564B",
            "gray": "#7F7F7F",
        }
        self.prompt_palette = {
            "grounded_factual": self.main_palette["blue"],
            "stylistic_continuation": self.main_palette["brown"],
            "hallucination_prone": self.main_palette["red"],
        }

    def _figure_size(self, layout: str = "single", rows: int = 1, height_scale: float = 1.0):
        width = self.cfg.figure_width_single if layout == "single" else self.cfg.figure_width_double
        height = self.cfg.figure_height_base * max(rows, 1) * height_scale
        return (width, height)

    def _short_mode_labels(self, n: int):
        return [f"M{i}" for i in range(n)]

    def _format_axes(self, ax, xlabel=None, ylabel=None, title=None, panel_label: str = None):
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        # If a panel label is provided, include it in the title string
        if panel_label is not None:
            if title is not None:
                title = f"{panel_label} {title}"
            else:
                title = panel_label
        if title is not None:
            ax.set_title(title, pad=6)
        ax.grid(True, alpha=0.5, linewidth=0.6)

    def _add_colorbar(self, fig, im, ax, label):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=7)
        return cbar

    def _clean_title_suffix(self, title_suffix: str) -> str:
        return title_suffix.replace("_", " ").strip()

    def _clean_filename_suffix(self, title_suffix: str) -> str:
        return title_suffix.replace(" ", "_").lower()

    def _save(self, fig, name):
        fig.align_ylabels()
        path = self.output_dir / f"{name}.{self.cfg.figure_format}"
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    # ------------------------------------------------------------------
    # 1. Momentum Spectrum Heatmap
    # ------------------------------------------------------------------
    def plot_momentum_spectrum(self, power, momenta, title_suffix=""):
        fig, ax = plt.subplots(figsize=self._figure_size("single", height_scale=1.35))
        sort_idx = momenta.argsort()
        p_sorted = momenta[sort_idx].numpy()
        power_sorted = power[:, sort_idx].numpy()
        power_log = np.log10(power_sorted + 1e-10)

        im = ax.imshow(power_log, aspect="auto", cmap="magma", origin="lower",
                       extent=[p_sorted[0], p_sorted[-1], 0, power.shape[0] - 1])
        self._format_axes(
            ax,
            xlabel=r"Lattice momentum $k$",
            ylabel="Layer index",
            title=f"Momentum power spectrum{self._clean_title_suffix(title_suffix)}",
        )
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        self._add_colorbar(fig, im, ax, r"$\log_{10} P(l,k)$")
        ax.axvline(x=0, color="white", linestyle="--", alpha=0.7, linewidth=0.8)
        self._save(fig, f"momentum_spectrum{self._clean_filename_suffix(title_suffix)}")

    # ------------------------------------------------------------------
    # 2. Spin Spectrum
    # ------------------------------------------------------------------
    def plot_spin_spectrum(self, sigma, title_suffix=""):
        fig, axes = plt.subplots(1, 2, figsize=self._figure_size("double", height_scale=1.1),
                                 gridspec_kw={"width_ratios": [3, 1]})
        sigma_np = sigma.numpy()
        sigma_norm = sigma_np / (sigma_np[:, 0:1] + 1e-10)

        im = axes[0].imshow(sigma_norm.T, aspect="auto", cmap="viridis", origin="lower")
        self._format_axes(
            axes[0],
            xlabel="Layer index",
            ylabel="SVD mode index",
            title=f"Spin spectrum{self._clean_title_suffix(title_suffix)}",
            panel_label="(a)",
        )
        self._add_colorbar(fig, im, axes[0], r"$\sigma_i / \sigma_0$")

        avg = sigma_np.mean(axis=0)
        axes[1].barh(range(len(avg)), avg, color=self.main_palette["blue"], alpha=0.85)
        self._format_axes(
            axes[1],
            xlabel=r"Mean singular value $\bar{\sigma}_i$",
            ylabel="SVD mode",
            title="Average mode spectrum",
            panel_label="(b)",
        )
        axes[1].invert_yaxis()
        fig.tight_layout()
        self._save(fig, f"spin_spectrum{self._clean_filename_suffix(title_suffix)}")

    # ------------------------------------------------------------------
    # 3. Mass Spectrum with Spectral Function (v5)
    # ------------------------------------------------------------------
    def plot_mass_spectrum(self, propagator, n_show=10, title_suffix=""):
        # Layout: left column has two stacked panels (a,b), right column is a tall panel (c)
        # Increase height_scale for a taller figure (improves stacked panel spacing)
        fig = plt.figure(figsize=self._figure_size("double", height_scale=1.40))
        # Increase vertical spacing to avoid title/legend overlap between stacked panels
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1], hspace=0.80, wspace=0.32)
        ax_a = fig.add_subplot(gs[0, 0])    # top-left: decay curves (a)
        ax_b = fig.add_subplot(gs[1, 0])    # bottom-left: spectral function (b)
        ax_c = fig.add_subplot(gs[:, 1])    # right (spans both rows): mass bar chart (c)

        n_show = min(n_show, propagator.decay_curves.shape[1])
        layers = np.arange(propagator.decay_curves.shape[0])

        colors_map = {"massless": self.main_palette["blue"], "light": self.main_palette["green"],
                      "medium": self.main_palette["orange"], "heavy": self.main_palette["red"]}

        # (a) Top-left: decay curves
        for i in range(n_show):
            curve = propagator.decay_curves[:, i].numpy()
            cat = propagator.mass_category[i]
            color = colors_map.get(cat, "gray")
            m = propagator.mass[i].item()
            ax_a.semilogy(layers, np.maximum(curve, 1e-10),
                          label=f"Mode {i} (m={m:.3f})", color=color,
                          linewidth=1.5, alpha=0.8)
        self._format_axes(
            ax_a,
            xlabel="Layer index",
            ylabel=r"Mode amplitude $A_i(\tau)$",
            title=f"Layerwise mode amplitudes{self._clean_title_suffix(title_suffix)}",
            panel_label="(a)",
        )
        # hide legend for panel (a) to avoid overlap with panel (b)
        if ax_a.legend_ is not None:
            ax_a.legend_.remove()

        # (b) Bottom-left: spectral function
        if hasattr(propagator, 'spectral_function') and propagator.spectral_function is not None:
            sf = propagator.spectral_function
            omega = np.linspace(0, np.pi, sf.shape[0])
            for i in range(min(5, n_show)):
                ax_b.plot(omega, sf[:, i].numpy(),
                          label=f"Mode {i}", linewidth=1, alpha=0.8)
                if i < len(getattr(propagator, 'pole_positions', [])):
                    pp = propagator.pole_positions[i]
                    ax_b.axvline(x=pp, color="red", linestyle=":", linewidth=0.5, alpha=0.8)
            self._format_axes(
                ax_b,
                xlabel=r"Frequency $\omega$",
                ylabel=r"Spectral density $\rho(\omega)$",
                title="Spectral function and pole locations",
                panel_label="(b)",
            )
            ax_b.legend(fontsize=6)
        else:
            ax_b.text(0.5, 0.5, "No spectral function available",
                      ha="center", va="center", fontsize=9)
            ax_b.axis("off")

        # (c) Right: mass bar chart (pole vs lattice)
        masses_pole = propagator.mass[:n_show].numpy()
        masses_lat = propagator.mass_lattice[:n_show].numpy()
        x = np.arange(n_show)
        w = 0.35
        cat_colors = [colors_map.get(c, "gray") for c in propagator.mass_category[:n_show]]
        ax_c.barh(x - w/2, masses_pole, w, color="#7874F0", alpha=0.85)
        ax_c.barh(x + w/2, masses_lat, w, color="#9ECAE1", alpha=0.9)
        for i, c in enumerate(cat_colors):
            ax_c.barh(x[i] - w/2, masses_pole[i], w, color="none", edgecolor=c, linewidth=1)
        legend_items = [
            Patch(facecolor="#7874F0", alpha=0.85, label="Pole mass"),
            Patch(facecolor="#9ECAE1", alpha=0.9, label="Lattice mass"),
        ]
        seen_cats = {}
        for c in propagator.mass_category[:n_show]:
            if c not in seen_cats:
                seen_cats[c] = colors_map.get(c, "gray")
        for cat_name, cat_color in seen_cats.items():
            legend_items.append(
                Patch(facecolor="none", edgecolor=cat_color, linewidth=1,
                      label=f"  {cat_name}")
            )
        # Place legend for panel (c) at bottom-right inside the axes
        ax_c.legend(handles=legend_items, fontsize=6, loc="lower right")
        self._format_axes(
            ax_c,
            xlabel="Mass",
            ylabel=None,
            title="Pole and lattice masses",
            panel_label="(c)",
        )
        ax_c.set_yticks(x)
        ax_c.set_yticklabels([f"Mode {i}" for i in range(n_show)])
        ax_c.invert_yaxis()

        fig.tight_layout()
        self._save(fig, f"mass_spectrum{self._clean_filename_suffix(title_suffix)}")

    # ------------------------------------------------------------------
    # 4. Lattice Dispersion Relation (v5 NEW)
    # ------------------------------------------------------------------
    def plot_dispersion_relation(self, disp: DispersionRelation, n_show=5):
        fig, axes = plt.subplots(1, 2, figsize=self._figure_size("double", height_scale=1.05))

        momenta = disp.momenta.numpy()
        sort_idx = np.argsort(momenta)
        k_sorted = momenta[sort_idx]

        # BZ inner-zone cutoff used during fitting
        k_cutoff = 2.0 * np.pi / 3.0

        # Left: E(k) for each mode
        for i in range(min(n_show, disp.energies.shape[1])):
            E_sorted = disp.energies[sort_idx, i].numpy()
            m_fit = disp.fitted_mass[i].item()
            r2 = disp.fit_quality[i].item()
            # Mark fallback masses (R²=0) differently
            src = "fit" if r2 > 0 else "E(0)"
            axes[0].plot(k_sorted, E_sorted, 'o-', markersize=4,
                         label=f"Mode {i} (m={m_fit:.3f}, {src}, R$^2$={r2:.2f})",
                         linewidth=1.5, alpha=0.8)
            # Plot fitted lattice dispersion curve only if fit actually succeeded
            k_fine = np.linspace(-np.pi, np.pi, 200)
            if r2 > 0:
                E_fit = np.sqrt(m_fit**2 + (2*np.sin(k_fine/2))**2)
                axes[0].plot(k_fine, E_fit, '--', alpha=0.4, linewidth=1)
            else:
                # Show horizontal line at rest mass for flat-dispersion modes
                axes[0].axhline(y=m_fit, linestyle=':', alpha=0.3, linewidth=1)

        # Show inner-BZ fitting region
        axes[0].axvline(x=-k_cutoff, color="gray", linestyle=":", alpha=0.5)
        axes[0].axvline(x=k_cutoff, color="gray", linestyle=":", alpha=0.5,
                        label=r"Fit region $|k| \leq 2\pi/3$")

        self._format_axes(
            axes[0],
            xlabel=r"Lattice momentum $k$",
            ylabel=r"Energy $E(k)$",
            title=r"Dispersion relation $E^2=m^2+(2\sin(k/2))^2$",
            panel_label="(a)",
        )
        axes[0].legend(fontsize=7)
        axes[0].set_xlim(-np.pi, np.pi)
        axes[0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axes[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

        # Right: group velocity
        for i in range(min(n_show, disp.group_velocity.shape[1])):
            vg = disp.group_velocity[sort_idx, i].numpy()
            axes[1].plot(k_sorted, vg, label=f"Mode {i}", linewidth=1.5)
        self._format_axes(
            axes[1],
            xlabel=r"Lattice momentum $k$",
            ylabel=r"Group velocity $v_g=dE/dk$",
            title="Group velocity",
            panel_label="(b)",
        )
        axes[1].legend(fontsize=8)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axes[1].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

        fig.tight_layout()
        self._save(fig, "dispersion_relation")

    # ------------------------------------------------------------------
    # 5. RG Flow Diagram (v5 NEW)
    # ------------------------------------------------------------------
    def plot_rg_flow(self, rg: RenormalizationFlow, n_show=8):
        fig, axes = plt.subplots(2, 2, figsize=self._figure_size("double", rows=2, height_scale=1.1))

        n_show = min(n_show, rg.running_mass.shape[1])
        layers = np.arange(rg.running_mass.shape[0])

        # (a) Running mass
        for i in range(n_show):
            fp = rg.fixed_point_layers[i].item()
            fp_type = rg.fixed_point_type[i]
            label = f"Mode {i} (FP={fp}, {fp_type})"
            axes[0, 0].plot(layers, rg.running_mass[:, i].numpy(),
                           label=label, linewidth=1.5)
            axes[0, 0].axvline(x=fp, linestyle=":", alpha=0.2)
        self._format_axes(
            axes[0, 0],
            xlabel="Layer index",
            ylabel=r"Running mass $m(l)$",
            title="RG flow of Intelliton masses",
            panel_label="(a)",
        )
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5,
                           label="m=0 (growth/decay boundary)")

        # (b) Beta function
        beta_layers = np.arange(rg.beta_function.shape[0])
        for i in range(n_show):
            axes[0, 1].plot(beta_layers, rg.beta_function[:, i].numpy(),
                           label=f"Mode {i}", linewidth=1.5, alpha=0.7)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        self._format_axes(
            axes[0, 1],
            xlabel="Layer index",
            ylabel=r"$\beta(l)=dm/dl$",
            title=r"RG β-function",
            panel_label="(b)",
        )
        axes[0, 1].legend(fontsize=7)

        # (c) Anomalous dimensions
        anom = rg.anomalous_dimension[:n_show].numpy()
        modes = [f"Mode {i}" for i in range(n_show)]
        colors = ["#4CAF50" if a > 0 else "#F44336" for a in anom]
        axes[1, 0].barh(modes, anom, color=colors)
        self._format_axes(
            axes[1, 0],
            xlabel=r"Anomalous dimension γ",
            ylabel=None,
            title="Anomalous dimensions",
            panel_label="(c)",
        )
        axes[1, 0].axvline(x=0, color="black", linestyle="-", alpha=0.3)
        axes[1, 0].invert_yaxis()

        # (d) Stability classification
        stab = rg.stability[:n_show].numpy()
        fp_types = rg.fixed_point_type[:n_show]
        type_colors = {"IR": "#2196F3", "UV": "#F44336",
                       "crossover": "#FF9800", "none": "#9E9E9E"}
        bar_colors = [type_colors.get(t, "#9E9E9E") for t in fp_types]
        axes[1, 1].barh(modes, stab, color=bar_colors)
        self._format_axes(
            axes[1, 1],
            xlabel="Stability eigenvalue",
            ylabel=None,
            title="Fixed-point stability (<0 indicates stable IR)",
            panel_label="(d)",
        )
        axes[1, 1].axvline(x=0, color="black", linestyle="-", alpha=0.3)
        axes[1, 1].invert_yaxis()

        legend_el = [Patch(facecolor=c, label=k) for k, c in type_colors.items()]
        axes[1, 1].legend(handles=legend_el, loc="lower right", fontsize=6)

        fig.tight_layout()
        self._save(fig, "rg_flow")

    # ------------------------------------------------------------------
    # 6. EFT Parameters (v5 NEW)
    # ------------------------------------------------------------------
    def plot_eft_parameters(self, eft: EFTParameters, n_show=10):
        fig, axes = plt.subplots(1, 3, figsize=self._figure_size("double", height_scale=1.05))
        n_show = min(n_show, eft.bare_mass.shape[0])
        modes = self._short_mode_labels(n_show)

        # (a) Bare vs Renormalized mass
        x = np.arange(n_show)
        w = 0.35
        axes[0].bar(x - w/2, eft.bare_mass[:n_show].numpy(), w,
               label="Bare (UV)", color=self.main_palette["orange"])
        axes[0].bar(x + w/2, eft.renormalized_mass[:n_show].numpy(), w,
               label="Renormalized (IR)", color=self.main_palette["blue"])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(modes, rotation=35, ha="right")
        self._format_axes(axes[0], ylabel="Mass", title="Bare and renormalized masses", panel_label="(a)")
        axes[0].legend()
        axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # (b) Mass shift
        shift = eft.mass_shift[:n_show].numpy()
        colors = [self.main_palette["green"] if s > 0 else self.main_palette["red"] for s in shift]
        axes[1].bar(modes, shift, color=colors)
        self._format_axes(axes[1], ylabel=r"$\Delta m = m_{\mathrm{ren}} - m_{\mathrm{bare}}$", title="Mass renormalization", panel_label="(b)")
        axes[1].tick_params(axis="x", rotation=35)
        for label in axes[1].get_xticklabels():
            label.set_ha("right")
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # (c) Wavefunction renormalization Z
        Z = eft.Z[:n_show].numpy()
        axes[2].bar(modes, Z, color=self.main_palette["blue"])
        self._format_axes(axes[2], ylabel="Wavefunction renormalization $Z$", title="Wavefunction renormalization", panel_label="(c)")
        axes[2].tick_params(axis="x", rotation=35)
        for label in axes[2].get_xticklabels():
            label.set_ha("right")
        axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5,
                        label="Z=1 (no renorm)")
        axes[2].legend()

        fig.tight_layout(pad=1.0, w_pad=1.2)
        self._save(fig, "eft_parameters")

    # ------------------------------------------------------------------
    # 7. Phase Transitions (v5 NEW)
    # ------------------------------------------------------------------
    def plot_phase_transitions(self, transitions: list, sigma: torch.Tensor):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=self._figure_size("double", height_scale=0.95),
            gridspec_kw={"width_ratios": [1.35, 1.0]}
        )

        # (a) Mass gap with transition markers
        n_layers = sigma.shape[0]
        n_modes = min(sigma.shape[1], 2)
        mass_gap = sigma[:, 0].numpy() - sigma[:, min(1, n_modes-1)].numpy()
        axes[0].plot(range(n_layers), mass_gap, 'b-', linewidth=1.5, label="mass_gap")
        for tr in transitions:
            axes[0].axvline(x=tr.layer, color="red", linestyle="--", linewidth=0.5, alpha=0.5, label="_nolegend_")

        self._format_axes(
            axes[0],
            xlabel="Layer index",
            ylabel=r"Mass gap $\sigma_0-\sigma_1$",
            title="Mass-gap transitions",
            panel_label="(a)",
        )
        # Make axes visually clear for publication: stronger spines, bolder ticks, no background grid
        if axes[0].legend_ is not None:
            axes[0].legend_.remove()
        axes[0].grid(False)
        for spine in ["left", "bottom"]:
            axes[0].spines[spine].set_visible(True)
            axes[0].spines[spine].set_linewidth(1.0)
            axes[0].spines[spine].set_color("#222222")
        for spine in ["top", "right"]:
            axes[0].spines[spine].set_visible(False)
        axes[0].tick_params(axis="both", which="major", labelsize=7, width=1)
        axes[0].xaxis.label.set_fontsize(7)
        axes[0].yaxis.label.set_fontsize(7)
        axes[0].title.set_fontsize(8)

        # (b) Transition details
        if transitions:
            tr_data = {
                "Layer": [t.layer for t in transitions],
                "Jump": [f"{t.order_parameter_jump:.3f}" for t in transitions],
                "Mass Gap\nBefore": [f"{t.mass_gap_before:.2f}" for t in transitions],
                "Mass Gap\nAfter": [f"{t.mass_gap_after:.2f}" for t in transitions],
            }
            axes[1].axis("off")
            table = axes[1].table(
                cellText=list(zip(*tr_data.values())),
                colLabels=list(tr_data.keys()),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.15, 2.3)
            n_data_rows = len(transitions)
            for (row, col), cell in table.get_celld().items():
                cell.set_linewidth(0.0)
                cell.visible_edges = ""
                if row == 0:
                    cell.set_text_props(weight="bold")
                    cell.visible_edges = "TB"
                    cell.set_linewidth(0.8)
                elif row == n_data_rows:
                    cell.visible_edges = "B"
                    cell.set_linewidth(0.8)
            self._format_axes(axes[1], title="Detected phase transitions", panel_label="(b)")
        else:
            axes[1].text(0.5, 0.5, "No phase transitions detected",
                        ha="center", va="center", fontsize=14)
            axes[1].axis("off")

        fig.tight_layout()
        bbox_left = axes[0].get_position()
        bbox_right = axes[1].get_position()
        axes[1].set_position([bbox_right.x0, bbox_left.y0, bbox_right.width, bbox_left.height])
        self._save(fig, "phase_transitions")

    # ------------------------------------------------------------------
    # 8. Polyakov Loops (v5 NEW)
    # ------------------------------------------------------------------
    def plot_polyakov_loops(self, polyakov: PolyakovLoopResult):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # (a) Per-token Polyakov loop
        P = polyakov.loop_values.numpy()
        axes[0].bar(range(len(P)), P, color="steelblue", alpha=0.8)
        axes[0].axhline(y=polyakov.expectation_value, color="red",
                       linestyle="--", label=f"<P>={polyakov.expectation_value:.4f}")
        axes[0].set_xlabel("Token Position")
        axes[0].set_ylabel("Polyakov Loop P(t)")
        axes[0].set_title(f"Polyakov Loop ({polyakov.phase.upper()} phase)")
        axes[0].legend()

        # (b) Layer contributions
        lc = polyakov.layer_contributions.numpy()
        axes[1].plot(range(len(lc)), lc, 'o-', color="darkorange")
        axes[1].set_xlabel("Layer Index")
        axes[1].set_ylabel("Layer Contribution")
        axes[1].set_title(f"Temporal Link (susceptibility={polyakov.susceptibility:.4f})")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        self._save(fig, "polyakov_loops")

    # ------------------------------------------------------------------
    # 9. Wilson Loops and Field Strength
    # ------------------------------------------------------------------
    def plot_wilson_loops(self, wilson):
        n_pairs = len(wilson.plaquettes)
        if n_pairs == 0:
            return
        n_show = min(n_pairs, 4)
        fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        for idx, (key, W) in enumerate(list(wilson.plaquettes.items())[:n_show]):
            im1 = axes[0, idx].imshow(W.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
            tr_val = wilson.plaquette_traces.get(key, 0.0)
            axes[0, idx].set_title(f"W {key}\nTr/T={tr_val:.3f}")
            fig.colorbar(im1, ax=axes[0, idx], fraction=0.046)

            F = wilson.field_strength[key]
            im2 = axes[1, idx].imshow(F.numpy(), cmap="hot", vmin=0, vmax=1)
            axes[1, idx].set_title(f"F {key}")
            fig.colorbar(im2, ax=axes[1, idx], fraction=0.046)

        fig.suptitle("Lattice Gauge Field: Wilson Loops & Field Strength",
                    fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "wilson_loops")

    # ------------------------------------------------------------------
    # 10. Helicity Conservation
    # ------------------------------------------------------------------
    def plot_helicity_conservation(self, helicity, n_show=10):
        fig, ax = plt.subplots(figsize=self._figure_size("single", height_scale=1.0))
        n_show = min(n_show, helicity.shape[1])
        layers = np.arange(helicity.shape[0])

        for i in range(n_show):
            ax.plot(layers, helicity[:, i].numpy(),
                   label=f"Mode {i}", linewidth=1.5, alpha=0.7)
        self._format_axes(ax, xlabel="Layer index", ylabel="Helicity", title="Helicity across layers")
        ax.legend(fontsize=8)
        self._save(fig, "helicity_conservation")

    def plot_momentum_and_helicity(self, power, momenta, helicity, title_suffix="", n_show=8):
        fig, axes = plt.subplots(1, 2, figsize=self._figure_size("double", height_scale=1.0))

        sort_idx = momenta.argsort()
        p_sorted = momenta[sort_idx].numpy()
        power_sorted = power[:, sort_idx].numpy()
        power_log = np.log10(power_sorted + 1e-10)

        im = axes[0].imshow(
            power_log,
            aspect="auto",
            cmap="magma",
            origin="lower",
            extent=[p_sorted[0], p_sorted[-1], 0, power.shape[0] - 1],
        )
        self._format_axes(
            axes[0],
            xlabel=r"Lattice momentum $k$",
            ylabel="Layer index",
            title=f"Momentum power spectrum{self._clean_title_suffix(title_suffix)}",
            panel_label="(a)",
        )
        axes[0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axes[0].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        axes[0].axvline(x=0, color="white", linestyle="--", alpha=0.7, linewidth=0.8)
        self._add_colorbar(fig, im, axes[0], r"$\log_{10} P(l,k)$")

        n_show = min(n_show, helicity.shape[1])
        layers = np.arange(helicity.shape[0])
        for i in range(n_show):
            axes[1].plot(layers, helicity[:, i].numpy(), label=f"Mode {i}")
        self._format_axes(
            axes[1],
            xlabel="Layer index",
            ylabel="Helicity",
            title="Helicity across layers",
            panel_label="(b)",
        )
        axes[1].legend(fontsize=7)

        fig.tight_layout()
        self._save(fig, f"momentum_helicity{self._clean_filename_suffix(title_suffix)}")

    # ------------------------------------------------------------------
    # 11. Parallel Transport
    # ------------------------------------------------------------------
    def plot_parallel_transport(self, transport):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im = axes[0].imshow(transport.alignment.numpy(), aspect="auto",
                           cmap="YlGn", vmin=0, vmax=1)
        axes[0].set_xlabel("SVD Mode")
        axes[0].set_ylabel("Attention Layer")
        axes[0].set_title("Parallel Transport Alignment")
        fig.colorbar(im, ax=axes[0])

        n = transport.avg_alignment_per_mode.shape[0]
        axes[1].bar(range(n), transport.avg_alignment_per_mode.numpy(),
                   color="seagreen")
        axes[1].set_xlabel("SVD Mode")
        axes[1].set_ylabel("Avg Alignment")
        axes[1].set_title("Spin Preservation per Mode")
        axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

        fig.tight_layout()
        self._save(fig, "parallel_transport")

    # ------------------------------------------------------------------
    # 12. RoPE Ablation Results (v5 NEW)
    # ------------------------------------------------------------------
    def plot_rope_ablation(self, ablation: RoPEAblationResult):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        n_abl = len(ablation.ablated_pairs)
        labels = [f"[{s}:{e}]" for s, e in ablation.ablated_pairs]

        # (a) KL divergence per ablation
        kl = ablation.kl_divergences.numpy()
        colors = ["#F44336" if c else "#4CAF50" for c in ablation.circuit_collapsed]
        axes[0].bar(labels, kl, color=colors)
        axes[0].set_xlabel("Ablated RoPE Frequency Pairs")
        axes[0].set_ylabel("KL Divergence from Baseline")
        axes[0].set_title("(a) RoPE Dimension Ablation Impact")
        axes[0].tick_params(axis="x", rotation=45)

        from matplotlib.patches import Patch
        legend_el = [Patch(facecolor="#F44336", label="Circuit collapsed"),
                    Patch(facecolor="#4CAF50", label="Circuit intact")]
        axes[0].legend(handles=legend_el)

        # (b) Mode sensitivity heatmap
        sens = ablation.mode_sensitivity.numpy()
        n_show_modes = min(10, sens.shape[1])
        im = axes[1].imshow(sens[:, :n_show_modes], aspect="auto", cmap="YlOrRd")
        axes[1].set_xlabel("SVD Mode Index")
        axes[1].set_ylabel("Ablation Group")
        axes[1].set_yticks(range(n_abl))
        axes[1].set_yticklabels(labels)
        axes[1].set_title("(b) Mode Sensitivity to RoPE Ablation")
        fig.colorbar(im, ax=axes[1], label="Fractional Energy in Ablated Dims")

        fig.tight_layout()
        self._save(fig, "rope_ablation")

    # ------------------------------------------------------------------
    # 13. Particle Table
    # ------------------------------------------------------------------
    def plot_particle_table(self, catalog: IntellitonCatalog):
        df = catalog.build_dataframe()
        if df.empty:
            return

        n_rows = len(df)
        n_cols = len(df.columns)

        # Compute figure size to closely match table dimensions (in inches).
        # Choose per-column width and per-row height tuned for readability in paper.
        per_col_w = 1.05
        per_row_h = 0.36
        header_h = 0.9
        width_in = max(6.0, n_cols * per_col_w)
        height_in = max(2.5, header_h + n_rows * per_row_h)

        fig = plt.figure(figsize=(width_in, height_in))
        # Create a single axis that fills most of the figure for the table.
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
        ax.axis("off")
        ax.set_title(
            "Intelliton particle table\n"
            f"$\\alpha$={catalog.eft_alpha:.4f}, $\\beta$={catalog.eft_beta:.4f}",
            fontsize=10, fontweight="bold", pad=12,
        )

        colors_map = {"massless": "#E8F1FB", "light": "#EAF6EA",
                      "medium": "#FFF1E6", "heavy": "#FDECEC"}
        cell_colors = []
        for _, row in df.iterrows():
            color = colors_map.get(row["Category"], "#FFFFFF")
            cell_colors.append([color] * n_cols)

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellColours=cell_colors, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        # Increase vertical spacing so long cells can wrap without overlap
        table.scale(1.0, 1.4)

        # Auto-adjust column widths
        try:
            table.auto_set_column_width(list(range(n_cols)))
        except Exception:
            pass

        # Enable wrapping for cells in the 'Active In' column (if present)
        active_in_idx = None
        try:
            active_in_idx = list(df.columns).index("Active In")
        except ValueError:
            active_in_idx = None

        for (r, c), cell in table.get_celld().items():
            # header styling
            if r == 0:
                cell.set_facecolor(self.main_palette["blue"])
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            # wrap long text in the Active In column
            if active_in_idx is not None and c == active_in_idx:
                try:
                    cell.get_text().set_wrap(True)
                    cell.get_text().set_ha("left")
                except Exception:
                    pass

        for j in range(n_cols):
            table[0, j].set_facecolor(self.main_palette["blue"])
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Resize figure to tightly fit the table content (preserve readability)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = table.get_window_extent(renderer)
            # Convert pixels -> inches using figure DPI
            width_in = bbox.width / float(fig.dpi)
            height_in = bbox.height / float(fig.dpi)
            # Add small margins for title/header and padding
            pad_w = 0.6
            pad_h = 0.9
            fig.set_size_inches(max(4.0, width_in + pad_w), max(2.5, height_in + pad_h))
            # Reposition axis to fill figure while leaving small margins
            ax.set_position([0.02, 0.02, 0.96, 0.94])
        except Exception:
            # Fallback: leave existing sizing
            pass

        # Save using the centralized saver
        self._save(fig, "particle_table")

    # ------------------------------------------------------------------
    # 14. Intervention Results
    # ------------------------------------------------------------------
    def plot_intervention_results(self, results_df):
        if results_df.empty:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # KL Divergence
        pivot = results_df.groupby(["Species", "Intervention"])["KL_Divergence"].mean()
        pivot.unstack(fill_value=0).plot(kind="bar", ax=axes[0, 0],
                                         color=["#2196F3", "#F44336"])
        axes[0, 0].set_title("(a) KL Divergence")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Cosine Similarity
        pivot2 = results_df.groupby(["Species", "Intervention"])["Cosine_Sim"].mean()
        pivot2.unstack(fill_value=0).plot(kind="bar", ax=axes[0, 1],
                                          color=["#2196F3", "#F44336"])
        axes[0, 1].set_title("(b) Cosine Similarity")
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Top-10 Overlap
        pivot3 = results_df.groupby(["Species", "Intervention"])["Top10_Overlap"].mean()
        pivot3.unstack(fill_value=0).plot(kind="bar", ax=axes[1, 0],
                                          color=["#2196F3", "#F44336"])
        axes[1, 0].set_title("(c) Top-10 Overlap")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Top Token Changed
        pivot4 = results_df.groupby(["Species", "Intervention"])["Top_Token_Changed"].mean()
        pivot4.unstack(fill_value=0).plot(kind="bar", ax=axes[1, 1],
                                          color=["#2196F3", "#F44336"])
        axes[1, 1].set_title("(d) Top Token Change Rate")
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].tick_params(axis="x", rotation=45)

        fig.suptitle("Gauge Intervention Effects", fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "intervention_results")

    # ------------------------------------------------------------------
    # 15. Fusion Tree (v5 NEW)
    # ------------------------------------------------------------------
    def plot_fusion_tree(self, fusion):
        """Plot fusion tracking results: coupling matrix, flow diagram, fusion rate."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # (a) Mode-coupling heatmap (average across attention layers)
        M = fusion.coupling_matrix  # [n_attn_layers, n_modes, n_modes]
        n_modes = M.shape[1]

        if M.shape[0] > 0:
            # Average coupling across attention layers
            M_avg = M.mean(dim=0).numpy()
            # Zero diagonal for clarity (self-coupling is trivially large)
            np.fill_diagonal(M_avg, 0)
            im = axes[0].imshow(M_avg, cmap="hot", aspect="equal",
                                vmin=0, vmax=max(M_avg.max(), 0.01))
            plt.colorbar(im, ax=axes[0], fraction=0.046)
            axes[0].set_xlabel("Mode j")
            axes[0].set_ylabel("Mode i")
            axes[0].set_title("(a) Attention-Mediated Mode Coupling\n"
                             r"$M_{ij} = |U_i^T A_{avg} U_j|$ (off-diagonal)")
        else:
            axes[0].text(0.5, 0.5, "No attention data", ha="center", va="center",
                        transform=axes[0].transAxes)
            axes[0].set_title("(a) Mode Coupling")

        # (b) Mode flow / alluvial diagram
        if fusion.mode_genealogy:
            n_layers_total = len(next(iter(fusion.mode_genealogy.values())))
            colors = plt.cm.tab10(np.linspace(0, 1, min(n_modes, 10)))

            for final_mode, ancestors_per_layer in fusion.mode_genealogy.items():
                if final_mode >= 10:
                    break
                for l in range(n_layers_total - 1):
                    for src in ancestors_per_layer[l]:
                        for tgt in ancestors_per_layer[l + 1]:
                            axes[1].plot(
                                [l, l + 1], [src, tgt],
                                color=colors[final_mode % 10],
                                alpha=0.3, linewidth=1.5,
                            )

            # Mark fusion events
            for event in fusion.fusion_events:
                for src in event.source_modes:
                    axes[1].plot(
                        event.layer, event.target_mode,
                        'v', color='red', markersize=6, zorder=5,
                    )

            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Mode Index")
            axes[1].set_title(f"(b) Mode Genealogy Flow\n"
                             f"({len(fusion.fusion_events)} fusion events)")
            axes[1].set_xlim(-0.5, n_layers_total - 0.5)
            axes[1].set_ylim(-0.5, n_modes - 0.5)
            axes[1].invert_yaxis()
        else:
            axes[1].text(0.5, 0.5, "No genealogy data", ha="center", va="center",
                        transform=axes[1].transAxes)
            axes[1].set_title("(b) Mode Flow")

        # (c) Fusion rate per layer transition
        fr = fusion.fusion_rate.numpy()
        layers = np.arange(len(fr))
        axes[2].bar(layers, fr, color="#FF7043", alpha=0.8)
        axes[2].set_xlabel("Layer Transition (l -> l+1)")
        axes[2].set_ylabel("Fusion Events")
        axes[2].set_title("(c) Fusion Rate")
        axes[2].grid(True, alpha=0.3, axis="y")

        # Secondary axis: active mode count
        ax2c = axes[2].twinx()
        active = fusion.n_active_modes_per_layer.numpy()
        ax2c.plot(np.arange(len(active)), active, 'b-o', markersize=3,
                  label="Active modes", alpha=0.7)
        ax2c.set_ylabel("Active Modes", color="blue")
        ax2c.tick_params(axis="y", labelcolor="blue")

        fig.suptitle("Intelliton Fusion Tracking", fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "fusion_tree")

    # ------------------------------------------------------------------
    # 16. Vacuum Expectation Value (v5 NEW)
    # ------------------------------------------------------------------
    def plot_vev(self, vacuum):
        """Plot VEV and symmetry breaking analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Color map for categories
        cats = sorted(vacuum.order_parameter_by_cat.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(cats), 1)))

        # (a) VEV norm across layers
        vev_norm = vacuum.vev_norm.numpy()
        layers = np.arange(len(vev_norm))
        axes[0, 0].plot(layers, vev_norm, 'k-', linewidth=2, label="||VEV||")
        axes[0, 0].set_xlabel("Layer")
        axes[0, 0].set_ylabel("||VEV||")
        axes[0, 0].set_title("(a) Vacuum State Norm")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # (b) Order parameter per category
        for i, cat in enumerate(cats):
            op = vacuum.order_parameter_by_cat[cat].numpy()
            axes[0, 1].plot(np.arange(len(op)), op, color=colors[i],
                           label=cat, linewidth=1.5)
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel(r"$||\langle\delta R\rangle||$")
        axes[0, 1].set_title("(b) Symmetry Breaking Order Parameter")
        axes[0, 1].legend(fontsize=7)
        axes[0, 1].grid(True, alpha=0.3)

        # (c) Excited-state top singular values
        n_show_sv = 5
        for i, cat in enumerate(cats):
            sigma_ex = vacuum.excited_sigma_by_cat[cat].numpy()  # [L+1, n_modes]
            # Plot top singular value
            axes[1, 0].plot(
                np.arange(sigma_ex.shape[0]),
                sigma_ex[:, 0],
                color=colors[i], label=f"{cat} (sv0)",
                linewidth=1.5, linestyle="-",
            )
            # Plot 5th singular value (dashed) for spectral width
            if sigma_ex.shape[1] >= n_show_sv:
                axes[1, 0].plot(
                    np.arange(sigma_ex.shape[0]),
                    sigma_ex[:, n_show_sv - 1],
                    color=colors[i], linewidth=1, linestyle="--", alpha=0.5,
                )
        axes[1, 0].set_xlabel("Layer")
        axes[1, 0].set_ylabel(r"$\sigma_{exc}$")
        axes[1, 0].set_title(r"(c) Excited-State Spectrum ($\sigma_1$ solid, $\sigma_5$ dashed)")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")

        # (d) VEV alignment
        for i, cat in enumerate(cats):
            align = vacuum.vev_alignment_by_cat[cat].numpy()
            axes[1, 1].plot(np.arange(len(align)), align, color=colors[i],
                           label=cat, linewidth=1.5)
        axes[1, 1].set_xlabel("Layer")
        axes[1, 1].set_ylabel(r"$\cos(\theta_{prompt}, \theta_{vac})$")
        axes[1, 1].set_title("(d) VEV Alignment (1=vacuum, 0=fully broken)")
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle("Vacuum Expectation Value & Symmetry Breaking",
                     fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "vev_analysis")

    # ------------------------------------------------------------------
    # 17. Intelliton Occupation Trajectories
    # ------------------------------------------------------------------
    def plot_intelliton_occupations(self, detail_df: pd.DataFrame, top_n: int = 5):
        if detail_df.empty:
            return

        prompt_types = list(detail_df["Prompt_Type"].dropna().unique())
        fig, axes = plt.subplots(1, len(prompt_types), figsize=(max(1, len(prompt_types)) * 2.4 + 0.6, 2.8), squeeze=False)
        axes = axes[0]

        for ax, prompt_type in zip(axes, prompt_types):
            sub = detail_df[detail_df["Prompt_Type"] == prompt_type]
            species_rank = (
                sub.groupby("Species_Name")["Occupation"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            for species_name in species_rank:
                species_sub = (
                    sub[sub["Species_Name"] == species_name]
                    .groupby("Step_Index", as_index=False)["Occupation"]
                    .mean()
                )
                ax.plot(species_sub["Step_Index"], species_sub["Occupation"], marker="o", label=species_name)
            self._format_axes(ax, xlabel="Generation step", ylabel="Occupation", title=prompt_type.replace("_", " "))
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        self._save(fig, "intelliton_occupation_trajectories")

    def plot_merged_trajectory_and_occupations(self, summary_df: pd.DataFrame, detail_df: pd.DataFrame, top_n: int = 5, output_name: str = "intelliton_trajectory_merged"):
        """Create a combined 2×3 figure with clear visual separation.

        Uses two independent 1×3 subplot grids positioned manually so that
        panel titles and shared legends never overlap with subplot content.
        """
        if summary_df.empty and detail_df.empty:
            return

        prompt_order = ["grounded_factual", "stylistic_continuation", "hallucination_prone"]

        # --- figure -----------------------------------------------------------------
        w = self.cfg.figure_width_double                  # 7.0 inches
        h = 6.0                                           # compact two-panel figure
        fig = plt.figure(figsize=(w, h))

        # ── Panel (a): top row ─────────────────────────────────────────────────────
        # Axes occupy [left, bottom, width, height] in figure-fraction coords.
        # Reserve top strip for title+legend, then three compact subplots.
        panel_a_top    = 0.88                             # top of subplot area
        panel_a_bottom = 0.62                             # bottom of subplot area
        panel_a_height = panel_a_top - panel_a_bottom     # 0.26
        col_width = 0.25
        col_gap   = 0.04
        left_margin = 0.10

        top_axes = []
        for i in range(3):
            left = left_margin + i * (col_width + col_gap)
            ax = fig.add_axes([left, panel_a_bottom, col_width, panel_a_height])
            top_axes.append(ax)

        metrics = [
            ("Mean_Mode_Activation_Shift", "Mode activation shift", "Shift"),
            ("Mean_Grounded_Deviation", "Grounded-sector deviation", "Deviation"),
            ("Mean_Top_Occupation", "Dominant Intelliton occupation", "Occupation"),
        ]

        handles_top, labels_top = [], []
        for pt in prompt_order:
            color = self.prompt_palette.get(pt, "#333333")
            sub = summary_df[summary_df["Prompt_Type"] == pt]
            if sub.empty:
                ln = Line2D([0], [0], color=color, linewidth=1.5)
                handles_top.append(ln); labels_top.append(pt)
                continue
            for ax, (col, _, _) in zip(top_axes, metrics):
                ln, = ax.plot(sub["Step_Index"], sub[col],
                              marker="o", color=color, linewidth=1.5, alpha=0.9)
            handles_top.append(ln); labels_top.append(pt)

        for ax, (_, title, ylabel) in zip(top_axes, metrics):
            ax.set_title(title, fontsize=8, pad=4)
            ax.set_xlabel("Generation step", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # (a) section title
        fig.text(0.5, 0.99, "(a) Intelliton trajectories overview",
                 ha="center", va="top", fontsize=10, fontweight="bold")
        # (a) shared legend – between title and subplot titles
        if handles_top:
            fig.legend(handles_top, labels_top,
                       loc="upper center", bbox_to_anchor=(0.5, 0.955),
                       ncol=len(handles_top), fontsize=7.5, frameon=False)

        # ── Panel (b): bottom row ──────────────────────────────────────────────────
        panel_b_top    = 0.38                             # top of subplot area
        panel_b_bottom = 0.08                             # bottom of subplot area
        panel_b_height = panel_b_top - panel_b_bottom     # 0.30 (similar to panel a)

        bottom_axes = []
        for i in range(3):
            left = left_margin + i * (col_width + col_gap)
            ax = fig.add_axes([left, panel_b_bottom, col_width, panel_b_height])
            bottom_axes.append(ax)

        species_handles = {}
        for ax, pt in zip(bottom_axes, prompt_order):
            sub = detail_df[detail_df["Prompt_Type"] == pt]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(pt.replace("_", " "), fontsize=8, pad=4)
                continue

            species_rank = (
                sub.groupby("Species_Name")["Occupation"]
                .mean().sort_values(ascending=False)
                .head(top_n).index.tolist()
            )
            for species_name in species_rank:
                species_sub = (
                    sub[sub["Species_Name"] == species_name]
                    .groupby("Step_Index", as_index=False)["Occupation"].mean()
                )
                ln, = ax.plot(species_sub["Step_Index"], species_sub["Occupation"],
                              marker="o")
                if species_name not in species_handles:
                    species_handles[species_name] = ln

            ax.set_title(pt.replace("_", " "), fontsize=8, pad=4)
            ax.set_xlabel("Generation step", fontsize=7)
            ax.set_ylabel("Occupation", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

        # (b) section title
        fig.text(0.5, 0.52, "(b) Intelliton occupation trajectories",
                 ha="center", va="top", fontsize=10, fontweight="bold")
        # (b) shared legend – between title and subplot titles
        if species_handles:
            sh_labels = list(species_handles.keys())
            sh_handles = [species_handles[k] for k in sh_labels]
            fig.legend(sh_handles, sh_labels,
                       loc="upper center", bbox_to_anchor=(0.5, 0.48),
                       ncol=min(6, len(sh_handles)), fontsize=7, frameon=False)

        # Save with explicit bbox_inches so rcParams 'tight' doesn't collapse headers
        fig.align_ylabels()
        path = self.output_dir / f"{output_name}.{self.cfg.figure_format}"
        fig.savefig(path, dpi=self.cfg.figure_dpi,
                    bbox_inches=None, pad_inches=0.05)
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    # ------------------------------------------------------------------
    # 18. Intelliton Species Transition Graph
    # ------------------------------------------------------------------
    def plot_intelliton_transition_graph(self, transition_df: pd.DataFrame, top_edges: int = 12):
        if transition_df.empty:
            return
        prompt_types = list(transition_df["Prompt_Type"].dropna().unique())
        # Determine a reasonable figure height based on the largest number of species
        max_nodes = 1
        for pt in prompt_types:
            sub_all = transition_df[transition_df["Prompt_Type"] == pt].head(top_edges)
            if sub_all.empty:
                continue
            nodes = set(sub_all["From_Species"]).union(set(sub_all["To_Species"]))
            max_nodes = max(max_nodes, len(nodes))

        # Increase base figure size and scale so labels are legible for paper
        fig_h = max(4.0, 0.45 * max_nodes)
        # Use a narrower per-panel width for paper-friendly layout
        fig_w = 4.0 * max(len(prompt_types), 1)
        fig, axes = plt.subplots(1, len(prompt_types), figsize=(fig_w, fig_h), squeeze=False)
        axes = axes[0]

        for ax, prompt_type in zip(axes, prompt_types):
            sub = transition_df[transition_df["Prompt_Type"] == prompt_type].head(top_edges)
            if sub.empty:
                ax.axis("off")
                continue

            # Build left/right species lists ordered by total transition counts
            left_species = list(sub.groupby("From_Species")["Count"].sum().sort_values(ascending=False).index)
            right_species = list(sub.groupby("To_Species")["Count"].sum().sort_values(ascending=False).index)
            n_left = len(left_species)
            n_right = len(right_species)
            n_nodes = max(n_left, n_right)

            # x positions (data coords 0..1) for left and right columns
            x_left, x_right = 0.08, 0.92

            # y positions: integers 0..n_left-1 (top-to-bottom) — we'll invert y-axis for display
            left_y = {name: i for i, name in enumerate(left_species)}
            right_y = {name: i for i, name in enumerate(right_species)}

            # Draw edges between corresponding node indices
            max_count = max(float(sub["Count"].max()), 1.0)
            for _, row in sub.iterrows():
                from_sp = row["From_Species"]
                to_sp = row["To_Species"]
                if from_sp not in left_y or to_sp not in right_y:
                    continue
                y0 = left_y[from_sp]
                y1 = right_y[to_sp]
                lw = 0.8 + 3.2 * float(row["Count"]) / max_count
                # Normalize a shift metric into [0,1] for colormap safely
                shift_val = row.get("Mean_Target_Mode_Activation_Shift", 0.0)
                try:
                    shift_val = float(shift_val)
                except Exception:
                    shift_val = 0.0
                cmap_val = max(0.0, min(0.99, shift_val))
                color = plt.cm.inferno(cmap_val)
                ax.plot([x_left, x_right], [y0, y1], color=color, linewidth=lw, alpha=0.75)

            # Draw node markers and labels (larger for paper)
            for name, y in left_y.items():
                ax.scatter([x_left], [y], s=120, color="#333333")
                ax.text(x_left - 0.055, y, name, ha="right", va="center", fontsize=12)
            for name, y in right_y.items():
                ax.scatter([x_right], [y], s=120, color="#333333")
                ax.text(x_right + 0.055, y, name, ha="left", va="center", fontsize=12)

            ax.set_title(f"{prompt_type}", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, n_nodes - 0.5)
            ax.set_xticks([x_left, x_right])
            ax.set_xticklabels(["Step t", "Step t+1"], fontsize=12)
            ax.tick_params(axis="x", labelsize=12)
            ax.set_yticks([])
            ax.invert_yaxis()
            ax.grid(False)

        fig.tight_layout()
        self._save(fig, "intelliton_transition_graph")

    # ------------------------------------------------------------------
    # 17. Gauge Covariance (v5 NEW)
    # ------------------------------------------------------------------
    def plot_gauge_covariance(self, gc):
        """Plot gauge covariance test results."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        layers = gc.layers_tested
        n_species = gc.kl_raw.shape[0]
        n_show = min(n_species, 6)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_show, 1)))

        # (a) KL divergence: raw vs transported
        for i in range(n_show):
            name = gc.species_names[i] if i < len(gc.species_names) else f"Sp{i}"
            axes[0].plot(layers, gc.kl_raw[i].numpy(), '--', color=colors[i],
                        alpha=0.5, linewidth=1.5)
            axes[0].plot(layers, gc.kl_transported[i].numpy(), '-', color=colors[i],
                        label=f"{name}", linewidth=2)

        # Mark reference layer
        ref = gc.layers_tested[len(gc.layers_tested) // 2] if gc.layers_tested else 18
        axes[0].axvline(x=ref, color="gray", linestyle=":", alpha=0.5,
                       label="Reference layer")
        axes[0].set_xlabel("Intervention Layer")
        axes[0].set_ylabel("KL Divergence")
        axes[0].set_title("(a) Raw (dashed) vs Transported (solid)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # (b) Gauge violation heatmap
        gv = gc.gauge_violation[:n_show].numpy()
        vmax = max(abs(gv).max(), 0.01)
        im = axes[1].imshow(gv, cmap="RdBu_r", aspect="auto",
                           vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_xticks(range(len(layers)))
        axes[1].set_xticklabels(layers)
        axes[1].set_yticks(range(n_show))
        axes[1].set_yticklabels(
            [gc.species_names[i] if i < len(gc.species_names) else f"Sp{i}"
             for i in range(n_show)]
        )
        axes[1].set_xlabel("Intervention Layer")
        axes[1].set_title(r"(b) Gauge Violation: $\Delta KL = KL_{raw} - KL_{PT}$")

        # (c) Transport fidelity
        for i in range(n_show):
            name = gc.species_names[i] if i < len(gc.species_names) else f"Sp{i}"
            axes[2].plot(layers, gc.transport_fidelity[i].numpy(), '-o',
                        color=colors[i], label=name, linewidth=1.5, markersize=4)
        axes[2].set_xlabel("Target Layer")
        axes[2].set_ylabel(r"$\cos(v_{raw}, v_{PT})$")
        axes[2].set_title("(c) Transport Fidelity")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Gauge Invariance of Interventions",
                     fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "gauge_covariance")

    # ------------------------------------------------------------------
    # 19. Topological Charge (v5 NEW)
    # ------------------------------------------------------------------
    def plot_topological_charge(self, topo: TopologicalChargeResult):
        """Plot topological charge and winding numbers."""
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # (a) Topological charge density per plaquette
        if topo.charge_density.numel() > 0:
            plaq_labels = list(topo.charge_per_plaquette.keys())
            q_vals = topo.charge_density.numpy()
            x = np.arange(len(q_vals))
            colors = ["#2196F3" if q >= 0 else "#F44336" for q in q_vals]
            axes[0].bar(x, q_vals, color=colors, alpha=0.8)
            if len(plaq_labels) <= 12:
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(plaq_labels, rotation=45, fontsize=7)
            axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
            axes[0].set_xlabel("Plaquette")
            axes[0].set_ylabel(r"$q_P = F_P / 2\pi$")
            axes[0].set_title(
                f"(a) Topological Charge Density\n"
                f"$Q = {topo.total_charge:.4f}$, "
                f"nearest int = {topo.nearest_integer}, "
                f"{'QUANTIZED' if topo.is_quantized else 'not quantized'}"
            )
            axes[0].grid(True, alpha=0.3, axis="y")

        # (b) Per-token winding number
        wn = topo.winding_numbers.numpy()
        T = len(wn)
        axes[1].plot(range(T), wn, 'b-', linewidth=1.5, alpha=0.8)
        axes[1].axhline(y=topo.mean_winding, color="red", linestyle="--",
                        alpha=0.7, label=f"mean = {topo.mean_winding:.3f}")
        # Mark integer values
        for n in range(int(np.floor(wn.min())), int(np.ceil(wn.max())) + 1):
            axes[1].axhline(y=n, color="gray", linestyle=":", alpha=0.3)
        axes[1].set_xlabel("Token Position")
        axes[1].set_ylabel(r"Winding Number $n(t) = \theta / 2\pi$")
        axes[1].set_title("(b) Polyakov Winding Numbers")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # (c) Quantization check: histogram of charge density
        axes[2].hist(wn, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        axes[2].set_xlabel("Winding Number")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"(c) Winding Number Distribution\n"
                         f"Quantization error: {topo.quantization_error:.4f}")
        # Mark integers
        for n in range(int(np.floor(wn.min())), int(np.ceil(wn.max())) + 1):
            axes[2].axvline(x=n, color="red", linestyle="--", alpha=0.5)

        fig.suptitle("Topological Charge Analysis",
                     fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "topological_charge")

    # ------------------------------------------------------------------
    # 20. Lorentzian Resonance Widths (v5 NEW)
    # ------------------------------------------------------------------
    def plot_resonance_widths(self, resonance, n_show=6):
        """Plot Lorentzian fits to spectral function peaks."""
        from src.lattice_field import ResonanceWidthResult
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))

        omega = resonance.omega.numpy()
        n_show = min(n_show, len(resonance.fits))

        def lorentzian(w, A, w0, gamma):
            return A / ((w - w0) ** 2 + (gamma / 2.0) ** 2)

        # Top row: individual Lorentzian fits for first n_show modes
        for idx in range(min(n_show, 3)):
            ax = axes[0, idx]
            fit = resonance.fits[idx]
            rho = resonance.spectral_function[:, idx].numpy()

            ax.plot(omega, rho, 'b-', linewidth=1.5, label="Data", alpha=0.8)
            # Fitted Lorentzian
            lor = lorentzian(omega, fit.amplitude, fit.omega_0, fit.gamma)
            ax.plot(omega, lor, 'r--', linewidth=2,
                    label=f"Lorentzian (R²={fit.fit_r_squared:.2f})")
            ax.axvline(x=fit.omega_0, color="green", linestyle=":", alpha=0.5,
                       label=f"ω₀={fit.omega_0:.3f}")

            # Mark FWHM
            half_max = fit.amplitude / ((fit.gamma / 2.0) ** 2)
            left_w = fit.omega_0 - fit.gamma / 2
            right_w = fit.omega_0 + fit.gamma / 2
            ax.axvspan(max(0, left_w), min(np.pi, right_w),
                       alpha=0.1, color="red")

            status = "SHARP" if resonance.is_well_defined[idx] else "BROAD"
            ax.set_title(f"Mode {idx}: Γ={fit.gamma:.3f}, τ={fit.lifetime:.1f} [{status}]")
            ax.set_xlabel(r"$\omega$")
            ax.set_ylabel(r"$\rho(\omega)$")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Bottom-left: Lifetime bar chart
        lifetimes = [f.lifetime for f in resonance.fits[:n_show]]
        well_defined = resonance.is_well_defined[:n_show]
        colors = ["#4CAF50" if wd else "#FF9800" for wd in well_defined]
        mode_labels = [f"Mode {i}" for i in range(n_show)]
        axes[1, 0].barh(mode_labels, lifetimes, color=colors)
        axes[1, 0].set_xlabel(r"Lifetime $\tau = 1/\Gamma$")
        axes[1, 0].set_title("(d) Quasiparticle Lifetimes")
        axes[1, 0].invert_yaxis()
        from matplotlib.patches import Patch
        legend_el = [Patch(facecolor="#4CAF50", label="Well-defined (sharp)"),
                     Patch(facecolor="#FF9800", label="Broad/unstable")]
        axes[1, 0].legend(handles=legend_el, fontsize=8)

        # Bottom-center: Width vs center frequency
        omegas = [f.omega_0 for f in resonance.fits[:n_show]]
        gammas = [f.gamma for f in resonance.fits[:n_show]]
        scatter_colors = ["#4CAF50" if wd else "#F44336" for wd in well_defined]
        axes[1, 1].scatter(omegas, gammas, c=scatter_colors, s=80, edgecolors="black")
        for i, (w, g) in enumerate(zip(omegas, gammas)):
            axes[1, 1].annotate(f"M{i}", (w, g), fontsize=8,
                                xytext=(3, 3), textcoords="offset points")
        # Sharpness threshold line: gamma/omega = 0.5
        w_range = np.linspace(0.01, np.pi, 100)
        axes[1, 1].plot(w_range, 0.5 * w_range, 'k--', alpha=0.3,
                        label=r"$\Gamma/\omega_0 = 0.5$")
        axes[1, 1].set_xlabel(r"$\omega_0$ (center frequency)")
        axes[1, 1].set_ylabel(r"$\Gamma$ (FWHM)")
        axes[1, 1].set_title("(e) Resonance Width vs Frequency")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Bottom-right: Quasiparticle weight
        qp_w = resonance.quasiparticle_weight[:n_show]
        axes[1, 2].bar(mode_labels, qp_w, color="steelblue", alpha=0.8)
        axes[1, 2].set_ylabel("Quasiparticle Weight Z_qp")
        axes[1, 2].set_title("(f) Spectral Weight under Lorentzian")
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
                           label="Z=0.5")
        axes[1, 2].legend()
        axes[1, 2].tick_params(axis="x", rotation=45)

        fig.suptitle("Lorentzian Resonance Width Analysis",
                     fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "resonance_widths")

    # ------------------------------------------------------------------
    # 20.5 Differential Spectroscopy (v5 EXPERIMENTAL)
    # ------------------------------------------------------------------
    def plot_differential_spectroscopy(self, result: DifferentialSpectroscopyResult, top_k=12):
        """Plot differential-spectrum quasiparticle candidates."""
        fig, axes = plt.subplots(2, 2, figsize=(22, 14))
        omega = result.omega.numpy() if len(result.omega) else np.array([0.0])

        categories = list(result.category_total_spectrum.keys())
        if categories:
            heat = np.stack([
                result.category_total_spectrum[cat].numpy() for cat in categories
            ], axis=0)
            im = axes[0, 0].imshow(
                np.log10(np.maximum(heat, 1e-10)),
                aspect="auto",
                cmap="magma",
                origin="lower",
                extent=[omega[0], omega[-1], 0, len(categories) - 1],
            )
            axes[0, 0].set_yticks(range(len(categories)))
            axes[0, 0].set_yticklabels(categories)
            axes[0, 0].set_xlabel(r"$\omega$")
            axes[0, 0].set_title("(a) Category Differential Spectra")
            fig.colorbar(im, ax=axes[0, 0], label=r"$\log_{10} \rho_\Delta(\omega)$")
        else:
            axes[0, 0].text(0.5, 0.5, "No spectra", ha="center", va="center")
            axes[0, 0].set_title("(a) Category Differential Spectra")

        if result.candidate_df.empty:
            axes[0, 1].text(0.5, 0.5, "No recurrent narrow peaks found",
                            ha="center", va="center")
            axes[0, 1].set_title("(b) Top Differential Candidates")
            axes[1, 0].text(0.5, 0.5, "No candidate scatter", ha="center", va="center")
            axes[1, 0].set_title("(c) Width vs Frequency")
            axes[1, 1].text(0.5, 0.5, "No mode scores", ha="center", va="center")
            axes[1, 1].set_title("(d) Category Mode Scores")
            fig.suptitle("Differential Spectroscopy", fontsize=16, fontweight="bold")
            fig.tight_layout()
            self._save(fig, "differential_spectroscopy")
            return

        top_df = result.candidate_df.head(top_k).copy()
        labels = [f"{row.category}:M{int(row.mode_index)}" for row in top_df.itertuples()]
        axes[0, 1].barh(labels, top_df["score"], color="steelblue", alpha=0.85)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel("Candidate Score")
        axes[0, 1].set_title("(b) Top Differential Candidates")

        scatter = axes[1, 0].scatter(
            result.candidate_df["omega_0"],
            result.candidate_df["gamma"],
            c=result.candidate_df["score"],
            s=60 + 140 * result.candidate_df["recurrence"],
            cmap="viridis",
            edgecolors="black",
            alpha=0.85,
        )
        axes[1, 0].set_xlabel(r"$\omega_0$")
        axes[1, 0].set_ylabel(r"$\Gamma$")
        axes[1, 0].set_title("(c) Width vs Frequency")
        fig.colorbar(scatter, ax=axes[1, 0], label="Score")

        score_rows = []
        for cat, tensor in result.category_mode_scores.items():
            arr = tensor.numpy()
            for mode_idx, score in enumerate(arr):
                score_rows.append((cat, mode_idx, score))
        if score_rows:
            score_df = pd.DataFrame(score_rows, columns=["category", "mode_index", "score"])
            pivot = score_df.pivot(index="category", columns="mode_index", values="score").fillna(0.0)
            sns.heatmap(pivot, cmap="YlGnBu", ax=axes[1, 1], cbar_kws={"label": "Mode score"})
            axes[1, 1].set_title("(d) Category Mode Scores")
            axes[1, 1].set_xlabel("Mode index")
            axes[1, 1].set_ylabel("")
        else:
            axes[1, 1].text(0.5, 0.5, "No mode scores", ha="center", va="center")
            axes[1, 1].set_title("(d) Category Mode Scores")

        fig.suptitle("Differential Spectroscopy", fontsize=16, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "differential_spectroscopy")

    # ------------------------------------------------------------------
    # 21. Hallucination Diagnostics (v5 NEW)
    # ------------------------------------------------------------------
    def plot_hallucination_diagnostics(self, diag):
        """Plot hallucination diagnostic results.

        1×3 layout focusing on the core spectral characterization:
          (a) Spectral divergence with bimodal peak annotation
          (b) Spectral entropy gap
          (c) Per-pair spectral divergence heatmap
        """
        fig = plt.figure(figsize=self._figure_size("double", rows=2, height_scale=1.0))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0], figure=fig)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[:, 1]),
        ]

        n_layers = diag.mean_spectral_divergence.shape[0]
        layers = np.arange(n_layers)

        # ── (a) Mean spectral divergence with bimodal peaks ──────────
        sd = diag.mean_spectral_divergence.numpy()
        axes[0].plot(layers, sd, color="#1565C0", linewidth=2.2)
        # Shade ±1 std band across pairs
        if diag.signatures:
            n_l = min(s.spectral_divergence.shape[0] for s in diag.signatures)
            sd_stack = torch.stack(
                [s.spectral_divergence[:n_l] for s in diag.signatures]
            ).numpy()
            sd_std = sd_stack.std(axis=0)
            axes[0].fill_between(
                layers[:n_l], sd[:n_l] - sd_std, sd[:n_l] + sd_std,
                color="#1565C0", alpha=0.12, label="±1 s.d. across pairs",
            )
        # Mark bimodal peaks
        axes[0].axvline(
            x=diag.early_peak_layer, color="#E65100", linestyle="--",
            alpha=0.8, linewidth=1.5,
            label=f"Early peak: layer {diag.early_peak_layer} "
                  f"(JSD={diag.early_peak_strength:.4f})",
        )
        axes[0].axvline(
            x=diag.late_peak_layer, color="#B71C1C", linestyle="--",
            alpha=0.8, linewidth=1.5,
            label=f"Late peak: layer {diag.late_peak_layer} "
                  f"(JSD={diag.late_peak_strength:.4f})",
        )
        self._format_axes(
            axes[0],
            xlabel="Layer index",
            ylabel="Jensen-Shannon divergence",
            title="Spectral divergence: grounded vs hallucination",
            panel_label="(a)",
        )
        axes[0].legend(fontsize=7.5, loc="upper left")

        # ── (b) Entropy gap ──────────────────────────────────────────
        eg = diag.mean_entropy_gap.numpy()
        axes[1].plot(layers, eg, color="#2E7D32", linewidth=2.2)
        axes[1].fill_between(
            layers, 0, eg, where=eg > 0, color="#E53935", alpha=0.15,
            label="Halluc more diffuse",
        )
        axes[1].fill_between(
            layers, 0, eg, where=eg < 0, color="#1565C0", alpha=0.15,
            label="Grounded more diffuse",
        )
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        # Annotate late-layer trend
        late_start = n_layers - n_layers // 4
        late_mean = float(eg[late_start:].mean())
        axes[1].annotate(
            f"Late-layer mean\n$\\Delta H$={late_mean:+.4f}",
            xy=(late_start + (n_layers - late_start) / 2, late_mean),
            fontsize=8, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )
        self._format_axes(
            axes[1],
            xlabel="Layer index",
            ylabel=r"$\Delta H = H_{\mathrm{halluc}} - H_{\mathrm{grounded}}$",
            title="Spectral entropy gap",
            panel_label="(b)",
        )
        axes[1].legend(fontsize=8)

        # ── (c) Per-pair spectral divergence heatmap ─────────────────
        if diag.signatures:
            n_pairs = len(diag.signatures)
            n_l = min(s.spectral_divergence.shape[0] for s in diag.signatures)
            sd_matrix = torch.stack(
                [s.spectral_divergence[:n_l] for s in diag.signatures]
            ).numpy()
            im = axes[2].imshow(
                sd_matrix, aspect="auto", cmap="YlOrRd", origin="lower",
            )
            self._add_colorbar(fig, im, axes[2], "JSD")
            self._format_axes(axes[2], xlabel="Layer index", title=f"Per-pair spectral divergence ({n_pairs} pairs)", panel_label="(c)")
            pair_cats = diag.pair_categories
            if n_pairs <= 25:
                axes[2].set_yticks(range(n_pairs))
                axes[2].set_yticklabels(pair_cats, fontsize=6)
            # Mark bimodal peaks on heatmap
            axes[2].axvline(
                x=diag.early_peak_layer, color="#E65100", linestyle=":",
                alpha=0.6, linewidth=1.2,
            )
            axes[2].axvline(
                x=diag.late_peak_layer, color="#B71C1C", linestyle=":",
                alpha=0.6, linewidth=1.2,
            )
        else:
            axes[2].text(
                0.5, 0.5, "No pairs analyzed", ha="center",
                va="center", transform=axes[2].transAxes,
            )

        fig.tight_layout()
        self._save(fig, "hallucination_diagnostics")

    # ------------------------------------------------------------------
    # 18. Summary Figure
    # ------------------------------------------------------------------
    def plot_summary(self, power, momenta, sigma, propagator, disp=None, rg=None):
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # (a) Momentum
        ax1 = fig.add_subplot(gs[0, 0])
        sort_idx = momenta.argsort()
        pl = np.log10(power[:, sort_idx].numpy() + 1e-10)
        ax1.imshow(pl, aspect="auto", cmap="inferno", origin="lower")
        ax1.set_title("(a) Momentum Spectrum")
        ax1.set_xlabel("k"); ax1.set_ylabel("Layer")

        # (b) Spin
        ax2 = fig.add_subplot(gs[0, 1])
        sn = sigma.numpy() / (sigma[:, 0:1].numpy() + 1e-10)
        ax2.imshow(sn.T, aspect="auto", cmap="viridis", origin="lower")
        ax2.set_title("(b) Spin Spectrum"); ax2.set_xlabel("Layer")

        # (c) Mass decay
        ax3 = fig.add_subplot(gs[0, 2])
        layers = np.arange(propagator.decay_curves.shape[0])
        for i in range(min(6, propagator.decay_curves.shape[1])):
            ax3.semilogy(layers, np.maximum(propagator.decay_curves[:, i].numpy(), 1e-10),
                        label=f"m={propagator.mass[i].item():.3f}")
        ax3.set_title("(c) Mode Amplitude"); ax3.legend(fontsize=7)

        # (d) Dispersion
        ax4 = fig.add_subplot(gs[1, 0])
        if disp is not None:
            ks = disp.momenta.numpy()
            si = np.argsort(ks)
            for i in range(min(4, disp.energies.shape[1])):
                ax4.plot(ks[si], disp.energies[si, i].numpy(), 'o-', markersize=2,
                        label=f"m={disp.fitted_mass[i].item():.3f}")
            ax4.set_title("(d) Dispersion E(k)"); ax4.legend(fontsize=7)
        else:
            ax4.text(0.5, 0.5, "N/A", ha="center")

        # (e) RG flow
        ax5 = fig.add_subplot(gs[1, 1])
        if rg is not None:
            for i in range(min(6, rg.running_mass.shape[1])):
                ax5.plot(rg.running_mass[:, i].numpy(), label=f"Mode {i}")
            ax5.set_title("(e) RG Mass Flow"); ax5.legend(fontsize=7)
        else:
            ax5.text(0.5, 0.5, "N/A", ha="center")

        # (f) Mass histogram
        ax6 = fig.add_subplot(gs[1, 2])
        cm = {"massless": "#2196F3", "light": "#4CAF50",
              "medium": "#FF9800", "heavy": "#F44336"}
        colors = [cm.get(c, "gray") for c in propagator.mass_category]
        ax6.bar(range(len(propagator.mass)), propagator.mass.numpy(), color=colors)
        ax6.set_title("(f) Mass Spectrum"); ax6.set_xlabel("Mode")

        fig.suptitle(f"Intelliton Spectrum Analysis -- {self.cfg.model_name}",
                    fontsize=18, fontweight="bold")
        self._save(fig, "summary")
