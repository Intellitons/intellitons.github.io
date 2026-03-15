"""Wigner Classification v5 -- Lattice Gauge Theory Intelliton Classifier.

Upgrades from v4:
1. Uses dispersion-relation mass (not just propagator decay)
2. Includes lattice momentum from Brillouin zone
3. Adds EFT bare vs renormalized mass to the catalog
4. RG flow stability classification (stable IR / unstable UV / resonance)
5. Enhanced PDG table with all new quantum numbers
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import IntellitonConfig
from src.lattice_field import (
    LatticeField, PropagatorResult, SpinDecomposition, DispersionRelation,
)
from src.eft_renormalization import RenormalizationFlow, EFTParameters


@dataclass
class IntellitonSpecies:
    """A classified Intelliton quasi-particle (v5)."""
    species_id: int
    name: str
    # Wigner quantum numbers
    spin: float
    mass: float                        # pole mass from propagator
    mass_lattice: float                # mass from lattice dispersion relation
    mass_bare: float                   # bare mass (early layers)
    mass_renormalized: float           # renormalized mass (late layers)
    mass_category: str
    dominant_momentum: float
    helicity: float
    # Lattice properties
    group_velocity: float              # dE/dk at dominant momentum
    dispersion_quality: float          # R^2 of lattice dispersion fit
    # RG flow properties
    fixed_point_layer: int
    fixed_point_type: str              # "IR" / "UV" / "crossover" / "none"
    anomalous_dimension: float
    wavefunction_Z: float              # wavefunction renormalization
    # Classification metadata
    svd_mode_index: int
    layer_range: Tuple[int, int]
    prompt_categories: List[str]
    amplitude: float
    fit_r_squared: float


@dataclass
class IntellitonCatalog:
    """PDG-style catalog of classified Intelliton species (v5)."""
    species: List[IntellitonSpecies]
    eft_alpha: float = 0.0
    eft_beta: float = 0.0
    summary_df: Optional[pd.DataFrame] = None

    def build_dataframe(self) -> pd.DataFrame:
        rows = []
        for sp in self.species:
            rows.append({
                "Name": sp.name,
                "Spin": f"{sp.spin:.2f}",
                "Mass(pole)": f"{sp.mass:.4f}",
                "Mass(lat)": f"{sp.mass_lattice:.4f}",
                "Mass(bare)": f"{sp.mass_bare:.4f}",
                "Mass(ren)": f"{sp.mass_renormalized:.4f}",
                "Category": sp.mass_category,
                "Momentum": f"{sp.dominant_momentum:.3f}",
                "Helicity": f"{sp.helicity:.2f}",
                "v_group": f"{sp.group_velocity:.3f}",
                "FP Layer": sp.fixed_point_layer,
                "FP Type": sp.fixed_point_type,
                "gamma": f"{sp.anomalous_dimension:.3f}",
                "Z": f"{sp.wavefunction_Z:.3f}",
                "Layers": f"{sp.layer_range[0]}-{sp.layer_range[1]}",
                "Amplitude": f"{sp.amplitude:.1f}",
                "Active In": ", ".join(sp.prompt_categories[:3]),
            })
        self.summary_df = pd.DataFrame(rows)
        return self.summary_df

    def build_compact_dataframe(self) -> pd.DataFrame:
        """Compact table for terminal display."""
        rows = []
        for sp in self.species:
            rows.append({
                "Name": sp.name,
                "s": f"{sp.spin:.1f}",
                "m_pole": f"{sp.mass:.3f}",
                "m_lat": f"{sp.mass_lattice:.3f}",
                "Cat": sp.mass_category[:4],
                "k": f"{sp.dominant_momentum:.2f}",
                "h": f"{sp.helicity:.1f}",
                "FP": sp.fixed_point_type[:2],
                "gamma": f"{sp.anomalous_dimension:.2f}",
                "Z": f"{sp.wavefunction_Z:.2f}",
                "Amp": f"{sp.amplitude:.0f}",
            })
        return pd.DataFrame(rows)

    def print_table(self):
        df = self.build_compact_dataframe()
        width = 110
        header = (
            "=" * width + "\n"
            "  INTELLITON PARTICLE TABLE -- Lattice Gauge Theory Classification\n"
            f"  EFT Parameters: alpha={self.eft_alpha:.4f}, beta={self.eft_beta:.4f}\n"
            "=" * width
        )
        print(header)
        print(df.to_string(index=False))
        print("=" * width)
        print(f"  {len(self.species)} species classified")

    def to_latex(self) -> str:
        df = self.build_dataframe()
        return df.to_latex(
            index=False,
            caption="Intelliton Particle Table (Lattice Gauge Theory)",
            label="tab:intelliton",
        )


class IntellitonClassifier:
    """Classify Intelliton species from spectral data.

    Algorithm:
    1. Per-category SVD analysis with dispersion relation
    2. Pole mass extraction + lattice dispersion mass
    3. EFT renormalization flow per mode
    4. Cross-category merging by Vh cosine similarity
    5. Build catalog with full quantum number assignments
    """

    def __init__(self, cfg: IntellitonConfig):
        self.cfg = cfg
        self.species_vectors: Dict[int, torch.Tensor] = {}

    def classify(
        self,
        fields_by_category: Dict[str, List[LatticeField]],
        rg_flow: Optional[RenormalizationFlow] = None,
        eft_params: Optional[EFTParameters] = None,
        dispersion: Optional[DispersionRelation] = None,
    ) -> IntellitonCatalog:
        n_modes = self.cfg.n_top_modes

        # Evaluate min modes across all fields in all categories to prevent zero-padding artifacts
        global_min_modes = self.cfg.n_top_modes
        for cat, fields in fields_by_category.items():
            for lf in fields:
                decomp = lf.compute_spin_decomposition(n_modes=self.cfg.n_top_modes)
                global_min_modes = min(global_min_modes, decomp.sigma.shape[1])
                
        # Phase 1: Per-category spectral analysis
        category_spectra = {}
        category_Vh = {}
        category_active_modes = {}

        for cat, fields in fields_by_category.items():
            cat_sigmas, cat_masses, cat_mass_lat = [], [], []
            cat_r2s, cat_momenta, cat_helicities, cat_spins = [], [], [], []
            cat_Vh_vectors = []

            for lf in fields:
                decomp = lf.compute_spin_decomposition()
                prop = lf.compute_propagator(decomp)
                helicity = lf.compute_helicity(decomp)
                spin_qn = lf.compute_spin_quantum_numbers(decomp)

                mid_start = self.cfg.num_layers // 4
                mid_end = 3 * self.cfg.num_layers // 4
                avg_sigma = decomp.sigma[mid_start:mid_end].mean(dim=0)

                cat_sigmas.append(avg_sigma[:global_min_modes])
                cat_masses.append(prop.mass[:global_min_modes])
                cat_mass_lat.append(prop.mass_lattice[:global_min_modes])
                cat_r2s.append(prop.fit_r_squared[:global_min_modes])
                cat_helicities.append(helicity.mean(dim=0)[:global_min_modes])
                cat_spins.append(spin_qn.float().mean(dim=0)[:global_min_modes])

                mid_momenta = torch.zeros_like(prop.mass)
                if dispersion is not None:
                    # Peak momentum per mode: k where E(k) is maximal
                    peak_indices = dispersion.energies.argmax(dim=0)  # [n_modes]
                    mid_momenta = dispersion.momenta[peak_indices]
                cat_momenta.append(mid_momenta[:global_min_modes])

                vh_layer = min(self.cfg.num_layers // 2, decomp.Vh.shape[0]-1)
                cat_Vh_vectors.append(decomp.Vh[vh_layer, :global_min_modes, :])

            category_spectra[cat] = {
                "sigma": torch.stack(cat_sigmas).mean(dim=0),
                "mass": torch.stack(cat_masses).mean(dim=0),
                "mass_lat": torch.stack(cat_mass_lat).mean(dim=0),
                "r2": torch.stack(cat_r2s).mean(dim=0),
                "helicity": torch.stack(cat_helicities).mean(dim=0),
                "spin": torch.stack(cat_spins).mean(dim=0),
                "momentum": torch.stack(cat_momenta).mean(dim=0),
            }
            category_Vh[cat] = torch.stack(cat_Vh_vectors).mean(dim=0)
            category_active_modes[cat] = self._find_active_modes(
                category_spectra[cat]["sigma"]
            )

        # Phase 2: Build candidate species
        candidates = []
        for cat in fields_by_category:
            spec = category_spectra[cat]
            for mode_i in category_active_modes[cat]:
                candidates.append({
                    "category": cat,
                    "mode_index": mode_i,
                    "sigma": spec["sigma"][mode_i].item(),
                    "mass": spec["mass"][mode_i].item(),
                    "mass_lat": spec["mass_lat"][mode_i].item(),
                    "r2": spec["r2"][mode_i].item(),
                    "spin": spec["spin"][mode_i].item(),
                    "helicity": spec["helicity"][mode_i].item(),
                    "momentum": spec["momentum"][mode_i].item(),
                    "Vh": category_Vh[cat][mode_i],
                })

        if not candidates:
            return IntellitonCatalog(species=[])

        # Phase 3: Merge by Vh cosine similarity
        Vh_stack = torch.stack([c["Vh"] for c in candidates])
        Vh_norm = Vh_stack / (Vh_stack.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = (Vh_norm @ Vh_norm.T).abs()
        clusters = self._cluster_by_similarity(
            sim_matrix, self.cfg.merge_cosine_threshold
        )

        # Phase 4: Build final species with v5 quantum numbers
        species_list = []
        self.species_vectors = {}

        for idx, cluster_indices in enumerate(clusters):
            cc = [candidates[i] for i in cluster_indices]
            active_cats = sorted(set(c["category"] for c in cc))

            avg_mass = float(np.mean([c["mass"] for c in cc]))
            avg_mass_lat = float(np.mean([c["mass_lat"] for c in cc]))
            avg_r2 = float(np.mean([c["r2"] for c in cc]))
            avg_spin = float(np.mean([c["spin"] for c in cc]))
            avg_helicity = float(np.mean([c["helicity"] for c in cc]))
            avg_momentum = float(np.mean([c["momentum"] for c in cc]))
            avg_amplitude = float(np.mean([c["sigma"] for c in cc]))

            best_cand = max(cc, key=lambda c: c["sigma"])
            mode_i = best_cand["mode_index"]

            avg_Vh = torch.stack([c["Vh"] for c in cc]).mean(dim=0)
            self.species_vectors[idx] = avg_Vh

            # Mass category
            abs_mass = abs(avg_mass)
            if abs_mass < 0.01:
                mass_cat = "massless"
            elif abs_mass < 0.1:
                mass_cat = "light"
            elif abs_mass < 0.5:
                mass_cat = "medium"
            else:
                mass_cat = "heavy"

            layer_range = self._find_layer_range(fields_by_category, mode_i, active_cats)

            # RG flow quantities (if available)
            fp_layer = 0
            fp_type = "none"
            anom_dim = 0.0
            Z = 1.0
            m_bare = avg_mass
            m_ren = avg_mass

            if rg_flow is not None and mode_i < rg_flow.running_mass.shape[1]:
                fp_layer = rg_flow.fixed_point_layers[mode_i].item()
                fp_type = rg_flow.fixed_point_type[mode_i]
                anom_dim = rg_flow.anomalous_dimension[mode_i].item()

            if eft_params is not None and mode_i < eft_params.bare_mass.shape[0]:
                m_bare = eft_params.bare_mass[mode_i].item()
                m_ren = eft_params.renormalized_mass[mode_i].item()
                Z = eft_params.Z[mode_i].item()

            # Group velocity
            v_g = 0.0
            if dispersion is not None and mode_i < dispersion.group_velocity.shape[1]:
                # At the dominant momentum
                momenta = dispersion.momenta
                k_idx = (momenta - avg_momentum).abs().argmin().item()
                v_g = dispersion.group_velocity[k_idx, mode_i].item()

            disp_quality = 0.0
            if dispersion is not None and mode_i < dispersion.fit_quality.shape[0]:
                disp_quality = dispersion.fit_quality[mode_i].item()

            species_list.append(IntellitonSpecies(
                species_id=idx,
                name=f"I_{idx}",
                spin=round(avg_spin, 2),
                mass=round(avg_mass, 4),
                mass_lattice=round(avg_mass_lat, 4),
                mass_bare=round(m_bare, 4),
                mass_renormalized=round(m_ren, 4),
                mass_category=mass_cat,
                dominant_momentum=round(avg_momentum, 3),
                helicity=round(avg_helicity, 2),
                group_velocity=round(v_g, 3),
                dispersion_quality=round(disp_quality, 3),
                fixed_point_layer=fp_layer,
                fixed_point_type=fp_type,
                anomalous_dimension=round(anom_dim, 3),
                wavefunction_Z=round(Z, 3),
                svd_mode_index=mode_i,
                layer_range=layer_range,
                prompt_categories=active_cats,
                amplitude=round(avg_amplitude, 2),
                fit_r_squared=round(avg_r2, 3),
            ))

        # Sort by amplitude and re-index
        species_list.sort(key=lambda s: s.amplitude, reverse=True)
        new_vectors = {}
        for i, sp in enumerate(species_list):
            old_id = sp.species_id
            sp.species_id = i
            sp.name = f"I_{i}"
            new_vectors[i] = self.species_vectors[old_id]
        self.species_vectors = new_vectors

        catalog = IntellitonCatalog(
            species=species_list,
            eft_alpha=eft_params.alpha if eft_params else 0.0,
            eft_beta=eft_params.beta if eft_params else 0.0,
        )
        return catalog

    def _find_active_modes(self, sigma: torch.Tensor) -> List[int]:
        if len(sigma) == 0 or sigma[0].item() == 0:
            return []
        threshold = sigma[0].item() * self.cfg.sv_threshold_ratio
        return [i for i in range(len(sigma)) if sigma[i].item() > threshold]

    def _cluster_by_similarity(
        self, sim_matrix: torch.Tensor, threshold: float
    ) -> List[List[int]]:
        n = sim_matrix.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j].item() >= threshold:
                    union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        return list(groups.values())

    def _find_layer_range(
        self, fields_by_category, mode_i, active_cats=None
    ) -> Tuple[int, int]:
        cats = active_cats or list(fields_by_category.keys())
        starts, ends = [], []
        for cat in cats:
            fields = fields_by_category.get(cat, [])
            if not fields:
                continue
            decomp = fields[0].compute_spin_decomposition()
            if mode_i >= decomp.sigma.shape[1]:
                continue
            col = decomp.sigma[:, mode_i]
            threshold = col.max().item() * 0.1
            active = (col > threshold).nonzero(as_tuple=True)[0]
            if len(active) > 0:
                starts.append(active[0].item())
                ends.append(active[-1].item())
        if not starts:
            return (0, self.cfg.num_layers)
        return (min(starts), max(ends))
