"""Lattice Field Theory v5 -- Discrete Quantum Field on the Token Lattice.

Major upgrades from v4:
  1. Discrete propagator with pole mass extraction via spectral function
  2. Lattice dispersion relation E(k) fitting
  3. Brillouin zone mass from dispersion curvature
  4. Improved spin quantum numbers via representation theory dimension
  5. Layer-wise renormalization group (RG) flow of masses

The field tensor phi[l, t, d] represents:
  - l: layer index = discrete Euclidean time
  - t: token position = spatial lattice site
  - d: channel index = internal (spin/color) degree of freedom
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from src.config import IntellitonConfig


@dataclass
class SpinDecomposition:
    """SVD decomposition of the residual stream per layer."""
    U: torch.Tensor       # [L+1, T, n_modes]
    sigma: torch.Tensor   # [L+1, n_modes]
    Vh: torch.Tensor      # [L+1, n_modes, D]
    matched_indices: Optional[List[torch.Tensor]] = None


@dataclass
class PropagatorResult:
    """Mass extracted from discrete propagator analysis."""
    mass: torch.Tensor                 # [n_modes] -- pole masses
    mass_lattice: torch.Tensor         # [n_modes] -- lattice masses from dispersion
    decay_curves: torch.Tensor         # [L+1, n_modes] -- normalized amplitude (growth or decay)
    spectral_function: torch.Tensor    # [n_freq, n_modes] -- spectral density rho(omega)
    fit_r_squared: torch.Tensor        # [n_modes]
    mass_category: List[str]
    pole_positions: List[float]        # pole positions in frequency space


@dataclass
class LorentzianFit:
    """Lorentzian fit result for a single spectral peak.

    L(omega) = A / ((omega - omega_0)^2 + (Gamma/2)^2)
    """
    omega_0: float          # peak center frequency
    gamma: float            # full width at half maximum (FWHM)
    amplitude: float        # peak height A
    lifetime: float         # tau = 1/Gamma (quasiparticle lifetime)
    fit_r_squared: float    # quality of Lorentzian fit


@dataclass
class ResonanceWidthResult:
    """Lorentzian resonance width analysis for all modes."""
    fits: List[LorentzianFit]                # per-mode best Lorentzian fit
    spectral_function: torch.Tensor          # [n_freq, n_modes]
    omega: torch.Tensor                      # [n_freq] frequency axis
    is_well_defined: List[bool]              # True if Gamma/omega_0 < threshold
    quasiparticle_weight: List[float]        # Z_qp = integral under Lorentzian / total


@dataclass
class DispersionRelation:
    """Lattice dispersion relation E(k) for each mode."""
    momenta: torch.Tensor          # [T] -- Brillouin zone momenta
    energies: torch.Tensor         # [T, n_modes] -- energy at each momentum
    fitted_mass: torch.Tensor      # [n_modes] -- mass from dispersion fit
    fit_quality: torch.Tensor      # [n_modes] -- R^2 of lattice dispersion fit
    group_velocity: torch.Tensor   # [T, n_modes] -- dE/dk


class LatticeField:
    """Discrete quantum field on the 1D token lattice (v5)."""

    def __init__(self, field_tensor: torch.Tensor, cfg: IntellitonConfig):
        self.field = field_tensor
        self.n_layers = field_tensor.shape[0]
        self.seq_len = field_tensor.shape[1]
        self.hidden_dim = field_tensor.shape[2]
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Brillouin Zone Momenta
    # ------------------------------------------------------------------

    def brillouin_zone_momenta(self) -> torch.Tensor:
        """Lattice momenta in the first Brillouin zone [-pi, pi)."""
        T = self.seq_len
        k = torch.arange(T, dtype=torch.float32)
        p = 2.0 * torch.pi * k / T
        p[p > torch.pi] -= 2.0 * torch.pi
        return p

    # ------------------------------------------------------------------
    # Discrete Fourier Transform (Lattice Momentum Spectrum)
    # ------------------------------------------------------------------

    def compute_momentum_spectrum(self) -> torch.Tensor:
        """DFT of the field over the token (spatial) dimension.
        Returns: complex tensor [L+1, T, D]"""
        field_clean = self.field.float()
        field_clean = torch.where(
            torch.isfinite(field_clean), field_clean,
            torch.zeros_like(field_clean)
        )
        return torch.fft.fft(field_clean, dim=1, norm="ortho")

    def momentum_power_spectrum(self) -> torch.Tensor:
        """P[l, k] = sum_d |R_tilde[l,k,d]|^2. Returns: [L+1, T]"""
        ft = self.compute_momentum_spectrum()
        return (ft.abs() ** 2).sum(dim=-1)

    # ------------------------------------------------------------------
    # SVD Spin Decomposition
    # ------------------------------------------------------------------

    def compute_spin_decomposition(
        self, n_modes: Optional[int] = None
    ) -> SpinDecomposition:
        """SVD of residual stream at each layer, keeping top n_modes."""
        n_modes = n_modes or self.cfg.n_top_modes
        n_modes = min(n_modes, self.seq_len, self.hidden_dim)
        U_all, sigma_all, Vh_all = [], [], []

        for l in range(self.n_layers):
            R_l = self.field[l].float()
            R_l = torch.where(torch.isfinite(R_l), R_l, torch.zeros_like(R_l))
            U, S, Vh = torch.linalg.svd(R_l, full_matrices=False)
            U_all.append(U[:, :n_modes])
            sigma_all.append(S[:n_modes])
            Vh_all.append(Vh[:n_modes, :])

        decomp = SpinDecomposition(
            U=torch.stack(U_all),
            sigma=torch.stack(sigma_all),
            Vh=torch.stack(Vh_all),
        )
        decomp.matched_indices = self._match_modes_across_layers(decomp)
        return decomp

    def _match_modes_across_layers(
        self, decomp: SpinDecomposition
    ) -> List[torch.Tensor]:
        """Match SVD modes across layers by Vh cosine similarity."""
        matched = []
        for l in range(self.n_layers - 1):
            v_curr = decomp.Vh[l]
            v_next = decomp.Vh[l + 1]
            v_curr_n = v_curr / (v_curr.norm(dim=-1, keepdim=True) + 1e-8)
            v_next_n = v_next / (v_next.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (v_curr_n @ v_next_n.T).abs()
            best = sim.argmax(dim=1)
            matched.append(best)
        return matched

    # ------------------------------------------------------------------
    # Spin Quantum Numbers (Representation Dimension)
    # ------------------------------------------------------------------

    def compute_spin_quantum_numbers(
        self, decomp: SpinDecomposition
    ) -> torch.Tensor:
        """Assign spin via spectral entropy of spatial Fourier profile.

        Low entropy (localized in k-space) -> spin 0 (scalar)
        Medium entropy -> spin 1 (vector)
        High entropy (delocalized) -> spin 2 (tensor)

        Returns: [L+1, n_modes] with continuous spin in [0, 2]
        """
        U = decomp.U.float()
        n_modes = U.shape[2]
        T = U.shape[1]
        eps = 1e-10
        log_T = np.log(T)
        spin = torch.zeros(self.n_layers, n_modes)

        for l in range(self.n_layers):
            for i in range(n_modes):
                u_ft = torch.fft.fft(U[l, :, i], norm="ortho")
                power = u_ft.abs() ** 2
                p = power / (power.sum() + eps)
                H = -(p * torch.log(p + eps)).sum().item()
                H_norm = max(0.0, min(1.0, H / log_T))
                spin[l, i] = H_norm * 2.0

        return spin

    # ------------------------------------------------------------------
    # Discrete Propagator -- Pole Mass Extraction (v5 NEW)
    # ------------------------------------------------------------------

    def compute_propagator(self, decomp: SpinDecomposition) -> PropagatorResult:
        """Extract mass from the discrete propagator G(tau) = <phi(0)phi(tau)>.

        v5 upgrade: Instead of simple exponential fit, we compute:
        1. Two-point correlator C(delta_l) across layers for each mode
        2. Spectral function rho(omega) via DFT of C(delta_l)
        3. Pole mass from dominant peak in rho(omega)
        4. Fallback to effective mass m_eff(l) = log(C(l)/C(l+1))
        """
        sigma = decomp.sigma.float()  # [L+1, n_modes]
        n_modes = sigma.shape[1]
        matched = decomp.matched_indices

        # Build matched decay curves (same as v4)
        decay_curves = torch.zeros(self.n_layers, n_modes)
        decay_curves[0] = sigma[0]
        current_idx = torch.arange(n_modes)
        for l in range(self.n_layers - 1):
            if matched is not None and l < len(matched):
                current_idx = matched[l][current_idx]
            for i in range(n_modes):
                decay_curves[l + 1, i] = sigma[l + 1, current_idx[i]]

        # Normalize
        l0 = self.cfg.mass_fit_skip_layers
        eps = 1e-10
        ref = decay_curves[l0].clamp(min=eps)
        norm_curves = decay_curves / ref.unsqueeze(0)

        # Compute spectral function via DFT over layer (time) dimension
        # Transformer modes typically GROW across layers (residual connections),
        # so the raw correlator C(l) is monotonically increasing.
        # We analyze C(l) directly, enabling pole
        # extraction. Poles of 1/C coincide with poles of C (same mass).
        C = norm_curves[l0:]  # [L-l0, n_modes]
        C_for_fft = C
        n_tau = C_for_fft.shape[0]
        # Zero-pad for better frequency resolution
        n_fft = max(64, n_tau * 4)
        C_padded = torch.zeros(n_fft, n_modes)
        C_padded[:n_tau] = C_for_fft

        spectral = torch.fft.rfft(C_padded, dim=0).abs() ** 2  # [n_fft//2+1, n_modes]
        omega = torch.linspace(0, torch.pi, spectral.shape[0])

        # Extract pole mass from spectral function peaks
        masses = torch.zeros(n_modes)
        pole_positions = []
        r_squared = torch.zeros(n_modes)

        for i in range(n_modes):
            rho = spectral[:, i].numpy()
            # Find peaks in spectral function
            peaks, properties = find_peaks(rho, height=rho.max() * 0.1, distance=2)

            if len(peaks) > 0:
                # Dominant pole = lowest-frequency peak (heaviest contribution)
                dominant_peak = peaks[0]
                pole_omega = omega[dominant_peak].item()
                # Lattice pole mass: m = 2 * sinh(omega/2) for lattice propagator
                # or simply m = omega for small omega
                masses[i] = 2.0 * np.sinh(pole_omega / 2.0)
                pole_positions.append(pole_omega)
            else:
                # Fallback: effective mass from absolute rate of change
                m_eff = self._effective_mass(C[:, i])
                masses[i] = m_eff
                pole_positions.append(m_eff)

            # R^2 from exponential fit on the reciprocal (decaying) correlator
            r_squared[i] = self._fit_exponential_r2(
                C_for_fft[:, i], masses[i].item()
            )

        # Lattice dispersion mass (from E(k=0))
        mass_lattice = self._compute_lattice_mass(decomp)

        # Mass categories
        categories = []
        for m in masses:
            mv = abs(m.item())
            if mv < 0.01:
                categories.append("massless")
            elif mv < 0.1:
                categories.append("light")
            elif mv < 0.5:
                categories.append("medium")
            else:
                categories.append("heavy")

        return PropagatorResult(
            mass=masses,
            mass_lattice=mass_lattice,
            decay_curves=norm_curves,
            spectral_function=spectral,
            fit_r_squared=r_squared,
            mass_category=categories,
            pole_positions=pole_positions,
        )

    def _effective_mass(self, C: torch.Tensor) -> float:
        """Effective mass: |<log(C(l)/C(l+1))>| averaged over layers.

        Returns the absolute rate of change -- positive for both decaying
        and growing modes (transformer residual streams typically grow).
        """
        eps = 1e-10
        C_np = C.clamp(min=eps).numpy()
        ratios = C_np[:-1] / C_np[1:]
        ratios = np.clip(ratios, eps, None)
        m_eff = np.mean(np.log(ratios))
        return abs(float(m_eff))

    def _fit_exponential_r2(self, curve: torch.Tensor, mass: float) -> float:
        """R^2 of exponential fit C(l) ~ exp(-m*l)."""
        eps = 1e-10
        l = np.arange(len(curve), dtype=np.float64)
        y = curve.clamp(min=eps).numpy().astype(np.float64)
        y_pred = np.exp(-mass * l)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + eps
        return float(1.0 - ss_res / ss_tot)

    def _compute_lattice_mass(self, decomp: SpinDecomposition) -> torch.Tensor:
        """Compute lattice mass from k-space propagator and dispersion relation.

        For each SVD mode i, compute the energy E(k) from the decay of the
        k-space two-point correlator across layers, then extract the rest
        mass from E(k=0) via the lattice dispersion relation:
            E^2(k) = m^2 + (2 sin(k/2))^2
        At k=0 this reduces to E(0) = m, giving the lattice rest mass.
        """
        n_modes = decomp.sigma.shape[1]
        T = self.seq_len
        mass_lat = torch.zeros(n_modes)
        eps = 1e-10
        l_start = self.cfg.mass_fit_skip_layers

        for i in range(n_modes):
            # Fourier-transform the spatial (U) profile at each layer
            ft_layers = []
            for l in range(self.n_layers):
                u_mode = decomp.U[l, :, i].float()
                u_ft = torch.fft.fft(u_mode, norm="ortho")
                ft_layers.append(u_ft)

            # E(k) = average of -log|phi_k(l) / phi_k(l+1)| over layers
            E_k = torch.zeros(T)
            count = 0
            for l in range(l_start, self.n_layers - 1):
                amp_l = ft_layers[l].abs()
                amp_next = ft_layers[l + 1].abs()
                ratio = (amp_next + eps) / (amp_l + eps)
                E_k += torch.log(ratio.clamp(min=eps))
                count += 1
            if count > 0:
                E_k /= count
            E_k = E_k.abs()  # energy is magnitude of rate of change

            # Rest mass = E at zero-momentum
            momenta = self.brillouin_zone_momenta()
            k_zero_idx = momenta.abs().argmin().item()
            E_rest = E_k[k_zero_idx].item()

            # Also try fitting: m^2 = E(k)^2 - (2 sin(k/2))^2 for all k
            # and take the median (more robust than single-k)
            k_np = momenta.numpy()
            E_np = E_k.numpy()
            m_sq_all = E_np ** 2 - (2.0 * np.sin(k_np / 2.0)) ** 2
            # Keep only positive values for median
            m_sq_pos = m_sq_all[m_sq_all > 0]
            if len(m_sq_pos) > 0:
                mass_lat[i] = float(np.sqrt(np.median(m_sq_pos)))
            else:
                # Fallback: rest mass from E(k=0)
                mass_lat[i] = E_rest

        return mass_lat

    def compute_resonance_widths(
        self, propagator: PropagatorResult,
        sharpness_threshold: float = 0.5,
    ) -> ResonanceWidthResult:
        """Fit Lorentzian profiles to spectral function peaks.

        For each mode, fits L(omega) = A / ((omega - omega_0)^2 + (Gamma/2)^2)
        to the dominant peak in rho(omega).

        A well-defined quasiparticle has Gamma/omega_0 << 1 (sharp peak).
        A broad peak (Gamma ~ omega_0) indicates a short-lived excitation.

        Args:
            propagator: PropagatorResult from compute_propagator
            sharpness_threshold: max Gamma/omega_0 for "well-defined" quasiparticle

        Returns: ResonanceWidthResult
        """
        sf = propagator.spectral_function  # [n_freq, n_modes]
        n_modes = sf.shape[1]
        omega = torch.linspace(0, float(torch.pi), sf.shape[0])
        omega_np = omega.numpy()

        fits = []
        is_well_defined = []
        qp_weights = []

        def lorentzian(w, A, w0, gamma):
            return A / ((w - w0) ** 2 + (gamma / 2.0) ** 2)

        for i in range(n_modes):
            rho = sf[:, i].numpy().astype(np.float64)
            rho_total = float(np.sum(rho))

            # Find peaks for initial guess
            peaks, props = find_peaks(rho, height=rho.max() * 0.1, distance=2)

            if len(peaks) == 0:
                # No peak: report trivial result
                fits.append(LorentzianFit(
                    omega_0=0.0, gamma=np.pi, amplitude=float(rho.max()),
                    lifetime=1.0 / np.pi, fit_r_squared=0.0,
                ))
                is_well_defined.append(False)
                qp_weights.append(0.0)
                continue

            # Use the dominant (first) peak
            pk = peaks[0]
            w0_init = omega_np[pk]
            A_init = float(rho[pk])
            # Estimate FWHM from half-max width
            half_max = A_init / 2.0
            left = pk
            while left > 0 and rho[left] > half_max:
                left -= 1
            right = pk
            while right < len(rho) - 1 and rho[right] > half_max:
                right += 1
            gamma_init = max(omega_np[right] - omega_np[left], 0.01)

            try:
                popt, _ = curve_fit(
                    lorentzian, omega_np, rho,
                    p0=[A_init * (gamma_init / 2.0) ** 2, w0_init, gamma_init],
                    bounds=([0, 0, 1e-6], [np.inf, np.pi, np.pi]),
                    maxfev=10000,
                )
                A_fit, w0_fit, gamma_fit = popt

                # R^2
                rho_pred = lorentzian(omega_np, *popt)
                ss_res = np.sum((rho - rho_pred) ** 2)
                ss_tot = np.sum((rho - rho.mean()) ** 2) + 1e-10
                r2 = max(0.0, 1.0 - ss_res / ss_tot)

                lifetime = 1.0 / max(gamma_fit, 1e-10)

                # Quasiparticle weight: fraction of spectral weight under Lorentzian
                lor_integral = float(np.sum(rho_pred))
                qp_w = lor_integral / max(rho_total, 1e-10)

                fits.append(LorentzianFit(
                    omega_0=float(w0_fit),
                    gamma=float(gamma_fit),
                    amplitude=float(A_fit),
                    lifetime=float(lifetime),
                    fit_r_squared=float(r2),
                ))
                is_well_defined.append(gamma_fit / max(w0_fit, 1e-10) < sharpness_threshold)
                qp_weights.append(min(qp_w, 1.0))

            except (RuntimeError, ValueError):
                # Fit failed: use peak-based estimate
                fits.append(LorentzianFit(
                    omega_0=w0_init,
                    gamma=gamma_init,
                    amplitude=A_init,
                    lifetime=1.0 / max(gamma_init, 1e-10),
                    fit_r_squared=0.0,
                ))
                is_well_defined.append(False)
                qp_weights.append(0.0)

        return ResonanceWidthResult(
            fits=fits,
            spectral_function=sf,
            omega=omega,
            is_well_defined=is_well_defined,
            quasiparticle_weight=qp_weights,
        )

    # ------------------------------------------------------------------
    # Lattice Dispersion Relation (v5 NEW)
    # ------------------------------------------------------------------

    def compute_dispersion_relation(
        self, decomp: SpinDecomposition
    ) -> DispersionRelation:
        """Compute the lattice dispersion relation E(k) for each SVD mode.

        For each mode i:
        1. At each layer l, FFT of U[l, :, i] gives amplitude at each k
        2. Energy E_i(k) = -log(|G_i(k, delta_l)|) where G is the
           k-space propagator between consecutive layers
        3. Fit to lattice dispersion: E^2 = m^2 + (2*sin(k/2))^2

        Returns: DispersionRelation with momenta, energies, fitted masses
        """
        n_modes = decomp.U.shape[2]
        T = self.seq_len
        momenta = self.brillouin_zone_momenta()

        # Compute k-space propagator: G(k, l->l+1) = <phi_k(l) phi_k(l+1)*>
        energies = torch.zeros(T, n_modes)
        fitted_mass = torch.zeros(n_modes)
        fit_quality = torch.zeros(n_modes)
        group_velocity = torch.zeros(T, n_modes)

        for i in range(n_modes):
            # Collect Fourier amplitudes across layers
            ft_layers = []
            for l in range(self.n_layers):
                u_mode = decomp.U[l, :, i].float()
                u_ft = torch.fft.fft(u_mode, norm="ortho")
                ft_layers.append(u_ft)

            # Compute energy from correlator decay in layer dimension
            # For each k, E(k) = -<log|phi_k(l)/phi_k(l+1)|> averaged
            E_k = torch.zeros(T)
            eps = 1e-10
            count = 0
            l_start = self.cfg.mass_fit_skip_layers
            for l in range(l_start, self.n_layers - 1):
                amp_l = ft_layers[l].abs()
                amp_next = ft_layers[l + 1].abs()
                ratio = (amp_next + eps) / (amp_l + eps)
                E_k += torch.log(ratio.clamp(min=eps))
                count += 1
            if count > 0:
                E_k /= count

            E_k = E_k.abs()  # energy is magnitude of rate of change
            energies[:, i] = E_k

            # Fit lattice dispersion: E(k)^2 = m^2 + (2*sin(k/2))^2
            # Use only low-momentum points (|k| < 2pi/3) where the lattice
            # dispersion relation is most reliable.  BZ-edge points are
            # dominated by noise for short sequences.
            k_np = momenta.numpy()
            E_np = E_k.numpy()
            k_zero_idx = int(np.argmin(np.abs(k_np)))
            E_rest = E_np[k_zero_idx]

            # Select inner Brillouin zone for fitting
            k_cutoff = 2.0 * np.pi / 3.0
            inner_mask = np.abs(k_np) <= k_cutoff
            k_inner = k_np[inner_mask]
            E_inner = E_np[inner_mask]

            if len(k_inner) >= 3:
                try:
                    def lattice_dispersion(k, m):
                        return np.sqrt(m ** 2 + (2.0 * np.sin(k / 2.0)) ** 2)

                    popt, _ = curve_fit(
                        lattice_dispersion, k_inner, E_inner,
                        p0=[max(E_rest, 0.01)], bounds=(0, np.inf),
                        maxfev=10000
                    )
                    fitted_mass[i] = float(popt[0])

                    # R^2 computed on inner points only
                    E_pred = lattice_dispersion(k_inner, popt[0])
                    ss_res = np.sum((E_inner - E_pred) ** 2)
                    ss_tot = np.sum((E_inner - E_inner.mean()) ** 2) + eps
                    r2 = 1.0 - ss_res / ss_tot

                    if r2 < 0:
                        # Poor fit — fall back to E(k=0) as rest mass
                        fitted_mass[i] = float(E_rest)
                        fit_quality[i] = 0.0
                    else:
                        fit_quality[i] = float(r2)
                except (RuntimeError, ValueError):
                    fitted_mass[i] = float(E_rest)
                    fit_quality[i] = 0.0
            else:
                # Too few inner-BZ points — use E(k=0) directly
                fitted_mass[i] = float(E_rest)
                fit_quality[i] = 0.0

            # Group velocity: dE/dk (finite difference)
            dk = 2 * torch.pi / T
            dE = torch.zeros(T)
            for j in range(T):
                j_next = (j + 1) % T
                j_prev = (j - 1) % T
                dE[j] = (E_k[j_next] - E_k[j_prev]) / (2 * dk)
            group_velocity[:, i] = dE

        return DispersionRelation(
            momenta=momenta,
            energies=energies,
            fitted_mass=fitted_mass,
            fit_quality=fit_quality,
            group_velocity=group_velocity,
        )

    # ------------------------------------------------------------------
    # Per-Mode Momentum
    # ------------------------------------------------------------------

    def compute_per_mode_momentum(self, decomp: SpinDecomposition) -> torch.Tensor:
        """Dominant lattice momentum for each mode at each layer.
        Returns: [L+1, n_modes]"""
        U = decomp.U.float()
        n_modes = U.shape[2]
        momenta = self.brillouin_zone_momenta()
        mode_momenta = torch.zeros(self.n_layers, n_modes)

        for l in range(self.n_layers):
            for i in range(n_modes):
                u_ft = torch.fft.fft(U[l, :, i], norm="ortho")
                k_peak = u_ft.abs().argmax().item()
                mode_momenta[l, i] = momenta[k_peak]

        return mode_momenta

    # ------------------------------------------------------------------
    # Helicity
    # ------------------------------------------------------------------

    def compute_helicity(self, decomp: SpinDecomposition) -> torch.Tensor:
        """Helicity = projection of spin onto momentum.
        Returns: [L+1, n_modes]"""
        U = decomp.U.float()
        spin_qn = self.compute_spin_quantum_numbers(decomp)
        momenta = self.brillouin_zone_momenta()
        n_modes = U.shape[2]
        helicity = torch.zeros(self.n_layers, n_modes)

        for l in range(self.n_layers):
            for i in range(n_modes):
                u_ft = torch.fft.fft(U[l, :, i], norm="ortho")
                k_peak = u_ft.abs().argmax().item()
                p_peak = momenta[k_peak].item()
                s = spin_qn[l, i].item()
                if abs(p_peak) < 1e-6:
                    helicity[l, i] = 0.0
                else:
                    helicity[l, i] = float(np.sign(p_peak)) * s

        return helicity

    # ------------------------------------------------------------------
    # EFT Renormalization Group Flow (v5 NEW)
    # ------------------------------------------------------------------

    def compute_rg_flow(self, decomp: SpinDecomposition) -> dict:
        """Track the Renormalization Group (RG) flow of effective mass
        across layers (Euclidean time).

        Inspired by Raju & Netrapalli (2026) EFT of Transformers:
        - Compute effective mass at each layer scale
        - Identify fixed points (stable Intellitons)
        - Detect phase transitions (grokking = mass gap opening)

        Returns dict with:
            effective_mass: [L+1, n_modes] -- running mass at each layer
            beta_function: [L, n_modes] -- dm/dl (RG beta function)
            fixed_points: [n_modes] -- layers where dm/dl ~ 0
            anomalous_dimension: [n_modes] -- gamma = d(log sigma)/d(log l)
        """
        sigma = decomp.sigma.float()
        n_modes = sigma.shape[1]
        w = self.cfg.eft_window_size
        eps = 1e-10

        # Signed running mass: m_eff = -d(log sigma)/dl
        # Positive = decaying mode (normal mass), Negative = growing mode (tachyonic)
        eff_mass = torch.zeros(self.n_layers, n_modes)
        for l in range(self.n_layers):
            l_start = max(0, l - w // 2)
            l_end = min(self.n_layers, l + w // 2 + 1)
            if l_end - l_start < 2:
                continue
            for i in range(n_modes):
                window = sigma[l_start:l_end, i].clamp(min=eps)
                log_window = torch.log(window)
                # Linear fit: log(sigma) = slope * l + c, so m = -slope
                x = torch.arange(l_end - l_start, dtype=torch.float32)
                xm = x.mean()
                ym = log_window.mean()
                slope = ((x - xm) * (log_window - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=eps)
                eff_mass[l, i] = -slope.item()

        # Beta function: dm/dl
        beta = torch.zeros(self.n_layers - 1, n_modes)
        for l in range(self.n_layers - 1):
            beta[l] = eff_mass[l + 1] - eff_mass[l]

        # Fixed points: layers where |beta| is minimal
        fixed_points = torch.zeros(n_modes, dtype=torch.long)
        for i in range(n_modes):
            abs_beta = beta[:, i].abs()
            # Find the layer in the second half (after "warm-up") with smallest |beta|
            fp_start = self.n_layers // 3
            fp_region = abs_beta[fp_start:]
            fixed_points[i] = fp_start + fp_region.argmin().item()

        # Anomalous dimension: gamma = d(log sigma) / d(log l)
        anomalous = torch.zeros(n_modes)
        for i in range(n_modes):
            mid_s = self.n_layers // 3
            mid_e = 2 * self.n_layers // 3
            log_sigma = torch.log(sigma[mid_s:mid_e, i].clamp(min=eps))
            log_l = torch.log(torch.arange(mid_s, mid_e, dtype=torch.float32) + 1)
            x = log_l
            y = log_sigma
            xm, ym = x.mean(), y.mean()
            slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=eps)
            anomalous[i] = slope.item()

        return {
            "effective_mass": eff_mass,
            "beta_function": beta,
            "fixed_points": fixed_points,
            "anomalous_dimension": anomalous,
        }
