"""
Steric Mass Action (SMA) Isotherm for Multi-Component Ion-Exchange Chromatography
==================================================================================

The SMA isotherm (Brooks & Cramer, 1992) models competitive adsorption in
ion-exchange chromatography by explicitly accounting for:
  - Electrostatic interactions (charge-based binding)
  - Steric shielding (bound proteins block neighboring sites)
  - Salt modulation of binding affinity

Isotherm equations (for component i, with salt as component 0):
--------------------------------------------------------------------
  q_i = K_i * c_i * (Lambda - sum_j(nu_j + sigma_j) * q_j)^nu_i / c_salt^nu_i

where:
  Lambda  : ionic capacity of the stationary phase [mM]
  K_i     : equilibrium constant for component i
  nu_i    : characteristic charge (number of binding sites used)
  sigma_i : steric shielding factor (sites blocked per bound molecule)
  c_salt  : salt concentration in mobile phase [mM]
  c_i     : mobile phase concentration of component i [g/L or mM]
  q_i     : stationary phase concentration of component i [g/L or mM]

References:
  Brooks, C.A. & Cramer, S.M. (1992). Steric mass-action ion exchange:
  Displacement profiles and induced salt gradients. AIChE Journal, 38(12).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class SMAParameters:
    """
    Parameters for the Steric Mass Action isotherm.

    Attributes
    ----------
    Lambda : float
        Ionic capacity of the stationary phase [mM or same units as q_salt].
    K : np.ndarray
        Equilibrium constants for each protein component. Shape: (n_components,).
    nu : np.ndarray
        Characteristic charges for each protein component. Shape: (n_components,).
    sigma : np.ndarray
        Steric shielding factors for each protein component. Shape: (n_components,).
    n_components : int
        Number of protein components (excludes salt).
    """
    Lambda: float
    K: np.ndarray
    nu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.nu = np.asarray(self.nu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)

        if not (self.K.shape == self.nu.shape == self.sigma.shape):
            raise ValueError(
                f"K, nu, sigma must have the same shape. "
                f"Got K={self.K.shape}, nu={self.nu.shape}, sigma={self.sigma.shape}"
            )
        if self.Lambda <= 0:
            raise ValueError(f"Lambda (ionic capacity) must be positive, got {self.Lambda}")

    @property
    def n_components(self) -> int:
        return len(self.K)


class SMAIsotherm:
    """
    Steric Mass Action (SMA) isotherm for competitive multi-component adsorption.

    This class provides both the equilibrium adsorption calculation and its
    Jacobian, which is needed for implicit ODE solvers in the TDM.

    The salt counter-ion is treated as component 0 and obeys a simple
    electro-neutrality constraint on the stationary phase:

        q_salt = Lambda - sum_i(nu_i * q_i)

    Usage
    -----
    >>> params = SMAParameters(Lambda=1200.0, K=[0.05], nu=[5.0], sigma=[50.0])
    >>> iso = SMAIsotherm(params)
    >>> c_salt, c_protein = 50.0, np.array([1.5])
    >>> q = iso.compute_equilibrium(c_salt, c_protein)
    """

    def __init__(self, params: SMAParameters):
        self.params = params

    def _free_sites(self, q_proteins: np.ndarray) -> float:
        """
        Compute the number of available binding sites (Q* in SMA literature).

        Q* = Lambda - sum_i( (nu_i + sigma_i) * q_i )

        Parameters
        ----------
        q_proteins : np.ndarray
            Stationary phase concentrations of all protein components.

        Returns
        -------
        float
            Available ionic capacity. Clipped to zero to avoid non-physical states.
        """
        occupied = np.sum((self.params.nu + self.params.sigma) * q_proteins)
        return max(self.params.Lambda - occupied, 0.0)

    def compute_equilibrium(
        self,
        c_salt: float,
        c_proteins: np.ndarray,
        q_proteins_guess: Optional[np.ndarray] = None,
        max_iter: int = 500,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """
        Compute stationary phase concentrations at equilibrium.

        The SMA isotherm is implicit in q because free sites Q* depends on q
        itself. The naive fixed-point iteration

            q^(k+1) = K * c * (Q*(q^(k)))^nu / c_salt^nu

        can oscillate (2-cycle) when the Jacobian spectral radius > 1.
        We instead solve the root-finding problem

            F(q) = q - K * c * (Q*(q))^nu / c_salt^nu = 0

        using **damped Newton–Raphson** with a bisection-based line search to
        guarantee a monotone decrease in ||F||, which always converges for the
        physically meaningful (non-negative, capacity-bounded) root.

        Parameters
        ----------
        c_salt : float
            Mobile phase salt concentration [mM]. Must be > 0.
        c_proteins : np.ndarray
            Mobile phase protein concentrations [g/L]. Shape: (n_components,).
        q_proteins_guess : np.ndarray, optional
            Initial guess for q. Defaults to a physically motivated estimate.
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on ||F(q)||_inf.

        Returns
        -------
        np.ndarray
            Equilibrium stationary phase concentrations. Shape: (n_components,).
        """
        if c_salt <= 0:
            raise ValueError(f"Salt concentration must be positive, got c_salt={c_salt}")

        c_proteins = np.asarray(c_proteins, dtype=float)
        n = self.params.n_components

        # --- Initial guess ---
        if q_proteins_guess is not None:
            q = np.array(q_proteins_guess, dtype=float)
        else:
            # Dilute-limit estimate: assume Q* ≈ Lambda (no loading yet)
            q = np.maximum(
                self.params.K * c_proteins * (self.params.Lambda ** self.params.nu)
                / (c_salt ** self.params.nu),
                0.0,
            )
            # Clip so we don't start in a physically impossible region
            q = self._clip_to_feasible(q)

        def residual(q_vec):
            q_star = self._free_sites(q_vec)
            rhs = self.params.K * c_proteins * (q_star ** self.params.nu) / (c_salt ** self.params.nu)
            return q_vec - np.maximum(rhs, 0.0)

        def jacobian(q_vec):
            """Analytical Jacobian of F(q) = q - g(q)."""
            q_star = self._free_sites(q_vec)
            if q_star <= 0:
                return np.eye(n)
            # dg_i/dq_j = K_i*c_i*nu_i*(Q*)^(nu_i-1) / c_salt^nu_i * (-(nu_j+sigma_j))
            dg_dQstar = (
                self.params.K * c_proteins * self.params.nu
                * (q_star ** (self.params.nu - 1.0))
                / (c_salt ** self.params.nu)
            )
            dQstar_dq = -(self.params.nu + self.params.sigma)
            # J_ij = delta_ij - dg_i/dQstar * dQstar/dq_j
            J = np.eye(n) - np.outer(dg_dQstar, dQstar_dq)
            return J

        for iteration in range(max_iter):
            F = residual(q)
            if np.max(np.abs(F)) < tol:
                return q

            # Newton step
            J = jacobian(q)
            try:
                dq = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(J, -F, rcond=None)[0]

            # Damped line search: try step sizes 1, 0.5, 0.25, ...
            step = 1.0
            F_norm = np.max(np.abs(F))
            for _ in range(10):
                q_trial = self._clip_to_feasible(q + step * dq)
                if np.max(np.abs(residual(q_trial))) < F_norm:
                    break
                step *= 0.5

            q = q_trial

        warnings.warn(
            f"SMA Newton solver did not fully converge after {max_iter} iterations. "
            f"Residual: {np.max(np.abs(residual(q))):.2e}. "
            f"Result may be approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return q

    def _clip_to_feasible(self, q: np.ndarray) -> np.ndarray:
        """
        Project q into the physically feasible region:
          q_i >= 0  and  sum((nu_i + sigma_i)*q_i) <= Lambda.

        Uses a simple proportional scaling if the capacity constraint is violated.
        """
        q = np.maximum(q, 0.0)
        occupied = np.sum((self.params.nu + self.params.sigma) * q)
        if occupied > self.params.Lambda:
            q = q * (self.params.Lambda / occupied) * 0.99
        return q

    def compute_dqdc(
        self,
        c_salt: float,
        c_proteins: np.ndarray,
        q_proteins: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Jacobian of q with respect to c (dq/dc matrix).

        This is needed by the TDM solver to compute the effective retardation
        factor for each component. Entry (i, j) = dq_i / dc_j.

        Derivation (implicit differentiation of the SMA equilibrium):
        ---------------------------------------------------------------
        At equilibrium for component i:
            q_i = K_i * c_i * (Q*)^{nu_i} / c_salt^{nu_i}    ...(*)

        where Q* = Lambda - sum_k( (nu_k + sigma_k) * q_k ).

        Differentiating (*) w.r.t. c_j (holding c_salt fixed):

            dq_i/dc_j = [K_i*(Q*)^{nu_i}/c_salt^{nu_i}] * delta_{ij}
                        + K_i*c_i*nu_i*(Q*)^{nu_i-1}/c_salt^{nu_i} * dQ*/dc_j

        where dQ*/dc_j = -sum_k( (nu_k + sigma_k) * dq_k/dc_j ).

        This yields the linear system (matrix form for all i, j):

            dq/dc = B + C * dq/dc

        where:
            B_{ij}  = [K_i*(Q*)^{nu_i}/c_salt^{nu_i}] * delta_{ij}   (direct term)
            C_{ij}  = -K_i*c_i*nu_i*(Q*)^{nu_i-1}/c_salt^{nu_i} * (nu_j + sigma_j)

        Rearranging: (I - C) * dq/dc = B  =>  dq/dc = (I - C)^{-1} * B

        Parameters
        ----------
        c_salt : float
            Mobile phase salt concentration.
        c_proteins : np.ndarray
            Mobile phase protein concentrations. Shape: (n_components,).
        q_proteins : np.ndarray
            Stationary phase protein concentrations at equilibrium. Shape: (n_components,).

        Returns
        -------
        np.ndarray
            Jacobian matrix dq/dc. Shape: (n_components, n_components).
        """
        n = self.params.n_components
        q_star = self._free_sites(q_proteins)

        if q_star <= 0:
            # All sites occupied: no sensitivity to mobile phase changes
            return np.zeros((n, n))

        # B: diagonal direct-sensitivity matrix
        # B_ii = K_i * (Q*)^nu_i / c_salt^nu_i
        b_diag = self.params.K * (q_star ** self.params.nu) / (c_salt ** self.params.nu)
        B = np.diag(b_diag)

        # C: coupling matrix through Q*
        # C_ij = -K_i*c_i*nu_i*(Q*)^(nu_i-1)/c_salt^nu_i * (nu_j + sigma_j)
        dgi_dQstar = (
            self.params.K
            * c_proteins
            * self.params.nu
            * (q_star ** (self.params.nu - 1.0))
            / (c_salt ** self.params.nu)
        )
        # C = outer(dg/dQ*, -(nu+sigma))
        C = -np.outer(dgi_dQstar, self.params.nu + self.params.sigma)

        # Solve: (I - C) * dqdc = B
        A = np.eye(n) - C
        try:
            dqdc = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            dqdc = np.linalg.lstsq(A, B, rcond=None)[0]

        return dqdc

    def salt_stationary_phase(self, q_proteins: np.ndarray) -> float:
        """
        Compute the salt counter-ion concentration on the stationary phase.

        From electro-neutrality on the stationary phase:
            q_salt = Lambda - sum_i(nu_i * q_i)

        Parameters
        ----------
        q_proteins : np.ndarray
            Stationary phase protein concentrations.

        Returns
        -------
        float
            Stationary phase salt concentration.
        """
        return max(self.params.Lambda - np.sum(self.params.nu * q_proteins), 0.0)
