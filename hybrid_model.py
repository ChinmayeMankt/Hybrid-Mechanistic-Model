"""
Hybrid Mechanistic-ML Model for Preparative Chromatography
===========================================================

This module is the top-level orchestrator that combines:
  1. The Transport-Dispersive Model (TDM) — mechanistic backbone
  2. The Gaussian Process Residual Corrector (GPR) — ML correction layer

Architecture
------------
The hybrid prediction follows a two-stage pipeline:

    Stage 1 (Mechanistic):
        Run TDM + SMA isotherm simulation to get the physics-based prediction
        of the chromatogram (concentration vs. time at column outlet).

    Stage 2 (ML Correction):
        Extract summary statistics from the mechanistic chromatogram.
        Build a feature vector (operating conditions + mechanistic summary).
        Query the fitted GPR to get:
            delta_mu    = predicted residual correction
            delta_sigma = uncertainty of the correction

    Stage 3 (Fusion):
        Corrected prediction = mechanistic_summary + delta_mu
        Uncertainty          = delta_sigma  (from GPR posterior)

The GPR is trained offline on experimental data: pairs of
(operating conditions, observed_summary - mechanistic_summary).
At inference time, it corrects the mechanistic prediction for the given
operating point without any additional experimental data.

Key Design Decisions
--------------------
- The correction is applied at the **summary statistic** level (e.g., peak
  area, retention time), not pixel-by-pixel on the chromatogram. This keeps
  the feature space low-dimensional and the GPR tractable.
- For full chromatogram correction, see the `ChromatogramCorrector` class
  which applies a time-warping + amplitude correction to the raw profile.
- The TDM is always run first; the GPR never replaces the mechanistic model,
  only nudges its output.

Usage
-----
See HybridChromatographyModel.predict() for the standard workflow.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .tdm_solver import TDMSolver, ColumnParameters, ChromatographyRun
from .sma_isotherm import SMAIsotherm, SMAParameters
from .residual_model import ResidualGPR, ResidualData, build_features_from_run, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class ChromatogramSummary:
    """
    Summary statistics extracted from a chromatogram (mechanistic or corrected).

    All values refer to a single protein component at the column outlet.

    Attributes
    ----------
    component_index : int
        Index of the protein component (1-based, 0 = salt).
    peak_retention_time_min : float
        Time of peak maximum [min].
    peak_height_gL : float
        Peak maximum concentration [g/L].
    peak_width_min : float
        Peak width at half-maximum (FWHM) [min].
    peak_area_gL_min : float
        Integrated peak area [g/L * min], proportional to recovered mass.
    resolution : float
        Resolution to the next peak (0.0 if only one component).
    """
    component_index: int
    peak_retention_time_min: float
    peak_height_gL: float
    peak_width_min: float
    peak_area_gL_min: float
    resolution: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peak_retention_time_min": self.peak_retention_time_min,
            "peak_height_gL": self.peak_height_gL,
            "peak_width_min": self.peak_width_min,
            "peak_area_gL_min": self.peak_area_gL_min,
            "resolution": self.resolution,
        }


@dataclass
class HybridPrediction:
    """
    Full output from the hybrid model for one chromatography run.

    Attributes
    ----------
    t : np.ndarray
        Time axis [min]. Shape: (n_t,).
    z : np.ndarray
        Spatial axis [cm]. Shape: (N,).
    c_mechanistic : np.ndarray
        Mechanistic outlet chromatogram. Shape: (n_t, n_total).
    c_corrected : np.ndarray or None
        GPR-corrected outlet chromatogram (if GPR is fitted). Shape: (n_t, n_total).
    mechanistic_summaries : list of ChromatogramSummary
        Per-component summary statistics from TDM.
    corrected_summaries : list of ChromatogramSummary or None
        GPR-corrected summary statistics.
    correction_uncertainty : np.ndarray or None
        Posterior std of GPR correction per output. Shape: (n_outputs,).
    gpr_was_applied : bool
        Whether the GPR correction was applied.
    """
    t: np.ndarray
    z: np.ndarray
    c_mechanistic: np.ndarray
    c_corrected: Optional[np.ndarray]
    mechanistic_summaries: list
    corrected_summaries: Optional[list]
    correction_uncertainty: Optional[np.ndarray]
    gpr_was_applied: bool


# ---------------------------------------------------------------------------
# Helper: chromatogram analysis
# ---------------------------------------------------------------------------

def extract_peak_summary(
    t: np.ndarray,
    c_outlet: np.ndarray,
    component_index: int,
) -> ChromatogramSummary:
    """
    Extract peak summary statistics from a single-component outlet profile.

    Parameters
    ----------
    t : np.ndarray
        Time vector [min]. Shape: (n_t,).
    c_outlet : np.ndarray
        Outlet concentration for all components. Shape: (n_t, n_total).
        Index 0 = salt; index k = protein k.
    component_index : int
        Component to analyze (1-based for proteins).

    Returns
    -------
    ChromatogramSummary
    """
    c = c_outlet[:, component_index]

    if c.max() < 1e-10:
        # Component never eluted (or at negligible concentration)
        return ChromatogramSummary(
            component_index=component_index,
            peak_retention_time_min=float("nan"),
            peak_height_gL=0.0,
            peak_width_min=float("nan"),
            peak_area_gL_min=float(np.trapezoid(c, t)),
            resolution=0.0,
        )

    # Peak maximum
    peak_idx = int(np.argmax(c))
    peak_time = float(t[peak_idx])
    peak_height = float(c[peak_idx])

    # Peak area (trapezoidal integration)
    peak_area = float(np.trapezoid(c, t))

    # FWHM: find half-maximum crossings
    half_max = peak_height / 2.0
    above = c >= half_max
    # Find first crossing (rising edge)
    rising = np.where(np.diff(above.astype(int)) > 0)[0]
    falling = np.where(np.diff(above.astype(int)) < 0)[0]

    if len(rising) > 0 and len(falling) > 0:
        t_rise = float(np.interp(half_max, [c[rising[0]], c[rising[0] + 1]], [t[rising[0]], t[rising[0] + 1]]))
        t_fall = float(np.interp(half_max, [c[falling[-1] + 1], c[falling[-1]]], [t[falling[-1] + 1], t[falling[-1]]]))
        fwhm = t_fall - t_rise
    else:
        fwhm = float("nan")

    return ChromatogramSummary(
        component_index=component_index,
        peak_retention_time_min=peak_time,
        peak_height_gL=peak_height,
        peak_width_min=fwhm,
        peak_area_gL_min=peak_area,
        resolution=0.0,
    )


def compute_resolution(summary_a: ChromatogramSummary, summary_b: ChromatogramSummary) -> float:
    """
    Compute chromatographic resolution between two peaks.

    Rs = 2 * |t_R,b - t_R,a| / (w_a + w_b)

    where w is the baseline peak width ≈ 4 * sigma ≈ 1.699 * FWHM.

    Returns 0.0 if either peak is missing FWHM.
    """
    if np.isnan(summary_a.peak_width_min) or np.isnan(summary_b.peak_width_min):
        return 0.0
    delta_t = abs(summary_b.peak_retention_time_min - summary_a.peak_retention_time_min)
    w_sum = (summary_a.peak_width_min + summary_b.peak_width_min) * 1.699  # FWHM → baseline width
    return 2.0 * delta_t / max(w_sum, 1e-10)


# ---------------------------------------------------------------------------
# Main hybrid model class
# ---------------------------------------------------------------------------

class HybridChromatographyModel:
    """
    Hybrid Mechanistic-ML Digital Twin for Preparative Chromatography.

    Combines the 1D Transport-Dispersive Model (TDM) with an optional
    Gaussian Process residual corrector. The GPR correction layer is
    calibrated from experimental data and improves prediction accuracy
    for non-ideal effects (e.g., buffer-specific binding, temperature
    variation, resin lot-to-lot variation).

    Parameters
    ----------
    column : ColumnParameters
        Physical and operational column parameters.
    sma_params : SMAParameters
        SMA isotherm parameters for the protein/resin system.
    residual_gpr : ResidualGPR, optional
        Pre-fitted GPR corrector. If None, the model operates in
        purely mechanistic mode.

    Example
    -------
    >>> col = ColumnParameters(length=20.0, diameter=2.6, void_fraction=0.35,
    ...                        flow_rate=5.0, n_grid_points=50)
    >>> sma = SMAParameters(Lambda=1200., K=[0.05, 0.02], nu=[5., 4.], sigma=[50., 30.])
    >>> model = HybridChromatographyModel(col, sma)

    # Purely mechanistic run:
    >>> result = model.predict(operating_conditions, inlet_fn, c_initial, t_end)

    # After calibration with experimental data:
    >>> model.fit_residual_model(X_train, y_residuals_train)
    >>> result = model.predict(operating_conditions, inlet_fn, c_initial, t_end)
    """

    def __init__(
        self,
        column: ColumnParameters,
        sma_params: SMAParameters,
        residual_gpr: Optional[ResidualGPR] = None,
    ):
        self.column = column
        self.sma_params = sma_params
        self.isotherm = SMAIsotherm(sma_params)
        self.solver = TDMSolver(column, self.isotherm)
        self.residual_gpr = residual_gpr

        self._n_proteins = sma_params.n_components

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit_residual_model(
        self,
        residual_data: ResidualData,
        kernel: str = "matern",
        n_restarts_optimizer: int = 5,
    ) -> "HybridChromatographyModel":
        """
        Fit the GPR residual corrector from experimental data.

        Parameters
        ----------
        residual_data : ResidualData
            Training data where y_residuals = y_observed - y_mechanistic.
            Feature matrix X should be built using build_features_from_run().
        kernel : str
            GPR kernel type ('matern' recommended, or 'rbf').
        n_restarts_optimizer : int
            Hyperparameter optimization restarts.

        Returns
        -------
        self (for chaining)
        """
        print(f"[HybridModel] Fitting GPR residual model on "
              f"{residual_data.X.shape[0]} samples, "
              f"{residual_data.y_residuals.shape[1]} output(s).")
        self.residual_gpr = ResidualGPR(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self.residual_gpr.fit(residual_data)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        operating_conditions: dict,
        c_inlet_fn: Callable[[float], np.ndarray],
        c_initial: np.ndarray,
        t_end: float,
        t_eval: Optional[np.ndarray] = None,
        apply_gpr: bool = True,
        verbose: bool = True,
    ) -> HybridPrediction:
        """
        Run the hybrid model for one chromatography experiment.

        Stage 1: Mechanistic TDM simulation.
        Stage 2: GPR residual correction (if model is fitted and apply_gpr=True).
        Stage 3: Fuse predictions and return structured output.

        Parameters
        ----------
        operating_conditions : dict
            Scalar descriptors of the run (salt gradient, load volume, etc.).
            Keys: see residual_model.FEATURE_NAMES for required fields.
        c_inlet_fn : Callable[[float], np.ndarray]
            Inlet concentration profile. Maps time [min] -> array of
            concentrations [g/L] with shape (n_proteins + 1,).
            Index 0 = salt, indices 1..n = proteins.
        c_initial : np.ndarray
            Initial column concentrations. Shape: (n_proteins + 1,).
        t_end : float
            Simulation end time [min].
        t_eval : np.ndarray, optional
            Time points for output. Defaults to 200 evenly spaced points.
        apply_gpr : bool
            Whether to apply the GPR correction (if fitted). Set to False
            to get the raw mechanistic prediction only.
        verbose : bool
            Print progress messages.

        Returns
        -------
        HybridPrediction
            Structured result containing chromatograms and summary statistics.
        """
        if t_eval is None:
            t_eval = np.linspace(0, t_end, 200)

        # ---- Stage 1: Mechanistic simulation ----
        if verbose:
            print("[HybridModel] Stage 1: Running mechanistic TDM simulation...")

        run = ChromatographyRun(
            t_end=t_end,
            c_inlet=c_inlet_fn,
            c_initial=np.asarray(c_initial),
        )
        mech_result = self.solver.solve(run, t_eval=t_eval, verbose=verbose)

        t = mech_result["t"]
        c_outlet_mech = mech_result["c_outlet"]  # shape: (n_t, n_total)

        # Extract mechanistic summary statistics per protein component
        mech_summaries = []
        for i in range(1, self._n_proteins + 1):
            summary = extract_peak_summary(t, c_outlet_mech, component_index=i)
            mech_summaries.append(summary)

        # Compute resolution between adjacent peaks (if multi-component)
        for idx in range(len(mech_summaries) - 1):
            res = compute_resolution(mech_summaries[idx], mech_summaries[idx + 1])
            mech_summaries[idx].resolution = res

        if verbose:
            for s in mech_summaries:
                print(f"  [Mech] Component {s.component_index}: "
                      f"t_R={s.peak_retention_time_min:.2f} min, "
                      f"h={s.peak_height_gL:.4f} g/L, "
                      f"area={s.peak_area_gL_min:.4f} g/L·min")

        # ---- Stage 2: GPR Correction ----
        gpr_applied = False
        c_corrected = None
        corrected_summaries = None
        correction_uncertainty = None

        if apply_gpr and self.residual_gpr is not None:
            if verbose:
                print("[HybridModel] Stage 2: Applying GPR residual correction...")

            # Build feature vector from operating conditions + mechanistic summary
            # Use first protein component as the primary summary source
            primary_summary = mech_summaries[0].to_dict()
            x_feat = build_features_from_run(operating_conditions, primary_summary)

            # GPR prediction: mean correction and uncertainty
            delta_mu, delta_sigma = self.residual_gpr.predict_single(x_feat, return_std=True)
            correction_uncertainty = delta_sigma

            if verbose:
                output_names = self.residual_gpr._output_names
                for k, name in enumerate(output_names):
                    print(f"  [GPR] {name}: delta={delta_mu[k]:+.4f} ± {delta_sigma[k]:.4f}")

            # Apply scalar corrections to summary statistics
            corrected_summaries = self._apply_corrections_to_summaries(
                mech_summaries, delta_mu, self.residual_gpr._output_names
            )

            # Apply correction to the full chromatogram (amplitude scaling only)
            # More sophisticated time-warping correction can be added in ChromatogramCorrector
            c_corrected = self._apply_chromatogram_correction(
                t, c_outlet_mech, mech_summaries, corrected_summaries
            )

            gpr_applied = True

        return HybridPrediction(
            t=t,
            z=mech_result["z"],
            c_mechanistic=c_outlet_mech,
            c_corrected=c_corrected,
            mechanistic_summaries=mech_summaries,
            corrected_summaries=corrected_summaries,
            correction_uncertainty=correction_uncertainty,
            gpr_was_applied=gpr_applied,
        )

    # ------------------------------------------------------------------
    # Calibration data generation
    # ------------------------------------------------------------------

    def generate_residual_training_data(
        self,
        experimental_runs: list[dict],
        verbose: bool = True,
    ) -> ResidualData:
        """
        Generate GPR training data by comparing mechanistic predictions
        to experimental observations for a set of runs.

        Each entry in experimental_runs must contain:
          - 'operating_conditions' : dict  (salt gradient, load, flow, pH, ...)
          - 'c_inlet_fn'           : Callable  (inlet concentration profile)
          - 'c_initial'            : np.ndarray
          - 't_end'                : float
          - 'observed_summary'     : dict  (measured peak statistics)
              Keys: 'peak_retention_time_min', 'peak_height_gL',
                    'peak_width_min', 'peak_area_gL_min'

        Parameters
        ----------
        experimental_runs : list of dict
            Each dict describes one experimental run.

        Returns
        -------
        ResidualData
            Ready to pass to fit_residual_model().
        """
        X_list, y_list = [], []

        for i, exp in enumerate(experimental_runs):
            if verbose:
                print(f"\n[HybridModel] Processing training run {i+1}/{len(experimental_runs)}...")

            # Run mechanistic model
            result = self.predict(
                operating_conditions=exp["operating_conditions"],
                c_inlet_fn=exp["c_inlet_fn"],
                c_initial=exp["c_initial"],
                t_end=exp["t_end"],
                apply_gpr=False,  # mechanistic only
                verbose=verbose,
            )

            # Build feature vector
            mech_s = result.mechanistic_summaries[0].to_dict()
            x_feat = build_features_from_run(exp["operating_conditions"], mech_s)

            # Compute residuals: observed - mechanistic
            obs = exp["observed_summary"]
            residuals = np.array([
                obs.get("peak_retention_time_min", 0.0) - mech_s["peak_retention_time_min"],
                obs.get("peak_height_gL", 0.0)          - mech_s["peak_height_gL"],
                obs.get("peak_width_min", 0.0)           - mech_s["peak_width_min"],
                obs.get("peak_area_gL_min", 0.0)         - mech_s["peak_area_gL_min"],
            ])

            X_list.append(x_feat)
            y_list.append(residuals)

            if verbose:
                print(f"  Residuals: Δt_R={residuals[0]:+.3f} min, "
                      f"Δh={residuals[1]:+.4f} g/L, "
                      f"Δarea={residuals[3]:+.4f} g/L·min")

        output_names = [
            "peak_retention_time_min",
            "peak_height_gL",
            "peak_width_min",
            "peak_area_gL_min",
        ]

        return ResidualData(
            X=np.array(X_list),
            y_residuals=np.array(y_list),
            feature_names=FEATURE_NAMES,
            output_names=output_names,
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save the fitted GPR to a directory (TDM parameters are lightweight)."""
        import joblib
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.residual_gpr is not None:
            self.residual_gpr.save(path / "residual_gpr.joblib")
        print(f"[HybridModel] Saved to {path}")

    def load_residual_model(self, path: str | Path) -> None:
        """Load a pre-fitted GPR residual model from disk."""
        self.residual_gpr = ResidualGPR.load(path)
        print(f"[HybridModel] GPR residual model loaded from {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_corrections_to_summaries(
        self,
        mech_summaries: list,
        delta_mu: np.ndarray,
        output_names: list[str],
    ) -> list:
        """
        Add GPR scalar corrections to the mechanistic summary statistics.

        Maps GPR output names to ChromatogramSummary fields and adds delta_mu.
        """
        import copy
        corrected = copy.deepcopy(mech_summaries)

        # Map GPR output names to ChromatogramSummary attribute names
        name_map = {
            "peak_retention_time_min": "peak_retention_time_min",
            "peak_height_gL":          "peak_height_gL",
            "peak_width_min":          "peak_width_min",
            "peak_area_gL_min":        "peak_area_gL_min",
        }

        # Apply corrections to the primary component (index 0)
        for k, name in enumerate(output_names):
            attr = name_map.get(name)
            if attr is not None and len(corrected) > 0:
                old_val = getattr(corrected[0], attr)
                if not np.isnan(old_val):
                    setattr(corrected[0], attr, old_val + float(delta_mu[k]))

        return corrected

    def _apply_chromatogram_correction(
        self,
        t: np.ndarray,
        c_outlet_mech: np.ndarray,
        mech_summaries: list,
        corrected_summaries: list,
    ) -> np.ndarray:
        """
        Apply a simple amplitude-scaling correction to the mechanistic chromatogram.

        For each protein component, scales the chromatogram so that the corrected
        peak height matches the GPR-corrected value. This is a first-order
        approximation; more accurate approaches use full-profile warping.

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        c_outlet_mech : np.ndarray
            Mechanistic outlet chromatogram. Shape: (n_t, n_total).
        mech_summaries, corrected_summaries : list of ChromatogramSummary

        Returns
        -------
        np.ndarray
            Corrected chromatogram. Shape: (n_t, n_total).
        """
        import copy
        c_corr = c_outlet_mech.copy()

        for i, (ms, cs) in enumerate(zip(mech_summaries, corrected_summaries)):
            comp_idx = i + 1  # 0 = salt

            if ms.peak_height_gL < 1e-10:
                continue  # skip if no peak

            # Amplitude correction factor
            scale = cs.peak_height_gL / ms.peak_height_gL
            scale = np.clip(scale, 0.1, 10.0)  # guard against extreme corrections

            c_corr[:, comp_idx] = c_outlet_mech[:, comp_idx] * scale

        return c_corr
