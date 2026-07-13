"""
Gaussian Process Residual Corrector
====================================

Trains a Gaussian Process Regression (GPR) model to predict the pointwise
residual between mechanistic TDM predictions and experimental observations.

The ML correction layer models the structured discrepancy:
    delta(x) = y_obs(x) - y_mech(x)

where x is a feature vector summarizing the operating conditions and
chromatographic state, and delta is the residual (error to correct).

Why Gaussian Processes?
-----------------------
- Provides **uncertainty quantification**: predictions come with confidence
  intervals, which is critical for bioprocess decision-making.
- Natural interpolation between observed operating conditions.
- Works well with limited experimental data (typical in bioprocessing).
- Kernel hyperparameters have physical interpretability (length scales,
  signal variance).

Feature Engineering
-------------------
The GPR feature vector x is constructed from:
  1. Operating conditions: salt gradient slope, peak salt concentration,
     load volume, flow rate.
  2. Mechanistic model summary statistics: predicted peak retention time,
     peak width, peak height from TDM simulation.
  3. Buffer conditions: pH (if available), temperature offset.

This hybrid feature space lets the GPR learn buffer-specific and
gradient-specific biases that the mechanistic model cannot capture.

Limitations
-----------
- GPR scales as O(n³) with training data; use sparse approximations
  (e.g., SparseGPR) if n > ~1000 data points.
- The residual model is calibrated to a specific column/protein system;
  extrapolation beyond training conditions carries high uncertainty.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    ConstantKernel,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


@dataclass
class ResidualData:
    """
    Container for residual training data.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix. Shape: (n_samples, n_features).
    y_residuals : np.ndarray
        Observed residuals (y_obs - y_mech). Shape: (n_samples, n_outputs)
        or (n_samples,) for single-output.
    feature_names : list of str
        Names corresponding to columns of X (for interpretability).
    output_names : list of str
        Names of residual outputs (e.g., ["protein_1_peak_area", ...]).
    """
    X: np.ndarray
    y_residuals: np.ndarray
    feature_names: list = field(default_factory=list)
    output_names: list = field(default_factory=list)

    def __post_init__(self):
        self.X = np.atleast_2d(np.asarray(self.X, dtype=float))
        self.y_residuals = np.asarray(self.y_residuals, dtype=float)
        if self.y_residuals.ndim == 1:
            self.y_residuals = self.y_residuals[:, np.newaxis]
        if self.X.shape[0] != self.y_residuals.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} rows but y_residuals has "
                f"{self.y_residuals.shape[0]} rows."
            )


class ResidualGPR:
    """
    Multi-output Gaussian Process residual corrector.

    Trains one independent GPR per output dimension (scikit-learn's GPR is
    inherently single-output; multi-output is handled via a list of models).

    Parameters
    ----------
    kernel : str
        Kernel type: 'rbf' (isotropic RBF), 'matern' (Matérn 5/2, recommended
        for smoother but non-infinitely-differentiable functions), or 'custom'
        (pass a sklearn kernel object via kernel_obj).
    kernel_obj : optional
        Custom sklearn kernel object (used when kernel='custom').
    n_restarts_optimizer : int
        Number of random restarts for hyperparameter optimization. Higher values
        reduce risk of local optima at the cost of fit time.
    alpha : float
        Regularization (nugget) added to the diagonal of the kernel matrix.
        Acts as observation noise; helps numerical stability.
    normalize_y : bool
        Whether to subtract the mean of y during fitting. Recommended True.

    Example
    -------
    >>> gpr = ResidualGPR(kernel='matern', n_restarts_optimizer=5)
    >>> data = ResidualData(X=X_train, y_residuals=resid_train)
    >>> gpr.fit(data)
    >>> mu, sigma = gpr.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        kernel: str = "matern",
        kernel_obj=None,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-4,
        normalize_y: bool = True,
    ):
        self.kernel_type = kernel
        self.kernel_obj = kernel_obj
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.normalize_y = normalize_y

        self._models: list[GaussianProcessRegressor] = []
        self._scaler = StandardScaler()
        self._output_names: list[str] = []
        self._is_fitted = False
        self._n_outputs: int = 0

    def _build_kernel(self, n_features: int):
        """
        Construct the GPR kernel.

        The kernel is:
            C(x, x') = k_signal(x, x') + k_noise(x, x')

        where k_signal captures structured variation and k_noise absorbs
        observation noise and unmodeled fast variation.

        A ConstantKernel * base_kernel structure allows the signal amplitude
        to be learned jointly with the length scales.
        """
        if self.kernel_obj is not None:
            return self.kernel_obj

        # Per-feature length scales (ARD: Automatic Relevance Determination)
        # This allows the model to effectively ignore irrelevant features
        length_scales = np.ones(n_features)
        length_scale_bounds = (1e-3, 1e3)

        if self.kernel_type == "rbf":
            base = RBF(
                length_scale=length_scales,
                length_scale_bounds=length_scale_bounds,
            )
        elif self.kernel_type == "matern":
            base = Matern(
                length_scale=length_scales,
                length_scale_bounds=length_scale_bounds,
                nu=2.5,  # smooth but allows some non-stationarity
            )
        else:
            raise ValueError(f"Unknown kernel type '{self.kernel_type}'. Choose 'rbf' or 'matern'.")

        # Signal amplitude kernel
        amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-4, 1e4))

        # White noise kernel (observation noise)
        noise = WhiteKernel(noise_level=self.alpha, noise_level_bounds=(1e-8, 1e0))

        return amplitude * base + noise

    def fit(self, data: ResidualData) -> "ResidualGPR":
        """
        Fit independent GPR models for each output dimension.

        Parameters
        ----------
        data : ResidualData
            Training data with features X and residuals y_residuals.

        Returns
        -------
        self
        """
        n_samples, n_features = data.X.shape
        self._n_outputs = data.y_residuals.shape[1]
        self._output_names = data.output_names or [f"output_{i}" for i in range(self._n_outputs)]

        if n_samples < 5:
            warnings.warn(
                f"Only {n_samples} training samples. GPR may overfit. "
                "Consider acquiring more experimental data.",
                UserWarning,
                stacklevel=2,
            )

        # Standardize features (critical for GPR: RBF/Matérn assume unit length scales)
        X_scaled = self._scaler.fit_transform(data.X)

        self._models = []
        for k in range(self._n_outputs):
            kernel = self._build_kernel(n_features)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=1e-10,        # numerical stability nugget (noise modeled by WhiteKernel)
                normalize_y=self.normalize_y,
                random_state=42,
            )
            gpr.fit(X_scaled, data.y_residuals[:, k])
            self._models.append(gpr)
            print(
                f"[ResidualGPR] Output '{self._output_names[k]}': "
                f"log-marginal-likelihood = {gpr.log_marginal_likelihood_value_:.3f}"
            )

        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict residuals and (optionally) their posterior standard deviations.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for new operating conditions. Shape: (n_new, n_features).
        return_std : bool
            Whether to return posterior standard deviations (uncertainty).

        Returns
        -------
        mu : np.ndarray
            Predicted residuals (mean). Shape: (n_new, n_outputs).
        sigma : np.ndarray or None
            Posterior standard deviations. Shape: (n_new, n_outputs). None if
            return_std=False.
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_scaled = self._scaler.transform(X)

        mu_list, sigma_list = [], []
        for gpr in self._models:
            if return_std:
                mu_k, sigma_k = gpr.predict(X_scaled, return_std=True)
                sigma_list.append(sigma_k)
            else:
                mu_k = gpr.predict(X_scaled, return_std=False)
            mu_list.append(mu_k)

        mu = np.column_stack(mu_list)
        sigma = np.column_stack(sigma_list) if return_std else None
        return mu, sigma

    def predict_single(
        self,
        x: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict for a single feature vector (convenience wrapper).

        Parameters
        ----------
        x : np.ndarray
            Feature vector. Shape: (n_features,).

        Returns
        -------
        mu : np.ndarray, shape (n_outputs,)
        sigma : np.ndarray or None, shape (n_outputs,)
        """
        mu, sigma = self.predict(x[np.newaxis, :], return_std=return_std)
        sigma_out = sigma[0] if sigma is not None else None
        return mu[0], sigma_out

    def get_hyperparameters(self) -> list[dict]:
        """
        Extract learned kernel hyperparameters for each output.

        Returns
        -------
        list of dicts, one per output, with kernel parameter names and values.
        """
        self._check_fitted()
        result = []
        for k, gpr in enumerate(self._models):
            params = gpr.kernel_.get_params()
            result.append({"output": self._output_names[k], **params})
        return result

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk using joblib."""
        self._check_fitted()
        joblib.dump(self, path)
        print(f"[ResidualGPR] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ResidualGPR":
        """Load a persisted ResidualGPR from disk."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a ResidualGPR, got {type(model)}")
        return model

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "ResidualGPR is not fitted yet. Call .fit(data) before predicting."
            )


def build_features_from_run(
    operating_conditions: dict,
    mech_summary: dict,
) -> np.ndarray:
    """
    Construct the GPR feature vector from operating conditions and
    mechanistic model summary statistics.

    This function defines the feature engineering protocol. Both training
    and prediction must use the same feature order.

    Parameters
    ----------
    operating_conditions : dict
        Keys: 'salt_start_mM', 'salt_end_mM', 'gradient_length_CV',
              'load_volume_CV', 'flow_rate_mL_min', 'ph' (optional, default 7.0).
    mech_summary : dict
        Keys: 'peak_retention_time_min', 'peak_height_gL', 'peak_width_min',
              'peak_area_gL_min', 'resolution' (if multi-component).

    Returns
    -------
    np.ndarray
        Feature vector. Shape: (n_features,).

    Notes
    -----
    Feature names (order):
      0: salt_start_mM
      1: salt_end_mM
      2: gradient_slope_mM_per_CV  (derived)
      3: load_volume_CV
      4: flow_rate_mL_min
      5: ph
      6: mech_peak_retention_time_min
      7: mech_peak_height_gL
      8: mech_peak_width_min
      9: mech_peak_area_gL_min
    """
    oc = operating_conditions
    ms = mech_summary

    salt_start = float(oc.get("salt_start_mM", 0.0))
    salt_end = float(oc.get("salt_end_mM", 500.0))
    grad_len = float(oc.get("gradient_length_CV", 10.0))
    grad_slope = (salt_end - salt_start) / max(grad_len, 1e-6)

    features = np.array([
        salt_start,
        salt_end,
        grad_slope,
        float(oc.get("load_volume_CV", 1.0)),
        float(oc.get("flow_rate_mL_min", 5.0)),
        float(oc.get("ph", 7.0)),
        float(ms.get("peak_retention_time_min", 0.0)),
        float(ms.get("peak_height_gL", 0.0)),
        float(ms.get("peak_width_min", 0.0)),
        float(ms.get("peak_area_gL_min", 0.0)),
    ])
    return features


FEATURE_NAMES = [
    "salt_start_mM",
    "salt_end_mM",
    "gradient_slope_mM_per_CV",
    "load_volume_CV",
    "flow_rate_mL_min",
    "ph",
    "mech_peak_retention_time_min",
    "mech_peak_height_gL",
    "mech_peak_width_min",
    "mech_peak_area_gL_min",
]
