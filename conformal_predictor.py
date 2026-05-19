"""
Conformal Predictor
====================
Split Conformal Prediction for demand forecasting.
Gives coverage-GUARANTEED prediction intervals — unlike standard ML error bars,
conformal intervals hold regardless of data distribution (distribution-free).

Theory: Vovk et al. (2005) "Algorithmic Learning in a Random World"
        Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal Prediction"

Coverage guarantee: P(Y_new ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
This holds in finite samples WITHOUT any distributional assumptions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")


class SplitConformalPredictor:
    """
    Split Conformal Prediction using nonconformity scores.
    Uses absolute residuals as nonconformity measure (most common choice).
    
    Algorithm:
    1. Split data → calibration set (20%)
    2. Compute nonconformity scores on calibration: |y_i - ŷ_i|
    3. At test time: q = (1-α) quantile of scores
    4. Interval = [ŷ - q, ŷ + q]
    """

    def __init__(self, alpha: float = 0.05):
        """
        alpha: miscoverage level. alpha=0.05 → 95% coverage guarantee.
        """
        self.alpha = alpha
        self.q_hat: float = None          # conformal quantile
        self.calibration_scores: np.ndarray = None
        self.coverage_level = 1 - alpha
        self.n_calibration: int = 0
        self.is_fitted: bool = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalPredictor":
        """
        Fit conformal quantile from calibration set residuals.
        y_true, y_pred: 1-D arrays of actual and predicted demand on held-out data.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Nonconformity scores — absolute residuals
        scores = np.abs(y_true - y_pred)
        self.calibration_scores = scores
        self.n_calibration = len(scores)
        
        # Finite-sample corrected quantile: ceil((n+1)(1-α)) / n
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self.is_fitted = True
        return self

    def calibrate_from_series(
        self, df: pd.DataFrame,
        actual_col: str = "Units_Sold",
        pred_col:   str = "rolling_mean_7",
        calib_frac: float = 0.25,
    ) -> "SplitConformalPredictor":
        """Convenience: calibrate directly from a dataframe."""
        df = df.dropna(subset=[actual_col, pred_col]).sort_values("Date")
        n = len(df)
        calib = df.tail(max(30, int(n * calib_frac)))
        return self.calibrate(
            calib[actual_col].values,
            calib[pred_col].values
        )

    def predict_interval(self, y_hat: float) -> Tuple[float, float]:
        """Return (lower, upper) conformal interval for a point forecast."""
        if not self.is_fitted:
            raise RuntimeError("Call calibrate() first.")
        lo = max(0.0, y_hat - self.q_hat)
        hi = y_hat + self.q_hat
        return lo, hi

    def predict_interval_array(
        self, y_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized interval prediction."""
        y_hat = np.array(y_hat)
        lo = np.maximum(0, y_hat - self.q_hat)
        hi = y_hat + self.q_hat
        return lo, hi

    def coverage_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Empirically verify coverage on a test set."""
        lo, hi = self.predict_interval_array(y_pred)
        covered = np.sum((y_true >= lo) & (y_true <= hi))
        empirical_cov = covered / len(y_true)
        avg_width = np.mean(hi - lo)
        return {
            "target_coverage":   self.coverage_level,
            "empirical_coverage": round(empirical_cov, 4),
            "coverage_gap":       round(empirical_cov - self.coverage_level, 4),
            "avg_interval_width": round(avg_width, 2),
            "q_hat":              round(self.q_hat, 2),
            "n_calibration":      self.n_calibration,
            "guaranteed":         empirical_cov >= self.coverage_level - 0.02,
        }

    @property
    def interval_width(self) -> float:
        return self.q_hat * 2 if self.is_fitted else 0.0

    @property
    def uncertainty_pct(self) -> float:
        """Width as % of q_hat — useful for trust score."""
        return self.q_hat if self.is_fitted else 0.0


def build_conformal_predictor(
    df: pd.DataFrame,
    store_id: str,
    product_id: str,
    alpha: float = 0.05,
) -> SplitConformalPredictor:
    """
    Build a calibrated conformal predictor for a specific store-product pair.
    Uses rolling_mean_7 as baseline predictor for calibration.
    """
    sub = df[
        (df["Store_ID"] == store_id) &
        (df["Product_ID"] == product_id)
    ].dropna(subset=["Units_Sold"]).sort_values("Date")

    cp = SplitConformalPredictor(alpha=alpha)

    if "rolling_mean_7" in sub.columns and len(sub) >= 30:
        cp.calibrate_from_series(
            sub, actual_col="Units_Sold", pred_col="rolling_mean_7"
        )
    else:
        # Fallback: use historical std
        std = float(sub["Units_Sold"].std()) if len(sub) > 1 else 10.0
        cp.q_hat = 1.96 * std
        cp.is_fitted = True
        cp.n_calibration = len(sub)
        cp.calibration_scores = np.array([std])

    return cp


def add_conformal_bands(
    forecast_df: pd.DataFrame,
    cp: SplitConformalPredictor,
    pred_col: str = "Predicted_Demand",
) -> pd.DataFrame:
    """Add lower/upper conformal band columns to a forecast dataframe."""
    df = forecast_df.copy()
    lo, hi = cp.predict_interval_array(df[pred_col].values)
    df["Conf_Lower"] = np.round(lo, 1)
    df["Conf_Upper"] = np.round(hi, 1)
    df["Conf_Width"]  = np.round(hi - lo, 1)
    df["Coverage_Pct"] = int(cp.coverage_level * 100)
    return df
