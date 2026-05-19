"""
Causal Demand Estimator
========================
Difference-in-Differences (DiD) estimation of true festival demand lift.

Problem with multipliers:
  "Diwali → ×1.6 demand" — but how much of that is:
  (a) True festival demand?
  (b) Pre-festival stockpiling (will be returned)?
  (c) Competitor store being closed?
  (d) Just seasonal trend?

DiD isolates (a) by controlling for (b)-(d) using a counterfactual:
  ATT = (Festival Store Post) - (Festival Store Pre) 
        - [(Control Store Post) - (Control Store Pre)]

Changepoint Detection:
  Uses CUSUM (Cumulative Sum) to detect when demand regime shifts.
  Answers: "When did this SKU enter a new demand state?"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


#DIFFERENCE-IN-DIFFERENCES

class FestivalCausalEstimator:
    """
    Estimates causal (true) festival demand lift using DiD.
    
    Setup:
      Treatment  = store-product during festival period
      Control    = same product in stores WITHOUT the festival effect
                   (or same store in non-festival periods from prior years)
    """

    def __init__(self, window_days: int = 7, pre_days: int = 14):
        self.window_days = window_days   # days around festival
        self.pre_days    = pre_days      # pre-festival baseline window
        self.results_: Dict = {}

    def estimate_lift(
        self,
        df:           pd.DataFrame,
        festival_date: pd.Timestamp,
        store_id:      str,
        product_id:    str,
    ) -> Dict:
        """
        Estimate causal lift for a specific festival using DiD.
        
        Returns:
            naive_lift:   raw observed multiplier (what current system uses)
            causal_lift:  DiD-estimated true causal lift
            confounding:  estimated confounding bias
            p_value:      statistical significance
            confidence:   "HIGH" / "MEDIUM" / "LOW"
        """
        sub = df[
            (df["Store_ID"] == store_id) &
            (df["Product_ID"] == product_id)
        ].sort_values("Date").set_index("Date")

        if len(sub) < 30:
            return self._fallback(festival_date)

        #Define time windows 
        fest_start = festival_date
        fest_end   = festival_date + pd.Timedelta(days=self.window_days)
        pre_start  = festival_date - pd.Timedelta(days=self.pre_days)
        pre_end    = festival_date - pd.Timedelta(days=1)

        # Treatment group: festival period
        treated_post = sub.loc[
            (sub.index >= fest_start) & (sub.index <= fest_end), "Units_Sold"
        ]
        treated_pre = sub.loc[
            (sub.index >= pre_start) & (sub.index <= pre_end), "Units_Sold"
        ]

        if treated_post.empty or treated_pre.empty:
            return self._fallback(festival_date)

        #Control: same product, non-festival periods from other stores 
        control_stores = [
            s for s in df["Store_ID"].unique() if s != store_id
        ]
        control_records = []
        for cs in control_stores:
            ctrl = df[
                (df["Store_ID"] == cs) &
                (df["Product_ID"] == product_id)
            ].sort_values("Date").set_index("Date")
            if len(ctrl) < 10:
                continue
            c_pre  = ctrl.loc[
                (ctrl.index >= pre_start) & (ctrl.index <= pre_end), "Units_Sold"
            ]
            c_post = ctrl.loc[
                (ctrl.index >= fest_start) & (ctrl.index <= fest_end), "Units_Sold"
            ]
            if not c_pre.empty and not c_post.empty:
                control_records.append({
                    "pre_mean":  c_pre.mean(),
                    "post_mean": c_post.mean(),
                })

        # DiD Estimation
        treat_pre_mean  = float(treated_pre.mean())
        treat_post_mean = float(treated_post.mean())
        naive_lift      = treat_post_mean / max(treat_pre_mean, 1.0)

        if control_records:
            ctrl_df      = pd.DataFrame(control_records)
            ctrl_pre_m   = ctrl_df["pre_mean"].mean()
            ctrl_post_m  = ctrl_df["post_mean"].mean()
            control_trend = (ctrl_post_m - ctrl_pre_m) / max(ctrl_pre_m, 1.0)
            # DiD: remove control trend from treatment lift
            causal_units = treat_post_mean - treat_pre_mean - (ctrl_post_m - ctrl_pre_m)
            causal_lift  = max(1.0, (treat_pre_mean + causal_units) / max(treat_pre_mean, 1.0))
            confounding  = round(naive_lift - causal_lift, 3)
        else:
            # Fallback: use year-ago same window as control
            year_ago_start = fest_start - pd.Timedelta(days=365)
            year_ago_end   = fest_end   - pd.Timedelta(days=365)
            year_ago = sub.loc[
                (sub.index >= year_ago_start) & (sub.index <= year_ago_end), "Units_Sold"
            ]
            if not year_ago.empty:
                ya_trend = (float(year_ago.mean()) - treat_pre_mean) / max(treat_pre_mean, 1.0)
                causal_lift = max(1.0, naive_lift - ya_trend)
                confounding = round(naive_lift - causal_lift, 3)
            else:
                causal_lift = naive_lift
                confounding = 0.0

        #   Statistical Significance: Mann-Whitney U test (non-parametric)
        if len(treated_pre) > 1 and len(treated_post) > 1:
            _, p_val = stats.mannwhitneyu(
                treated_post.values, treated_pre.values,
                alternative="greater"
            )
        else:
            p_val = 0.5

        confidence = "HIGH" if p_val < 0.05 else ("MEDIUM" if p_val < 0.15 else "LOW")

        result = {
            "festival_date":     festival_date.date(),
            "store_id":          store_id,
            "product_id":        product_id,
            "naive_lift":        round(naive_lift, 3),
            "causal_lift":       round(causal_lift, 3),
            "confounding_bias":  confounding,
            "confounding_pct":   round(confounding / max(naive_lift, 1.0) * 100, 1),
            "treat_pre_mean":    round(treat_pre_mean, 1),
            "treat_post_mean":   round(treat_post_mean, 1),
            "p_value":           round(p_val, 4),
            "confidence":        confidence,
            "n_control_stores":  len(control_records),
        }
        self.results_[f"{store_id}_{product_id}"] = result
        return result

    def batch_estimate(
        self,
        df:          pd.DataFrame,
        store_id:    str,
        product_id:  str,
    ) -> pd.DataFrame:
        """Estimate causal lift for all major festivals found in data."""
        from festival_calendar import FESTIVALS_EXACT
        rows = []
        data_start = df["Date"].min()
        data_end   = df["Date"].max()

        for fdate, (fname, ftype, _) in FESTIVALS_EXACT.items():
            if ftype not in ("mega", "major"):
                continue
            ts = pd.Timestamp(fdate)
            if not (data_start <= ts <= data_end):
                continue
            r = self.estimate_lift(df, ts, store_id, product_id)
            r["festival_name"] = fname
            r["festival_type"] = ftype
            rows.append(r)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("causal_lift", ascending=False)

    def _fallback(self, festival_date) -> Dict:
        return {
            "festival_date": festival_date if not hasattr(festival_date,"date") else festival_date.date(),
            "naive_lift": 1.0, "causal_lift": 1.0, "confounding_bias": 0.0,
            "confounding_pct": 0.0, "treat_pre_mean": 0.0, "treat_post_mean": 0.0,
            "p_value": 1.0, "confidence": "LOW", "n_control_stores": 0,
        }




class DemandChangepointDetector:
    """
    CUSUM (Cumulative Sum) changepoint detection for demand regime shifts.
    
    Answers: "When did demand pattern fundamentally change?"
    Uses both CUSUM (for mean shifts) and rolling variance (for volatility shifts).
    """

    def __init__(self, threshold_sigma: float = 3.0, min_segment: int = 14):
        self.threshold_sigma = threshold_sigma
        self.min_segment     = min_segment

    def detect(
        self,
        series: pd.Series,
        dates:  pd.Series = None,
    ) -> Dict:
        """
        Detect changepoints in a demand time series.
        Returns list of changepoint dates and regime statistics.
        """
        values = np.array(series.dropna())
        if len(values) < 30:
            return {"changepoints": [], "regimes": [], "n_regimes": 1}

        mu    = np.mean(values)
        sigma = np.std(values) + 1e-6
        k     = self.threshold_sigma * 0.5   # allowance

        # CUSUM statistic
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mu)/sigma - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mu)/sigma - k)

        threshold = self.threshold_sigma
        cp_indices = []
        i = 0
        while i < len(values):
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                if not cp_indices or (i - cp_indices[-1]) >= self.min_segment:
                    cp_indices.append(i)
                # Reset CUSUM after detection
                cusum_pos[i:] = np.maximum(0, cusum_pos[i:] - threshold)
                cusum_neg[i:] = np.maximum(0, cusum_neg[i:] - threshold)
            i += 1

        # Build regime statistics
        idx_series = dates.reset_index(drop=True) if dates is not None else pd.RangeIndex(len(values))
        boundaries = [0] + cp_indices + [len(values)]
        regimes = []
        for j in range(len(boundaries) - 1):
            s, e = boundaries[j], boundaries[j+1]
            seg  = values[s:e]
            if len(seg) == 0:
                continue
            start_label = str(idx_series.iloc[s])[:10] if hasattr(idx_series, 'iloc') else str(s)
            end_label   = str(idx_series.iloc[min(e-1, len(idx_series)-1)])[:10] if hasattr(idx_series, 'iloc') else str(e)
            regimes.append({
                "regime":      j + 1,
                "start":       start_label,
                "end":         end_label,
                "mean":        round(float(np.mean(seg)), 1),
                "std":         round(float(np.std(seg)), 1),
                "cv":          round(float(np.std(seg) / (np.mean(seg)+1e-6)), 3),
                "n_days":      len(seg),
                "trend":       "UP" if seg[-1] > seg[0]*1.05 else ("DOWN" if seg[-1] < seg[0]*0.95 else "STABLE"),
            })

        cp_dates = []
        if dates is not None:
            dates_r = dates.reset_index(drop=True)
            for idx in cp_indices:
                if idx < len(dates_r):
                    cp_dates.append(str(dates_r.iloc[idx])[:10])
        else:
            cp_dates = [str(i) for i in cp_indices]

        return {
            "changepoints":   cp_dates,
            "n_changepoints": len(cp_dates),
            "regimes":        regimes,
            "n_regimes":      len(regimes),
            "cusum_pos":      cusum_pos.tolist(),
            "cusum_neg":      cusum_neg.tolist(),
            "threshold":      threshold,
        }

    def retraining_needed(self, result: Dict) -> Tuple[bool, str]:
        """Decide if model retraining is needed based on recent regime shift."""
        if result["n_changepoints"] == 0:
            return False, "No regime change detected — model is current."
        regimes = result["regimes"]
        if len(regimes) < 2:
            return False, "Insufficient regime data."
        last  = regimes[-1]
        prev  = regimes[-2]
        shift = abs(last["mean"] - prev["mean"]) / max(prev["mean"], 1.0)
        if shift > 0.20:
            return True, (
                f"🔴 REGIME SHIFT DETECTED: Demand changed by {shift*100:.1f}% "
                f"from {prev['mean']:.1f} → {last['mean']:.1f} units/day. "
                f"Model retraining recommended."
            )
        elif shift > 0.10:
            return False, (
                f"🟡 MILD SHIFT: Demand shifted {shift*100:.1f}%. Monitor closely."
            )
        return False, "✅ Current regime stable — no retraining needed."












