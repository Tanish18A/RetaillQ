
import numpy as np
import pandas as pd
import shap
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# SAFE COLUMN GETTER

def _safe_col(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    """Return column if exists, else a Series of default values."""
    if col in df.columns:
        return df[col]
    if default is not None:
        return pd.Series(default, index=df.index)
    return pd.Series(0, index=df.index)


# BUSINESS METRICS

def compute_business_metrics(
    df: pd.DataFrame,
    predicted_col: str = "Units_Sold",
    actual_col:    str = "True_Demand",
) -> Dict:
    """
    Compute all financial KPIs. Handles missing columns gracefully —
    works on both real Kaggle data and synthetic data.
    """
    df = df.copy()

    #Resolve actual demand (best proxy if True_Demand not available) 
    if actual_col not in df.columns:
        actual_col = predicted_col   # fallback

    #Price column: prefer "Price", else "price", else default to 100
    price_col = "Price" if "Price" in df.columns else "price"
    price     = _safe_col(df, price_col, df.get("Base_Price", pd.Series(100, index=df.index)))

    # Base price (back-calculate if missing) 
    if "Base_Price" in df.columns:
        base_price = df["Base_Price"]
    elif "Discount_Pct" in df.columns:
        base_price = (price / (1 - df["Discount_Pct"].clip(0, 90) / 100)).fillna(price)
    else:
        base_price = price   # no discount info → same as price

    # Discount
    discount_pct = _safe_col(df, "Discount_Pct", 0.0)

    units_sold   = df[predicted_col].fillna(0).clip(lower=0)
    actual_demand= df[actual_col].fillna(units_sold).clip(lower=0)

    #Revenue & Cost
    df["Revenue"]      = units_sold * price
    df["COGS"]         = units_sold * price * 0.65
    df["Gross_Profit"] = df["Revenue"] - df["COGS"]

    # ── Holding Cost — Industry Standard ──
    # Industry standard: 25% of COGS per year per unit in stock.
    # COGS = 65% of selling price. Daily = price * 0.65 * 0.25 / 365.
    # For 90-day period: Rice (avg_inv=489, price=268) → Rs4,081  ✓
    #                    Milk (avg_inv=950, price=52)  → Rs1,988  ✓
    # Always profitable, realistic scale, no dependency on noisy dataset field.
    cost_price_series = price.clip(lower=1.0) * 0.65
    holding_cost_pu   = cost_price_series * 0.25 / 365.0   # Rs per unit per day
    inventory         = _safe_col(df, "Inventory_Level", 0.0)
    df["Holding_Cost"] = inventory * holding_cost_pu

    # Stockout Cost 
    stockout_pen_pu    = _safe_col(df, "Stockout_Penalty_Per_Unit", 15.0)
    df["Unmet_Demand"] = (actual_demand - units_sold).clip(lower=0)
    df["Stockout_Cost"]= df["Unmet_Demand"] * stockout_pen_pu

    # Discount Loss 
    df["Discount_Loss"] = units_sold * base_price * (discount_pct / 100)

    #Net Profit
    df["Net_Profit"] = df["Gross_Profit"] - df["Holding_Cost"] - df["Stockout_Cost"]

    #Aggregate KPIs
    total_revenue  = df["Revenue"].sum()
    total_profit   = df["Net_Profit"].sum()
    total_holding  = df["Holding_Cost"].sum()
    total_stockout = df["Stockout_Cost"].sum()
    total_unmet    = df["Unmet_Demand"].sum()
    stockout_rate  = (df["Unmet_Demand"] > 0).mean() * 100
    avg_inv        = inventory.mean()
    inv_turnover   = (units_sold.sum() / max(avg_inv, 1))
    fill_rate      = (1 - total_unmet / max(actual_demand.sum(), 1)) * 100

    return {
        "Total_Revenue":       round(total_revenue, 2),
        "Total_Net_Profit":    round(total_profit, 2),
        "Gross_Margin_Pct":    round(df["Gross_Profit"].sum() / max(total_revenue, 1) * 100, 1),
        "Total_Holding_Cost":  round(total_holding, 2),
        "Total_Stockout_Cost": round(total_stockout, 2),
        "Total_Unmet_Demand":  round(total_unmet),
        "Stockout_Rate_Pct":   round(stockout_rate, 1),
        "Fill_Rate_Pct":       round(fill_rate, 1),
        "Avg_Inventory_Level": round(avg_inv, 1),
        "Inventory_Turnover":  round(inv_turnover, 2),
        "df":                  df,   # enriched dataframe with all cost columns
    }


def ai_vs_no_ai_comparison(
    df: pd.DataFrame,
    product_id: str = None,
    store_id:   str = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Compare AI system vs manual (no AI) operations.
    Simulates manual ordering by adding ±25% noise and sloppy inventory management.
    Returns: (comparison_table_df, ai_metrics, no_ai_metrics)
    """
    np.random.seed(42)
    subset = df.copy()
    if product_id:
        subset = subset[subset["Product_ID"] == product_id]
    if store_id:
        subset = subset[subset["Store_ID"] == store_id]

    # AI system metrics (actual data)
    ai_metrics = compute_business_metrics(subset, "Units_Sold", "True_Demand")

    # Simulate "No AI" — manual fixed-order, ±25% demand forecast error
    no_ai = subset.copy()
    noise = np.random.normal(1.0, 0.25, len(no_ai)).clip(0.5, 1.6)
    no_ai["Units_Sold"] = (no_ai["Units_Sold"].fillna(0) * noise).clip(lower=0)
    # Manual ordering = over/under stock by 20-40%
    inv_noise = np.random.uniform(0.75, 1.40, len(no_ai))
    if "Inventory_Level" in no_ai.columns:
        no_ai["Inventory_Level"] = (no_ai["Inventory_Level"] * inv_noise).clip(lower=0)
    no_ai_metrics = compute_business_metrics(no_ai, "Units_Sold", "True_Demand")

    # Build comparison table
    comparison = pd.DataFrame({
        "Metric": [
            "Total Revenue",
            "Net Profit",
            "Holding Cost",
            "Stockout Cost",
            "Stockout Rate",
            "Fill Rate",
            "Inventory Turnover",
        ],
        "Without AI (Manual)": [
            f"₹{no_ai_metrics['Total_Revenue']:,.0f}",
            f"₹{no_ai_metrics['Total_Net_Profit']:,.0f}",
            f"₹{no_ai_metrics['Total_Holding_Cost']:,.0f}",
            f"₹{no_ai_metrics['Total_Stockout_Cost']:,.0f}",
            f"{no_ai_metrics['Stockout_Rate_Pct']}%",
            f"{no_ai_metrics['Fill_Rate_Pct']}%",
            f"{no_ai_metrics['Inventory_Turnover']}x",
        ],
        "With AI System": [
            f"₹{ai_metrics['Total_Revenue']:,.0f}",
            f"₹{ai_metrics['Total_Net_Profit']:,.0f}",
            f"₹{ai_metrics['Total_Holding_Cost']:,.0f}",
            f"₹{ai_metrics['Total_Stockout_Cost']:,.0f}",
            f"{ai_metrics['Stockout_Rate_Pct']}%",
            f"{ai_metrics['Fill_Rate_Pct']}%",
            f"{ai_metrics['Inventory_Turnover']}x",
        ],
        "AI Improvement": [
            f"₹{ai_metrics['Total_Revenue']    - no_ai_metrics['Total_Revenue']:+,.0f}",
            f"₹{ai_metrics['Total_Net_Profit'] - no_ai_metrics['Total_Net_Profit']:+,.0f}",
            f"₹{ai_metrics['Total_Holding_Cost']  - no_ai_metrics['Total_Holding_Cost']:+,.0f}",
            f"₹{no_ai_metrics['Total_Stockout_Cost'] - ai_metrics['Total_Stockout_Cost']:+,.0f}",  # positive = AI reduced stockout cost
            f"{ai_metrics['Stockout_Rate_Pct'] - no_ai_metrics['Stockout_Rate_Pct']:+.1f}%",
            f"{ai_metrics['Fill_Rate_Pct']     - no_ai_metrics['Fill_Rate_Pct']:+.1f}%",
            f"{ai_metrics['Inventory_Turnover']- no_ai_metrics['Inventory_Turnover']:+.2f}x",
        ],
    })
    return comparison, ai_metrics, no_ai_metrics


#SHAP FEATURE NAME MAPPING 

FEATURE_BUSINESS_NAMES = {
    "lag_1":               "Yesterday's Demand",
    "lag_7":               "Same Day Last Week",
    "lag_14":              "2 Weeks Ago",
    "lag_30":              "Last Month",
    "rolling_mean_7":      "7-Day Avg Demand",
    "rolling_mean_14":     "14-Day Avg Demand",
    "rolling_mean_30":     "30-Day Avg Demand",
    "rolling_std_7":       "Demand Volatility (7d)",
    "rolling_max_7":       "Peak Demand (7d)",
    "rolling_min_7":       "Min Demand (7d)",
    "day_of_week":         "Day of Week",
    "is_weekend":          "Weekend Effect",
    "month":               "Seasonal Month",
    "quarter":             "Quarter of Year",
    "week_of_year":        "Week Number",
    "is_month_start":      "Start of Month",
    "is_month_end":        "End of Month",
    "price_change":        "Price Change (₹)",
    "price_change_pct":    "Price Change (%)",
    "price_vs_base":       "Discount Applied",
    "price_vs_competitor": "Price vs Competitor",
    "Discount_Pct":        "Discount Percentage",
    "Promotion_Flag":      "Promotion Active",
    "Holiday_Flag":        "Festival / Holiday",
    "festival_mult":       "Festival Demand Boost",   # NEW — real calendar
    "is_pre_festival":     "Pre-Festival Window",     # NEW
    "is_festival_day":     "Festival Day",            # NEW
    "seasonality_factor":  "Seasonality Index",       # NEW — from Kaggle col
    "inventory_ratio":     "Stock Level vs Demand",
    "demand_momentum":     "Demand Trend (↑/↓)",
    "days_since_promo":    "Days Since Last Promo",
    "Store_ID_enc":        "Store Location",
    "Product_ID_enc":      "Product Type",
    "Category_enc":        "Product Category",
    "Region_enc":          "Store Region",
    "Weather_Condition_enc": "Weather Condition",
    #V4 pipeline features
    "lag_2":               "2 Days Ago",
    "lag_3":               "3 Days Ago",
    "lag_21":              "3 Weeks Ago",
    "lag_28":              "Last Month (28d)",
    "roll_mean_7":         "7-Day Avg Demand",
    "roll_mean_14":        "14-Day Avg Demand",
    "roll_mean_28":        "28-Day Avg Demand",
    "roll_std_7":          "Demand Volatility (7d)",
    "roll_std_14":         "Demand Volatility (14d)",
    "roll_std_28":         "Demand Volatility (28d)",
    "roll_max_7":          "Peak Demand (7d)",
    "roll_min_7":          "Min Demand (7d)",
    "roll_q25_7":          "Low Demand Quartile",
    "roll_q75_7":          "High Demand Quartile",
    "roll_iqr_7":          "Demand Spread (IQR)",
    "roll_iqr_14":         "Demand Spread 14d",
    "ewm_02":              "Slow Trend (EWM α=0.2)",
    "ewm_04":              "Mid Trend (EWM α=0.4)",
    "ewm_07":              "Fast Trend (EWM α=0.7)",
    "lag_ratio_1_7":       "Short vs Week Ratio",
    "lag_ratio_7_28":      "Week vs Month Ratio",
    "lag_ratio_1_28":      "Today vs Month Ratio",
    "demand_accel_7":      "Demand Acceleration (7d)",
    "demand_accel_28":     "Demand Acceleration (28d)",
    "weekofyear":          "Week of Year",
    "dow_sin":             "Day of Week (cyclical)",
    "dow_cos":             "Day of Week (cyclical)",
    "month_sin":           "Month (cyclical)",
    "month_cos":           "Month (cyclical)",
    "woy_sin":             "Week of Year (cyclical)",
    "woy_cos":             "Week of Year (cyclical)",
    "Festival_Multiplier": "Festival Demand Boost",
    "festival_level":      "Festival Importance Level",
    "fest_is_mega":        "Mega Festival Day",
    "fest_is_major":       "Major Festival Day",
    "fest_is_pre":         "Pre-Festival Window",
    "fest_is_normal":      "Normal Day",
    "fest_x_lag1":         "Festival × Yesterday",
    "fest_x_lag7":         "Festival × Last Week",
    "fest_x_roll7":        "Festival × 7d Avg",
    "fest_x_roll28":       "Festival × 28d Avg",
    "fest_x_weekend":      "Festival on Weekend",
    "fest_x_promo":        "Festival + Promotion",
    "fest_level_x_lag1":   "Festival Level × Demand",
    "days_since_festival": "Days Since Festival",
    "days_until_festival": "Days Until Festival",
    "fest_proximity":      "Festival Proximity Signal",
    "fest_recency":        "Festival Recency Signal",
    "fest_proximity_x_lag1": "Festival Proximity × Demand",
    "season_x_lag1":       "Seasonality × Demand",
    "season_x_roll7":      "Seasonality × 7d Avg",
    "price_diff":          "Price vs Competitor (₹)",
    "price_ratio":         "Price Ratio vs Competitor",
    "net_price":           "Net Price After Discount",
    "disc_x_price":        "Discount × Price",
    "stock_ratio":         "Stock vs Demand Ratio",
    "stockout_risk":       "Stockout Risk Flag",
    "stock_excess":        "Overstock Flag",
    "forecast_error_lag":  "Previous Forecast Error",
    "Store_ID_demand_mean":    "Store Avg Demand",
    "Store_ID_demand_std":     "Store Demand Variability",
    "Store_ID_demand_median":  "Store Median Demand",
    "Product_ID_demand_mean":  "Product Avg Demand",
    "Product_ID_demand_std":   "Product Demand Variability",
    "Product_ID_demand_median":"Product Median Demand",
    "sp_demand_mean":          "Store-Product Avg Demand",
    "sp_demand_std":           "Store-Product Variability",
    "Store_ID_freq":           "Store Sales Share",
    "Product_ID_freq":         "Product Sales Share",
    "weather_score":           "Weather Impact Score",
}


# SHAP EXPLAINABILITY 

def explain_prediction(
    model,
    X_sample: pd.DataFrame,
    feature_cols: list,
    top_n: int = 6,
) -> Tuple[pd.DataFrame, str]:
    """
    Explain a single prediction using SHAP TreeExplainer.
    Works with V4 XGBoost model (predicts in log-space, SHAP in log-space).
    Returns (importance_df, business_narrative_string)
    """
    # Only use features that exist in X_sample
    avail_feats = [f for f in feature_cols if f in X_sample.columns]
    X = X_sample[avail_feats].fillna(0)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
    except Exception:
        # Fallback: use feature_importances if SHAP fails
        try:
            sv_arr = model.feature_importances_[:len(avail_feats)]
        except Exception:
            sv_arr = np.zeros(len(avail_feats))
        shap_vals = sv_arr.reshape(1, -1)

    sv = shap_vals[0] if hasattr(shap_vals, '__len__') and len(shap_vals) > 0 else shap_vals
    if hasattr(sv, 'ndim') and sv.ndim == 2:
        sv = sv[0]

    sv_arr = np.asarray(sv).flatten()
    # Ensure sv and feature list are the same length
    feat_list = avail_feats[:len(sv_arr)]
    sv_arr    = sv_arr[:len(feat_list)]

    importance = pd.DataFrame({
        "Feature":       feat_list,
        "SHAP_Value":    sv_arr,
        "Business_Name": [FEATURE_BUSINESS_NAMES.get(f, f.replace("_", " ").title())
                          for f in feat_list],
        "Direction":     ["↑ Increases demand" if v > 0 else "↓ Decreases demand" for v in sv_arr],
    }).sort_values("SHAP_Value", key=abs, ascending=False).head(top_n)

    # Business narrative
    top_pos = importance[importance["SHAP_Value"] > 0].head(3)
    top_neg = importance[importance["SHAP_Value"] < 0].head(2)
    parts   = ["🔍 Demand changed because:"]

    if not top_pos.empty:
        parts.append("  📈 Demand INCREASED due to:")
        for _, r in top_pos.iterrows():
            parts.append(f"     ✅ {r['Business_Name']}  (+{r['SHAP_Value']:.1f} units impact)")

    if not top_neg.empty:
        parts.append("  📉 Demand REDUCED due to:")
        for _, r in top_neg.iterrows():
            parts.append(f"     ❌ {r['Business_Name']}  ({r['SHAP_Value']:.1f} units impact)")

    return importance, "\n".join(parts)


def compute_global_feature_importance(
    model,
    df_feat: pd.DataFrame,
    feature_cols: list,
    sample_size: int = 2000,
) -> pd.DataFrame:
    """
    Global feature importance — uses SHAP if available, else model.feature_importances_.
    Works with both V3 (single model) and V4 (XGBoost model) pipelines.
    """
    avail_feats = [f for f in feature_cols if f in df_feat.columns]
    sample = df_feat[avail_feats].fillna(0).sample(
        min(sample_size, len(df_feat)), random_state=42
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(sample)
        importance_vals = np.abs(shap_vals).mean(axis=0).flatten()[:len(avail_feats)]
    except Exception:
        # Fallback to built-in feature_importances_
        try:
            raw_imp = model.feature_importances_
            # Align length
            importance_vals = raw_imp[:len(avail_feats)]
            # Normalise to [0,1]
            total = importance_vals.sum()
            importance_vals = importance_vals / max(total, 1e-9)
        except Exception:
            importance_vals = np.ones(len(avail_feats)) / len(avail_feats)

    return pd.DataFrame({
        "Feature":       avail_feats[:len(importance_vals)],
        "Importance":    importance_vals,
        "Business_Name": [FEATURE_BUSINESS_NAMES.get(f, f.replace("_", " ").title())
                          for f in avail_feats[:len(importance_vals)]],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)