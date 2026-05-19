"""
Decision Engine
================
Translates ML demand forecasts → actionable inventory orders.

Updated (v2):
  - Safety Stock uses last-30-day σ_d (more stable than forecast variance)
  - Formula: Safety Stock = Z × σ_d × √(Lead_Time)
  - ROP = (avg_demand × L) + safety_stock
  - demand_std_30d parameter: pass from app for historical σ
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

SERVICE_LEVEL_Z = {90: 1.28, 95: 1.645, 97: 1.88, 99: 2.326, 99.9: 3.09}


@dataclass
class InventoryConfig:
    """All parameters for one store-product pair."""
    product_id:                str
    product_name:              str
    store_id:                  str
    current_inventory:         float
    lead_time_days:            int
    supplier_limit:            int
    min_order_qty:             int
    reorder_point:             float
    service_level_pct:         float = 95.0
    holding_cost_per_unit:     float = 2.0
    stockout_penalty_per_unit: float = 15.0
    unit_price:                float = 100.0
    cost_price:                float = 70.0

    @property
    def z_score(self) -> float:
        sl   = min(self.service_level_pct, 99.9)
        keys = sorted(SERVICE_LEVEL_Z.keys())
        for k in keys:
            if sl <= k:
                return SERVICE_LEVEL_Z[k]
        return SERVICE_LEVEL_Z[99.9]


@dataclass
class OrderDecision:
    product_id:           str
    product_name:         str
    store_id:             str
    date:                 str
    predicted_demand_7d:  float
    avg_daily_demand:     float
    demand_std:           float
    safety_stock:         float
    reorder_point:        float
    recommended_order:    float
    constrained_order:    float
    current_inventory:    float
    projected_inventory:  float
    stockout_risk:        str
    overstock_risk:       str
    stockout_risk_pct:    float
    overstock_risk_pct:   float
    days_of_stock:        float
    alert_message:        str
    festival_warning:     str = ""
    std_source:           str = "forecast"   # "historical_30d" or "forecast"

    def to_dict(self) -> dict:
        return self.__dict__


# CORE FORMULAS

def calculate_safety_stock(demand_std: float, lead_time: int, z: float) -> float:
    """Safety Stock = Z × σ_d × √(L)"""
    return z * demand_std * np.sqrt(max(lead_time, 1))


def calculate_order_quantity(
    predicted_7d:      float,
    safety_stock:      float,
    current_inventory: float,
    reorder_point:     float,
) -> float:
    """
    Order = Predicted_Demand + Safety_Stock − Current_Inventory
    Only triggered when inventory ≤ reorder_point.
    """
    if current_inventory > reorder_point:
        return 0.0
    return max(0.0, predicted_7d + safety_stock - current_inventory)


def apply_constraints(raw_order: float, supplier_limit: int, min_order_qty: int) -> float:
    """Enforce real-world supplier constraints."""
    if raw_order <= 0:
        return 0.0
    constrained = min(raw_order, supplier_limit)
    if 0 < constrained < min_order_qty:
        constrained = min_order_qty
    return round(constrained)


def assess_risk(
    current_inventory: float,
    avg_daily_demand:  float,
    safety_stock:      float,
    projected_inv:     float,
    lead_time:         int,
) -> tuple:
    """Returns (stockout_label, overstock_label, stockout_pct, overstock_pct)"""
    if avg_daily_demand <= 0:
        return "LOW", "LOW", 0.0, 0.0

    days_stock  = current_inventory / avg_daily_demand
    excess_days = projected_inv     / avg_daily_demand

    if current_inventory <= 0:
        so_risk, so_pct = "CRITICAL", 95.0
    elif current_inventory < safety_stock:
        so_risk, so_pct = "HIGH",     75.0
    elif days_stock < lead_time * 1.5:
        so_risk, so_pct = "MEDIUM",   45.0
    else:
        so_risk, so_pct = "LOW",      10.0

    if excess_days > 21:
        ov_risk, ov_pct = "HIGH",   70.0
    elif excess_days > 14:
        ov_risk, ov_pct = "MEDIUM", 40.0
    else:
        ov_risk, ov_pct = "LOW",    10.0

    return so_risk, ov_risk, so_pct, ov_pct


def check_festival_window(forecast_df: pd.DataFrame) -> str:
    if "Festival_Mult" not in forecast_df.columns:
        return ""
    fest_days = forecast_df[forecast_df["Festival_Mult"] > 1.0]
    if fest_days.empty:
        return ""
    max_mult = fest_days["Festival_Mult"].max()
    n_days   = len(fest_days)
    if max_mult >= 1.5:
        return (f"🎊 FESTIVAL in forecast window ({n_days} days, peak ×{max_mult:.2f}) — "
                f"System has increased order quantity automatically.")
    elif max_mult >= 1.1:
        return (f"📅 Pre-festival window detected ({n_days} days) — "
                f"Mild demand increase expected.")
    return ""


# MAIN DECISION FUNCTION

def make_order_decision(
    forecast_df:     pd.DataFrame,
    config:          InventoryConfig,
    decision_date:   str   = None,
    demand_std_30d:  float = None,   # NEW: σ from last 30 days of actual sales
) -> OrderDecision:
    """
    Core decision: 7-day forecast + config → final order recommendation.

    Safety Stock formula:
        σ_d  = std of last 30 days actual sales (if provided) else forecast std
        SS   = Z × σ_d × √(L)
        ROP  = (avg_daily_demand × L) + SS
    """
    if decision_date is None:
        decision_date = str(forecast_df["Date"].max().date())

    predicted_7d = float(forecast_df["Predicted_Demand"].sum())
    avg_daily    = predicted_7d / 7

    # ── Use 30-day historical σ if provided (more robust) 
    std_source = "forecast"
    if demand_std_30d is not None and demand_std_30d > 0:
        demand_std = demand_std_30d
        std_source = "historical_30d"
    else:
        demand_std = float(np.std(forecast_df["Predicted_Demand"].values)) if len(forecast_df) > 1 \
                     else avg_daily * 0.15

    safety_stock  = calculate_safety_stock(demand_std, config.lead_time_days, config.z_score)
    reorder_point = avg_daily * config.lead_time_days + safety_stock

    fest_warning = check_festival_window(forecast_df)
    fest_boost   = 1.0
    if "Festival_Mult" in forecast_df.columns:
        max_mult  = forecast_df["Festival_Mult"].max()
        if max_mult >= 1.5:
            fest_boost = 1.30
        elif max_mult >= 1.2:
            fest_boost = 1.15

    raw_order   = calculate_order_quantity(
        predicted_7d * fest_boost, safety_stock,
        config.current_inventory, reorder_point
    )
    constrained = apply_constraints(raw_order, config.supplier_limit, config.min_order_qty)
    projected   = max(0, config.current_inventory + constrained - predicted_7d)
    days_stock  = (config.current_inventory / avg_daily) if avg_daily > 0 else 999

    so_r, ov_r, so_p, ov_p = assess_risk(
        config.current_inventory, avg_daily, safety_stock, projected, config.lead_time_days
    )

    alerts = []
    if fest_warning:
        alerts.append(fest_warning)
    if so_r in ("HIGH", "CRITICAL"):
        alerts.append(f"⚠️ STOCKOUT RISK {so_r}: {days_stock:.1f} days of stock left!")
    if ov_r == "HIGH":
        alerts.append("📦 OVERSTOCK: Consider pausing future orders.")
    if constrained != round(raw_order) and constrained > 0:
        alerts.append(f"🚚 SUPPLIER CAP: Order capped at {constrained} (requested {round(raw_order)}).")
    if constrained == 0 and so_r == "LOW":
        alerts.append("✅ Stock sufficient. No order needed this cycle.")
    alert_msg = " | ".join(alerts) if alerts else "✅ Normal operations."

    return OrderDecision(
        product_id=config.product_id,
        product_name=config.product_name,
        store_id=config.store_id,
        date=decision_date,
        predicted_demand_7d=round(predicted_7d, 1),
        avg_daily_demand=round(avg_daily, 1),
        demand_std=round(demand_std, 2),
        safety_stock=round(safety_stock, 1),
        reorder_point=round(reorder_point, 1),
        recommended_order=round(raw_order),
        constrained_order=constrained,
        current_inventory=config.current_inventory,
        projected_inventory=round(projected, 1),
        stockout_risk=so_r,
        overstock_risk=ov_r,
        stockout_risk_pct=so_p,
        overstock_risk_pct=ov_p,
        days_of_stock=round(days_stock, 1),
        alert_message=alert_msg,
        festival_warning=fest_warning,
        std_source=std_source,
    )


def run_all_decisions(
    df_feat:       pd.DataFrame,
    forecast_fn    = None,
    model          = None,
    feature_cols:  list = None,
    service_level: float = 95.0,
) -> pd.DataFrame:
    if forecast_fn is None:
        try:
            from v4_integration import forecast_next_7_days_v4
            _forecast_fn = lambda df, sid, pid: forecast_next_7_days_v4(df, store_id=sid, product_id=pid)
        except ImportError:
            raise RuntimeError("v4_integration.py not found and no forecast_fn provided.")
    else:
        _forecast_fn = lambda df, sid, pid: forecast_fn(df, model, feature_cols, sid, pid)

    results   = []
    base_cols  = ["Store_ID", "Product_ID"]
    extra_cols = ["Product_Name", "Supplier_Limit", "Lead_Time_Days",
                  "Min_Order_Qty", "Inventory_Level", "Holding_Cost_Per_Unit",
                  "Stockout_Penalty_Per_Unit", "Price"]
    avail_cols = base_cols + [c for c in extra_cols if c in df_feat.columns]
    combos     = df_feat[avail_cols].drop_duplicates(subset=base_cols)

    for _, row in combos.iterrows():
        try:
            fcast = _forecast_fn(df_feat, row["Store_ID"], row["Product_ID"])
            hist30 = df_feat[
                (df_feat["Store_ID"] == row["Store_ID"]) &
                (df_feat["Product_ID"] == row["Product_ID"])
            ].sort_values("Date")["Units_Sold"].tail(30)
            std_30d = float(hist30.std()) if len(hist30) > 1 else None

            cfg = InventoryConfig(
                product_id   = row["Product_ID"],
                product_name = row.get("Product_Name", row["Product_ID"]),
                store_id     = row["Store_ID"],
                current_inventory        = float(row.get("Inventory_Level", 0)),
                lead_time_days           = int(row.get("Lead_Time_Days", 3)),
                supplier_limit           = int(row.get("Supplier_Limit", 300)),
                min_order_qty            = int(row.get("Min_Order_Qty", 20)),
                reorder_point            = float(row.get("Inventory_Level", 0)) * 0.3,
                service_level_pct        = service_level,
                holding_cost_per_unit    = float(row.get("Holding_Cost_Per_Unit", 2.0)),
                stockout_penalty_per_unit= float(row.get("Stockout_Penalty_Per_Unit", 15.0)),
                unit_price  = float(row.get("Price", 100.0)),
                cost_price  = float(row.get("Price", 100.0)) * 0.65,
            )
            decision = make_order_decision(fcast, cfg, demand_std_30d=std_30d)
            results.append(decision.to_dict())
        except Exception as e:
            print(f"   ⚠️ Decision error {row['Store_ID']}-{row['Product_ID']}: {e}")

    return pd.DataFrame(results)
