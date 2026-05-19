"""
Retail Store Inventory Forecasting — Synthetic Dataset Generator
=================================================================
Kaggle Dataset: anirudhchauhan/retail-store-inventory-forecasting-dataset
Schema: EXACT match to the real Kaggle CSV columns

Real festival calendar injected (Diwali, Holi, Eid change every year).
Seasonal patterns, weather, promotions, price elasticity — all modelled.
"""

import numpy as np
import pandas as pd
from datetime import date as date_type
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

STORES = {
    "Store_1": {"region": "North", "city": "Delhi",   "size": "large"},
    "Store_2": {"region": "South", "city": "Chennai", "size": "medium"},
    "Store_3": {"region": "West",  "city": "Mumbai",  "size": "large"},
    "Store_4": {"region": "East",  "city": "Kolkata", "size": "small"},
    "Store_5": {"region": "North", "city": "Lucknow", "size": "medium"},
}

PRODUCTS = {
    "P001": {"name": "Rice 5kg",        "category": "Grocery",       "base_price": 280, "base_demand": 60,  "holding_cost": 2, "stockout_penalty": 15, "elasticity": 0.8},
    "P002": {"name": "Cooking Oil 1L",  "category": "Grocery",       "base_price": 130, "base_demand": 45,  "holding_cost": 3, "stockout_penalty": 12, "elasticity": 0.7},
    "P003": {"name": "Detergent 500g",  "category": "Household",     "base_price": 95,  "base_demand": 35,  "holding_cost": 2, "stockout_penalty": 10, "elasticity": 0.6},
    "P004": {"name": "Biscuits 200g",   "category": "Snacks",        "base_price": 40,  "base_demand": 80,  "holding_cost": 1, "stockout_penalty": 8,  "elasticity": 1.0},
    "P005": {"name": "Shampoo 200ml",   "category": "Personal Care", "base_price": 160, "base_demand": 25,  "holding_cost": 4, "stockout_penalty": 14, "elasticity": 0.5},
    "P006": {"name": "Milk 1L",         "category": "Dairy",         "base_price": 55,  "base_demand": 120, "holding_cost": 5, "stockout_penalty": 20, "elasticity": 0.3},
    "P007": {"name": "Atta 10kg",       "category": "Grocery",       "base_price": 380, "base_demand": 50,  "holding_cost": 3, "stockout_penalty": 18, "elasticity": 0.6},
    "P008": {"name": "Chips 100g",      "category": "Snacks",        "base_price": 20,  "base_demand": 100, "holding_cost": 1, "stockout_penalty": 5,  "elasticity": 1.2},
    "P009": {"name": "Toothpaste 100g", "category": "Personal Care", "base_price": 75,  "base_demand": 40,  "holding_cost": 2, "stockout_penalty": 10, "elasticity": 0.4},
    "P010": {"name": "Sugar 1kg",       "category": "Grocery",       "base_price": 45,  "base_demand": 70,  "holding_cost": 2, "stockout_penalty": 12, "elasticity": 0.5},
}

SIZE_MULT = {"large": 1.30, "medium": 1.0, "small": 0.75}

WEATHER_TYPES   = ["Sunny", "Cloudy", "Rainy", "Stormy"]
WEATHER_IMPACTS = {"Sunny": 1.05, "Cloudy": 1.00, "Rainy": 0.88, "Stormy": 0.72}

MONTH_SEASONALITY = {
    1: 1.10, 2: 1.00, 3: 1.12, 4: 1.05, 5: 0.95,
    6: 0.90, 7: 0.88, 8: 0.92, 9: 1.00,
    10: 1.18, 11: 1.22, 12: 1.08,
}


def generate_dataset(
    start_date: str = "2022-01-01",
    end_date: str = "2024-06-30",
    inject_noise: bool = True,
    inject_missing: bool = True,
) -> pd.DataFrame:
    """
    Generates synthetic dataset matching the exact Kaggle column schema.

    Exact Kaggle columns produced:
      Product ID, Store ID, Date, Category, Region,
      Inventory Level, Units Sold, Units Ordered, Demand Forecast,
      Price, Discount, Weather Condition, Holiday/Promotion,
      Competitor Pricing, Seasonality Factor, Lead Time Days,
      Order Fulfillment Days
    """
    from festival_calendar import build_festival_map, get_festival_info

    s_d = date_type.fromisoformat(start_date)
    e_d = date_type.fromisoformat(end_date)
    festival_map = build_festival_map(s_d, e_d)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for store_id, store_info in STORES.items():
        for prod_id, prod in PRODUCTS.items():
            lead_time   = int(np.random.choice([2, 3, 4, 5], p=[0.15, 0.40, 0.35, 0.10]))
            fulfillment = lead_time + int(np.random.choice([0, 1], p=[0.70, 0.30]))
            size_f      = SIZE_MULT[store_info["size"]]
            sup_limit   = int(prod["base_demand"] * size_f * np.random.uniform(4, 7))
            min_order   = int(prod["base_demand"] * 0.5)
            inventory   = int(prod["base_demand"] * size_f * 10)

            for ts in dates:
                d = ts.date()
                dow = ts.dayofweek
                is_weekend = dow >= 5

                season_f  = MONTH_SEASONALITY[ts.month]
                weekend_b = 1.20 if is_weekend else 1.00

                # Real festival data
                fest_name, fest_mult, fest_flag = get_festival_info(d, festival_map)

                # Weather
                m = ts.month
                if m in [6, 7, 8]:   wprobs = [0.15, 0.25, 0.45, 0.15]
                elif m in [12, 1, 2]: wprobs = [0.30, 0.45, 0.20, 0.05]
                else:                 wprobs = [0.50, 0.25, 0.20, 0.05]
                weather    = np.random.choice(WEATHER_TYPES, p=wprobs)
                weather_i  = WEATHER_IMPACTS[weather]

                # Promotion — higher near festivals
                promo_p    = 0.30 if fest_mult > 1.2 else 0.12
                promo_flag = int(np.random.random() < promo_p)

                # Discount
                if promo_flag or fest_mult > 1.2:
                    discount = float(np.random.choice([10, 15, 20, 25], p=[0.30, 0.35, 0.25, 0.10]))
                else:
                    discount = float(np.random.choice([0, 0, 0, 5, 10], p=[0.50, 0.15, 0.15, 0.12, 0.08]))

                base_price       = prod["base_price"]
                effective_price  = round(base_price * (1 - discount / 100), 2)
                elasticity_eff   = 1 + (discount / 100) * prod["elasticity"]
                competitor_price = round(base_price * np.random.uniform(0.90, 1.15), 2)
                comp_edge        = max(1.0, 1 + (competitor_price - effective_price) / base_price * 0.4)

                # Noise
                if inject_noise:
                    if np.random.random() < 0.04:
                        noise = np.random.uniform(1.20, 1.30)
                    else:
                        noise = float(np.clip(np.random.normal(1.0, 0.07), 0.65, 1.40))
                else:
                    noise = 1.0

                # True demand
                true_demand = max(0, round(
                    prod["base_demand"] * size_f * season_f * weekend_b
                    * fest_mult * weather_i
                    * (1.15 if promo_flag else 1.0)
                    * elasticity_eff * comp_edge * noise
                ))

                # Missing data simulation
                if inject_missing and np.random.random() < 0.018:
                    units_sold = np.nan
                else:
                    units_sold = float(min(true_demand, inventory))

                # Naive forecast
                demand_forecast = max(0, round(
                    prod["base_demand"] * size_f * season_f
                    * weekend_b * float(np.random.normal(1.0, 0.10))
                ))

                seasonality_factor = round(season_f * fest_mult * weekend_b, 3)

                # Reorder
                reorder_point = prod["base_demand"] * size_f * lead_time * 1.5
                if inventory <= reorder_point:
                    units_ordered = min(max(int(demand_forecast * 7), min_order), sup_limit)
                else:
                    units_ordered = 0

                actual_sold = 0 if np.isnan(units_sold) else units_sold
                inventory   = int(np.clip(
                    inventory - actual_sold + units_ordered,
                    0, prod["base_demand"] * size_f * 30
                ))

                holiday_promo = 1 if (fest_flag or promo_flag) else 0

                records.append({
                    # ── EXACT KAGGLE COLUMNS ───────────────────────────────
                    "Product ID":             prod_id,
                    "Store ID":               store_id,
                    "Date":                   ts,
                    "Category":               prod["category"],
                    "Region":                 store_info["region"],
                    "Inventory Level":        inventory,
                    "Units Sold":             units_sold,
                    "Units Ordered":          units_ordered,
                    "Demand Forecast":        demand_forecast,
                    "Price":                  effective_price,
                    "Discount":               discount,
                    "Weather Condition":      weather,
                    "Holiday/Promotion":      holiday_promo,
                    "Competitor Pricing":     competitor_price,
                    "Seasonality Factor":     seasonality_factor,
                    "Lead Time Days":         lead_time,
                    "Order Fulfillment Days": fulfillment,
                    # ── ENRICHMENT ─────────────────────────────────────────
                    "Product_Name":           prod["name"],
                    "Base_Price":             base_price,
                    "Promotion_Flag":         promo_flag,
                    "Holiday_Flag":           fest_flag,
                    "Holiday_Festival":       fest_name,
                    "Festival_Multiplier":    round(fest_mult, 3),
                    "Supplier_Limit":         sup_limit,
                    "Min_Order_Qty":          min_order,
                    "Holding_Cost_Per_Unit":  prod["holding_cost"],
                    "Stockout_Penalty_Per_Unit": prod["stockout_penalty"],
                    "True_Demand":            true_demand,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    n_fest = (df["Holiday_Flag"] == 1).sum()
    n_miss = df["Units Sold"].isna().sum()
    print(f"✅ Dataset generated: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Date range            : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"   Stores: {df['Store ID'].nunique()}  |  Products: {df['Product ID'].nunique()}")
    print(f"   Festival-affected rows: {n_fest:,} ({n_fest/len(df)*100:.1f}%)")
    print(f"   Missing Units Sold    : {n_miss} (simulated gaps)")
    return df