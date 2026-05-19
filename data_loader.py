"""
Smart Data Loader
==================
Accepts the real Kaggle CSV (retail_store_inventory.csv) if present,
OR generates a perfectly matching synthetic dataset.

Kaggle Dataset: anirudhchauhan/retail-store-inventory-forecasting-dataset
Exact column names matched.

HOW TO USE YOUR REAL DATASET:
  1. Download from Kaggle → retail_store_inventory.csv
  2. Drop it in the same folder as this file
  3. Run — it auto-detects and loads it

"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#EXACT KAGGLE COLUMN NAMES
# These are the real column names from the Kaggle dataset
KAGGLE_COLUMNS = [
    "Product ID",
    "Store ID",
    "Date",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality Factor",
    "Lead Time Days",
    "Order Fulfillment Days",
]

# Canonical internal names we use throughout the pipeline
# (maps Kaggle → internal)
COLUMN_MAP = {
    "Product ID":            "Product_ID",
    "Store ID":              "Store_ID",
    "Date":                  "Date",
    "Category":              "Category",
    "Region":                "Region",
    "Inventory Level":       "Inventory_Level",
    "Units Sold":            "Units_Sold",
    "Units Ordered":         "Units_Ordered",
    "Demand Forecast":       "Demand_Forecast",
    "Price":                 "Price",
    "Discount":              "Discount_Pct",
    "Weather Condition":     "Weather_Condition",
    "Holiday/Promotion":     "Holiday_Promotion",
    "Competitor Pricing":    "Competitor_Price",
    "Seasonality Factor":    "Seasonality_Factor",
    "Lead Time Days":        "Lead_Time_Days",
    "Order Fulfillment Days": "Order_Fulfillment_Days",
}

# Columns we ADD on top of the Kaggle schema (engineering additions)
EXTRA_COLS_DEFAULTS = {
    "Product_Name":              None,   # inferred from Product_ID
    "Base_Price":                None,   # inferred from Price + Discount_Pct
    "Promotion_Flag":            None,   # derived from Holiday_Promotion
    "Holiday_Flag":              None,   # derived from Holiday_Promotion
    "Holiday_Festival":          "None",
    "Min_Order_Qty":             None,   # inferred = 0.5 × avg daily demand
    "Supplier_Limit":            None,   # inferred
    "Holding_Cost_Per_Unit":     2.0,
    "Stockout_Penalty_Per_Unit": 15.0,
    "True_Demand":               None,   # = Units_Sold (best proxy)
}

# Known search paths for the Kaggle CSV
KAGGLE_CSV_NAMES = [
    "retail_store_inventory.csv",
    "retail_store_inventory_forecasting.csv",
    "inventory_forecasting.csv",
    "dataset.csv",
    "train.csv",
]


#LOADER

def find_kaggle_csv(base_dir: str) -> str | None:
    """Search common locations for the downloaded Kaggle CSV."""
    search_dirs = [
        base_dir,
        os.path.dirname(base_dir),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "dataset"),
        os.path.expanduser("~/Downloads"),
    ]
    for d in search_dirs:
        for name in KAGGLE_CSV_NAMES:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    return None


def load_kaggle_csv(path: str) -> pd.DataFrame:
    """
    Load and standardise the real Kaggle dataset.
    Handles Kaggle's exact column names → internal names.
    """
    print(f"📂 Loading real Kaggle dataset from: {path}")
    df = pd.read_csv(path)
    print(f"   Raw shape: {df.shape}")
    print(f"   Columns  : {list(df.columns)}")

    # Rename columns
    rename = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped in COLUMN_MAP:
            rename[col] = COLUMN_MAP[stripped]
    df = df.rename(columns=rename)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["Date"])

    #  Derive missing columns
    _enrich_kaggle_df(df)

    print(f"   Enriched shape : {df.shape}")
    print(f"   Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def _enrich_kaggle_df(df: pd.DataFrame) -> None:
    """
    Add all columns missing from Kaggle CSV that our pipeline needs.
    Modifies df IN PLACE.
    """
    #Product_Name from Product_ID
    if "Product_Name" not in df.columns:
        product_name_map = {
            pid: f"Product_{pid}" for pid in df["Product_ID"].unique()
        }
        df["Product_Name"] = df["Product_ID"].map(product_name_map)

    # ── Base_Price (back-calculate from Price and Discount) ──
    if "Base_Price" not in df.columns:
        if "Discount_Pct" in df.columns:
            df["Base_Price"] = (df["Price"] / (1 - df["Discount_Pct"].clip(0, 90) / 100)).round(2)
        else:
            df["Base_Price"] = df["Price"]

    # ── Holiday / Promotion flags ──
    if "Holiday_Promotion" in df.columns:
        hp = df["Holiday_Promotion"]
        # Could be 0/1 int OR string like "Holiday", "Promotion", "None"
        if hp.dtype == object:
            df["Holiday_Flag"]   = (hp.str.strip().str.lower().isin(["holiday","yes","1","true"])).astype(int)
            df["Promotion_Flag"] = (hp.str.strip().str.lower().isin(["promotion","promo","yes","1","true"])).astype(int)
            df["Holiday_Festival"] = hp.where(hp.str.strip().str.lower() != "none", "None")
        else:
            df["Holiday_Flag"]    = hp.fillna(0).astype(int)
            df["Promotion_Flag"]  = hp.fillna(0).astype(int)
            df["Holiday_Festival"] = "None"
    else:
        df["Holiday_Flag"]    = 0
        df["Promotion_Flag"]  = 0
        df["Holiday_Festival"] = "None"

    # ── INJECT REAL FESTIVAL DATA INTO KAGGLE DATASET ──
    _inject_real_festivals(df)

    # ── Supplier constraints ──
    if "Supplier_Limit" not in df.columns:
        avg_demand = df.groupby(["Store_ID","Product_ID"])["Units_Sold"].transform("mean").fillna(50)
        df["Supplier_Limit"] = (avg_demand * 6).round().astype(int).clip(lower=10)

    if "Min_Order_Qty" not in df.columns:
        avg_demand = df.groupby(["Store_ID","Product_ID"])["Units_Sold"].transform("mean").fillna(50)
        df["Min_Order_Qty"] = (avg_demand * 0.5).round().astype(int).clip(lower=5)

    # ── Cost columns ──
    if "Holding_Cost_Per_Unit" not in df.columns:
        df["Holding_Cost_Per_Unit"] = 2.0
    if "Stockout_Penalty_Per_Unit" not in df.columns:
        df["Stockout_Penalty_Per_Unit"] = 15.0

    # ── True_Demand (best proxy = Units_Sold + stockout estimate) ──
    if "True_Demand" not in df.columns:
        df["True_Demand"] = df["Units_Sold"]

    # ── Order_Fulfillment_Days → Lead_Time_Days fallback ──
    if "Lead_Time_Days" not in df.columns and "Order_Fulfillment_Days" in df.columns:
        df["Lead_Time_Days"] = df["Order_Fulfillment_Days"]
    elif "Lead_Time_Days" not in df.columns:
        df["Lead_Time_Days"] = 3


def _inject_real_festivals(df: pd.DataFrame) -> None:
    """
    CORE FUNCTION: Overwrite Holiday_Flag and Holiday_Festival
    using our real Indian festival calendar for every row.
    Modifies df IN PLACE.
    """
    from festival_calendar import build_festival_map
    from datetime import date as date_type

    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    festival_map = build_festival_map(date_min, date_max)

    # Map date → festival info
    def get_info(dt):
        d = dt.date() if hasattr(dt, 'date') else dt
        if d in festival_map:
            name, mult = festival_map[d]
            return name, 1, mult
        return "None", 0, 1.0

    result = df["Date"].apply(get_info)
    df["Holiday_Festival"]     = result.apply(lambda x: x[0])
    df["Holiday_Flag"]         = result.apply(lambda x: x[1])
    df["Festival_Multiplier"]  = result.apply(lambda x: x[2])

    injected = (df["Holiday_Flag"] == 1).sum()
    print(f"   ✅ Real festivals injected: {injected:,} rows marked as festival days")


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def load_data(base_dir: str = None) -> pd.DataFrame:
    """
    Master loader. Priority:
      1. retail_store_inventory.csv (Kaggle download)  ← YOUR REAL FILE
      2. retail_data.csv (previously generated)
      3. Generate fresh synthetic dataset

    Always returns a DataFrame with all required columns.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Try real Kaggle file first ──
    kaggle_path = find_kaggle_csv(base_dir)
    if kaggle_path:
        df = load_kaggle_csv(kaggle_path)
        # Save enriched version
        out_path = os.path.join(base_dir, "retail_data.csv")
        df.to_csv(out_path, index=False)
        print(f"   💾 Enriched dataset saved → {out_path}")
        return df

    # ── Try cached synthetic ──
    cache_path = os.path.join(base_dir, "retail_data.csv")
    if os.path.exists(cache_path):
        print(f"📂 Loading cached synthetic dataset from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        # Re-inject latest real festival data in case calendar was updated
        _inject_real_festivals(df)
        return df

    # ── Generate fresh ──
    print("🔧 No dataset found. Generating realistic synthetic dataset...")
    from data_generator import generate_dataset
    df = generate_dataset()
    df.to_csv(cache_path, index=False)
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    df = load_data(base)
    print(f"\n Dataset ready: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    fest_days = df[df['Holiday_Flag']==1]['Holiday_Festival'].value_counts().head(10)
    print(f"\nTop festival days:\n{fest_days}")