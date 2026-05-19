"""
RetailIQ — Training Pipeline     
Trains: XGBoost + LightGBM ensemble 
Saves to: output/ (models, features, metrics)
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, BASE_DIR)

OUT_DIR = os.path.join(BASE_DIR, "output_v4")
os.makedirs(OUT_DIR, exist_ok=True)


#STEP 1: LOAD DATA

def load_and_preprocess():
    print("Loading & Preprocessing Data")
    from data_loader import load_data
    df = load_data(BASE_DIR)
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    # Required columns with defaults
    defaults = {
        "Promotion_Flag":         0,
        "Holiday_Flag":           0,
        "Competitor_Price":       df.get("Price", pd.Series(100.0)),
        "Base_Price":             df.get("Price", pd.Series(100.0)),
        "Discount_Pct":           0.0,
        "Festival_Multiplier":    1.0,
        "Inventory_Level":        100.0,
        "Lead_Time_Days":         3,
        "Supplier_Limit":         300,
        "Min_Order_Qty":          20,
        "Holding_Cost_Per_Unit":  2.0,
        "Stockout_Penalty_Per_Unit": 15.0,
        "Product_Name":           df.get("Product_ID", pd.Series("P001")),
        "Category":               "General",
        "Region":                 "India",
        "Weather_Condition":      "Sunny",
        "Seasonality_Factor":     1.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Fill missing Units_Sold
    df["Units_Sold"] = (
        df.groupby(["Store_ID", "Product_ID"])["Units_Sold"]
        .transform(lambda s: s.ffill().bfill().fillna(s.mean()))
    )
    df["Units_Sold"]      = df["Units_Sold"].clip(lower=0)
    df["Inventory_Level"] = df["Inventory_Level"].clip(lower=0)
    df["Price"]           = df["Price"].clip(lower=1)

    # Remove duplicates
    df = df.drop_duplicates(subset=["Date", "Store_ID", "Product_ID"], keep="last")
    df = df.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)

    print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Stores: {df['Store_ID'].nunique()}  |  Products: {df['Product_ID'].nunique()}")
    print(f"   Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# STEP 2: FEATURE ENGINEERING

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Feature Engineering")
    df = df.copy().sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)
    grp = df.groupby(["Store_ID", "Product_ID"])["Units_Sold"]

    #Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling statistics
    for w in [7, 14, 28]:
        shifted = grp.transform(lambda s: s.shift(1))
        df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=2).std().fillna(0)
        df[f"roll_min_{w}"]  = shifted.rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"]  = shifted.rolling(w, min_periods=1).max()
        df[f"roll_q25_{w}"]  = shifted.rolling(w, min_periods=1).quantile(0.25)
        df[f"roll_q75_{w}"]  = shifted.rolling(w, min_periods=1).quantile(0.75)

    df["roll_iqr_7"]  = df["roll_q75_7"]  - df["roll_q25_7"]
    df["roll_iqr_14"] = df["roll_q75_14"] - df["roll_q25_14"]

    # Keep pipeline.py-compatible names as aliases
    df["rolling_mean_7"]  = df["roll_mean_7"]
    df["rolling_mean_14"] = df["roll_mean_14"]
    df["rolling_mean_30"] = df.get("roll_mean_28", df["roll_mean_28"])
    df["rolling_std_7"]   = df["roll_std_7"]
    df["rolling_max_7"]   = df["roll_max_7"]
    df["rolling_min_7"]   = df["roll_min_7"]

    #EWM features
    for alpha in [0.2, 0.4, 0.7]:
        col = f"ewm_{str(alpha).replace('.','')}"
        df[col] = grp.transform(
            lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )

    #Time features
    df["day_of_week"]   = df["Date"].dt.dayofweek
    df["day_of_month"]  = df["Date"].dt.day
    df["week_of_year"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["weekofyear"]    = df["week_of_year"]
    df["month"]         = df["Date"].dt.month
    df["quarter"]       = df["Date"].dt.quarter
    df["year"]          = df["Date"].dt.year
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"]= df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"]  = df["Date"].dt.is_month_end.astype(int)

    # Cyclical encodings
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"]  / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"]  / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]        / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]        / 12)
    df["woy_sin"]   = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]   = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Festival features (real Indian calendar) 
    fm_col = "Festival_Multiplier"
    df["festival_mult"] = df[fm_col].fillna(1.0)
    df["Festival_Multiplier"] = df["festival_mult"]

    df["festival_level"]  = df["festival_mult"].apply(
        lambda x: 5 if x >= 1.5 else (3 if x >= 1.2 else (1 if x > 1.0 else 0))
    )
    df["fest_is_mega"]    = (df["festival_mult"] >= 1.50).astype(int)
    df["fest_is_major"]   = ((df["festival_mult"] >= 1.20) & (df["festival_mult"] < 1.50)).astype(int)
    df["fest_is_pre"]     = ((df["festival_mult"] >  1.00) & (df["festival_mult"] < 1.20)).astype(int)
    df["fest_is_normal"]  = (df["festival_mult"] <= 1.00).astype(int)
    df["is_festival_day"] = df["fest_is_mega"]
    df["is_pre_festival"] = df["fest_is_pre"]

    # Interaction: festival × demand
    df["fest_x_lag1"]      = df["festival_mult"] * df["lag_1"].fillna(0)
    df["fest_x_lag7"]      = df["festival_mult"] * df["lag_7"].fillna(0)
    df["fest_x_roll7"]     = df["festival_mult"] * df["roll_mean_7"].fillna(0)
    df["fest_x_roll28"]    = df["festival_mult"] * df["roll_mean_28"].fillna(0)
    df["fest_x_weekend"]   = df["festival_mult"] * df["is_weekend"]
    df["fest_x_promo"]     = df["festival_mult"] * df["Promotion_Flag"].fillna(0)
    df["fest_level_x_lag1"]= df["festival_level"] * df["lag_1"].fillna(0)

    # Days proximity to festival (simplified)
    df["days_until_festival"] = (df["festival_mult"] > 1.0).astype(int) * 0
    df["days_since_festival"] = (df["festival_mult"] > 1.0).astype(int) * 0
    df["fest_proximity"]      = np.exp(-df["days_until_festival"] / 5)
    df["fest_recency"]        = np.exp(-df["days_since_festival"] / 3)
    df["fest_proximity_x_lag1"] = df["fest_proximity"] * df["lag_1"].fillna(0)

    # Demand ratio & acceleration features
    eps = 1.0
    df["lag_ratio_1_7"]   = df["lag_1"].fillna(0)  / (df["lag_7"].fillna(0)  + eps)
    df["lag_ratio_7_28"]  = df["lag_7"].fillna(0)  / (df["lag_28"].fillna(0) + eps)
    df["lag_ratio_1_28"]  = df["lag_1"].fillna(0)  / (df["lag_28"].fillna(0) + eps)
    df["demand_accel_7"]  = (df["lag_1"].fillna(0) - df["lag_7"].fillna(0))  / (df["lag_7"].fillna(0)  + eps)
    df["demand_accel_28"] = (df["lag_7"].fillna(0) - df["lag_28"].fillna(0)) / (df["lag_28"].fillna(0) + eps)
    df["demand_momentum"] = df["lag_1"].fillna(0) - df["lag_7"].fillna(0)

    #Price features
    price_grp = df.groupby(["Store_ID", "Product_ID"])["Price"]
    df["price_change"]        = price_grp.transform(lambda s: s.diff().fillna(0))
    df["price_change_pct"]    = price_grp.transform(
        lambda s: s.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    )
    df["price_vs_base"]       = (df["Price"] / df["Base_Price"].replace(0, 1)).clip(0.5, 1.5)
    df["price_vs_competitor"] = (
        (df["Competitor_Price"] - df["Price"]) / df["Price"].replace(0, 1)
    ).clip(-0.5, 0.5)
    df["Discount_Pct"]        = df["Discount_Pct"].fillna(0)

    #Other business features
    df["inventory_ratio"] = (
        df["Inventory_Level"] / df["roll_mean_7"].replace(0, 1)
    ).clip(0, 20)
    df["seasonality_factor"] = df.get("Seasonality_Factor", pd.Series(1.0, index=df.index)).fillna(1.0)

    def days_since_promo(series):
        result, last = [], np.nan
        for i, v in enumerate(series):
            if v == 1: last = i
            result.append(i - last if not np.isnan(last) else 30)
        return result

    df["days_since_promo"] = df.groupby(["Store_ID", "Product_ID"])["Promotion_Flag"].transform(
        days_since_promo
    )

    #Categorical encodings
    for col in ["Store_ID", "Product_ID", "Category", "Region", "Weather_Condition"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Drop rows where core lags are missing
    df = df.dropna(subset=["lag_7", "roll_mean_7"]).reset_index(drop=True)
    print(f"   Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# STEP 3: DEFINE FEATURES 

def get_feature_cols(df: pd.DataFrame) -> list:
    """All engineered features available for training."""
    candidates = [
        # Lags
        "lag_1","lag_2","lag_3","lag_7","lag_14","lag_21","lag_28",
        # Rolling stats
        "roll_mean_7","roll_std_7","roll_min_7","roll_max_7","roll_q25_7","roll_q75_7","roll_iqr_7",
        "roll_mean_14","roll_std_14","roll_min_14","roll_max_14","roll_q25_14","roll_q75_14","roll_iqr_14",
        "roll_mean_28","roll_std_28","roll_min_28","roll_max_28","roll_q25_28","roll_q75_28",
        # EWM
        "ewm_02","ewm_04","ewm_07",
        # Time
        "day_of_week","day_of_month","week_of_year","weekofyear","month","quarter","year",
        "is_weekend","is_month_start","is_month_end",
        "dow_sin","dow_cos","month_sin","month_cos","woy_sin","woy_cos",
        # Festival
        "festival_mult","festival_level","Festival_Multiplier",
        "fest_is_mega","fest_is_major","fest_is_pre","fest_is_normal",
        "is_festival_day","is_pre_festival","Holiday_Flag",
        "fest_x_lag1","fest_x_lag7","fest_x_roll7","fest_x_roll28",
        "fest_x_weekend","fest_x_promo","fest_level_x_lag1",
        "days_until_festival","days_since_festival",
        "fest_proximity","fest_recency","fest_proximity_x_lag1",
        # Demand dynamics
        "lag_ratio_1_7","lag_ratio_7_28","lag_ratio_1_28",
        "demand_accel_7","demand_accel_28","demand_momentum",
        # Price
        "price_change","price_change_pct","price_vs_base","price_vs_competitor",
        "Discount_Pct","Promotion_Flag",
        # Business
        "inventory_ratio","seasonality_factor","days_since_promo",
        # Encodings
        "Store_ID_enc","Product_ID_enc","Category_enc","Region_enc","Weather_Condition_enc",
    ]
    return [c for c in candidates if c in df.columns]


# STEP 4: TRAIN XGB + LGB ENSEMBLE

def train_ensemble(df: pd.DataFrame, feature_cols: list):
    print("Training XGBoost + LightGBM Ensemble")
    import xgboost as xgb
    import lightgbm as lgb

    TARGET = "Units_Sold"
    X = df[feature_cols].fillna(0)
    y = np.log1p(df[TARGET].fillna(0))   # log1p target for stability

    tscv    = TimeSeriesSplit(n_splits=5, test_size=max(500, int(len(df)*0.08)))
    folds   = list(tscv.split(X))

    #XGBoost 
    print("   Training XGBoost…")
    xgb_cv_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.035,
            subsample=0.80, colsample_bytree=0.75,
            min_child_weight=5, reg_alpha=0.15, reg_lambda=1.5,
            objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=40,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.expm1(m.predict(X_val)).clip(0)
        true  = np.expm1(y_val)
        mae   = mean_absolute_error(true, preds)
        rmse  = np.sqrt(mean_squared_error(true, preds))
        mape  = np.mean(np.abs((true - preds) / (true + 1))) * 100
        xgb_cv_metrics.append({"fold":fold+1,"model":"XGB","MAE":mae,"RMSE":rmse,"MAPE":mape})
        print(f"   XGB Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

    xgb_final = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.035,
        subsample=0.80, colsample_bytree=0.75,
        min_child_weight=5, reg_alpha=0.15, reg_lambda=1.5,
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_final.fit(X, y, verbose=False)

    #LightGBM
    print("   Training LightGBM…")
    lgb_cv_metrics = []
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = lgb.LGBMRegressor(
            n_estimators=600, max_depth=7, learning_rate=0.035,
            subsample=0.80, colsample_bytree=0.75,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            objective="regression", random_state=42, n_jobs=-1, verbosity=-1,
        )
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(-1)])
        preds = np.expm1(m.predict(X_val)).clip(0)
        true  = np.expm1(y_val)
        mae   = mean_absolute_error(true, preds)
        rmse  = np.sqrt(mean_squared_error(true, preds))
        mape  = np.mean(np.abs((true - preds) / (true + 1))) * 100
        lgb_cv_metrics.append({"fold":fold+1,"model":"LGB","MAE":mae,"RMSE":rmse,"MAPE":mape})
        print(f"   LGB Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

    lgb_final = lgb.LGBMRegressor(
        n_estimators=600, max_depth=7, learning_rate=0.035,
        subsample=0.80, colsample_bytree=0.75,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        objective="regression", random_state=42, n_jobs=-1, verbosity=-1,
    )
    lgb_final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    #  Ensemble weights (validation MAE-based) 
    xgb_mae = np.mean([m["MAE"] for m in xgb_cv_metrics])
    lgb_mae = np.mean([m["MAE"] for m in lgb_cv_metrics])
    total   = xgb_mae + lgb_mae + 1e-9
    # Lower MAE → higher weight
    w_xgb = round(lgb_mae / total, 3)
    w_lgb = round(xgb_mae / total, 3)
    # Normalize
    s = w_xgb + w_lgb
    w_xgb = round(w_xgb / s, 3)
    w_lgb = round(1 - w_xgb, 3)

    print(f"\n   XGB CV → MAE={xgb_mae:.2f}")
    print(f"   LGB CV → MAE={lgb_mae:.2f}")
    print(f"   Ensemble Weights: XGB={w_xgb}  LGB={w_lgb}")

    all_metrics = pd.DataFrame(xgb_cv_metrics + lgb_cv_metrics)
    return xgb_final, lgb_final, {"xgb": w_xgb, "lgb": w_lgb}, all_metrics


#STEP 5: SAVE ARTIFACTS

def save_artifacts(xgb_model, lgb_model, feature_cols, weights, df_feat, metrics):
    print("Saving Artifacts → output")

    # XGBoost model
    xgb_path = os.path.join(OUT_DIR, "xgb_model.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"   ✅ {xgb_path}")

    # LightGBM model
    lgb_path = os.path.join(OUT_DIR, "lgb_model.pkl")
    with open(lgb_path, "wb") as f:
        pickle.dump(lgb_model, f)
    print(f"   ✅ {lgb_path}")

    # Feature list
    feat_path = os.path.join(OUT_DIR, "features_used.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"   ✅ {feat_path}  ({len(feature_cols)} features)")

    # Ensemble weights
    w_path = os.path.join(OUT_DIR, "ensemble_weights.json")
    with open(w_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"   ✅ {w_path}  (xgb={weights['xgb']}  lgb={weights['lgb']})")

    # Full feature dataset (used by app for forecasting)
    csv_path = os.path.join(OUT_DIR, "data_features.csv")
    # Keep only essential columns to reduce file size
    keep_cols = (
        ["Date", "Store_ID", "Product_ID", "Units_Sold",
         "Inventory_Level", "Price", "Holiday_Flag",
         "Festival_Multiplier", "Promotion_Flag",
         "Lead_Time_Days", "Supplier_Limit", "Min_Order_Qty",
         "Holding_Cost_Per_Unit", "Stockout_Penalty_Per_Unit",
         "Product_Name"]
        + [c for c in feature_cols if c in df_feat.columns]
    )
    keep_cols = list(dict.fromkeys(keep_cols))   # deduplicate preserving order
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    df_feat[keep_cols].to_csv(csv_path, index=False)
    print(f"   ✅ {csv_path}  ({len(df_feat):,} rows)")

    # CV metrics
    m_path = os.path.join(OUT_DIR, "cv_metrics_v4.csv")
    metrics.to_csv(m_path, index=False)
    print(f"   ✅ {m_path}")

    # Also save a V3-compatible model.pkl so old code doesn't break
    v3_path = os.path.join(BASE_DIR, "model.pkl")
    with open(v3_path, "wb") as f:
        pickle.dump({"model": xgb_model, "feature_cols": feature_cols}, f)
    print(f"   ✅ {v3_path}  (V3-compatible fallback)")


#  STEP 6: QUICK SANITY CHECK 

def sanity_check(xgb_model, lgb_model, feature_cols, weights, df_feat):
    print(" Sanity Check")
    try:
        from v4_integration import load_v4_models, forecast_next_7_days_v4
        xgb_m, lgb_m, fc, wt = load_v4_models(OUT_DIR)

        store   = df_feat["Store_ID"].iloc[0]
        product = df_feat["Product_ID"].iloc[0]
        fc_df   = forecast_next_7_days_v4(df_feat, store_id=store, product_id=product)

        print(f"\n   Sample 7-day forecast for {store} / {product}:")
        print(f"   {fc_df[['Day_Label','Predicted_Demand','Festival']].to_string(index=False)}")
        print(f"\n   ✅ Sanity check PASSED — app.py is ready to run!")
    except Exception as e:
        print(f"   ⚠️ Sanity check warning: {e}")
        print(f"   Models saved successfully — try running app.py")


#MAIN

def main():
    print("\n" + "═"*65)
    print("  RetailIQ — V4 Training Pipeline")
    print("═"*65 + "\n")

    df_raw  = load_and_preprocess()
    df_feat = engineer_features(df_raw)

    feature_cols = get_feature_cols(df_feat)
    print(f"\n   Using {len(feature_cols)} features for training\n")

    xgb_model, lgb_model, weights, metrics = train_ensemble(df_feat, feature_cols)
    save_artifacts(xgb_model, lgb_model, feature_cols, weights, df_feat, metrics)
    sanity_check(xgb_model, lgb_model, feature_cols, weights, df_feat)

    print("\n" + "═"*65)
    print("PIPELINE COMPLETE")
  


if __name__ == "__main__":
    main()
