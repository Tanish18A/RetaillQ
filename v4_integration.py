

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import timedelta
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
sys.path.insert(0, BASE_DIR)


_v4_cache = {}

def load_v4_models(model_dir="output_v4"):
    """Cache V4 models so they are loaded only once."""
    if "xgb" in _v4_cache:
        return _v4_cache["xgb"], _v4_cache["lgb"], _v4_cache["features"], _v4_cache["weights"]

    mp = os.path.join(BASE_DIR, model_dir)
    xgb_path  = os.path.join(mp, "xgb_model.pkl")
    lgb_path  = os.path.join(mp, "lgb_model.pkl")
    feat_path = os.path.join(mp, "features_used.txt")
    w_path    = os.path.join(mp, "ensemble_weights.json")
    v3_path   = os.path.join(BASE_DIR, "model.pkl")

    if os.path.exists(xgb_path) and os.path.exists(lgb_path):
        import xgboost as xgb, lightgbm as lgb
        with open(xgb_path, "rb") as f: xgb_model = pickle.load(f)
        with open(lgb_path,  "rb") as f: lgb_model  = pickle.load(f)
        with open(feat_path)       as f: features   = [l.strip() for l in f if l.strip()]
        with open(w_path)          as f: weights    = json.load(f)
        _v4_cache.update({"xgb": xgb_model, "lgb": lgb_model,
                          "features": features, "weights": weights, "version": "v4"})
        print(f"✅ V4 models loaded ({len(features)} features)")
    elif os.path.exists(v3_path):
        with open(v3_path, "rb") as f: art = pickle.load(f)
        _v4_cache.update({"xgb": art["model"], "lgb": None,
                          "features": art["feature_cols"], "weights": {"xgb": 1.0, "lgb": 0.0},
                          "version": "v3"})
        print("⚠️ V4 not found — using V3 model as fallback")
    else:
        raise FileNotFoundError("No trained model found. Run pipeline_v4.py first.")

    return _v4_cache["xgb"], _v4_cache["lgb"], _v4_cache["features"], _v4_cache["weights"]


def predict_v4(X: pd.DataFrame, xgb_model, lgb_model, weights, features) -> np.ndarray:
    """Run V4 ensemble prediction on a feature matrix."""
    X_safe = X[[f for f in features if f in X.columns]].fillna(0)
    for f in features:
        if f not in X_safe.columns:
            X_safe[f] = 0.0
    X_safe = X_safe[features]
    xgb_pred = np.expm1(xgb_model.predict(X_safe)).clip(0)
    if lgb_model is not None and weights.get("lgb", 0) > 0:
        lgb_pred = np.expm1(lgb_model.predict(X_safe)).clip(0)
        return weights["xgb"] * xgb_pred + weights["lgb"] * lgb_pred
    return xgb_pred


def forecast_next_7_days_v4(
    df_feat: pd.DataFrame,
    model_dir: str = "output_v4",
    store_id:   str = None,
    product_id: str = None,
    n_days:     int = 7,
) -> pd.DataFrame:
    """
    V4-powered demand forecast for n_days ahead (default=7, supports 14).
    Rolling predictions feed back into lag features for multi-step accuracy.
    """
    from festival_calendar import build_festival_map, get_festival_info
    from datetime import date as date_type

    xgb_model, lgb_model, features, weights = load_v4_models(model_dir)

    grp = df_feat.copy()
    if store_id:
        grp = grp[grp["Store_ID"] == store_id]
    if product_id:
        grp = grp[grp["Product_ID"] == product_id]
    grp = grp.sort_values("Date").reset_index(drop=True)

    last_date = grp["Date"].max()
    last_row  = grp.iloc[-1].copy()
    recent    = list(grp["Units_Sold"].tail(30))

    fc_start = (last_date + timedelta(days=1)).date()
    fc_end   = (last_date + timedelta(days=n_days)).date()
    fm = build_festival_map(
        date_type(fc_start.year, 1, 1),
        date_type(fc_end.year + 1, 12, 31)
    )

    forecast_rows = []
    for day in range(1, n_days + 1):
        fdate = last_date + timedelta(days=day)
        row   = last_row.copy()

        row["day_of_week"]   = fdate.dayofweek
        row["day_of_month"]  = fdate.day
        row["week_of_year"]  = fdate.isocalendar()[1]
        row["weekofyear"]    = fdate.isocalendar()[1]
        row["month"]         = fdate.month
        row["quarter"]       = (fdate.month - 1) // 3 + 1
        row["year"]          = fdate.year
        row["is_weekend"]    = int(fdate.dayofweek >= 5)
        row["is_month_end"]  = int(fdate.day == pd.Period(fdate, "M").days_in_month)
        row["is_month_start"]= int(fdate.day == 1)

        row["dow_sin"]   = np.sin(2 * np.pi * row["day_of_week"]  / 7)
        row["dow_cos"]   = np.cos(2 * np.pi * row["day_of_week"]  / 7)
        row["month_sin"] = np.sin(2 * np.pi * row["month"]        / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"]        / 12)
        row["woy_sin"]   = np.sin(2 * np.pi * row["week_of_year"] / 52)
        row["woy_cos"]   = np.cos(2 * np.pi * row["week_of_year"] / 52)

        fest_name, fest_mult, fest_flag = get_festival_info(fdate.date(), fm)
        row["Festival_Multiplier"]  = fest_mult
        row["festival_level"]       = 5 if fest_mult >= 1.5 else (3 if fest_mult >= 1.2 else (1 if fest_mult > 1.0 else 0))
        row["fest_is_mega"]         = int(fest_mult >= 1.50)
        row["fest_is_major"]        = int(1.20 <= fest_mult < 1.50)
        row["fest_is_pre"]          = int(1.01 <= fest_mult < 1.20)
        row["fest_is_normal"]       = int(fest_mult <= 1.00)
        row["Holiday_Flag"]         = fest_flag

        n = len(recent)
        row["lag_1"]  = recent[-1]  if n >= 1  else 0
        row["lag_2"]  = recent[-2]  if n >= 2  else 0
        row["lag_3"]  = recent[-3]  if n >= 3  else 0
        row["lag_7"]  = recent[-7]  if n >= 7  else 0
        row["lag_14"] = recent[-14] if n >= 14 else 0
        row["lag_21"] = recent[-21] if n >= 21 else 0
        row["lag_28"] = recent[-28] if n >= 28 else 0

        for w in [7, 14, 28]:
            slice_ = recent[-w:] if len(recent) >= w else recent
            row[f"roll_mean_{w}"] = np.mean(slice_) if slice_ else 0
            row[f"roll_std_{w}"]  = np.std(slice_)  if len(slice_) > 1 else 0
            row[f"roll_min_{w}"]  = min(slice_)     if slice_ else 0
            row[f"roll_max_{w}"]  = max(slice_)     if slice_ else 0
            row[f"roll_q25_{w}"]  = np.percentile(slice_, 25) if slice_ else 0
            row[f"roll_q75_{w}"]  = np.percentile(slice_, 75) if slice_ else 0
        row["roll_iqr_7"]  = row.get("roll_q75_7",  0) - row.get("roll_q25_7",  0)
        row["roll_iqr_14"] = row.get("roll_q75_14", 0) - row.get("roll_q25_14", 0)

        for alpha in [0.2, 0.4, 0.7]:
            col = f"ewm_{str(alpha).replace('.','')}"
            if recent:
                s = pd.Series(recent)
                row[col] = float(s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
            else:
                row[col] = 0

        lag1 = row["lag_1"]; lag7 = row["lag_7"]; lag28 = row["lag_28"]
        eps  = 1.0
        row["lag_ratio_1_7"]   = lag1  / (lag7  + eps)
        row["lag_ratio_7_28"]  = lag7  / (lag28 + eps)
        row["lag_ratio_1_28"]  = lag1  / (lag28 + eps)
        row["demand_accel_7"]  = (lag1 - lag7)  / (lag7  + eps)
        row["demand_accel_28"] = (lag7 - lag28) / (lag28 + eps)
        row["demand_momentum"] = lag1 - lag7

        row["fest_x_lag1"]       = fest_mult * lag1
        row["fest_x_lag7"]       = fest_mult * row["lag_7"]
        row["fest_x_roll7"]      = fest_mult * row["roll_mean_7"]
        row["fest_x_roll28"]     = fest_mult * row["roll_mean_28"]
        row["fest_x_weekend"]    = fest_mult * row["is_weekend"]
        row["fest_x_promo"]      = fest_mult * row.get("Promotion_Flag", 0)
        row["fest_level_x_lag1"] = row["festival_level"] * lag1

        row["days_until_festival"] = 0 if fest_flag else min(30, 7)
        row["days_since_festival"] = 0 if fest_flag else min(30, 7)
        row["fest_proximity"]      = np.exp(-row["days_until_festival"] / 5)
        row["fest_recency"]        = np.exp(-row["days_since_festival"] / 3)
        row["fest_proximity_x_lag1"] = row["fest_proximity"] * lag1

        X_pred = pd.DataFrame([dict(row)])
        pred   = float(predict_v4(X_pred, xgb_model, lgb_model, weights, features)[0])
        pred   = max(0, round(pred))

        forecast_rows.append({
            "Date":             fdate,
            "Store_ID":         store_id,
            "Product_ID":       product_id,
            "Predicted_Demand": pred,
            "Day_Label":        fdate.strftime("%a %d %b"),
            "Festival":         fest_name if fest_flag else "—",
            "Festival_Mult":    fest_mult,
        })
        recent.append(pred)

    return pd.DataFrame(forecast_rows)
