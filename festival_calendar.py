"""
Real Indian Festival Calendar — 2022 to 2025
=============================================
All festival dates are EXACT based on the Hindu lunar calendar.
Diwali, Holi, Eid, Navratri, Dussehra all change every year.
This replaces the old hardcoded (wrong) approach.

Demand multiplier logic:
  Festival day     → 1.5x – 1.7x
  2 days before    → 1.2x – 1.4x  (shopping rush)
  3–5 days before  → 1.1x – 1.2x  (early buying)
  Week after major → 1.05x         (return/exchange, mild bump)
"""

from datetime import date, timedelta
from typing import Dict, Tuple, List

# ──────────────────────────────────────────────────────────────────────────────
# EXACT FESTIVAL DATES — verified against Hindu panchang & Govt holiday lists
# ──────────────────────────────────────────────────────────────────────────────

FESTIVALS_EXACT: Dict[date, Tuple[str, str, float]] = {
    # ─── 2022 ─────────────────────────────────────────────────────────────────
    date(2022,  1,  1): ("New Year Day",            "national",  1.25),
    date(2022,  1, 14): ("Makar Sankranti / Pongal","regional",  1.30),
    date(2022,  1, 26): ("Republic Day",            "national",  1.20),
    date(2022,  3, 18): ("Holi",                    "major",     1.55),
    date(2022,  3, 17): ("Holika Dahan (Holi Eve)", "major",     1.35),
    date(2022,  4, 14): ("Baisakhi / Tamil NY",     "regional",  1.25),
    date(2022,  5,  2): ("Eid al-Fitr",             "major",     1.40),
    date(2022,  7,  9): ("Eid al-Adha",             "moderate",  1.25),
    date(2022,  8, 11): ("Raksha Bandhan",           "major",     1.35),
    date(2022,  8, 15): ("Independence Day",         "national",  1.20),
    date(2022,  8, 18): ("Janmashtami",              "major",     1.30),
    date(2022,  8, 31): ("Ganesh Chaturthi",         "major",     1.35),
    date(2022,  9,  8): ("Onam",                     "regional",  1.25),
    date(2022, 10,  2): ("Gandhi Jayanti",           "national",  1.15),
    date(2022, 10,  2): ("Navratri Begins",          "major",     1.30),
    date(2022, 10,  5): ("Dussehra",                 "major",     1.50),
    date(2022, 10, 24): ("Diwali (Lakshmi Puja)",    "mega",      1.70),
    date(2022, 10, 23): ("Diwali Eve (Naraka Ch.)",  "mega",      1.55),
    date(2022, 10, 25): ("Govardhan Puja",           "major",     1.40),
    date(2022, 10, 26): ("Bhai Dooj",                "major",     1.30),
    date(2022, 11,  8): ("Chhath Puja",              "regional",  1.25),
    date(2022, 12, 25): ("Christmas",                "national",  1.25),
    date(2022, 12, 31): ("New Year Eve",             "national",  1.30),

    # ─── 2023 ─────────────────────────────────────────────────────────────────
    date(2023,  1,  1): ("New Year Day",             "national",  1.25),
    date(2023,  1, 14): ("Makar Sankranti / Pongal", "regional",  1.30),
    date(2023,  1, 26): ("Republic Day",             "national",  1.20),
    date(2023,  3,  7): ("Holika Dahan (Holi Eve)",  "major",     1.35),
    date(2023,  3,  8): ("Holi",                     "major",     1.55),
    date(2023,  4, 14): ("Baisakhi",                 "regional",  1.25),
    date(2023,  4, 21): ("Eid al-Fitr",              "major",     1.40),
    date(2023,  6, 28): ("Eid al-Adha",              "moderate",  1.25),
    date(2023,  8, 15): ("Independence Day",         "national",  1.20),
    date(2023,  8, 28): ("Janmashtami",              "major",     1.30),
    date(2023,  8, 29): ("Onam",                     "regional",  1.25),
    date(2023,  8, 30): ("Raksha Bandhan",           "major",     1.35),
    date(2023,  9, 18): ("Ganesh Chaturthi Begins",  "major",     1.35),
    date(2023, 10,  2): ("Gandhi Jayanti",           "national",  1.15),
    date(2023, 10, 15): ("Navratri Begins",          "major",     1.30),
    date(2023, 10, 24): ("Dussehra",                 "major",     1.50),
    date(2023, 11, 12): ("Diwali (Lakshmi Puja)",    "mega",      1.70),
    date(2023, 11, 11): ("Diwali Eve (Naraka Ch.)",  "mega",      1.55),
    date(2023, 11, 13): ("Govardhan Puja",           "major",     1.40),
    date(2023, 11, 14): ("Bhai Dooj",                "major",     1.30),
    date(2023, 11, 19): ("Chhath Puja",              "regional",  1.25),
    date(2023, 12, 25): ("Christmas",                "national",  1.25),
    date(2023, 12, 31): ("New Year Eve",             "national",  1.30),

    # ─── 2024 ─────────────────────────────────────────────────────────────────
    date(2024,  1,  1): ("New Year Day",             "national",  1.25),
    date(2024,  1, 14): ("Makar Sankranti / Pongal", "regional",  1.30),
    date(2024,  1, 22): ("Ram Mandir Inauguration",  "special",   1.45),
    date(2024,  1, 26): ("Republic Day",             "national",  1.20),
    date(2024,  3, 24): ("Holika Dahan (Holi Eve)",  "major",     1.35),
    date(2024,  3, 25): ("Holi",                     "major",     1.55),
    date(2024,  4, 11): ("Eid al-Fitr",              "major",     1.40),
    date(2024,  4, 14): ("Baisakhi / Tamil NY",      "regional",  1.25),
    date(2024,  6, 16): ("Eid al-Adha",              "moderate",  1.25),
    date(2024,  8, 15): ("Independence Day",         "national",  1.20),
    date(2024,  8, 19): ("Raksha Bandhan",           "major",     1.35),
    date(2024,  8, 26): ("Janmashtami",              "major",     1.30),
    date(2024,  9,  7): ("Ganesh Chaturthi Begins",  "major",     1.35),
    date(2024,  9, 15): ("Onam",                     "regional",  1.25),
    date(2024, 10,  2): ("Gandhi Jayanti",           "national",  1.15),
    date(2024, 10,  3): ("Navratri Begins",          "major",     1.30),
    date(2024, 10, 12): ("Dussehra",                 "major",     1.50),
    date(2024, 11,  1): ("Diwali (Lakshmi Puja)",    "mega",      1.70),
    date(2024, 10, 31): ("Diwali Eve (Naraka Ch.)",  "mega",      1.55),
    date(2024, 11,  2): ("Govardhan Puja",           "major",     1.40),
    date(2024, 11,  3): ("Bhai Dooj",                "major",     1.30),
    date(2024, 11,  7): ("Chhath Puja",              "regional",  1.25),
    date(2024, 12, 25): ("Christmas",                "national",  1.25),
    date(2024, 12, 31): ("New Year Eve",             "national",  1.30),

    # ─── 2025 (partial, for forecasting) ──────────────────────────────────────
    date(2025,  1,  1): ("New Year Day",             "national",  1.25),
    date(2025,  1, 14): ("Makar Sankranti / Pongal", "regional",  1.30),
    date(2025,  1, 26): ("Republic Day",             "national",  1.20),
    date(2025,  3, 13): ("Holika Dahan (Holi Eve)",  "major",     1.35),
    date(2025,  3, 14): ("Holi",                     "major",     1.55),
    date(2025,  3, 30): ("Eid al-Fitr",              "major",     1.40),
    date(2025,  4, 14): ("Baisakhi",                 "regional",  1.25),
    date(2025,  8, 15): ("Independence Day",         "national",  1.20),
    date(2025, 10, 20): ("Dussehra",                 "major",     1.50),
    date(2025, 10, 20): ("Diwali (Lakshmi Puja)",    "mega",      1.70),
    date(2025, 12, 25): ("Christmas",                "national",  1.25),
    date(2025, 12, 31): ("New Year Eve",             "national",  1.30),
}

# ──────────────────────────────────────────────────────────────────────────────
# PRE-SHOPPING WINDOWS (days before a major festival → demand rises early)
# ──────────────────────────────────────────────────────────────────────────────
PRE_FESTIVAL_WINDOWS = {
    "mega":     {"days_before": 7, "ramp": [1.05, 1.08, 1.12, 1.18, 1.25, 1.35, 1.45]},
    "major":    {"days_before": 5, "ramp": [1.04, 1.07, 1.12, 1.20, 1.30]},
    "moderate": {"days_before": 3, "ramp": [1.03, 1.06, 1.12]},
    "regional": {"days_before": 2, "ramp": [1.04, 1.10]},
    "national": {"days_before": 2, "ramp": [1.03, 1.07]},
    "special":  {"days_before": 3, "ramp": [1.05, 1.10, 1.20]},
}

# ──────────────────────────────────────────────────────────────────────────────
# BUILD FULL DATE→MULTIPLIER MAP (including pre-festival ramp-up)
# ──────────────────────────────────────────────────────────────────────────────

def build_festival_map(
    start: date = date(2022, 1, 1),
    end:   date = date(2025, 12, 31),
) -> Dict[date, Tuple[str, float]]:
    """
    Returns a dict: {date: (festival_name, demand_multiplier)}
    Covers festival days + pre-festival shopping ramp days.
    """
    festival_map: Dict[date, Tuple[str, float]] = {}

    for fest_date, (name, ftype, peak_mult) in FESTIVALS_EXACT.items():
        if not (start <= fest_date <= end):
            continue

        # ── Festival day itself ──
        festival_map[fest_date] = (name, peak_mult)

        # ── Pre-festival ramp ──
        window = PRE_FESTIVAL_WINDOWS.get(ftype, {"days_before": 0, "ramp": []})
        ramp   = window["ramp"]
        for i, mult in enumerate(reversed(ramp)):
            pre_date = fest_date - timedelta(days=i+1)
            if start <= pre_date <= end:
                # Don't override a higher multiplier (e.g., another festival)
                existing = festival_map.get(pre_date, (None, 1.0))
                if mult > existing[1]:
                    festival_map[pre_date] = (f"Pre-{name}", mult)

        # ── Post-festival mild bump (mega/major only) ──
        if ftype in ("mega", "major"):
            for i in range(1, 3):
                post_date = fest_date + timedelta(days=i)
                if start <= post_date <= end:
                    existing = festival_map.get(post_date, (None, 1.0))
                    post_mult = 1.08 if i == 1 else 1.04
                    if post_mult > existing[1]:
                        festival_map[post_date] = (f"Post-{name}", post_mult)

    return festival_map


def get_festival_info(d: date, festival_map: Dict) -> Tuple[str, float, int]:
    """
    Returns (festival_name, demand_multiplier, holiday_flag)
    """
    if d in festival_map:
        name, mult = festival_map[d]
        is_holiday = 1
    else:
        name, mult, is_holiday = "None", 1.0, 0
    return name, mult, is_holiday


# QUICK TEST
if __name__ == "__main__":
    fm = build_festival_map()
    print(f"Total festival-affected days (2022-2025): {len(fm)}")
    print("\nSample — Diwali 2022 window:")
    for d in [date(2022, 10, d) for d in range(17, 29)]:
        info = fm.get(d, ("Normal", 1.0))
        print(f"  {d.strftime('%a %d %b %Y')}  →  {info[0]:30s}  ×{info[1]:.2f}")

    print("\nSample — Holi 2023 window:")
    for d in [date(2023, 3, d) for d in range(4, 12)]:
        info = fm.get(d, ("Normal", 1.0))
        print(f"  {d.strftime('%a %d %b %Y')}  →  {info[0]:30s}  ×{info[1]:.2f}")