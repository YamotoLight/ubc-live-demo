# app.py — UBC LIVE (3 routes) with TransLink by stop ONLY
from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests

# ================== STREAMLIT SETUP ==================
st.set_page_config(page_title="UBC LIVE — Bus lineup waits", layout="wide")
st.title("UBC LIVE — Bus lineup waits")
st.caption("2-week simulated baseline (5–10 obs per 30-min block per day) + user reports + TransLink shown. All waits in minutes.")

# ================== CONFIG ==================
# just stopNo now
ROUTES = [
    {"key": "R4",  "label": "R4 to Joyce",    "base_mean": 6.0, "base_sd": 1.5, "stopNo": "60162"},
    {"key": "99",  "label": "99 B-Line",      "base_mean": 8.0, "base_sd": 2.0, "stopNo": "61395"},  # example
    {"key": "25",  "label": "25 Brentwood",   "base_mean": 5.0, "base_sd": 1.2, "stopNo": "50330"},
]

START_HOUR = 8
END_HOUR = 18
NUM_DAYS = 14
OBS_MIN = 5
OBS_MAX = 10

# ================== KEY ==================
def get_translink_key():
    try:
        return st.secrets["TRANSLINK_API_KEY"]
    except Exception:
        return os.getenv("TRANSLINK_API_KEY")

TRANSLINK_API_KEY = get_translink_key()

# ================== SLOTS ==================
def make_slots():
    slots = []
    slot_id = 0
    for h in range(START_HOUR, END_HOUR):
        for m in (0, 30):
            label = f"{h:02d}:{m:02d}–{h:02d}:{(m+30)%60:02d}"
            slots.append({"slot_id": slot_id, "label": label, "hour": h, "minute": m})
            slot_id += 1
    return slots

SLOTS = make_slots()

def slot_demand_multiplier(hour: int) -> float:
    if 8 <= hour < 10:
        return 1.4
    elif 10 <= hour < 14:
        return 1.2
    elif 14 <= hour < 17:
        return 1.1
    else:
        return 0.9

# ================== BASELINE SIM ==================
def simulate_baseline():
    rows = []
    rng = np.random.default_rng(42)
    for route in ROUTES:
        for day in range(1, NUM_DAYS + 1):
            for slot in SLOTS:
                n_obs = rng.integers(OBS_MIN, OBS_MAX + 1)  # 5–10 per day per block
                mult = slot_demand_multiplier(slot["hour"])
                mu = route["base_mean"] * mult
                sd = route["base_sd"]
                waits = rng.normal(loc=mu, scale=sd, size=int(n_obs))
                waits = np.clip(waits, 0.5, None)
                for w in waits:
                    rows.append(
                        {
                            "route_key": route["key"],
                            "route_label": route["label"],
                            "day": day,
                            "slot_id": slot["slot_id"],
                            "slot_label": slot["label"],
                            "wait_min": float(w),
                        }
                    )
    return pd.DataFrame(rows)

baseline_df = simulate_baseline()

# ================== USER REPORTS ==================
if "user_reports" not in st.session_state:
    st.session_state["user_reports"] = pd.DataFrame(
        columns=["route_key", "slot_id", "slot_label", "reported_min"]
    )

with st.expander("Add a user wait-time report"):
    c1, c2, c3 = st.columns(3)
    with c1:
        route_choice = st.selectbox("Route", [r["label"] for r in ROUTES])
    with c2:
        slot_choice = st.selectbox("Time block", [s["label"] for s in SLOTS])
    with c3:
        reported_min = st.number_input("Wait (minutes)", min_value=0.5, max_value=60.0, value=5.0, step=0.5)

    if st.button("Submit wait time"):
        route_key = [r["key"] for r in ROUTES if r["label"] == route_choice][0]
        slot_id = [s["slot_id"] for s in SLOTS if s["label"] == slot_choice][0]
        new_row = {
            "route_key": route_key,
            "slot_id": slot_id,
            "slot_label": slot_choice,
            "reported_min": float(reported_min),
        }
        st.session_state["user_reports"] = pd.concat(
            [st.session_state["user_reports"], pd.DataFrame([new_row])],
            ignore_index=True,
        )
        st.success("Added!")

with st.expander("See all user reports"):
    st.dataframe(st.session_state["user_reports"], use_container_width=True)

# ================== TRANSLINK FETCH (stop-only) ==================
def fetch_translink_wait(stop_no: str) -> float | None:
    if not TRANSLINK_API_KEY:
        return None
    try:
        url = f"https://api.translink.ca/rttiapi/v1/stops/{stop_no}/estimates?apiKey={TRANSLINK_API_KEY}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # data is usually a list of route objects; pick the first schedule of the first route
            if data and data[0].get("Schedules"):
                mins = data[0]["Schedules"][0].get("ExpectedCountdown")
                if mins is not None:
                    return float(mins)
        return None
    except Exception:
        return None

# ================== STATS ==================
def summarize_slot(route_key: str, slot_id: int):
    base = baseline_df[(baseline_df.route_key == route_key) & (baseline_df.slot_id == slot_id)]
    user = st.session_state["user_reports"]
    user = user[(user.route_key == route_key) & (user.slot_id == slot_id)]

    base_n = len(base)
    base_mean = base["wait_min"].mean() if base_n > 0 else math.nan
    base_var = base["wait_min"].var(ddof=1) if base_n > 1 else math.nan

    user_n = len(user)
    user_mean = user["reported_min"].mean() if user_n > 0 else math.nan

    if base_n + user_n > 0:
        parts = []
        if base_n > 0:
            parts.append(base_mean * base_n)
        if user_n > 0:
            parts.append(user_mean * user_n)
        combined_mean = sum(parts) / (base_n + user_n)
    else:
        combined_mean = math.nan

    if base_n > 1:
        se = math.sqrt(base_var / base_n)
        ci_low = combined_mean - 1.96 * se
        ci_high = combined_mean + 1.96 * se
        sd = math.sqrt(base_var)
        pi_low = combined_mean - 1.96 * sd
        pi_high = combined_mean + 1.96 * sd
    else:
        ci_low = ci_high = pi_low = pi_high = math.nan

    return {
        "base_n": int(base_n),
        "user_n": int(user_n),
        "combined_mean": round(float(combined_mean), 2) if not math.isnan(combined_mean) else None,
        "ci_low": round(float(ci_low), 2) if not math.isnan(ci_low) else None,
        "ci_high": round(float(ci_high), 2) if not math.isnan(ci_high) else None,
        "pi_low": round(float(pi_low), 2) if not math.isnan(pi_low) else None,
        "pi_high": round(float(pi_high), 2) if not math.isnan(pi_high) else None,
    }

# ================== UI ==================
cols = st.columns(3)
for i, route in enumerate(ROUTES):
    with cols[i]:
        st.subheader(route["label"])

        # 1 TransLink fetch per route, by stop only
        tl_wait = fetch_translink_wait(route["stopNo"])

        records = []
        for slot in SLOTS:
            stats = summarize_slot(route["key"], slot["slot_id"])
            records.append(
                {
                    "Time block": slot["label"],
                    "Baseline n": stats["base_n"],
                    "User n": stats["user_n"],
                    "Est. wait (min)": stats["combined_mean"],
                    "95% CI": f"{stats['ci_low']}–{stats['ci_high']}" if stats["ci_low"] is not None else "",
                    "95% PI": f"{stats['pi_low']}–{stats['pi_high']}" if stats["pi_low"] is not None else "",
                    "TransLink (min)": tl_wait if tl_wait is not None else "",
                }
            )
        df_show = pd.DataFrame(records)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

st.info(
    "TransLink column now shown. If it's blank, it means the real time API isn't happening in this time block."
