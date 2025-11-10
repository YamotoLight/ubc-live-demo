# app.py — UBC LIVE (3 routes) with TransLink status line (no table column)
from __future__ import annotations
import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import requests

# ================== STREAMLIT SETUP ==================
st.set_page_config(page_title="UBC LIVE — Bus lineup waits", layout="wide")
st.title("UBC LIVE — Bus lineup waits")
st.caption("2-week simulated baseline (5–10 obs per 30-min block per day) + user reports. Live TransLink ETA shown under each title. All waits in minutes.")

# ================== CONFIG ==================
ROUTES = [
    {"key": "R4",  "label": "R4 to Joyce",  "base_mean": 6.0, "base_sd": 1.5, "stopNo": "60162"},
    {"key": "99",  "label": "99 B-Line",    "base_mean": 8.0, "base_sd": 2.0, "stopNo": "61395"},
    {"key": "25",  "label": "25 Brentwood", "base_mean": 5.0, "base_sd": 1.2, "stopNo": "50330"},
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
            end_h = h if m == 0 else h + 1
            end_m = (m + 30) % 60
            label = f"{h:02d}:{m:02d}–{end_h:02d}:{end_m:02d}"
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

# ----- current block helpers (for the status line) -----
def current_block_label_and_index():
    now = datetime.now()
    day_start = datetime(now.year, now.month, now.day, START_HOUR, 0)
    day_end   = datetime(now.year, now.month, now.day, END_HOUR, 0)
    if not (day_start <= now < day_end):
        return None, None
    mins = int((now - day_start).total_seconds() // 60)
    idx = mins // 30
    start_dt = day_start + timedelta(minutes=idx * 30)
    end_dt   = start_dt + timedelta(minutes=30)
    label = f"{start_dt.hour:02d}:{start_dt.minute:02d}–{end_dt.hour:02d}:{end_dt.minute:02d}"
    return label, idx

def fmt(x):
    return "—" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else f"{x:.2f}"

# ================== BASELINE SIM ==================
def simulate_baseline():
    rows = []
    rng = np.random.default_rng(42)
    for route in ROUTES:
        for day in range(1, NUM_DAYS + 1):
            for slot in SLOTS:
                n_obs = rng.integers(OBS_MIN, OBS_MAX + 1)
                mult = slot_demand_multiplier(slot["hour"])
                mu = route["base_mean"] * mult
                sd = route["base_sd"]
                waits = rng.normal(loc=mu, scale=sd, size=int(n_obs))
                waits = np.clip(waits, 0.5, None)
                for w in waits:
                    rows.append(
                        {
                            "route_key": route["key"],
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
        r = requests.get(url, timeout=5, headers={"accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
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
        "combined_mean": float(combined_mean) if not math.isnan(combined_mean) else None,
        "ci_low": float(ci_low) if not math.isnan(ci_low) else None,
        "ci_high": float(ci_high) if not math.isnan(ci_high) else None,
        "pi_low": float(pi_low) if not math.isnan(pi_low) else None,
        "pi_high": float(pi_high) if not math.isnan(pi_high) else None,
    }

# ================== UI ==================
cols = st.columns(3)
for i, route in enumerate(ROUTES):
    with cols[i]:
        st.subheader(route["label"])

        # Live TransLink (one call per route)
        tl_wait = fetch_translink_wait(route["stopNo"])

        # Current 30-min window + estimated wait for that window
        cur_label, cur_idx = current_block_label_and_index()
        if cur_label is not None:
            cur_stats = summarize_slot(route["key"], cur_idx)
            est_cur = fmt(cur_stats["combined_mean"])
        else:
            est_cur = "—"

        # Status line under the title (this replaces the table column)
        tl_str = "—" if tl_wait is None else f"{int(round(tl_wait))} min"
        cur_label = cur_label or "—"
        st.caption(f"Current window: {cur_label}  •  Estimated wait: {est_cur}  •  Bus arrives in: {tl_str}")

        # Build the per-stop table (no TransLink column)
        records = []
        for slot in SLOTS:
            stats = summarize_slot(route["key"], slot["slot_id"])
            records.append(
                {
                    "Time block": slot["label"],
                    "Baseline n": stats["base_n"],
                    "User n": stats["user_n"],
                    "Est. wait (min)": None if stats["combined_mean"] is None else round(stats["combined_mean"], 2),
                    "95% CI": "" if stats["ci_low"] is None else f"{round(stats['ci_low'],2)}–{round(stats['ci_high'],2)}",
                    "95% PI": "" if stats["pi_low"] is None else f"{round(stats['pi_low'],2)}–{round(stats['pi_high'],2)}",
                }
            )
        df_show = pd.DataFrame(records)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

st.info("TransLink ETA is shown only in the status line. If it shows “—”, no live countdown was returned for that stop/time.")
