# app.py — UBC LIVE demo (no sidebar, click-to-open sections)
# 2-week baseline (simulated) + live user reports + optional TransLink + optional Monte Carlo
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from datetime import datetime

# ---------------------------------------------------------
# STREAMLIT SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="UBC LIVE — End-to-end demo",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("UBC LIVE — End-to-end demo")
st.caption("2-week simulated baseline + user reports (live) + TransLink proxy waits + Monte Carlo synthetic waits (minutes).")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def t_multiplier(level: float, df: int) -> float:
    try:
        import scipy.stats as sps
        return float(sps.t.ppf(0.5 * (1 + level), df))
    except Exception:
        return 1.96  # 95% normal fallback

def make_slots():
    """30-min blocks from 08:00 to 18:00."""
    slots = []
    start_h, end_h = 8, 18
    for h in range(start_h, end_h):
        slots.append(f"{h:02d}:00–{h:02d}:30")
        slots.append(f"{h:02d}:30–{h+1:02d}:00")
    return slots

def simulate_baseline(num_days: int = 14, seed: int = 123, return_raw: bool = True):
    """
    2-week fake survey. Each day we 'collect' 3–6 timing obs per slot.
    Noon-ish is busier. Outputs:
      - agg: slot-level n0, mu0, sd0
      - raw (optional): per-observation rows (day, slot, wait_min)
    """
    rng = np.random.default_rng(seed)
    slots = make_slots()
    rows = []

    def base_wait_from_slot(slot_label: str) -> float:
        hour = int(slot_label[:2])
        if 11 <= hour <= 13:
            return 7.5   # lunch spike
        elif 9 <= hour < 11:
            return 6.0
        elif 13 < hour <= 15:
            return 5.5
        else:
            return 4.0

    for day in range(num_days):
        day_bump = 0.4 if day % 7 in [0, 1, 2, 3, 4] else 0.0  # tiny weekday-ish bump
        for slot in slots:
            center = base_wait_from_slot(slot) + day_bump
            k = rng.integers(3, 7)  # 3–6 observations this slot this day
            waits = center + rng.normal(0, 1.0, size=k)
            waits = np.clip(waits, 0.5, None)
            for w in waits:
                rows.append({"day": day, "slot": slot, "wait_min": float(w)})

    raw_df = pd.DataFrame(rows)
    agg_df = (
        raw_df.groupby("slot")["wait_min"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n0", "mean": "mu0", "std": "sd0"})
        .sort_values("slot")
        .reset_index(drop=True)
    )
    agg_df["sd0"] = agg_df["sd0"].fillna(0.9)

    return (agg_df, raw_df) if return_raw else (agg_df, None)

def ci_and_pi(mean: float, sd: float, n: int, level: float = 0.95):
    df = max(n - 1, 1)
    t = t_multiplier(level, df)
    ci_half = t * sd / math.sqrt(n)                # CI for mean
    pi_half = t * sd * math.sqrt(1 + 1 / n)        # PI for new observation
    return (mean - ci_half, mean + ci_half), (mean - pi_half, mean + pi_half)

def fetch_translink_wait(api_key: str, stop_id: str, route_no: str | None = None):
    """
    Ping TransLink GTFS-RT v3 to verify the key. We don't parse the feed here;
    we return a dummy wait (4 min) so the rest of the app can fuse it.
    """
    if not api_key:
        return None
    url = f"https://gtfsapi.translink.ca/v3/gtfsrealtime?apikey={api_key}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return 4.0  # pretend live wait is 4 minutes
    except Exception:
        return None

def simulate_mm1_waits(lam: float, mu: float, n: int, seed: int = 42):
    """
    Simple M/M/1 queue simulation. Returns waiting times (minutes).
    """
    if lam <= 0 or mu <= 0 or lam >= mu:
        return np.array([])  # unstable/invalid
    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1.0 / lam, size=n)
    services = rng.exponential(1.0 / mu, size=n)

    arrival_times = np.cumsum(interarrivals)
    start_service = np.zeros(n)
    finish_service = np.zeros(n)
    waits = np.zeros(n)

    for i in range(n):
        start_service[i] = arrival_times[i] if i == 0 else max(arrival_times[i], finish_service[i - 1])
        finish_service[i] = start_service[i] + services[i]
        waits[i] = start_service[i] - arrival_times[i]

    return waits

def fuse_sources(slot_baseline, user_waits, translink_waits, mc_waits):
    """
    Treat each source as observations and pool the means by sample size.
    """
    n0 = float(slot_baseline["n0"])
    mu0 = float(slot_baseline["mu0"])
    sd0 = float(slot_baseline["sd0"])

    parts = [(n0, mu0)]  # baseline

    if len(user_waits) > 0:
        uw = np.array(user_waits, dtype=float)
        parts.append((len(uw), float(uw.mean())))

    if len(translink_waits) > 0:
        tw = np.array(translink_waits, dtype=float)
        parts.append((len(tw), float(tw.mean())))

    if len(mc_waits) > 0:
        mw = np.array(mc_waits, dtype=float)
        parts.append((len(mw), float(mw.mean())))

    total_n = sum(p[0] for p in parts)
    pooled_mean = sum(p[0] * p[1] for p in parts) / total_n

    return {"n": int(total_n), "mean": pooled_mean, "sd": sd0 if sd0 > 0 else 1.0}

# ---------------------------------------------------------
# SETTINGS (on main page, hidden by default)
# ---------------------------------------------------------
with st.expander("Settings (optional — click to open)"):
    user_sd_input = st.number_input("Fallback SD for user/sim (min)", value=3.0, step=0.5)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "translink_waits" not in st.session_state:
    st.session_state["translink_waits"] = []
if "mc_waits" not in st.session_state:
    st.session_state["mc_waits"] = []
if "user_reports" not in st.session_state:
    st.session_state["user_reports"] = {s: [] for s in make_slots()}

# ---------------------------------------------------------
# 1) BASELINE (always computed; viewing is click-to-open)
# ---------------------------------------------------------
bdf, raw_baseline = simulate_baseline(num_days=14, seed=123, return_raw=True)

with st.expander("Baseline data (click to open)"):
    st.markdown("**Aggregated baseline (per 30-min block)**")
    st.dataframe(bdf, use_container_width=True, height=300)

    st.markdown("**Explore baseline for a specific block**")
    col_b1, col_b2 = st.columns([2, 2])
    with col_b1:
        slot_view = st.selectbox("Choose block", make_slots(), index=make_slots().index("11:00–11:30"))
    with col_b2:
        show_raw = st.checkbox("Show raw per-day samples for this block", value=False)

    # show the selected slot's summary row
    sel_row = bdf[bdf["slot"] == slot_view].iloc[0]
    st.write(
        f"**{slot_view}** → n0 = {int(sel_row['n0'])}, "
        f"mu0 = {sel_row['mu0']:.2f} min, sd0 = {sel_row['sd0']:.2f} min"
    )

    if show_raw:
        sub = raw_baseline[raw_baseline["slot"] == slot_view][["day", "wait_min"]].copy()
        sub = sub.sort_values(["day"]).reset_index(drop=True)
        st.markdown("**Raw baseline samples for selected block**")
        st.dataframe(sub, use_container_width=True, height=240)

st.write("---")

# ---------------------------------------------------------
# 2) USER INPUT (click to open)
# ---------------------------------------------------------
with st.expander("User input — add observations (click to open)"):
    u1, u2, u3, u4 = st.columns([2, 2, 2, 1])
    with u1:
        slot_choice = st.selectbox("30-min block", make_slots(), index=make_slots().index("11:00–11:30"))
    with u2:
        user_wait = st.number_input("Observed wait (min)", min_value=0.0, step=0.1, value=6.0)
    with u3:
        add_clicked = st.button("Add user report")
    with u4:
        clear_slot = st.button("Clear slot")

    if add_clicked:
        st.session_state["user_reports"][slot_choice].append(float(user_wait))
        st.success(f"Added {user_wait:.1f} min to {slot_choice} at {datetime.now().strftime('%H:%M:%S')}")
    if clear_slot:
        st.session_state["user_reports"][slot_choice] = []
        st.info(f"Cleared user reports for {slot_choice}")

    # Summary table across all slots
    summary_rows = []
    for s, vals in st.session_state["user_reports"].items():
        n = len(vals)
        mean = float(np.mean(vals)) if n > 0 else np.nan
        summary_rows.append({"30-min block": s, "User n": n, "User mean (min)": None if np.isnan(mean) else round(mean, 2)})
    user_summary_df = pd.DataFrame(summary_rows).sort_values("30-min block").reset_index(drop=True)

    st.markdown("**User reports — summary (by slot)**")
    st.dataframe(user_summary_df, use_container_width=True, height=260)

    # Detailed table for the currently selected slot
    detail_df = pd.DataFrame({"Observed wait (min)": st.session_state["user_reports"][slot_choice]})
    st.markdown(f"**User reports — details for {slot_choice}**")
    st.dataframe(detail_df if not detail_df.empty else pd.DataFrame({"Observed wait (min)": []}), use_container_width=True)

st.write("---")

# ---------------------------------------------------------
# 3) TRANSLINK (click to open)
# ---------------------------------------------------------
with st.expander("TransLink next-bus countdowns (proxy waits) — click to open"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        api_key = st.text_input("TransLink API Key", type="password")
    with c2:
        stop_id = st.text_input("Stop ID (e.g. 61395)")
    with c3:
        route_no = st.text_input("Route No (optional)")
    with c4:
        fetch_btn = st.button("Fetch TransLink")

    if fetch_btn:
        tw = fetch_translink_wait(api_key, stop_id, route_no)
        if tw is None:
            st.warning("Could not fetch TransLink (GTFS-RT) — leaving this source empty.")
        else:
            st.session_state["translink_waits"].append(float(tw))
            st.success(f"TransLink proxy wait added: {tw:.1f} min")

    if len(st.session_state["translink_waits"]) > 0:
        st.caption(f"TransLink observations stored this run: {st.session_state['translink_waits']}")

st.write("---")

# ---------------------------------------------------------
# 4) MONTE CARLO (click to open)
# ---------------------------------------------------------
with st.expander("Monte Carlo (M/M/1) — extra synthetic waits (click to open)"):
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        lam = st.number_input("Arrivals/min λ", value=0.60, step=0.05)
    with m2:
        mu = st.number_input("Services/min μ", value=0.80, step=0.05)
    with m3:
        n_mc = st.number_input("N customers", value=200, step=50)
    with m4:
        mc_seed = st.number_input("Seed", value=42, step=1)

    if st.button("Run Monte Carlo"):
        mcw = simulate_mm1_waits(lam, mu, int(n_mc), seed=int(mc_seed))
        if mcw.size == 0:
            st.warning("Monte Carlo params invalid or unstable (need μ > λ).")
        else:
            st.session_state["mc_waits"] = list(mcw)
            st.success(f"Monte Carlo generated {len(mcw)} synthetic waits.")
            st.caption(f"Example waits (first 5): {[round(x,2) for x in mcw[:5]]}")

st.write("---")

# ---------------------------------------------------------
# 5) PREDICTION TABLE (always visible; updates live)
# ---------------------------------------------------------
st.subheader("Prediction for next waiting-time reading (fused sources)")
st.caption("Order of trust: baseline > user > TransLink > Monte Carlo. We pool them as observations.")

rows_out = []
for _, row in bdf.iterrows():
    slot = row["slot"]
    user_waits = st.session_state["user_reports"].get(slot, [])
    translink_waits = st.session_state["translink_waits"]
    mc_waits = st.session_state["mc_waits"]

    fused = fuse_sources(row, user_waits, translink_waits, mc_waits)
    n = fused["n"]
    mean = fused["mean"]
    sd = fused["sd"] if fused["sd"] > 0 else user_sd_input

    ci, pi = ci_and_pi(mean, sd, n, level=0.95)

    rows_out.append(
        {
            "30-min block": slot,
            "Samples (n)": n,
            "Avg wait (min)": round(mean, 2),
            "95% CI (min)": f"[{ci[0]:.1f}, {ci[1]:.1f}]",
            "95% PI (min)": f"[{pi[0]:.1f}, {pi[1]:.1f}]",
            "User n": len(user_waits),
        }
    )

out_df = pd.DataFrame(rows_out)
st.dataframe(out_df, use_container_width=True)
