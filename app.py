#!/usr/bin/env python3
# ================================================================
#  Stress-Ribbon Bridge Cable Selector  – Streamlit edition
#  Dual-input (Manual / CSV) + CSV-driven MOORA settings
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
# ================================================================

import math, os, json
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# Persistent credit line
# ---------------------------------------------------------------
CREDIT = "Authors : Vijaykumar Parmar & Dr. K. B. Parikh"

# ---------------------------------------------------------------
# Default MOORA-criterion configuration
# ---------------------------------------------------------------
@dataclass
class CriterionConfig:
    enabled: bool
    is_cost: bool          # True = cost, False = benefit
    shape: str             # "linear" | "exponential"
    trigger: str           # "above" | "below"
    threshold: float
    slope: float
    exponent: float

DEFAULT_CRIT: Dict[str, CriterionConfig] = {
    "Utilisation"   : CriterionConfig(True,  True , "exponential", "below", 0.8  , 1.0, 6.0),
    "Slope_pct"     : CriterionConfig(True,  False, "linear"     , "below", 2.5  , 1.0, 1.0),
    "Cable_Dia_mm"  : CriterionConfig(True,  True , "linear"     , "above", 150  , 0.5, 1.0),
    "N_Cables"      : CriterionConfig(True,  True , "exponential", "above", 5    , 1.0, 1.2),
    "NatFreq_Hz"    : CriterionConfig(True,  False, "linear"     , "above", 2.0  , 1.0, 1.0),
    "Tension_kN"    : CriterionConfig(True,  True , "linear"     , "above", 0.0  , 1.0, 1.0),
    "Sag_m"         : CriterionConfig(True,  False, "exponential", "below", 0.003, 1.0, 3.0),
}

# ----------------------------------------------------------------
# Numeric-range validator for primary bridge inputs (not MOORA)
# ----------------------------------------------------------------
CSV_RANGES = {
    "Span L (m)"       : (10, 500),
    "UDL w (kN/m)"     : (10, 1000),
    "Bridge width (m)" : (1, 10),
    "Base Cable Diam"  : (5, 300),
    "Base Cables"      : (2, 20),
    "Strength (MPa)"   : (200, 3000),
    "Density (kN/m3)"  : (50, 90),
    "Δ cables"         : (0, 5),
}

# ----------------------------------------------------------------
# Build template CSV (headers + one demo row) on first run
# ----------------------------------------------------------------
def _template_rows() -> List[Dict[str, Any]]:
    """Return one demo row for the template CSV."""
    row = {
        "Bridge Name"       : "Demo-Bridge",
        "Span L (m)"        : 80,
        "UDL w (kN/m)"      : 15,
        "Bridge width (m)"  : 3,
        "Base Cable Diam"   : 30,
        "Base Cables"       : 4,
        "Strength (MPa)"    : 1860,
        "Density (kN/m3)"   : 77,
        "Δ cables"          : 1,
    }
    # Append default MOORA settings
    for crit, cfg in DEFAULT_CRIT.items():
        row[f"{crit} Enabled"]   = int(cfg.enabled)         # 1 / 0
        row[f"{crit} Type"]      = "Cost" if cfg.is_cost else "Benefit"
        row[f"{crit} Shape"]     = cfg.shape
        row[f"{crit} Trigger"]   = cfg.trigger
        row[f"{crit} Threshold"] = cfg.threshold
        row[f"{crit} Slope"]     = cfg.slope
        row[f"{crit} Exponent"]  = cfg.exponent
    return [row]

if not os.path.exists("template_input.csv"):
    demo_df = pd.DataFrame(_template_rows())
    demo_df.to_csv("template_input.csv", index=False)

# ===============================================================
# -------- Engineering helper & MOORA logic (unchanged) ---------
# ===============================================================
def _area_mm2(d_mm: float) -> float:
    return math.pi * (d_mm / 2) ** 2

def cable_metrics(span_m, udl_kNpm, n_cables, dia_mm,
                  strength_MPa, utilisation, density_kNpm3):
    area_mm2  = _area_mm2(dia_mm)
    H_kN      = n_cables * area_mm2 * utilisation * strength_MPa / 1_000
    sag_m     = udl_kNpm * span_m ** 2 / (8 * H_kN) if H_kN else 0
    V_kN      = udl_kNpm * span_m / 2
    T_kN      = math.hypot(H_kN, V_kN)
    area_m2   = area_mm2 * 1e-6
    rho       = density_kNpm3 * 1_000 / 9.81
    mu_kgpm   = rho * area_m2
    omega2    = (H_kN * 1_000) / (mu_kgpm * n_cables) if mu_kgpm and n_cables else 0
    nat_f     = (1 / (2 * span_m)) * math.sqrt(omega2) if omega2 else 0
    mass_kg   = mu_kgpm * span_m * n_cables
    return {
        "Cable_Dia_mm": dia_mm,
        "Utilisation" : utilisation,
        "N_Cables"    : n_cables,
        "Slope_pct"   : sag_m / span_m * 100,
        "Tension_kN"  : T_kN,
        "Sag_m"       : sag_m,
        "NatFreq_Hz"  : nat_f,
        "CableMass_kg": mass_kg,
    }

def _pb_value(x: float, cfg: CriterionConfig) -> float:
    if not cfg.enabled:
        return 0.0
    diff = (x - cfg.threshold) if cfg.trigger == "above" else (cfg.threshold - x)
    if diff <= 0:
        return 0.0
    if cfg.shape == "linear":
        return cfg.slope * diff
    if cfg.shape == "exponential":
        return math.exp(cfg.exponent * diff) - 1
    return 0.0

def generate_alternatives(span, udl, base_n, base_dia, strength, density,
                          bridge_w, util_grid, dia_factors, n_delta):
    recs = []
    n_opts = [max(2, base_n + i) for i in range(-n_delta, n_delta + 1)]
    for fac in dia_factors:
        dia = round(base_dia * (1 + fac), 3)
        if dia < 5:
            continue
        for util in util_grid:
            for n in n_opts:
                r = cable_metrics(span, udl, n, dia, strength, util, density)
                r["Cable_Spacing_m"]   = bridge_w / n
                r["UDL_perCable_kNpm"] = udl / n
                recs.append(r)
    return pd.DataFrame(recs).round(6)

def moora_rank(df: pd.DataFrame, cfg_map: Dict[str, CriterionConfig]) -> pd.DataFrame:
    benefit, cost = [], []
    for crit, cfg in cfg_map.items():
        if not cfg.enabled:
            continue
        col = f"PB_{crit}"
        df[col] = df[crit].apply(lambda v: _pb_value(v, cfg))
        (cost if cfg.is_cost else benefit).append(col)
    for col in benefit + cost:
        norm = np.sqrt((df[col] ** 2).sum())
        df[f"N_{col}"] = df[col] / norm if norm else 0
    df["MOORA_Score"] = (
        df[[f"N_{c}" for c in benefit]].sum(axis=1)
        - df[[f"N_{c}" for c in cost]].sum(axis=1)
    )
    ranked = df.sort_values("MOORA_Score", ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked["Rank"] = ranked.index
    return ranked

# ----------------------------------------------------------------
# Pretty plots (unchanged from previous version, omitted here to
# save space – keep your existing contour / parallel functions)
# ----------------------------------------------------------------
# ...  (cable_profile_fig, contour_fig, parallel_fig) ...

# ===============================================================
# -------------------------  STREAMLIT  --------------------------
# ===============================================================
st.set_page_config("SRB Cable Selector – MOORA", layout="wide")
st.title("Stress-Ribbon Bridge Cable Selector (MOORA)")

# Initialise session placeholders
for k in ["span", "udl", "width", "base_n", "base_d",
          "strength", "density", "n_delta", "bridge_id", "cfg_map"]:
    st.session_state.setdefault(k, None)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("Input mode")
    input_mode = st.radio("Choose input method", ["Manual Input", "CSV Input"])

    # ================  MANUAL  =================
    if input_mode == "Manual Input":
        st.subheader("Bridge parameters – manual")
        st.session_state.span   = st.number_input("Span L (m)",       10.0, 500.0, 50.0)
        st.session_state.udl    = st.number_input("UDL w (kN/m)",     10.0, 1000.0, 100.0)
        st.session_state.width  = st.number_input("Bridge width (m)", 1.0, 10.0, 3.0)
        st.session_state.base_n = st.number_input("Base number of cables", 2, 20, 2)
        st.session_state.base_d = st.number_input("Base cable diameter (mm)", 5.0, 300.0, 20.0)
        st.session_state.strength = st.number_input("Cable strength σ (MPa)", 200.0, 3000.0, 1600.0)
        st.session_state.density  = st.number_input("Density γ (kN/m³)", 50.0, 90.0, 77.0)
        st.session_state.n_delta  = st.slider("± range around base #Cables", 0, 5, 1)
        st.session_state.bridge_id = None

        # ------ Manual MOORA widgets ------
        st.markdown("---")
        st.subheader("MOORA criterion settings")
        cfg_map: Dict[str, CriterionConfig] = {}
        for name, cfg in DEFAULT_CRIT.items():
            with st.expander(name, expanded=False):
                en = st.checkbox("Enabled", value=cfg.enabled, key=name+"en")
                ct = st.radio("Type", ["Cost", "Benefit"], 0 if cfg.is_cost else 1, key=name+"ct")
                sh = st.selectbox("Shape", ["linear", "exponential"],
                                  0 if cfg.shape == "linear" else 1, key=name+"sh")
                tr = st.selectbox("Trigger", ["above", "below"],
                                  0 if cfg.trigger == "above" else 1, key=name+"tr")
                th = st.number_input("Threshold", value=cfg.threshold, key=name+"th")
                sl = st.number_input("Slope (linear)", value=cfg.slope, key=name+"sl")
                ex = st.number_input("Exponent (exp)", value=cfg.exponent, key=name+"ex")
            cfg_map[name] = CriterionConfig(
                en, ct == "Cost", sh, tr, th, sl, ex
            )
        st.session_state.cfg_map = cfg_map

    # ================  CSV  ===================
    else:
        st.subheader("Bridge parameters – CSV")
        with open("template_input.csv", "rb") as tf:
            st.download_button("📥 Download sample template.csv", tf,
                               file_name="template_input.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload single-row CSV", type="csv")
        if uploaded:
            df_csv = pd.read_csv(uploaded)

            # -------- Highlight out-of-range numeric inputs -------
            def _validate_row(row):
                styles = []
                for col, val in row.items():
                    if col in CSV_RANGES:
                        lo, hi = CSV_RANGES[col]
                        styles.append("background-color: red"
                                      if not (lo <= val <= hi) else "")
                    else:
                        styles.append("")
                return styles

            st.dataframe(df_csv.style.apply(_validate_row, axis=1), height=260)

            row_idx = st.number_input(
                "Row index to load (starting at 2 = first data row)",
                min_value=2, max_value=len(df_csv) + 1, value=2, step=1
            )

            if st.button("Load input"):
                try:
                    row = df_csv.iloc[row_idx - 2]
                except IndexError:
                    st.error("Selected row is out of range.")
                else:
                    # ---- Bridge core parameters -------------------
                    st.session_state.span      = float(row["Span L (m)"])
                    st.session_state.udl       = float(row["UDL w (kN/m)"])
                    st.session_state.width     = float(row["Bridge width (m)"])
                    st.session_state.base_d    = float(row["Base Cable Diam"])
                    st.session_state.base_n    = int(row["Base Cables"])
                    st.session_state.strength  = float(row["Strength (MPa)"])
                    st.session_state.density   = float(row["Density (kN/m3)"])
                    st.session_state.n_delta   = int(row["Δ cables"])
                    st.session_state.bridge_id = str(row["Bridge Name"])

                    # ---- Build MOORA cfg from CSV (fallback → default)
                    cfg_map_csv: Dict[str, CriterionConfig] = {}
                    for crit, def_cfg in DEFAULT_CRIT.items():
                        enabled   = bool(row.get(f"{crit} Enabled", def_cfg.enabled))
                        is_cost   = str(row.get(f"{crit} Type", "Cost" if def_cfg.is_cost else "Benefit")).strip().lower() == "cost"
                        shape     = str(row.get(f"{crit} Shape", def_cfg.shape)).strip().lower()
                        trigger   = str(row.get(f"{crit} Trigger", def_cfg.trigger)).strip().lower()
                        threshold = float(row.get(f"{crit} Threshold", def_cfg.threshold))
                        slope     = float(row.get(f"{crit} Slope", def_cfg.slope))
                        exponent  = float(row.get(f"{crit} Exponent", def_cfg.exponent))
                        cfg_map_csv[crit] = CriterionConfig(
                            enabled, is_cost, shape, trigger, threshold, slope, exponent
                        )
                    st.session_state.cfg_map = cfg_map_csv

                    st.success(f"Loaded row {row_idx} for **{st.session_state.bridge_id}**")

    # ---------------- Run button (always visible) ---------------
    run_clicked = st.button("Run analysis", type="primary")

# ----------------------  ANALYSIS  -----------------------------
if run_clicked:
    # Verify required inputs
    keys = ["span", "udl", "width", "base_n", "base_d",
            "strength", "density", "n_delta", "cfg_map"]
    if any(st.session_state[k] is None for k in keys):
        st.error("❗ Please complete the input (load CSV or fill manual values) before running.")
    else:
        util_grid   = [0.6,0.7,0.8,0.9,0.95,0.99,1.0]
        dia_factors = np.linspace(-0.5, 0.5, 11)

        df_alts = generate_alternatives(
            st.session_state.span, st.session_state.udl,
            st.session_state.base_n, st.session_state.base_d,
            st.session_state.strength, st.session_state.density,
            st.session_state.width, util_grid, dia_factors,
            st.session_state.n_delta,
        )
        ranked = moora_rank(df_alts.copy(), st.session_state.cfg_map)
        st.session_state["ranked_df"] = ranked
        st.session_state["results_ready"] = True

# --------------------  RESULTS DISPLAY  ------------------------
if st.session_state.get("results_ready"):
    ranked = st.session_state["ranked_df"]
    best = ranked.iloc[0]

    title = "### Preferred alternative"
    if st.session_state.bridge_id:
        title += f" for **{st.session_state.bridge_id}**"
    st.markdown(
        f"{title}  \n"
        f"* Diameter: **{best.Cable_Dia_mm:.1f} mm**  \n"
        f"* Cables : **{int(best.N_Cables)}**  \n"
        f"* Utilisation: **{best.Utilisation:.2f}**  \n"
        f"* MOORA score: **{best.MOORA_Score:.3f}**  \n\n"
        f"**{CREDIT}**"
    )

    param_df = pd.DataFrame({
        "Parameter": ["Span", "UDL", "Bridge width", "Base cables",
                      "Base diameter", "Strength", "Density"],
        "Value"    : [st.session_state.span, st.session_state.udl,
                      st.session_state.width, st.session_state.base_n,
                      st.session_state.base_d, st.session_state.strength,
                      st.session_state.density],
        "Unit"     : ["m", "kN/m", "m", "", "mm", "MPa", "kN/m³"],
    })
    st.table(param_df)

    tab1, tab2, tab3 = st.tabs(
        ["Cable profile & contour", "Parallel plot", "Full table"]
    )

    # --- TAB 1 --------------------------------------------------
    with tab1:
        st.pyplot(cable_profile_fig(st.session_state.span, best.Sag_m))

        vars_list = ["Utilisation", "Cable_Dia_mm", "N_Cables",
                     "NatFreq_Hz", "Sag_m", "Tension_kN", "CableMass_kg"]
        c1, c2, _ = st.columns([3, 3, 1])
        x_sel = c1.selectbox("X variable", vars_list, key="xsel")
        y_sel = c2.selectbox("Y variable", vars_list, index=1, key="ysel")
        colA, colB = st.columns([1, 2])
        gen_single = colA.button("Generate")
        gen_all    = colB.button("Generate All Charts")

        if gen_single:
            if x_sel == y_sel:
                st.warning("Select two different variables.")
            else:
                st.pyplot(contour_fig(ranked, x_sel, y_sel))

        if gen_all:
            st.markdown("### All Contour Plots")
            for i, x in enumerate(vars_list):
                for j, y in enumerate(vars_list):
                    if i >= j:
                        continue
                    st.subheader(f"Contour: {x} vs {y}")
                    st.pyplot(contour_fig(ranked, x, y))

    # --- TAB 2 --------------------------------------------------
    with tab2:
        st.plotly_chart(parallel_fig(ranked), use_container_width=True)

    # --- TAB 3 --------------------------------------------------
    with tab3:
        st.dataframe(ranked)
        st.download_button(
            "Download CSV", ranked.to_csv(index=False).encode(),
            file_name="srb_results.csv", mime="text/csv"
        )
else:
    st.info("Set parameters (manual or CSV) and click **Run analysis**.")
