#!/usr/bin/env python3
# ================================================================
#  Stress‑Ribbon Bridge Cable Selector  – Streamlit edition
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
# ================================================================

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# Penalty / benefit configuration model
# ---------------------------------------------------------------
@dataclass
class CriterionConfig:
    enabled: bool = True
    is_cost: bool = True          # True = cost, False = benefit
    shape: str = "linear"         # "linear" | "exponential"
    trigger: str = "above"        # "above" | "below"
    threshold: float = 0.0
    slope: float = 1.0
    exponent: float = 1.0

DEFAULT_CRIT: Dict[str, CriterionConfig] = {
    "Utilisation"   : CriterionConfig(True, True , "exponential", "below", 0.8  , slope=1.0 , exponent=6.0),
    "Slope_pct"     : CriterionConfig(True, True , "linear"     , "above", 2.5  , slope=1.0 ),
    "Cable_Dia_mm"  : CriterionConfig(True, True , "linear"     , "above", 150  , slope=0.5 ),
    "N_Cables"      : CriterionConfig(True, True , "exponential", "above", 5    , exponent=1.2),
    "NatFreq_Hz"    : CriterionConfig(True, False, "linear"     , "above", 2.0  , slope=1.0 ),
    "Tension_kN"    : CriterionConfig(True, True , "linear"     , "above", 0.0  , slope=1.0 ),
    "Sag_m"         : CriterionConfig(True, True , "exponential", "below", 0.003, exponent=3.0),
}

CREDIT = "Authors : Vijaykumar Parmar & Dr. K. B. Parikh"

# ---------------------------------------------------------------
# Engineering helper functions
# ---------------------------------------------------------------
def _area_mm2(d_mm: float) -> float:
    return math.pi * (d_mm / 2) ** 2

def cable_metrics(
    span_m: float,
    udl_kNpm: float,
    n_cables: int,
    dia_mm: float,
    strength_MPa: float,
    utilisation: float,
    density_kNpm3: float,
) -> Dict[str, float]:
    area_mm2 = _area_mm2(dia_mm)
    H_kN = n_cables * area_mm2 * utilisation * strength_MPa / 1_000
    sag_m = udl_kNpm * span_m ** 2 / (8 * H_kN) if H_kN else 0
    V_kN  = udl_kNpm * span_m / 2
    T_kN  = math.hypot(H_kN, V_kN)
    area_m2 = area_mm2 * 1e-6
    rho = density_kNpm3 * 1_000 / 9.81
    mu_kgpm = rho * area_m2
    omega2 = (H_kN * 1_000) / (mu_kgpm * n_cables) if mu_kgpm and n_cables else 0
    nat_f = (1 / (2 * span_m)) * math.sqrt(omega2) if omega2 else 0
    mass_kg = mu_kgpm * span_m * n_cables
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

# ---------------------------------------------------------------
# Penalty / benefit magnitude
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Design‑space generation
# ---------------------------------------------------------------
def generate_alternatives(
    span, udl, base_n, base_dia, strength, density,
    bridge_w, util_grid, dia_factors, n_delta,
) -> pd.DataFrame:
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

# ---------------------------------------------------------------
# MOORA ranking
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------
def cable_profile_fig(span, sag) -> Figure:
    x = np.linspace(0, span, 200)
    y = -4 * sag * (x / span) * (1 - x / span)
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.plot(x, y, color="tab:blue")
    ax.set_xlabel("Span position (m)")
    ax.set_ylabel("Elevation (m, downward)")
    ax.set_title("Cable elevation profile")
    ax.grid(alpha=0.3, linestyle="--")
    fig.text(0.5, -0.1, CREDIT, ha="center", fontsize=8)
    return fig

# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------
def contour_fig(df: pd.DataFrame, xvar: str, yvar: str) -> Figure:
    """
    Draw a MOORA‑score contour plot for any X/Y variable pair.
    Uses a 7‑colour custom map:
        Black → Dark Red → Purple → Blue → Sky Blue → Light Green → Yellow
    Shows integer ticks if Y‑axis is N_Cables.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Grid for interpolation -------------------------------------------------
    xi = np.linspace(df[xvar].min(), df[xvar].max(), 120)

    if yvar == "N_Cables":                       # keep cable count integer
        yi = np.array(sorted(df[yvar].unique()))
    else:
        yi = np.linspace(df[yvar].min(), df[yvar].max(), 120)

    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(
        (df[xvar], df[yvar]),
        df["MOORA_Score"],
        (Xi, Yi),
        method="cubic",
    )

    # 7‑colour vivid map ------------------------------------------------------
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_moora",
        ["black", "#8B0000", "purple", "blue",
         "skyblue", "lightgreen", "yellow"],
        N=256,
    )

    # Plot --------------------------------------------------------------------
    fig = Figure(figsize=(6, 4))
    ax  = fig.add_subplot(111)

    cs = ax.contourf(Xi, Yi, Zi, levels=15, cmap=custom_cmap)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title("MOORA score contour")

    fig.colorbar(cs, ax=ax, label="MOORA Score")
    fig.text(0.5, -0.08, CREDIT, ha="center", fontsize=8)

    # Force integer ticks for N_Cables
    if yvar == "N_Cables":
        ax.set_yticks(yi)

    return fig


def parallel_fig(df: pd.DataFrame):
    top50 = df.head(50)               # limit to best 50
    fig = px.parallel_coordinates(
        top50,
        dimensions=[
            "Cable_Dia_mm", "Utilisation", "N_Cables", "NatFreq_Hz",
            "Sag_m", "Tension_kN", "CableMass_kg", "MOORA_Score",
        ],
        color="MOORA_Score",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel coordinates – top 50 alternatives",
    )
    fig.add_annotation(
        text=CREDIT, x=0.5, y=-0.12, xref="paper", yref="paper",
        showarrow=False, font=dict(size=10)
    )
    fig.update_layout(font=dict(size=11))
    return fig

# ===============================================================
# Streamlit UI
# ===============================================================
st.set_page_config("SRB Cable Selector – MOORA", layout="wide")
st.title("Stress‑Ribbon Bridge Cable Selector (MOORA)")

# -------------------------------- Sidebar inputs ---------------
with st.sidebar:
    st.header("Bridge parameters")
    span   = st.number_input("Span L (m)", 10.0, 500.0, 50.0)
    udl    = st.number_input("UDL w (kN/m)", 10.0, 1000.0, 100.0)
    width  = st.number_input("Bridge width (m)", 1.0, 10.0, 3.0)
    base_n = st.number_input("Base number of cables", 2, 20, 2)
    base_d = st.number_input("Base cable diameter (mm)", 5.0, 300.0, 20.0)
    strength = st.number_input("Cable strength σ (MPa)", 200.0, 3000.0, 1600.0)
    density  = st.number_input("Density γ (kN/m³)", 50.0, 90.0, 77.0)
    n_delta  = st.slider("± range around base #Cables", 0, 5, 1)

    st.markdown("---")
    st.subheader("MOORA criterion settings")
    cfg_map: Dict[str, CriterionConfig] = {}
    for name, cfg in DEFAULT_CRIT.items():
        with st.expander(name, expanded=False):
            en = st.checkbox("Enabled", value=cfg.enabled, key=name+"en")
            ct = st.radio("Type", ["Cost", "Benefit"], 0 if cfg.is_cost else 1, key=name+"ct")
            sh = st.selectbox("Shape", ["linear", "exponential"], 0 if cfg.shape=="linear" else 1, key=name+"sh")
            tr = st.selectbox("Trigger", ["above", "below"], 0 if cfg.trigger=="above" else 1, key=name+"tr")
            th = st.number_input("Threshold", value=cfg.threshold, key=name+"th")
            sl = st.number_input("Slope (linear)", value=cfg.slope, key=name+"sl")
            ex = st.number_input("Exponent (exp)", value=cfg.exponent, key=name+"ex")
        cfg_map[name] = CriterionConfig(en, ct=="Cost", sh, tr, th, sl, ex)

    run_clicked = st.button("Run analysis", type="primary")

# ---------------------------------------------------------------
# Run analysis & store results in session_state
# ---------------------------------------------------------------
if run_clicked:
    util_grid   = [0.6,0.7,0.8,0.9,0.95,0.99,1.0]
    dia_factors = np.linspace(-0.5,0.5,11)
    df_alts = generate_alternatives(
        span, udl, base_n, base_d, strength, density,
        width, util_grid, dia_factors, n_delta,
    )
    ranked = moora_rank(df_alts.copy(), cfg_map)
    st.session_state["ranked_df"] = ranked
    st.session_state["results_ready"] = True

# ---------------------------------------------------------------
# Display results if available
# ---------------------------------------------------------------
if st.session_state.get("results_ready"):
    ranked = st.session_state["ranked_df"]
    best = ranked.iloc[0]

    st.markdown(
        f"### Preferred alternative  \n"
        f"* Diameter: **{best.Cable_Dia_mm:.1f} mm**  \n"
        f"* Cables: **{int(best.N_Cables)}**  \n"
        f"* Utilisation: **{best.Utilisation:.2f}**  \n"
        f"* MOORA score: **{best.MOORA_Score:.3f}**  \n\n"
        f"**{CREDIT}**"
    )

    recap_df = pd.DataFrame({
        "Parameter": ["Span", "UDL", "Bridge width", "Base cables",
                      "Base diameter", "Strength", "Density"],
        "Value": [span, udl, width, base_n, base_d, strength, density],
        "Unit": ["m", "kN/m", "m", "", "mm", "MPa", "kN/m³"],
    })
    st.table(recap_df)

    tab1, tab2, tab3 = st.tabs(["Cable profile & contour", "Parallel plot", "Full table"])

    with tab1:
        st.pyplot(cable_profile_fig(span, best.Sag_m))

        st.markdown("#### Contour plot generator")
        vars_list = [
            "Utilisation", "Cable_Dia_mm", "N_Cables", "NatFreq_Hz",
            "Sag_m", "Tension_kN", "CableMass_kg",
        ]
        c1, c2, c3 = st.columns([3, 3, 1])
        x_sel = c1.selectbox("X variable", vars_list, key="xsel")
        y_sel = c2.selectbox("Y variable", vars_list, index=1, key="ysel")

        colA, colB = st.columns([1, 2])
        gen_single = colA.button("Generate")
        gen_all = colB.button("Generate All Charts")

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
                        continue  # avoid duplicates and same-variable pairs
                    st.subheader(f"Contour: {x} vs {y}")
                    st.pyplot(contour_fig(ranked, x, y))


    with tab2:
        st.plotly_chart(parallel_fig(ranked), use_container_width=True)

    with tab3:
        st.dataframe(ranked)
        st.download_button(
            "Download CSV",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name="srb_results.csv",
            mime="text/csv",
        )
else:
    st.info("Set parameters in the sidebar and click **Run analysis**.")
