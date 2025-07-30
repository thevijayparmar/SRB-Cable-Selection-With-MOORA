#!/usr/bin/env python3
# ================================================================
#  Stress‑Ribbon Bridge Cable Selector  – Streamlit edition
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
#  Licence : MIT‑style – use, modify, share with credit.
# ================================================================

# ---------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------
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
# 2. Penalty / benefit configuration model
# ---------------------------------------------------------------
@dataclass
class CriterionConfig:
    """User‑tunable settings for MOORA penalty / benefit."""
    enabled: bool = True
    is_cost: bool = True           # True = cost, False = benefit
    shape: str = "linear"          # "linear" | "exponential"
    trigger: str = "above"         # "above" | "below" threshold
    threshold: float = 0.0
    slope: float = 1.0
    exponent: float = 1.0

# Default settings
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
# 3. Engineering helper functions
# ---------------------------------------------------------------
def _area_mm2(d_mm: float) -> float:
    """Cross‑sectional area of a round cable (mm²)."""
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
    """Compute structural responses for one alternative."""
    area_mm2 = _area_mm2(dia_mm)
    # Horizontal tension component
    H_kN = n_cables * area_mm2 * utilisation * strength_MPa / 1_000
    # Sag using parabolic approximation
    sag_m = udl_kNpm * span_m ** 2 / (8 * H_kN) if H_kN else 0
    # Vertical reaction
    V_kN = udl_kNpm * span_m / 2
    # Total anchor tension
    T_kN = math.hypot(H_kN, V_kN)
    # Natural frequency (simplified taut‑string expression)
    area_m2 = area_mm2 * 1e-6
    rho_kgpm3 = density_kNpm3 * 1_000 / 9.81
    mu_kgpm = rho_kgpm3 * area_m2
    omega_sq = (H_kN * 1_000) / (mu_kgpm * n_cables) if mu_kgpm and n_cables else 0
    nat_f_Hz = (1 / (2 * span_m)) * math.sqrt(omega_sq) if omega_sq else 0
    # Total cable mass
    mass_kg = mu_kgpm * span_m * n_cables

    return {
        "Cable_Dia_mm": dia_mm,
        "Utilisation" : utilisation,
        "N_Cables"    : n_cables,
        "Slope_pct"   : sag_m / span_m * 100,
        "Tension_kN"  : T_kN,
        "Sag_m"       : sag_m,
        "NatFreq_Hz"  : nat_f_Hz,
        "CableMass_kg": mass_kg,
    }

# ---------------------------------------------------------------
# 4. Penalty / benefit magnitude
# ---------------------------------------------------------------
def _pb_value(x: float, cfg: CriterionConfig) -> float:
    """
    Return non‑negative penalty / benefit magnitude.
    """
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
# 5. Design‑space generation
# ---------------------------------------------------------------
def generate_alternatives(
    span: float,
    udl: float,
    base_n: int,
    base_dia: float,
    strength: float,
    density: float,
    bridge_w: float,
    util_grid: List[float],
    dia_factors: List[float],
    n_delta: int,
) -> pd.DataFrame:
    """Return DataFrame of every candidate configuration."""
    recs = []
    n_options = [max(2, base_n + i) for i in range(-n_delta, n_delta + 1)]
    for fac in dia_factors:
        dia = round(base_dia * (1 + fac), 3)
        if dia < 5:                       # omit unrealistic diameters
            continue
        for util in util_grid:
            for n in n_options:
                row = cable_metrics(span, udl, n, dia, strength, util, density)
                row["Cable_Spacing_m"]   = bridge_w / n
                row["UDL_perCable_kNpm"] = udl / n
                recs.append(row)
    return pd.DataFrame(recs).round(6)

# ---------------------------------------------------------------
# 6. MOORA scoring & ranking
# ---------------------------------------------------------------
def moora_rank(df: pd.DataFrame, cfg_map: Dict[str, CriterionConfig]) -> pd.DataFrame:
    """Add PB columns, normalise, score, and return ranked DataFrame."""
    benefit, cost = [], []
    # Penalty / benefit columns
    for crit, cfg in cfg_map.items():
        if not cfg.enabled:
            continue
        col = f"PB_{crit}"
        df[col] = df[crit].apply(lambda v: _pb_value(v, cfg))
        (cost if cfg.is_cost else benefit).append(col)
    # Vector normalise
    for col in benefit + cost:
        norm = np.sqrt((df[col] ** 2).sum())
        df[f"N_{col}"] = df[col] / norm if norm else 0
    # MOORA score
    df["MOORA_Score"] = (
        df[[f"N_{c}" for c in benefit]].sum(axis=1)
        - df[[f"N_{c}" for c in cost]].sum(axis=1)
    )
    ranked = df.sort_values("MOORA_Score", ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked["Rank"] = ranked.index
    return ranked

# ---------------------------------------------------------------
# 7. Plot helpers
# ---------------------------------------------------------------
def cable_profile_fig(span, sag, label="Cable"):
    """Matplotlib figure of sagging cable (downward)."""
    x = np.linspace(0, span, 200)
    y = -4 * sag * (x / span) * (1 - x / span)  # downward sag
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.plot(x, y, label=label, color="tab:blue")
    ax.set_xlabel("Span position (m)")
    ax.set_ylabel("Elevation (m, downward)")
    ax.set_title("Cable elevation profile")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.text(0.5, -0.1, CREDIT, ha="center", fontsize=8)
    return fig

def contour_fig(df: pd.DataFrame, xvar: str, yvar: str):
    """Matplotlib contour figure for MOORA score."""
    xi = np.linspace(df[xvar].min(), df[xvar].max(), 120)
    yi = np.linspace(df[yvar].min(), df[yvar].max(), 120)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((df[xvar], df[yvar]), df["MOORA_Score"], (Xi, Yi), method="cubic")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    cs = ax.contourf(Xi, Yi, Zi, levels=15, cmap=cm.viridis)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title("MOORA score contour")
    fig.colorbar(cs, ax=ax, label="MOORA Score")
    fig.text(0.5, -0.08, CREDIT, ha="center", fontsize=8)
    return fig

def parallel_fig(df: pd.DataFrame):
    """Plotly parallel‑coordinates of all alternatives."""
    fig = px.parallel_coordinates(
        df,
        dimensions=[
            "Cable_Dia_mm", "Utilisation", "N_Cables", "NatFreq_Hz",
            "Sag_m", "Tension_kN", "CableMass_kg", "MOORA_Score",
        ],
        color="MOORA_Score",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel coordinates – all alternatives",
    )
    fig.add_annotation(
        text=CREDIT, x=0.5, y=-0.12, xref="paper", yref="paper",
        showarrow=False, font=dict(size=10)
    )
    fig.update_layout(font=dict(size=11))
    return fig

# ===============================================================
# 8. Streamlit UI
# ===============================================================
st.set_page_config(
    page_title="SRB Cable Selector – MOORA",
    layout="wide",
)

st.title("Stress‑Ribbon Bridge Cable Selector (MOORA)")

# -------------  Sidebar inputs ---------------------------------
with st.sidebar:
    st.header("Bridge parameters")
    span   = st.number_input("Span L (m)", 10.0, 500.0, 50.0, step=1.0)
    udl    = st.number_input("UDL w (kN/m)", 10.0, 1000.0, 100.0, step=10.0)
    width  = st.number_input("Bridge width (m)", 1.0, 10.0, 3.0, step=0.1)
    base_n = st.number_input("Base number of cables", 2, 20, 2, step=1)
    base_d = st.number_input("Base cable diameter (mm)", 5.0, 300.0, 20.0, step=1.0)
    strength = st.number_input("Cable strength σ (MPa)", 200.0, 3000.0, 1600.0, step=50.0)
    density  = st.number_input("Density γ (kN/m³)", 50.0, 90.0, 77.0, step=1.0)
    n_delta  = st.slider("± range around base #Cables", 0, 5, 1)

    st.markdown("---")
    st.subheader("MOORA criterion settings")

    # Collect criterion UI into dict of configs
    cfg_map: Dict[str, CriterionConfig] = {}
    for name, default_cfg in DEFAULT_CRIT.items():
        with st.expander(name, expanded=False):
            enabled   = st.checkbox("Enabled", value=default_cfg.enabled, key=name+"_en")
            is_cost   = st.radio("Type", ["Cost", "Benefit"],
                                 index=0 if default_cfg.is_cost else 1, key=name+"_type")
            shape     = st.selectbox("Shape", ["linear", "exponential"],
                                     index=0 if default_cfg.shape=="linear" else 1,
                                     key=name+"_shape")
            trigger   = st.selectbox("Trigger", ["above", "below"],
                                     index=0 if default_cfg.trigger=="above" else 1,
                                     key=name+"_trig")
            threshold = st.number_input("Threshold", value=default_cfg.threshold, key=name+"_thr")
            slope     = st.number_input("Slope (linear)", value=default_cfg.slope, key=name+"_slope")
            exponent  = st.number_input("Exponent (exp)", value=default_cfg.exponent, key=name+"_exp")
        cfg_map[name] = CriterionConfig(
            enabled,
            is_cost == "Cost",
            shape,
            trigger,
            threshold,
            slope,
            exponent,
        )

    run_clicked = st.button("Run analysis", type="primary")

# ---------------------------------------------------------------
# 9. Perform analysis on button press
# ---------------------------------------------------------------
if run_clicked:
    # Generate design space
    util_grid   = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    dia_factors = np.linspace(-0.5, 0.5, 11)
    df_alts = generate_alternatives(
        span, udl, base_n, base_d, strength, density,
        width, util_grid, dia_factors, n_delta,
    )

    ranked = moora_rank(df_alts.copy(), cfg_map)

    # Cache for later use (contour plot)
    st.session_state["ranked_df"] = ranked

    best = ranked.iloc[0]

    st.markdown(
        f"### Preferred alternative  \n"
        f"* Diameter: **{best.Cable_Dia_mm:.1f} mm**  \n"
        f"* Cables: **{int(best.N_Cables)}**  \n"
        f"* Utilisation: **{best.Utilisation:.2f}**  \n"
        f"* MOORA score: **{best.MOORA_Score:.3f}**  \n\n"
        f"**{CREDIT}**"
    )

    # Input recap
    recap_df = pd.DataFrame({
        "Parameter": ["Span", "UDL", "Bridge width", "Base cables",
                      "Base diameter", "Strength", "Density"],
        "Value": [span, udl, width, base_n, base_d, strength, density],
        "Unit": ["m", "kN/m", "m", "", "mm", "MPa", "kN/m³"],
    })
    st.table(recap_df)

    # Layout results (tabs)
    tab1, tab2, tab3 = st.tabs(["Cable profile", "Parallel plot", "Full table"])

    with tab1:
        st.pyplot(cable_profile_fig(span, best.Sag_m, "Best alternative"))

        # Contour generator
        st.markdown("#### Contour plot generator")
        vars_for_contour = [
            "Utilisation", "Cable_Dia_mm", "N_Cables", "NatFreq_Hz",
            "Sag_m", "Tension_kN", "CableMass_kg",
        ]
        colx, coly, colbtn = st.columns([3, 3, 1])
        x_var = colx.selectbox("X variable", vars_for_contour, key="xvar")
        y_var = coly.selectbox("Y variable", vars_for_contour, index=1, key="yvar")
        if colbtn.button("Generate contour"):
            if x_var == y_var:
                st.warning("Choose two different variables.")
            else:
                st.pyplot(contour_fig(ranked, x_var, y_var))

    with tab2:
        st.plotly_chart(parallel_fig(ranked), use_container_width=True)

    with tab3:
        st.dataframe(ranked)

        # CSV download
        csv_data = ranked.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="srb_results.csv",
            mime="text/csv",
        )

else:
    st.info("Adjust parameters in the sidebar and click **Run analysis**.")
