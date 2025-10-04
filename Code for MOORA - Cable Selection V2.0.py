#!/usr/bin/env python3
# ================================================================
#  Stress‑Ribbon Bridge Cable Selector  – MOORA‑based analysis
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
#  Note: This version is adapted for Streamlit from the original
#        Jupyter/IPython implementation.
# ================================================================

# ---------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Dict, List
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import plotly.graph_objects as go

# ---------------------------------------------------------------
# 2. Penalty / Benefit configuration
# ---------------------------------------------------------------
@dataclass
class CriterionConfig:
    """Settings controlling penalty / benefit for a criterion."""
    enabled: bool = True
    is_cost: bool = True
    shape: str = "linear"
    trigger: str = "above"
    threshold: float = 0.0
    slope: float = 1.0
    exponent: float = 1.0

# Default settings (pre‑populate UI)
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
# 3. Engineering helper functions (No changes made here)
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
    """Return responses for one alternative."""
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
# 4. Penalty / Benefit magnitude (No changes made here)
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
# 5. Generate alternatives (No changes made here)
# ---------------------------------------------------------------
def generate_alternatives(
    span, udl, base_n, base_dia, strength, density,
    bridge_w, util_grid, dia_factors, n_delta
) -> pd.DataFrame:
    recs = []
    n_options = [max(2, base_n + i) for i in range(-n_delta, n_delta + 1)]
    for fac in dia_factors:
        dia = round(base_dia * (1 + fac), 3)
        if dia < 5:
            continue
        for util in util_grid:
            for n in n_options:
                r = cable_metrics(span, udl, n, dia, strength, util, density)
                r["Cable_Spacing_m"]   = bridge_w / n
                r["UDL_perCable_kNpm"] = udl / n
                recs.append(r)
    return pd.DataFrame(recs).round(6)

# ---------------------------------------------------------------
# 6. MOORA ranking (No changes made here)
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
# 7. Plot helpers
# ---------------------------------------------------------------
def cable_profile_plot(span, sag, label, equal_scale=False) -> Figure:
    xs = np.linspace(0, span, 200)
    ys = -4 * sag * (xs / span) * (1 - xs / span)
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.margins(0)
    ax.plot(xs, ys, color="tab:blue", label=label)
    if equal_scale:
        ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Span position (m)")
    ax.set_ylabel("Elevation (m, downward)")
    ax.set_title("Cable elevation profile")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.text(0.5, -0.1, CREDIT, ha="center", fontsize=8)
    return fig

def contour_plot(df: pd.DataFrame, xvar: str, yvar: str) -> Figure:
    xi = np.linspace(df[xvar].min(), df[xvar].max(), 140)
    yi = np.linspace(df[yvar].min(), df[yvar].max(), 140)
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

def parallel_plot(df: pd.DataFrame):
    """
    CORRECTED: This function is completely rewritten to use a single, stable
    Parcoords trace with a custom colorscale. This fixes the ValueError.
    """
    top10 = df.head(10).copy()
    if top10.empty:
        return go.Figure() # Return empty figure if no data

    dims = [
        "Cable_Dia_mm", "Utilisation", "N_Cables", "NatFreq_Hz",
        "Sag_m", "Tension_kN", "CableMass_kg", "MOORA_Score",
    ]

    # Map ranks to a numeric value for the colorscale
    def map_rank_to_color_val(rank):
        if rank == 1: return 1.0
        if rank == 2: return 0.8
        if rank == 3: return 0.6
        return 0.1 # Ranks 4-10

    color_vals = top10['Rank'].apply(map_rank_to_color_val)

    # Define the colorscale: numeric value -> color
    custom_colorscale = [
        [0.0, '#D3D3D3'], [0.1, '#D3D3D3'], # Ranks 4-10 -> Grey
        [0.11, 'blue'],   [0.6, 'blue'],     # Rank 3 -> Blue
        [0.61, 'green'],  [0.8, 'green'],    # Rank 2 -> Green
        [0.81, 'yellow'], [1.0, 'yellow'],   # Rank 1 -> Yellow
    ]

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_vals,
            colorscale=custom_colorscale,
            showscale=False, # Hide the color bar
        ),
        dimensions=[dict(
            label=col,
            values=top10[col]
        ) for col in dims]
    ))
            
    fig.update_layout(
        title="Parallel coordinates – top 10 alternatives",
        font=dict(size=12)
    )
    fig.add_annotation(
        text=CREDIT, x=0.5, y=-0.12, xref="paper", yref="paper",
        showarrow=False, font=dict(size=10)
    )
    return fig

# ---------------------------------------------------------------
# 8. UI layout and Backend Logic (Rewritten for Streamlit)
# ---------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Stress-Ribbon Bridge Cable Selector")

# Use session state to store results between reruns
if 'ranked_df' not in st.session_state:
    st.session_state.ranked_df = None

# --- Sidebar for inputs ---
st.sidebar.header("Bridge Inputs")
span_w = st.sidebar.number_input("Span (m)", value=50.0)
udl_w = st.sidebar.number_input("UDL (kN/m)", value=100.0)
width_w = st.sidebar.number_input("Bridge width (m)", value=3.0)
baseN_w = st.sidebar.number_input("Base #Cables", value=2, step=1)
baseD_w = st.sidebar.number_input("Base Ø (mm)", value=20.0)
strength_w = st.sidebar.number_input("Strength (MPa)", value=1600.0)
density_w = st.sidebar.number_input("Density (kN/m³)", value=77.0)
nDelta_w = st.sidebar.slider("Δ cables range", 0, 5, 1)

st.sidebar.header("MOORA Criterion Settings")
cfg_map = {}
for name, cfg in DEFAULT_CRIT.items():
    with st.sidebar.expander(name):
        enabled = st.checkbox("Enabled", value=cfg.enabled, key=f"{name}_enabled")
        is_cost = st.radio("Type", ["Cost", "Benefit"], index=0 if cfg.is_cost else 1, key=f"{name}_type") == "Cost"
        shape = st.selectbox("Shape", ["linear", "exponential"], index=["linear", "exponential"].index(cfg.shape), key=f"{name}_shape")
        trigger = st.selectbox("Trigger", ["above", "below"], index=["above", "below"].index(cfg.trigger), key=f"{name}_trigger")
        threshold = st.number_input("Threshold", value=cfg.threshold, key=f"{name}_threshold")
        slope = st.number_input("Slope", value=cfg.slope, key=f"{name}_slope")
        exponent = st.number_input("Exponent", value=cfg.exponent, key=f"{name}_exponent")
        cfg_map[name] = CriterionConfig(enabled, is_cost, shape, trigger, threshold, slope, exponent)

if st.sidebar.button("Run Analysis", type="primary"):
    # Generate alternatives
    util_grid = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    dia_factors = np.linspace(-0.5, 0.5, 11)
    df_alts = generate_alternatives(
        span_w, udl_w, baseN_w, baseD_w, strength_w, density_w,
        width_w, util_grid, dia_factors, nDelta_w
    )
    # Apply MOORA and store in session state
    st.session_state.ranked_df = moora_rank(df_alts.copy(), cfg_map)

# --- Main page for outputs ---
if st.session_state.ranked_df is not None:
    ranked = st.session_state.ranked_df
    best = ranked.iloc[0]

    st.header("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preferred Alternative")
        st.markdown(
            f"* **Diameter**: {best.Cable_Dia_mm:.1f} mm\n"
            f"* **Cables**: {int(best.N_Cables)}\n"
            f"* **Utilisation**: {best.Utilisation:.2f}\n"
            f"* **MOORA score**: {best.MOORA_Score:.3f}"
        )
        st.caption(CREDIT)
        
        # Add download button for the results
        csv = ranked.to_csv(index=False).encode('utf-8')
        st.download_button(
             label="Download Results as CSV",
             data=csv,
             file_name="srb_results.csv",
             mime="text/csv",
         )

    with col2:
        st.subheader("Cable Elevation Profile")
        profile_fig = cable_profile_plot(span_w, best.Sag_m, "Best alternative")
        st.pyplot(profile_fig)

    st.header("Design Space Exploration")
    
    st.subheader("Parallel Coordinates Plot")
    st.info("Visualizing trade-offs for the top 10 alternatives.")
    parallel_fig = parallel_plot(ranked)
    st.plotly_chart(parallel_fig, use_container_width=True)

    st.subheader("MOORA Score Contour Plot")
    vars_list = ["Utilisation", "Cable_Dia_mm", "N_Cables", "NatFreq_Hz", "Sag_m", "Tension_kN", "CableMass_kg"]
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        x_var = st.selectbox("X variable", vars_list, index=0)
    with c_col2:
        y_var = st.selectbox("Y variable", vars_list, index=1)
    
    if x_var == y_var:
        st.warning("Please choose two different variables for the contour plot.")
    else:
        contour_fig = contour_plot(ranked, x_var, y_var)
        st.pyplot(contour_fig)

    st.header("Full Ranking Table")
    st.dataframe(ranked)

else:
    st.info("👈 Configure inputs in the sidebar and click 'Run Analysis' to see the results.")
