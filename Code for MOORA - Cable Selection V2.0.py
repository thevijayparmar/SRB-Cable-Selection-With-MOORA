#!/usr/bin/env python3
# ================================================================
#  Stress‑Ribbon Bridge Cable Selector  – MOORA‑based analysis
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
# ================================================================

# ---------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Dict, List
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from scipy.interpolate import griddata

import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output

# ---------------------------------------------------------------
# 2. Penalty / Benefit configuration
# ---------------------------------------------------------------
@dataclass
class CriterionConfig:
    """Settings controlling penalty / benefit for a criterion."""
    enabled: bool = True            # whether used in MOORA
    is_cost: bool = True            # True = cost, False = benefit
    shape: str = "linear"           # "linear" or "exponential"
    trigger: str = "above"          # "above" or "below"
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
# 3. Engineering helper functions
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
# 4. Penalty / Benefit magnitude
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
# 5. Generate alternatives
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
# 6. MOORA ranking
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
def cable_profile_plot(span, sag, label) -> Figure:
    xs = np.linspace(0, span, 200)
    ys = -4 * sag * (xs / span) * (1 - xs / span)  # downward sag
    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color="tab:blue", label=label)
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
    fig.show()

# ---------------------------------------------------------------
# 8. UI widgets
# ---------------------------------------------------------------
# --- Bridge parameters
span_w    = widgets.FloatText(value=50.0 , description="Span (m)")
udl_w     = widgets.FloatText(value=100.0, description="UDL (kN/m)")
width_w   = widgets.FloatText(value=3.0  , description="Bridge width (m)")
baseN_w   = widgets.IntText  (value=2    , description="Base #Cables")
baseD_w   = widgets.FloatText(value=20.0 , description="Base Ø (mm)")
strength_w= widgets.FloatText(value=1600.0, description="Strength (MPa)")
density_w = widgets.FloatText(value=77.0 , description="Density (kN/m³)")
nDelta_w  = widgets.IntSlider(value=1, min=0, max=5, step=1,
                              description="Δ cables range")

bridge_box = widgets.VBox([
    span_w, udl_w, width_w, baseN_w, baseD_w,
    strength_w, density_w, nDelta_w
])

# --- MOORA criterion controls
crit_controls = {}
for name, cfg in DEFAULT_CRIT.items():
    crit_controls[name] = widgets.Accordion(children=[
        widgets.VBox([
            widgets.Checkbox(value=cfg.enabled, description="Enabled"),
            widgets.RadioButtons(options=["Cost", "Benefit"],
                                 value="Cost" if cfg.is_cost else "Benefit",
                                 description="Type"),
            widgets.Dropdown(options=["linear", "exponential"],
                             value=cfg.shape, description="Shape"),
            widgets.Dropdown(options=["above", "below"],
                             value=cfg.trigger, description="Trigger"),
            widgets.FloatText(value=cfg.threshold, description="Threshold"),
            widgets.FloatText(value=cfg.slope, description="Slope"),
            widgets.FloatText(value=cfg.exponent, description="Exponent"),
        ])
    ])
    crit_controls[name].set_title(0, name)

moora_box = widgets.VBox([crit_controls[k] for k in crit_controls])

# --- Variable selector for contour plot
vars_list = [
    "Utilisation", "Cable_Dia_mm", "N_Cables", "NatFreq_Hz",
    "Sag_m", "Tension_kN", "CableMass_kg"
]
x_var_dd = widgets.Dropdown(options=vars_list, description="X variable")
y_var_dd = widgets.Dropdown(options=vars_list, description="Y variable")
gen_contour_btn = widgets.Button(description="Generate contour")
contour_out = widgets.Output()

# --- Outputs
out_best      = widgets.Output()
out_recap     = widgets.Output()
out_profile   = widgets.Output()
out_parallel  = widgets.Output()
out_table     = widgets.Output()
csv_button    = widgets.Button(description="Download CSV")

# ---------------------------------------------------------------
# 9. Backend logic
# ---------------------------------------------------------------
def read_moora_settings() -> Dict[str, CriterionConfig]:
    """Collect MOORA settings from UI widgets."""
    cfg_map = {}
    for name, acc in crit_controls.items():
        box: widgets.VBox = acc.children[0]  # inner VBox
        enabled   = box.children[0].value
        is_cost   = box.children[1].value == "Cost"
        shape     = box.children[2].value
        trigger   = box.children[3].value
        threshold = box.children[4].value
        slope     = box.children[5].value
        exponent  = box.children[6].value
        cfg_map[name] = CriterionConfig(
            enabled, is_cost, shape, trigger,
            threshold, slope, exponent
        )
    return cfg_map

def run_analysis_clicked(_):
    """Handle Run analysis button."""
    # 1️⃣  Clear previous outputs
    for o in [out_best, out_recap, out_profile, out_parallel,
              out_table, contour_out]:
        o.clear_output()

    # 2️⃣  Read inputs
    span   = span_w.value
    udl    = udl_w.value
    width  = width_w.value
    base_n = baseN_w.value
    base_d = baseD_w.value
    strength = strength_w.value
    density  = density_w.value
    n_delta  = nDelta_w.value

    # 3️⃣  Generate alternatives
    util_grid   = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    dia_factors = np.linspace(-0.5, 0.5, 11)
    df_alts = generate_alternatives(
        span, udl, base_n, base_d, strength, density,
        width, util_grid, dia_factors, n_delta
    )

    # 4️⃣  Apply MOORA
    cfg_map = read_moora_settings()
    ranked = moora_rank(df_alts.copy(), cfg_map)

    # 5️⃣  Save CSV
    ranked.to_csv("srb_results.csv", index=False)

    # 6️⃣  Outputs
    best = ranked.iloc[0]

    with out_best:
        display(Markdown(
            f"## Preferred alternative  \n"
            f"* Diameter: **{best.Cable_Dia_mm:.1f} mm**  \n"
            f"* Cables: **{int(best.N_Cables)}**  \n"
            f"* Utilisation: **{best.Utilisation:.2f}**  \n"
            f"* MOORA score: **{best.MOORA_Score:.3f}**  \n\n"
            f"**{CREDIT}**"
        ))

    with out_recap:
        recap = pd.DataFrame({
            "Parameter": ["Span", "UDL", "Bridge width", "Base cables",
                          "Base diameter", "Strength", "Density"],
            "Value": [span, udl, width, base_n, base_d, strength, density],
            "Unit": ["m", "kN/m", "m", "", "mm", "MPa", "kN/m³"],
        })
        display(recap)

    with out_profile:
        display(cable_profile_plot(span, best.Sag_m, "Best alternative"))

    with out_parallel:
        parallel_plot(ranked)

    with out_table:
        display(ranked)

    # enable CSV button
    csv_button.disabled = False

    # store ranked globally for contour use
    global _RANKED_DF
    _RANKED_DF = ranked

def generate_contour_clicked(_):
    """Plot contour for selected variables."""
    contour_out.clear_output()
    xvar = x_var_dd.value
    yvar = y_var_dd.value
    if xvar == yvar:
        with contour_out:
            print("Choose two different variables.")
        return
    if "_RANKED_DF" not in globals():
        with contour_out:
            print("Run analysis first.")
        return
    with contour_out:
        display(contour_plot(_RANKED_DF, xvar, yvar))

def download_csv_clicked(_):
    """Download CSV in Colab."""
    try:
        from google.colab import files
        files.download("srb_results.csv")
    except Exception as e:
        print("Download failed:", e)

# ---------------------------------------------------------------
# 10. UI layout
# ---------------------------------------------------------------
run_btn = widgets.Button(description="Run analysis", button_style="success")
run_btn.on_click(run_analysis_clicked)
gen_contour_btn.on_click(generate_contour_clicked)
csv_button.on_click(download_csv_clicked)
csv_button.disabled = True

# Layout display
display(Markdown("### Bridge inputs"))
display(bridge_box)
display(Markdown("### MOORA criterion settings"))
display(moora_box)
display(run_btn)

display(out_best, out_recap, out_profile)

# Contour controls
display(widgets.HBox([x_var_dd, y_var_dd, gen_contour_btn]))
display(contour_out)

display(out_parallel, out_table, csv_button)
