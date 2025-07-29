"""
Stress‑Ribbon Bridge – MOORA Profiler (Streamlit Edition)
© Vijaykumar Parmar & Dr. K. B. Parikh, 2025
----------------------------------------------------------
Run locally:  streamlit run app.py
Deployed:     Streamlit Cloud will autoload this file.
"""

# 🛠️ Imports (same gang, plus streamlit)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as spi
import streamlit as st
from pandas.plotting import parallel_coordinates
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa – 3‑D side‑effects

# ----------------------------------------------------------
# 🧠 Core helpers – identical algebra, wrapped in functions
# ----------------------------------------------------------
def cable_area_mm2(d_mm):
    return math.pi * (d_mm / 2) ** 2

def generate_alternatives(
    span, udl, n_cables, sigma, base_dia, gamma_kN_m3
):
    dia_factors = np.array([-0.50,-0.40,-0.30,-0.20,-0.10,
                              0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
    util_factors = np.array([0.60,0.70,0.80,0.90,0.95,0.99,1.00])
    rho = gamma_kN_m3 * 1_000 / 9.81  # kg/m³

    rows = []
    for f in dia_factors:
        dia = round(base_dia * (1 + f), 3)
        if dia < 10:           # sanity guard
            continue
        A_mm = cable_area_mm2(dia)
        A_m  = A_mm * 1e-6
        for util in util_factors:
            H_kN = n_cables * A_mm * sigma * util / 1_000
            H_N  = H_kN * 1_000
            sag  = udl * span**2 / (8 * H_kN)
            V_kN = udl * span / 2
            T_kN = math.hypot(H_kN, V_kN)
            mu   = rho * A_m
            mass = mu * span * n_cables
            f1   = (1 / (2 * span)) * math.sqrt(H_N / (mu * n_cables))
            rows.append(
                dict(
                    Cable_Dia_mm=dia,
                    Utilisation=util,
                    Slope_pct=sag / span * 100,
                    Tension_kN=T_kN,
                    CableMass_kg=mass,
                    NatFreq_Hz=f1,
                )
            )
    return pd.DataFrame(rows).round(3)

def moora_score(df):
    """Vector‑normalised MOORA with NatFreq as a straight benefit."""
    crit = {
        "Slope_pct": False,
        "Tension_kN": False,
        "CableMass_kg": False,
        "Cable_Dia_mm": False,
        "NatFreq_Hz": True,
    }
    for c in crit:
        norm = np.sqrt((df[c] ** 2).sum())
        df[f"N_{c}"] = 0 if norm == 0 else df[c] / norm
    benefit = [f"N_{c}" for c, b in crit.items() if b]
    cost    = [f"N_{c}" for c, b in crit.items() if not b]
    df["MOORA_Score"] = df[benefit].sum(axis=1) - df[cost].sum(axis=1)
    return df.round(3)

# ------------- Streamlit UI starts here -------------------------------------
st.set_page_config(page_title="SRB MOORA Profiler", layout="wide")
st.title("Stress‑Ribbon Bridge – MOORA Profiler 🏗️")

with st.sidebar:
    st.header("🔧 Input parameters")
    span      = st.number_input("Span L (m)",          10.0, 500.0, 50.0, 5.0)
    udl       = st.number_input("UDL w (kN/m)",        1.0,  50.0, 5.0,  0.5)
    n_cables  = st.slider("No. of Cables n",           1,    6,    2)
    spacing   = st.number_input("Cable Spacing (m)",   0.2,  5.0,  1.0, 0.1)
    sigma     = st.number_input("Cable Strength σ (MPa)", 500, 2500, 1600, 50)
    base_dia  = st.number_input("Base Cable Ø (mm)",   6.0,  60.0, 20.0, 0.5)
    gamma     = st.number_input("Cable Density γ (kN/m³)", 50.0, 90.0, 77.0, 1.0)
    st.caption("*(Strength & density defaults = typical high‑strength steel)*")

# 🏃‍♂️ Crunch numbers (cached because math never changes unless inputs do)
@st.cache_data(show_spinner=False)
def run_model(span, udl, n_cables, spacing, sigma, base_dia, gamma):
    df = generate_alternatives(span, udl, n_cables, sigma, base_dia, gamma)
    df = moora_score(df)
    df = df.sort_values("MOORA_Score", ascending=False).reset_index(drop=True)
    df.index += 1
    df["Rank"] = df.index
    return df

df_ranked = run_model(span, udl, n_cables, spacing, sigma, base_dia, gamma)

# 📊 Tabs for tasty visuals
tab1, tab2, tab3, tab4 = st.tabs(
    ["Table", "2‑D Contours", "3‑D Surface", "Parallel Coords"]
)

with tab1:
    st.subheader("🏆 Ranked alternatives (top 20)")
    st.dataframe(df_ranked.head(20), height=400, use_container_width=True)

def contour_plot(x, y, z, title, clabel, cmap="viridis", overlay=None):
    xi = np.linspace(x.min(), x.max(), 140)
    yi = np.linspace(y.min(), y.max(), 140)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = spi.griddata((x, y), z, (Xi, Yi), method="cubic")
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(Xi, Yi, Zi, 15, cmap=cmap)
    ax.contour(Xi, Yi, Zi, 15, colors="k", linewidths=0.25)
    if overlay:
        for val, style in overlay:
            ax.contour(Xi, Yi, Zi, [val], colors="w", linestyles=style, linewidths=1)
    ax.set_xlabel("Cable Ø (mm)")
    ax.set_ylabel("Utilisation")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=clabel)
    st.pyplot(fig)

with tab2:
    st.subheader("Filled contour maps (Ø × Utilisation)")
    col1, col2, col3 = st.columns(3)
    with col1:
        contour_plot(df_ranked["Cable_Dia_mm"], df_ranked["Utilisation"],
                     df_ranked["MOORA_Score"], "MOORA Score", "Score")
    with col2:
        contour_plot(df_ranked["Cable_Dia_mm"], df_ranked["Utilisation"],
                     df_ranked["CableMass_kg"], "Cable Mass (kg)", "kg", cmap="magma_r")
    with col3:
        contour_plot(df_ranked["Cable_Dia_mm"], df_ranked["Utilisation"],
                     df_ranked["NatFreq_Hz"], "Natural Frequency (Hz)", "Hz",
                     cmap="plasma", overlay=[(3,"--"),(5,":")])

with tab3:
    st.subheader("3‑D Surface (NatFreq; colour = Slope)")
    xi = np.linspace(df_ranked["Cable_Dia_mm"].min(), df_ranked["Cable_Dia_mm"].max(), 60)
    yi = np.linspace(df_ranked["Utilisation"].min(), df_ranked["Utilisation"].max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = spi.griddata((df_ranked["Cable_Dia_mm"], df_ranked["Utilisation"]),
                      df_ranked["NatFreq_Hz"], (Xi, Yi), method="cubic")
    Si = spi.griddata((df_ranked["Cable_Dia_mm"], df_ranked["Utilisation"]),
                      df_ranked["Slope_pct"], (Xi, Yi), method="cubic")

    norm = mcolors.Normalize(vmin=df_ranked["Slope_pct"].min(),
                             vmax=df_ranked["Slope_pct"].max())
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xi, Yi, Zi, facecolors=cm.RdYlGn_r(norm(Si)),
                    rstride=1, cstride=1, antialiased=False)
    ax.contour(Xi, Yi, Zi, zdir="z", offset=Zi.min()-0.5,
               cmap="plasma", levels=10, linewidths=0.5)
    ax.set_xlabel("Cable Ø (mm)"); ax.set_ylabel("Utilisation"); ax.set_zlabel("f₁ (Hz)")
    ax.set_title("NatFreq surface (colour = deck slope)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdYlGn_r"),
                 ax=ax, shrink=0.6, label="Slope %")
    st.pyplot(fig)

with tab4:
    st.subheader("Parallel‑coordinates of top 10 (cost axes inverted)")
    pc = df_ranked.head(10).copy()
    pc["InvSlope"]   = -pc["Slope_pct"]
    pc["InvTension"] = -pc["Tension_kN"]
    pc["InvMass"]    = -pc["CableMass_kg"]
    pc["InvØ"]       = -pc["Cable_Dia_mm"]
    plot_cols = ["InvSlope","InvTension","InvMass","InvØ","NatFreq_Hz","MOORA_Score"]
    pc_norm = pc[plot_cols].apply(lambda s:(s - s.min())/(s.max() - s.min()))
    pc_norm["Alt"] = "A"+pc_norm.index.astype(str)
    fig, ax = plt.subplots(figsize=(10,4))
    parallel_coordinates(pc_norm, "Alt", colormap="viridis", linewidth=2, ax=ax)
    ax.set_ylabel("Normalised (0‑1)"); ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    st.pyplot(fig)

# 🎉 Footer summary
best = df_ranked.iloc[0]
st.success(
    f"**Best design:** Ø {best.Cable_Dia_mm:.1f} mm, util {best.Utilisation:.2f}, "
    f"slope {best.Slope_pct:.2f} %, tension {best.Tension_kN:.0f} kN, "
    f"mass {best.CableMass_kg:.0f} kg, f₁ {best.NatFreq_Hz:.2f} Hz "
    f"(MOORA {best.MOORA_Score:.3f})"
)
st.caption("If result does not suits you; try with different inputs.")
