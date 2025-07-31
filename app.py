#!/usr/bin/env python3
# ================================================================
#  Stress-Ribbon Bridge Cable Selector (MOORA) – Streamlit version
#  - Manual or CSV input
#  - MOORA ranking + plots
#  - One-page A4 PDF export (fpdf2 core fonts, ASCII-safe)
# ----------------------------------------------------------------
#  Authors : Vijaykumar Parmar & Dr. K. B. Parikh   (© 2025)
# ================================================================

import math, os, io, tempfile
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import griddata
import plotly.express as px
import streamlit as st
from fpdf import FPDF    # pre-installed via requirements.txt

CREDIT = "Authors : Vijaykumar Parmar & Dr. K. B. Parikh"

# ---------------------------------------------------------------------------
# Utility: replace Unicode punctuation not supported by core fonts
# ---------------------------------------------------------------------------
def _ascii_safe(text: str) -> str:
    return (text.replace("–", "-")
                .replace("—", "-")
                .replace("‘", "'")
                .replace("’", "'")
                .replace("“", '"')
                .replace("”", '"'))

# ---------------------------------------------------------------------------
# Template CSV (extra leading Bridge Name column)
# ---------------------------------------------------------------------------
_TEMPLATE = {
    "Bridge Name":       "Demo-Bridge",
    "Span L (m)":        80,
    "UDL w (kN/m)":      15,
    "Bridge width (m)":  3,
    "Base Cable Diam":   30,
    "Base Cables":       4,
    "Strength (MPa)":    1860,
    "Density (kN/m3)":   77,
    "Δ cables":          1,
}
if not os.path.exists("template_input.csv"):
    pd.DataFrame([_TEMPLATE]).to_csv("template_input.csv", index=False)

CSV_RANGES = {           # for validation styling
    "Span L (m)"       : (10, 500),
    "UDL w (kN/m)"     : (10, 1000),
    "Bridge width (m)" : (1, 10),
    "Base Cable Diam"  : (5, 300),
    "Base Cables"      : (2, 20),
    "Strength (MPa)"   : (200, 3000),
    "Density (kN/m3)"  : (50, 90),
    "Δ cables"         : (0, 5),
}

# ---------------------------------------------------------------------------
# Criterion configuration
# ---------------------------------------------------------------------------
@dataclass
class CriterionConfig:
    enabled: bool
    is_cost: bool
    shape: str         # 'linear' | 'exponential'
    trigger: str       # 'above' | 'below'
    threshold: float
    slope: float
    exponent: float

DEFAULT_CRIT: Dict[str, CriterionConfig] = {
    "Utilisation"   : CriterionConfig(True, True , "exponential", "below", 0.8 , 1.0, 6.0),
    "Slope_pct"     : CriterionConfig(True, False, "linear"     , "below", 2.5 , 1.0, 1.0),
    "Cable_Dia_mm"  : CriterionConfig(True, True , "linear"     , "above", 150 , 0.5, 1.0),
    "N_Cables"      : CriterionConfig(True, True , "exponential", "above", 5   , 1.0, 1.2),
    "NatFreq_Hz"    : CriterionConfig(True, False, "linear"     , "above", 2.0 , 1.0, 1.0),
    "Tension_kN"    : CriterionConfig(True, True , "linear"     , "above", 0.0 , 1.0, 1.0),
    "Sag_m"         : CriterionConfig(True, False, "exponential", "below", 0.003,1.0,3.0),
}

# ---------------------------------------------------------------------------
# Engineering helpers
# ---------------------------------------------------------------------------
def _area_mm2(d_mm: float) -> float:
    return math.pi * (d_mm/2) ** 2

def cable_metrics(span, udl, n_c, dia, strength, util, density):
    A_mm2 = _area_mm2(dia)
    H_kN  = n_c * A_mm2 * util * strength / 1_000
    sag   = udl * span**2 / (8 * H_kN) if H_kN else 0
    V_kN  = udl * span / 2
    T_kN  = math.hypot(H_kN, V_kN)
    A_m2  = A_mm2 * 1e-6
    rho   = density * 1_000 / 9.81
    mu    = rho * A_m2
    omega = ((H_kN*1_000)/(mu*n_c))**0.5 if mu and n_c else 0
    f_nat = omega/(2*span) if omega else 0
    return {
        "Cable_Dia_mm": dia,
        "Utilisation" : util,
        "N_Cables"    : n_c,
        "Slope_pct"   : sag/span*100,
        "Tension_kN"  : T_kN,
        "Sag_m"       : sag,
        "NatFreq_Hz"  : f_nat,
    }

def _penalty(x, cfg: CriterionConfig) -> float:
    if not cfg.enabled: return 0.0
    diff = (x - cfg.threshold) if cfg.trigger=="above" else (cfg.threshold - x)
    if diff <= 0: return 0.0
    return cfg.slope*diff if cfg.shape=="linear" else math.exp(cfg.exponent*diff)-1

# ---------------------------------------------------------------------------
# Design-space + MOORA
# ---------------------------------------------------------------------------
def generate_alts(span, udl, base_n, base_d, strength, density, width, n_delta):
    util_grid = [0.6,0.7,0.8,0.9,0.95,0.99,1.0]
    dia_factors = np.linspace(-.5, .5, 11)
    alts=[]
    n_opts=[max(2, base_n+i) for i in range(-n_delta, n_delta+1)]
    for fac in dia_factors:
        dia = round(base_d*(1+fac),3)
        if dia < 5: continue
        for util in util_grid:
            for n in n_opts:
                r=cable_metrics(span,udl,n,dia,strength,util,density)
                r["Cable_Spacing_m"]=width/n
                r["UDL_perCable_kNpm"]=udl/n
                alts.append(r)
    return pd.DataFrame(alts).round(6)

def moora_rank(df: pd.DataFrame, cfg_map):
    benefit,cost=[],[]
    for crit,cfg in cfg_map.items():
        if not cfg.enabled: continue
        col=f"PB_{crit}"
        df[col]=df[crit].apply(lambda v:_penalty(v,cfg))
        (cost if cfg.is_cost else benefit).append(col)
    for col in benefit+cost:
        norm=(df[col]**2).sum()**0.5
        df[f"N_{col}"]=df[col]/norm if norm else 0
    df["MOORA_Score"]=df[[f"N_{c}"for c in benefit]].sum(1)-df[[f"N_{c}"for c in cost]].sum(1)
    ranked=df.sort_values("MOORA_Score",ascending=False).reset_index(drop=True)
    ranked.index+=1
    ranked["Rank"]=ranked.index
    return ranked

# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def cable_profile_fig(span,sag) -> Figure:
    x=np.linspace(0,span,200)
    y=-4*sag*(x/span)*(1-x/span)
    fig=Figure(figsize=(6,3))
    ax=fig.add_subplot(111)
    ax.plot(x,y)
    ax.set_xlabel("Span position (m)")
    ax.set_ylabel("Elevation (m, downward)")
    ax.set_title("Cable elevation profile")
    ax.grid(ls="--",alpha=.3)
    fig.text(.5,-.12,CREDIT,ha="center",fontsize=8)
    return fig

# ---------------------------------------------------------------------------
# PDF builder (ASCII-safe strings)
# ---------------------------------------------------------------------------
def build_pdf(inp_df: pd.DataFrame, best: pd.Series,
              ranked: pd.DataFrame, prof_fig: Figure,
              bridge_id: str|None)->bytes:
    pdf=FPDF(format="A4",unit="mm")
    pdf.set_auto_page_break(True,10)
    pdf.add_page()
    pdf.set_font("Helvetica","B",14)
    title="SRB Cable-Selection Report"
    if bridge_id: title+=f" - {bridge_id}"
    pdf.cell(0,10,_ascii_safe(title),ln=1,align="C")

    pdf.set_font("Helvetica","",11)
    pdf.ln(1); pdf.cell(0,7,"Input Parameters:",ln=1)
    for p,v,u in zip(inp_df["Parameter"],inp_df["Value"],inp_df["Unit"]):
        pdf.cell(0,6,_ascii_safe(f" • {p}: {v} {u}"),ln=1)

    pdf.ln(1); pdf.cell(0,7,"Preferred Alternative:",ln=1)
    labels={"Cable_Dia_mm":"Diameter (mm)","N_Cables":"Cables",
            "Utilisation":"Utilisation","Sag_m":"Sag (m)",
            "Slope_pct":"Slope (%)","Tension_kN":"Tension (kN)",
            "MOORA_Score":"MOORA Score"}
    show=["Cable_Dia_mm","N_Cables","Utilisation",
          "Sag_m","Slope_pct","Tension_kN","MOORA_Score"]
    for fld in show:
        pdf.cell(0,6,_ascii_safe(f" • {labels[fld]}: {best[fld]:.3f}"),ln=1)

    with tempfile.NamedTemporaryFile(suffix=".png",delete=False) as tmp:
        prof_fig.savefig(tmp.name,dpi=140,bbox_inches="tight")
        pdf.image(tmp.name,w=180)

    pdf.ln(2); pdf.cell(0,7,"Ranking Table (top 10):",ln=1)
    pdf.set_font("Helvetica","",9)
    for _,r in ranked.head(10).iterrows():
        pdf.cell(0,5,_ascii_safe(
            f"#{int(r['Rank'])} – Dia {r['Cable_Dia_mm']} mm, "
            f"{int(r['N_Cables'])} cables, Score {r['MOORA_Score']:.3f}"),
            ln=1)
    if len(ranked)>10:
        pdf.cell(0,4,"(table truncated)",ln=1)
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config("SRB Cable Selector – MOORA",layout="wide")
st.title("Stress-Ribbon Bridge Cable Selector (MOORA)")

state=st.session_state   # shorthand

# initialise placeholders
for k in ("span","udl","width","base_n","base_d","strength","density",
          "n_delta","bridge_id","ranked_df"):
    state.setdefault(k,None)

# ---------------- Sidebar ----------------------------------------------------
with st.sidebar:
    mode=st.radio("Input mode",["Manual Input","CSV Input"])

    if mode=="Manual Input":
        state.span  = st.number_input("Span L (m)",10.,500.,80.)
        state.udl   = st.number_input("UDL w (kN/m)",10.,1000.,15.)
        state.width = st.number_input("Bridge width (m)",1.,10.,3.)
        state.base_d= st.number_input("Base cable diameter (mm)",5.,300.,30.)
        state.base_n= st.number_input("Base cables",2,20,4)
        state.strength=st.number_input("Strength σ (MPa)",200.,3000.,1860.)
        state.density =st.number_input("Density γ (kN/m³)",50.,90.,77.)
        state.n_delta =st.slider("±Δ cables",0,5,1)
        state.bridge_id=None

    else:  # CSV Input
        with open("template_input.csv","rb") as tf:
            st.download_button("📥 template_input.csv",tf,"template_input.csv","text/csv")
        csv_file=st.file_uploader("Upload CSV",type="csv")
        if csv_file:
            df_csv=pd.read_csv(csv_file)
            def _colour(row):
                return ["background-color:red"
                        if (c in CSV_RANGES and not CSV_RANGES[c][0]<=v<=CSV_RANGES[c][1])
                        else "" for c,v in row.items()]
            st.dataframe(df_csv.style.apply(_colour,axis=1),height=250)
            idx=st.number_input("Row index (≥2)",2,len(df_csv)+1,2,step=1)
            if st.button("Load row"):
                try: row=df_csv.iloc[idx-2]
                except IndexError:
                    st.error("Index out of range")
                else:
                    state.span   = float(row["Span L (m)"])
                    state.udl    = float(row["UDL w (kN/m)"])
                    state.width  = float(row["Bridge width (m)"])
                    state.base_d = float(row["Base Cable Diam"])
                    state.base_n = int(row["Base Cables"])
                    state.strength=float(row["Strength (MPa)"])
                    state.density =float(row["Density (kN/m3)"])
                    state.n_delta =int(row["Δ cables"])
                    state.bridge_id=str(row["Bridge Name"])
                    st.success(f"Row {idx} loaded ({state.bridge_id})")

    run=st.button("Run analysis",type="primary")

# ---------------- Analysis ----------------------------------------------------
if run:
    if None in (state.span,state.udl,state.width,state.base_n,state.base_d,
                state.strength,state.density,state.n_delta):
        st.error("Incomplete inputs.")
    else:
        alts=generate_alts(state.span,state.udl,state.base_n,state.base_d,
                           state.strength,state.density,state.width,state.n_delta)
        state.ranked_df=moora_rank(alts.copy(),DEFAULT_CRIT)

# ---------------- Results -----------------------------------------------------
if state.get("ranked_df") is not None:
    ranked=state.ranked_df
    best=ranked.iloc[0]
    fig=cable_profile_fig(state.span,best.Sag_m)

    hdr="### Preferred alternative"
    if state.bridge_id: hdr+=f" — **{state.bridge_id}**"
    st.markdown(hdr)
    st.markdown(
        f"* Dia **{best.Cable_Dia_mm:.1f} mm** / {int(best.N_Cables)} cables  \n"
        f"* Sag **{best.Sag_m:.3f} m** (Slope {best.Slope_pct:.2f} %)  \n"
        f"* Tension **{best.Tension_kN:.0f} kN**, Util {best.Utilisation:.2f}  \n"
        f"* MOORA score **{best.MOORA_Score:.3f}**"
    )

    inp_df=pd.DataFrame({
        "Parameter":["Span","UDL","Bridge width","Base cables",
                     "Base diameter","Strength","Density"],
        "Value":[state.span,state.udl,state.width,
                 state.base_n,state.base_d,state.strength,state.density],
        "Unit":["m","kN/m","m","","mm","MPa","kN/m³"]
    })

    tabs=st.tabs(["Profile & contour","Parallel","Full table"])
    with tabs[0]:
        st.pyplot(fig)
    with tabs[1]:
        st.plotly_chart(px.parallel_coordinates(
            ranked.head(50),
            dimensions=["Cable_Dia_mm","Utilisation","N_Cables",
                        "Sag_m","Slope_pct","Tension_kN","MOORA_Score"],
            color="MOORA_Score"),use_container_width=True)
    with tabs[2]:
        st.dataframe(ranked)

    pdf_bytes=build_pdf(inp_df,best,ranked,fig,state.bridge_id)
    st.download_button("📄 Download full PDF report",
                       pdf_bytes,"SRB_report.pdf","application/pdf")
else:
    st.info("Enter inputs and click **Run analysis**.")
