# =============================================================
#  Stress‑Ribbon Bridge – MOORA Profiler 🏗️🎓  (v4.0)
#  © Vijaykumar Parmar & Dr. K. B. Parikh, 2025
#  -------------------------------------------------------------
#  This notebook auto‑spawns 77 bridge variants, ranks them by
#  multi‑criteria MOORA goodness, and doodles a few snazzy plots
#  — all while cracking the occasional dad‑level engineering joke.
# =============================================================

# %% 🛠️ Imports (a.k.a. the usual suspects)
import math, textwrap, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D           # noqa: imported for side‑effects
from matplotlib import cm, colors as mcolors
from pandas.plotting import parallel_coordinates
from scipy.interpolate import griddata
from IPython.display import clear_output, display
%matplotlib inline

# %% 📥 Interactive inputs — ask the human, blame the human  ------------------
def ask_f(prompt, default):
    """One‑liner input helper; returns float or the default."""
    txt = input(f"{prompt} [default {default}]: ").strip()
    if not txt: return default
    try:  return float(txt)
    except ValueError:
        print("⚠️  That wasn’t a number — sticking with the default.")
        return default

print(textwrap.dedent("""
   🔧  Feed me some bridge parameters (press Enter to coast on defaults):
   ---------------------------------------------------------------------
"""))
SPAN      = ask_f("Span  L  (m)",                     50.0)
UDL       = ask_f("UDL   w  (kN/m)",                   5.0)
N_CABLES  = int(ask_f("No. of Cables  n",              2))
SPACING   = ask_f("Cable Spacing  (m)",                1.0)
SIGMA_A   = ask_f("Cable Strength σ (MPa)",         1600.0)
BASE_DIA  = ask_f("Desired Cable Ø (mm)",             20.0)
GAMMA     = ask_f("Cable Density γ (kN/m³)",          77.0)  # typical steel
clear_output()

print("📋 **Input recap**  (Because forgetting is human.)")
for lab,val,u in [("Span L",SPAN,"m"),("UDL w",UDL,"kN/m"),
                  ("Cables",N_CABLES,""),("Spacing",SPACING,"m"),
                  ("Strength σ",SIGMA_A,"MPa"),
                  ("Base Ø",BASE_DIA,"mm"),("Density γ",GAMMA,"kN/m³")]:
    print(f"   {lab:<13}: {val:.3f} {u}")
print()

# %% 🔢 Design grid — 11 diameters × 7 utilisations = 77 alts  -----------------
DIA_FACT  = np.array([-0.50,-0.40,-0.30,-0.20,-0.10,
                       0.00, 0.10, 0.20, 0.30, 0.40, 0.50])
UTIL_FACT = np.array([0.60,0.70,0.80,0.90,0.95,0.99,1.00])

area = lambda d: math.pi*(d/2)**2                  # mm² (circle 101)
kN2N = 1_000.0
rho  = GAMMA * 1_000/9.81                          # kg/m³ (γ→ρ cheat code)

def metrics(L,w,n,dia,sigma,util):
    """Crunch sag, tension, mass, nat‑freq, etc.  No fancy FEM, just old‑school."""
    A_mm = area(dia); A_m = A_mm * 1e-6            # mm² → m²
    H_kN = n * A_mm * sigma * util / 1_000         # horizontal comp. (kN)
    H_N  = H_kN * kN2N
    sag  = w * L**2 / (8 * H_kN)                   # parabolic sag ≈ fine
    V_kN = w * L / 2
    T_kN = math.hypot(H_kN, V_kN)                  # full tension at support
    mu   = rho * A_m                               # kg/m per cable
    mass = mu * L * n                              # kg, all cables
    f1   = (1/(2*L)) * math.sqrt(H_N / (mu * n))   # Hz, small‑sag string
    return {"Cable_Dia_mm":dia,"Utilisation":util,
            "Slope_pct":sag/L*100,"Tension_kN":T_kN,
            "CableMass_kg":mass,"NatFreq_Hz":f1}

records=[]
for f in DIA_FACT:
    dia = round(BASE_DIA * (1 + f), 3)
    if dia < 10: continue                          # reality checkpoint
    for util in UTIL_FACT:
        records.append(metrics(SPAN,UDL,N_CABLES,dia,SIGMA_A,util))

df = pd.DataFrame(records).round(3)

# %% 🧮 MOORA scoring — like GPA but for bridges  ------------------------------
CRIT = {                      # True = Benefit, False = Cost
    "Slope_pct"    : False,   # flatter decks = yay
    "Tension_kN"   : False,
    "CableMass_kg" : False,
    "Cable_Dia_mm" : False,
    "NatFreq_Hz"   : True     # higher vibration freq = sturdier vibes
}

for col in CRIT:
    vec = np.sqrt((df[col]**2).sum())
    df[f"N_{col}"] = 0 if vec==0 else df[col]/vec   # safe divide by hero‑0

benefit = [f"N_{c}" for c,b in CRIT.items() if b]
cost    = [f"N_{c}" for c,b in CRIT.items() if not b]
df["MOORA_Score"] = (df[benefit].sum(axis=1) - df[cost].sum(axis=1)).round(3)

ranked = df.sort_values("MOORA_Score", ascending=False).reset_index(drop=True)
ranked.index += 1; ranked["Rank"] = ranked.index   # humans start at 1

display(ranked.head(15))                           # top‑15 brag board
ranked.to_csv("srb_moora_results.csv", index=False)

# %% 🖼️ Contour helper — because copy‑pasting code is soooo 2024  --------------
def contour2d(x,y,z,title,xlab,ylab,clab,cmap='viridis',levels=15,
              overlay=None, annotate=None):
    xi, yi = np.linspace(x.min(),x.max(),140), np.linspace(y.min(),y.max(),140)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x,y), z, (Xi,Yi), method='cubic')
    plt.figure(figsize=(10,6))
    cs = plt.contourf(Xi,Yi,Zi,levels,cmap=cmap)
    plt.contour(Xi,Yi,Zi,levels,colors='k',linewidths=.25)
    plt.colorbar(cs,label=clab)
    if overlay:
        for val,style in overlay:
            plt.contour(Xi,Yi,Zi,[val],colors='w',linestyles=style,linewidths=1.1)
    if annotate:
        for (x0,y0,txt) in annotate: plt.text(x0,y0,txt,fontsize=8)
    plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab)
    plt.text(0.99,0.01,"© Vijaykumar Parmar & Dr. K. B. Parikh",
             ha='right',va='bottom',transform=plt.gcf().transFigure,fontsize=7)
    plt.show()

X, Y = df["Cable_Dia_mm"], df["Utilisation"]

# 1️⃣ MOORA Score landscape
contour2d(X,Y,df["MOORA_Score"],
          "MOORA Score Contours  (Utilisation × Cable Ø)",
          "Cable Diameter (mm)","Utilisation Ratio","MOORA Score")

# 2️⃣ Cable Mass landscape (magma because heavy = scary)
contour2d(X,Y,df["CableMass_kg"],
          "Cable Mass Contours  (Utilisation × Cable Ø)",
          "Cable Diameter (mm)","Utilisation Ratio","Cable Mass (kg)",
          cmap='magma_r')

# 3️⃣ Natural Freq landscape with 3 Hz and 5 Hz ring‑fences
contour2d(X,Y,df["NatFreq_Hz"],
          "Natural Frequency Contours  (Utilisation × Cable Ø)",
          "Cable Diameter (mm)","Utilisation Ratio","f₁ (Hz)",
          cmap='plasma',
          overlay=[(3,'--'),(5,':')],
          annotate=[(X.min(),3.05,"3 Hz line"),(X.min(),5.05,"5 Hz line")])

# %% 🌄 3‑D surface — NatFreq coloured by slope (Michelangelo meets MATLAB) ----
xi, yi = np.linspace(X.min(),X.max(),60), np.linspace(Y.min(),Y.max(),60)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((X,Y), df["NatFreq_Hz"],  (Xi,Yi), method='cubic')
Si = griddata((X,Y), df["Slope_pct"],   (Xi,Yi), method='cubic')

fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111, projection='3d')
norm = mcolors.Normalize(vmin=df["Slope_pct"].min(), vmax=df["Slope_pct"].max())
surface = ax.plot_surface(Xi, Yi, Zi,
                          facecolors=cm.RdYlGn_r(norm(Si)),
                          rstride=1, cstride=1, antialiased=False)
ax.contour(Xi, Yi, Zi, zdir='z', offset=Zi.min()-0.5,
           cmap='plasma', levels=10, linewidths=0.6)

ax.set_xlabel("Cable Ø (mm)"); ax.set_ylabel("Utilisation Ratio"); ax.set_zlabel("f₁ (Hz)")
ax.set_title("3‑D Surface: NatFreq (colour = Deck Slope %)")

cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'),
                  ax=ax, shrink=0.6, label="Deck Slope (%)")
fig.text(0.99,0.01,"© Vijaykumar Parmar & Dr. K. B. Parikh",
         ha='right',va='bottom',fontsize=7)
plt.show()

# %% 📈 Parallel‑coordinates — spaghetti for nerds ----------------------------
pc = ranked.head(10).copy()
pc["InvSlope"], pc["InvTension"], pc["InvMass"], pc["InvØ"] = \
    -pc["Slope_pct"], -pc["Tension_kN"], -pc["CableMass_kg"], -pc["Cable_Dia_mm"]
cols_pc = ["InvSlope","InvTension","InvMass","InvØ","NatFreq_Hz","MOORA_Score"]
pc_norm = pc[cols_pc].apply(lambda s:(s-s.min())/(s.max()-s.min()))
pc_norm["Alt"] = "A"+pc.index.astype(str)

plt.figure(figsize=(11,5))
parallel_coordinates(pc_norm, "Alt", colormap="viridis",
                     linewidth=2, alpha=0.75)
plt.title("Parallel‑Coordinates of Top 10 Alternatives\n(cost axes inverted)")
plt.ylabel("Normalised scale (0‑1)"); plt.xticks(rotation=25)
plt.legend(bbox_to_anchor=(1.03,1), loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()

# %% 🏁 Epilogue — one‑liner brag & gentle reminder ----------------------------
best = ranked.iloc[0]
print(f"\n🎯 **Top dog:** Ø {best.Cable_Dia_mm:.2f} mm, util {best.Utilisation:.2f}, "
      f"slope {best.Slope_pct:.2f} %, tension {best.Tension_kN:.2f} kN, "
      f"mass {best.CableMass_kg:.0f} kg, f₁ {best.NatFreq_Hz:.2f} Hz, "
      f"MOORA {best.MOORA_Score:.3f}  — print it, frame it, build it.")
print("📂  Full CSV → *srb_moora_results.csv* | 🚧")
