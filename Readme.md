# Stress‑Ribbon Bridge – MOORA Profiler 🏗️🎓

Interactive, browser‑based tool that helps engineers (and the occasional curious architect) compare **77 cable–utilisation design variants** for a stress‑ribbon footbridge.  
Behind the scenes it crunches classic sag/tension physics and ranks each design with the **MOORA multi‑criteria method**.

[![Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR‑STREAMLIT‑URL) <!-- replace after first deploy -->

---

## ✨ Key features

| What you get | Why it matters |
|--------------|----------------|
| **Real‑time sliders** for span, load, cable strength, etc. | Instant “what‑if” exploration |
| **MOORA score** (slope ↓, tension ↓, mass ↓, Ø ↓, frequency ↑) | Balanced decision metric |
| **2‑D contour maps** | Spot ridges & valleys of good designs |
| **3‑D surface (Nat‑Freq)** coloured by deck slope | See dynamic behaviour at a glance |
| **Parallel‑coordinates spaghetti** for top 10 | Visualise trade‑offs across *all* criteria |
| 100 % pure **Python/Matplotlib/Streamlit** | No exotic JS, easy to fork |

---

## 🚀 Run it locally

```bash
git clone https://github.com/your‑handle/srb‑moora-app.git
cd srb‑moora-app
pip install -r requirements.txt
streamlit run app.py

🧑‍💻 Authors
Vijaykumar Parmar
Dr. K. B. Parikh
