# Stress‑Ribbon Bridge Cable Selector (MOORA) &nbsp;🚧🔗

[![Streamlit App](https://img.shields.io/badge/Try%20it‑on-Streamlit‑Cloud-ff4b4b?logo=streamlit&logoColor=white)](https://mooraforsrb.streamlit.app/)
[![License](https://img.shields.io/github/license/your‑org/srb‑cable‑selector)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/thevijayparmar/)


> **Authors:** Vijaykumar Parmar & Dr. K. B. Parikh  
> **© 2025 – All rights reserved**

A lean, interactive **Streamlit** tool that helps bridge engineers shortlist optimal cable configurations for a **Stress‑Ribbon Bridge (SRB)** using the **MOORA (Multi‑Objective Optimisation on the basis of Ratio Analysis)** ranking method.

---

## ✨ Key Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Fast design‑space explorer** | Generates hundreds of cable alternatives by varying diameter, utilisation and number of cables. |
| 2 | **MOORA‑based ranking** | Converts engineering responses into cost/benefit scores and produces an overall MOORA ranking. |
| 3 | **Interactive plots** | • Cable profile • Contour plots (custom 7‑colour map) • Parallel‑coordinate plot for the top 50 designs. |
| 4 | **“Generate All Charts”** | One‑click batch creation of every valid X–Y contour combination. |
| 5 | **Configurable penalties/benefits** | Linear / exponential, threshold triggers, enable/disable toggle – all from the sidebar. |
| 6 | **CSV export** | Download the full ranked table for further processing. |

---

## 📚 Theory in a Nutshell

1. **Stress‑Ribbon Bridge (SRB)**  
   Slender concrete deck in tension, supported by post‑tensioned cables. Key variables: span **L**, cable diameter **d**, utilisation **u**, number of cables **n**.

2. **MOORA Method**  
   Normalises penalty/benefit values, sums benefits, subtracts costs → yields a single **MOORA Score** (higher is better).

3. **Default Criterion Settings**

| Criterion          | Type    | Default Trigger | Shape |
|--------------------|---------|-----------------|-------|
| Utilisation        | **Cost** | Below 0.8       | Exponential |
| Slope %            | **Benefit** | Below 2.5 %     | Linear |
| Cable Diameter mm  | **Cost** | Above 150 mm    | Linear |
| Number of Cables   | **Cost** | Above 5         | Exponential |
| Natural Freq Hz    | **Benefit** | Above 2.0 Hz    | Linear |
| Tension kN         | **Cost** | Above 0         | Linear |
| Sag m              | **Benefit** | Below L × 0.003 | Exponential |

*(All of these can be edited live in the app.)*

---

## 🖥️ Quick Start

### 1 · Clone & install

```bash
git clone https://github.com/<your‑org>/srb‑cable‑selector.git
cd srb‑cable‑selector
python -m venv .venv && source .venv/bin/activate   # optional virtualenv
pip install -r requirements.txt
