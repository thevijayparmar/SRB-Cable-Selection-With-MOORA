# Stress‑Ribbon Bridge Cable Selector (MOORA) &nbsp;🚧🔗

[![Streamlit App](https://img.shields.io/badge/Try%20it‑on-Streamlit‑Cloud-ff4b4b?logo=streamlit&logoColor=white)](https://share.streamlit.io/your‑repo/srb‑cable‑selector/main/app.py)
[![License](https://img.shields.io/github/license/your‑repo/srb‑cable‑selector)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)

> **Authors:** Vijaykumar Parmar & Dr. K. B. Parikh  
> **© 2025 – All rights reserved**

A lean, interactive **Streamlit** web‑tool that helps bridge engineers shortlist optimal cable configurations for a **Stress‑Ribbon Bridge (SRB)** using the **MOORA (Multi‑Objective Optimisation on the basis of Ratio Analysis)** ranking method.

---

## ✨ Key Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Fast design‑space explorer** | Generates hundreds of cable alternatives by varying diameter, utilisation and number of cables. |
| 2 | **MOORA‑based ranking** | Converts engineering responses into cost/benefit scores and produces an overall MOORA ranking. |
| 3 | **Interactive plots** | • Cable profile • Contour plots (with custom 7‑colour map) • Parallel‑coordinate plot for the top 50 designs. |
| 4 | **“Generate All Charts”** | One‑click batch creation of every valid X–Y contour combination. |
| 5 | **Configurable penalties/benefits** | Linear / exponential shapes, threshold triggers, enable/disable toggle – all from the sidebar. |
| 6 | **CSV export** | Download the full ranked table for further processing. |

---

## 📚 Theory in a Nutshell

1. **Stress‑Ribbon Bridge (SRB)**  
   A slender concrete deck that acts in tension, supported by post‑tensioned cables. Key design variables are span **L**, cable diameter **d**, utilisation **u**, and number of cables **n**.

2. **MOORA Method**  
   Normalises penalty/benefit values, sums benefits, subtracts costs → yields a single **MOORA Score**. Higher score = better alternative. See [J. Brauers, 2004].

3. **Criteria Implemented**

| Criterion          | Type    | Default Trigger | Shape |
|--------------------|---------|-----------------|-------|
| Utilisation        | Cost    | Below 0.8       | Exponential |
| Slope %            | Cost    | Above 2.5 %     | Linear |
| Cable Diameter mm  | Cost    | Above 150 mm    | Linear |
| Number of Cables   | Cost    | Above 5         | Exponential |
| Natural Freq Hz    | Benefit | Above 2.0 Hz    | Linear |
| Tension kN         | Cost    | Above 0         | Linear |
| Sag m              | Cost    | Below L × 0.003 | Exponential |

*(All are editable in the app.)*

---

## 🖥️ Quick Start

### 1 · Clone & install

```bash
git clone https://github.com/<your‑org>/srb‑cable‑selector.git
cd srb‑cable‑selector
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt


🧑‍💻 Authors
Vijaykumar Parmar
Dr. K. B. Parikh
