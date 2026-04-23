# ⚾ MLB Free Agent Value Calculator

> **ACC102 Mini Assignment · Track 4 – Interactive Data Analysis Tool**  
> A data-driven contract valuation tool for MLB front offices, built with Python and Streamlit.

🔗 **[Live App →](https://acc102-project-uyfq8flp9ktunqfd5vxzht.streamlit.app/)**  
📓 **[Analysis Notebook →](analysis.ipynb)**

---

## 1. Problem & User

MLB teams spend over **$4 billion per year** on player salaries, yet there is no universally accessible, open-source tool that lets a front-office analyst quickly stress-test a free-agent contract against historical market data.

This tool answers the question: **How much should an MLB free agent be paid, given his recent performance and age — and which historical contracts were significantly over- or under-valued?**

**Primary users:** MLB front-office staff (General Managers, salary analysts) evaluating free-agent offers.  
**Secondary users:** Player agents, sports-finance journalists, and sports analytics students.

---

## 2. Data

| Field | Detail |
|---|---|
| **Source** | Lahman Baseball Database (Chadwick Bureau Baseball Databank) |
| **URL** | https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd |
| **Licence** | Creative Commons Attribution-ShareAlike 3.0 |
| **Date accessed** | April.2026 |
| **Files used** | `People.csv`, `Batting.csv`, `Pitching.csv`, `Salaries.csv`, `Teams.csv` |
| **Analysis range** | 2000–2016 (period with complete salary records in Lahman) |
| **Records** | ~26,400 salary rows; ~128,600 batting rows; ~57,600 pitching rows |

---

## 3. Methods

The full analytical workflow is documented step-by-step in `analysis.ipynb`. The main Python steps are:

1. **Data loading & cleaning** — load five Lahman CSVs, filter to 2000–2016, handle missing values and multi-team seasons.
2. **Simplified WAR computation** — wOBA-based batting WAR for position players; FIP-based pitching WAR for pitchers (using the standard 10 runs-per-win conversion).
3. **League $/WAR market price** — compute the cost per Win Above Replacement for each year, capturing the growth of the free-agent market.
4. **Aging curve** — fit a quadratic polynomial regression (scikit-learn) to WAR vs. age data; the curve peaks at ~age 26–28, consistent with sabermetric literature.
5. **Contract valuation function** — combine the aging curve multipliers with the current $/WAR price to produce a year-by-year fair value for any proposed contract.
6. **Monte Carlo simulation** — run 500–5,000 simulations with configurable WAR volatility (σ) to show the *distribution* of fair-value outcomes, not just a point estimate.
7. **Surplus value analysis** — compute surplus (expected value − actual salary) for every player-season; identify the biggest bargains and busts.
8. **Export** — save three processed Parquet files (`master_dataset`, `aging_curve`, `dollar_per_war`) for fast loading in the Streamlit app.

---

## 4. Key Findings

- **The MLB free-agent market grew substantially** from 2000 to 2016, driven by expanding TV deals; the $/WAR benchmark roughly doubled over this period.
- **Player performance peaks in the late 20s** (aging curve peak: age 26 in this dataset), meaning multi-year deals signed at age 30+ systematically include below-peak seasons.
- **Pre-arbitration players carry the largest surplus** — their salaries are structurally suppressed under MLB's collective-bargaining rules, making them the highest-value contracts in the dataset by design.
- **The worst contracts share a pattern:** long-term deals (5+ years) signed when the player was already past peak age, followed by injury or rapid performance decline.
- **A simple open-data model is surprisingly powerful** — using only Lahman data and back-of-envelope sabermetrics, the tool reproduces the same contract rankings that professional analysts spend significant resources to compute.

---

## 5. How to Run

### Prerequisites
- Python 3.10 or higher
- The Lahman Database CSV files (free download)

### Step 1 — Download the Lahman Database

```bash
# Manual download
# Go to https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd
# Download the ZIP, unzip, and copy all CSV files into data/raw/
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the analysis notebook

Open `analysis.ipynb` in Jupyter and run all cells from top to bottom. This generates the three processed Parquet files in `data/processed/`.

```bash
jupyter notebook analysis.ipynb
# Run all cells (Kernel → Restart & Run All)
```

### Step 4 — Launch the Streamlit app

```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 6. Product Link / Demo

🔗 **Live Streamlit App:** [https://acc102-project-uyfq8flp9ktunqfd5vxzht.streamlit.app/]  
🎬 **Demo Video:** [https://youtu.be/s_zgNQToyRU]

The app has three tabs:
- **CONTRACT VALUATOR** — input a player's recent WAR, age, and proposed contract terms; get a fair-value estimate with year-by-year breakdown and Monte Carlo uncertainty analysis.
- **AGING CURVE** — explore how player WAR changes by age according to the fitted quadratic model.
- **MARKET EXPLORER** — interactive scatter plot of all 2000–2016 player-seasons coloured by contract surplus; filter by year range, minimum salary, and player type (batter/pitcher).

---

## 7. Limitations & Next Steps

**Current limitations:**
- Simplified WAR ignores defence, baserunning, and park effects; real fWAR/bWAR from Baseball Reference or FanGraphs would be more accurate.
- Salary data ends in 2016 in the public Lahman release; extending to recent seasons requires licensed data (Spotrac / Cot's Contracts).
- The aging curve is a league-wide average; individual players age differently (e.g. high-walk hitters age more gracefully than speed-dependent players).
- No injury or injury-history data, which is a major source of contract risk in practice.

**Possible next steps:**
- Integrate the Baseball Reference or FanGraphs API for more accurate WAR figures.
- Add a player name auto-complete that pulls recent statistics live.
- Extend the surplus analysis to pitcher-specific metrics (ERA-, xFIP).
- Add a positional scarcity adjustment to the $/WAR calculation.

---

## 8. Repository Structure

```
├── app.py                    # Streamlit app (entry point)
├── analysis.ipynb            # Full analytical workflow notebook
├── requirements.txt          # Python dependencies
├── README.md
├── data/
│   ├── raw/                  # Lahman CSV files (not committed — download separately)
│   └── processed/            # Parquet outputs generated by the notebook
│       ├── master_dataset.parquet
│       ├── aging_curve.parquet
│       └── dollar_per_war.parquet
└── figures/                  # Charts saved by the notebook
    ├── aging_curve.png
    ├── dollar_per_war.png
    └── surplus_value_map.png
```

> **Note:** Raw CSV files are not committed to this repository to keep file sizes small. Please follow the download instructions in Step 1 above.

---

*Data: Lahman Baseball Database (CC BY-SA 3.0) · Built with Streamlit, Plotly, pandas, scikit-learn*
