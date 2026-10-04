# ⛏️ MINEX: MOIL AI Mining Intelligence Platform
### Smart India Hackathon 2026 — Problem Statement #26009
> **"Using AI/ML and Space Technology to Identify Manganese Reserves and Overcome Production Shortfalls"**  
> **Organization:** Ministry of Steel | **Department:** MOIL Ltd. | **Theme:** Space Technology  
> **GitHub Repository:** [https://github.com/Sanil-Samir-Mhatre/Minex_26009](https://github.com/Sanil-Samir-Mhatre/Minex_26009)

---

## 📌 Executive Summary

India is one of the world's leading producers of manganese ore, a critical raw material indispensable for steel manufacturing, ferroalloys, and the burgeoning clean energy transition (EV lithium-ion cathode chemistries like NMC and LMFP). **MOIL Limited** (formerly Manganese Ore India Limited), a Miniratna state-owned enterprise under the Ministry of Steel, operates both deep underground and open-cast manganese mines across the Central Indian belt.

However, MOIL faces two systemic challenges:
1. **Exploration Inefficiencies:** Conventional ground exploration is geographically expansive, time-consuming, and resource-heavy. Identifying new high-potential manganese-bearing zones requires modern satellite spectral intelligence before committing millions of rupees into exploratory drilling.
2. **Production Volatility & Operational Shortfalls:** Active mining operations consistently face daily production shortfalls driven by heavy monsoon flooding, unplanned heavy earth-moving machinery (HEMM) breakdowns, blasting schedule delays, and fluctuating ore grade feeds.

**MINEX** is an enterprise-grade AI/ML and Space Technology intelligence platform that bridges the gap between **space-based exploration targeting** and **ground-level production optimization**.

```
                ┌─────────────────────────────────────────────────────────────────┐
                │             MINEX MINING INTELLIGENCE PLATFORM                  │
                └────────────────────────────────┬────────────────────────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  ▼                                                             ▼
     【PIPELINE A: SPACE EXPLORATION】                             【PIPELINE B: OPERATIONAL CONTINUITY】
  Sentinel-2 Multispectral + GSI Geoinformation                 Daily SCADA Fleet Metrics + ERA5 Weather
                  │                                                             │
                  ▼                                                             ▼
  Spectral Ratios (Iron, Clay, BSI, SWIR) +                     Operational Regressor & Risk Classifier
  Structural Lineaments & Fault Proximity                                       │
                  │                                                             ▼
                  ▼                                                Daily Tonnage Forecast (t/day) +
  Manganese Prospectivity Score [0.0 - 1.0]                     Early Risk Alerts (LOW / MED / HIGH / CRIT)
  with Bootstrap Uncertainty & SHAP Drivers                                     │
                  │                                                             ▼
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                             【PRESCRIPTIVE DECISION & VALUE LINK】
                     Target Priority Scoring (Top-K Drilling Targets) +
                     Autonomous Operational Mitigation Dispatch Engine
```

---

## 💡 Why We Are Doing What We Are Doing

### 1. The Strategic Imperative for Manganese
Manganese cannot be substituted in basic oxygen and electric arc furnace steelmaking (it acts as a deoxidizer and desulfurizer, with ~10 kg required per tonne of crude steel). As India aims for 300 million tonnes of annual steel production under the National Steel Policy 2017, MOIL must ramp up domestic extraction while discovering fresh reserves to avoid import dependency.

### 2. Space Technology: From Blind Drilling to Precision Targeting
Drilling a single exploratory diamond core borehole can cost hundreds of thousands of rupees. By utilizing multispectral satellite bands (Sentinel-2 MSI) and geological remote sensing (band ratios like B4/B2 for iron oxides, B11/B12 for hydroxyl/clay alterations, and SWIR indices for gondite formations), our platform detects the surface alterations and host-rock signatures characteristically associated with manganese deposits. This narrows down regional prospectivity from hundreds of square kilometers to high-priority target clusters.

### 3. Predictive Operations: Preventing Unplanned Shortfalls
In mining, unplanned downtime of shovel-dumper fleets, water-logging in open pits during monsoon months (Nagpur/Balaghat receives intense rainfall from July to September), and blasting delays immediately cascade into unmet blast-to-crusher targets. By predicting shortfalls 24–48 hours in advance, mine managers can proactively adjust shovel allocations, divert haul routes, initiate preventive dewatering, or blend grades.

---

## 🔬 Scope & Realism: To What Extent We Scaled It Down and Why

In software hackathons and prototype deployments, transparency regarding data fidelity and physical limits is the hallmark of real engineering. Here is our deliberate scoping:

### 1. The Geographic Study Area
We localized the platform to the **Central Indian Manganese Belt (Sausar Group)** spanning:
- **Maharashtra:** Nagpur District (Mansar, Gumgaon, Kandri) and Bhandara District (Dongri Buzurg).
- **Madhya Pradesh:** Balaghat District (Balaghat/Bharweli, Ukwa, Tirodi).
This region accounts for over 50% of India's high-grade manganese ore reserves.

### 2. Why High-Fidelity Synthetic / Proxy Data Was Used
* **Proprietary PSU Operational Data:** MOIL's real-time SCADA sensor streams, shift dumper tonnage logs, blasting manifests, and fleet telemetry are classified internal operational records not accessible publicly.
* **Geological Survey of India (GSI) Confidential Layers:** Detailed 1:25,000 scale structural strike, fault lineaments, and borehole assay tables are restricted under national mineral spatial data protocols.
* **Cloud & Server Bandwidth Constraints:** Live ingestion and on-the-fly raster processing of multi-gigabyte Sentinel-2 L2A 12-band tiles across regional swathes would exceed free-tier cloud limits, requiring large multi-node GPU clusters.

### 3. Preserved Domain Physics & Scientific Rigor
Rather than arbitrary random numbers, our data pipeline implements **physically and geologically constrained data modeling**:
- **Spectral Physics:** Band values mirror real Sentinel-2 surface reflectance across B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR-1), and B12 (SWIR-2).
- **Geological Association:** Points located near Sausar Group formations (Mansar mica schists, gondites, calc-silicates) and closer to lineament fault zones exhibit realistic spectral alteration and higher manganese association.
- **Meteorological Coupling:** Rainfall follows central Indian monsoon seasonality (monsoon peaks in July–August, dry winters), directly influencing soil saturation and hauling delays.
- **Operational Reality:** Modeled after MOIL's real capacity ranges (Dongri Buzurg opencast: ~1,200–2,500 t/day; Balaghat underground: ~1,500–3,000 t/day; Ukwa: ~500–1,200 t/day).

### 4. Critical Scientific Distinction: Prospectivity vs. Reserves
> ⚠️ **Scientific & Regulatory Note:** Multispectral remote sensing measures optical and shortwave-infrared surface reflectance (the top few millimeters of soil, rock outcrop, and vegetation canopy). **Satellites cannot "see" underground ore bodies.** 
> Therefore, this platform provides **Manganese Prospectivity Scores (0.0 to 1.0)** and **Structural Favorability**, highlighting where exploration teams should focus geochemical sampling, trenching, and geophysical testing. It does **NOT** declare confirmed reserves or replace UNFC (United Nations Framework Classification) / JORC-compliant core drilling.

---

## 🖥️ Multi-Page Application Breakdown

The platform is organized into 9 specialized, interconnected modules accessible via the Streamlit sidebar:

| Page | Module Title | Primary Functionality & Value |
|------|--------------|-------------------------------|
| **Home** | `app.py` | Platform orientation, system health indicators, credentials check, and quick-start guide. |
| **01** | `1_dashboard.py` | **Executive Overview:** High-level KPIs (average daily output, shortfall alert count, top prospective zone), interactive GIS overview map, and fleet health trackers. |
| **02** | `2_prospectivity.py` | **Exploration Intelligence:** Interactive geospatial prospectivity map, satellite spectral band inspection (NDVI, NDWI, Iron/Clay/SWIR ratios), top-K prioritized drilling coordinates, and SHAP feature importance. |
| **03** | `3_production.py` | **Operational Shortfall Forecaster:** Daily planned vs. predicted tonnage, historical trend analysis, interactive operational parameters slider, and shortfall probability classifier. |
| **04** | `4_risk_recommendations.py` | **Prescriptive Decision Engine:** Real-time operational risk register (Critical, High, Medium, Low) with automated root-cause attribution and immediate corrective dispatch guidelines. |
| **05** | `5_data_explorer.py` | **Data Auditing & Ingestion Hub:** Complete transparency into underlying satellite, geological, weather, and operational datasets with statistical distributions, filters, and CSV export. |
| **06** | `6_model_performance.py` | **AI/ML Diagnostics:** Confusion matrices, ROC-AUC curves, cross-validation scores, regression error residuals (MAE, RMSE, R²), and multi-model benchmark comparisons. |
| **07** | `7_architecture.py` | **System Architecture & Literature:** Complete data pipeline flowcharts, peer-reviewed scientific literature citations (Zhao et al. 2025, Singh et al. 2023), and mathematical formulations. |
| **08** | `8_whatif_simulator.py` | **Interactive Operational Digital Twin:** Adjust haul truck availability, simulate a sudden 60mm monsoon storm, or test grade changes to forecast immediate financial and tonnage impact. |
| **09** | `9_exploration_ops_link.py` | **Strategic Value Linkage:** Bridges long-term satellite exploration hits with short-term pit operations, calculating reserve replenishment ratios and feeder lifespan. |

---

## 🤖 Machine Learning Pipeline & Explainability

### Model 1: Manganese Prospectivity Classifier
- **Architecture:** Soft-voting ensemble combining **Random Forest** (200 trees, balanced weights), **XGBoost** (gradient boosted trees), and **Gaussian Naive Bayes**.
- **Class Balancing:** Synthetic Minority Over-sampling Technique (**SMOTE**) to handle extreme positive-unlabeled imbalance typical of mineral occurrence targets.
- **Key Predictors (17 Features):**
  - Structural Density & Fault Proximity (Zhao et al., 2025 — proved to be the single most influential structural predictor).
  - Sentinel-2 Mineral Ratios: Iron Oxide Ratio ($B4/B2$), Clay Alteration Index ($B11/B12$), SWIR Ratio ($B11/B8$).
  - Vegetation & Soil Indices: NDVI, NDWI, Bare Soil Index (BSI).
  - Topographic & Morphometric: SRTM DEM Elevation, Slope Gradient.
  - Proximity to existing producing MOIL pitheads.
- **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** waterfall and summary plots explain exact spectral drivers behind each pixel's prospectivity.

### Model 2: Operational Production Forecaster
- **Architecture:** **XGBoost Regressor** tuned for non-linear daily tonnage prediction based on equipment availability, planned targets, effective working hours, blasting delays, and weather parameters.
- **Performance:** Achieves strong $R^2 > 0.90$ with Mean Absolute Error under 40 tonnes/day on operational testing splits.

### Model 3: Operational Risk Classifier
- **Architecture:** Multi-class XGBoost Classifier categorizing operational days into 4 risk tiers:
  - 🟢 **LOW RISK:** Target achievement $\ge 95\%$
  - 🟡 **MEDIUM RISK:** Minor shortfall between $5\% - 15\%$
  - 🟠 **HIGH RISK:** Severe shortfall between $15\% - 25\%$
  - 🔴 **CRITICAL RISK:** Major disruption with shortfall $> 25\%$

---

## 🚀 Quick Start & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/Sanil-Samir-Mhatre/Minex_26009.git
cd Minex_26009
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets (Optional)
```bash
# Copy example configuration
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
*(The app runs seamlessly in Demo Mode even without API keys!)*

### 5. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Step-by-Step Streamlit Cloud Deployment Guide

Follow these simple steps to deploy this repository to **Streamlit Community Cloud**:

1. **Push Repository to GitHub:**
   Ensure all files are committed and pushed to your public repository:
   `https://github.com/Sanil-Samir-Mhatre/Minex_26009`

2. **Log into Streamlit Community Cloud:**
   Visit [share.streamlit.io](https://share.streamlit.io) and authenticate using your GitHub account.

3. **Deploy New App:**
   - Click **"Create app"** (or **"New app"**).
   - Select **"I already have an app"**.
   - Fill in repository details:
     - **Repository:** `Sanil-Samir-Mhatre/Minex_26009`
     - **Branch:** `main`
     - **Main file path:** `app.py`
     - **App URL:** Customize your subdomain (e.g. `minex-moil-intelligence.streamlit.app`)

4. **Configure Secrets (App Settings):**
   - Click **"Advanced settings..."** at the bottom of the deployment modal.
   - In the **Secrets** text box, paste your configuration from `.streamlit/secrets.toml.example`:
     ```toml
     API_KEY = "your_secret_api_key_here"
     GOOGLE_MAPS_API_KEY = "your_google_maps_key_if_available"
     ```
   - *Note: If no Google Maps key is provided, the platform automatically falls back to interactive Folium/OpenStreetMap overlays without errors.*

5. **Click Deploy!**
   Streamlit Cloud will provision the container, install packages from `requirements.txt`, and launch the app in 1–2 minutes.

---

## 🔭 Future Scope & Industrial Roadmap

When transitioning from this prototype to live enterprise deployment at MOIL Ltd., the following technical extensions are planned:

1. **Spaceborne & Airborne Hyperspectral Integration:**
   - Transitioning from Sentinel-2's 12 broad bands to spaceborne hyperspectral sensors (**PRISMA**, **EnMAP**, and **NASA EMIT**) with 200+ narrow spectral channels (400–2500 nm). This enables direct diagnostic absorption mapping of manganese oxyhydroxides (pyrolusite, psilomelane, braunite, cryptomelane).
   - Drone-mounted LiDAR and multispectral sensors for micro-topography and bench-scale alteration mapping in active opencast pits.

2. **3D Subsurface Geophysical Data Inversion:**
   - Ingesting ground-penetrating radar (GPR), electrical resistivity tomography (ERT), and gravity-magnetic surveys.
   - Coupling 2D satellite surface prospectivity with 3D geological block modeling (using GemPy / Leapfrog) to generate drill-ready volumetric targets.

3. **Enterprise SCADA & IoT Telemetry Ingestion:**
   - Integration with MOIL's dispatch systems via standard industrial protocols (OPC-UA / MQTT).
   - Automated CAN-bus telematics on Komatsu / Caterpillar dumper-shovel fleets for real-time engine temperature, fuel burn, and payload cycle monitoring.

4. **Automated UNFC Resource Categorization:**
   - As exploratory core holes are drilled, automated logging of assay grades into a centralized PostGIS spatial database.
   - Algorithmic classification into UNFC code categories (G1: Measured, G2: Indicated, G3: Inferred).

---

## 📁 Repository Structure

```
Minex_26009/
├── .streamlit/
│   ├── config.toml                     # Dark UI theme, fonts & layout settings
│   └── secrets.toml.example            # Template for secret keys (never commit secrets.toml)
├── data/
│   └── README.txt                      # Data catalog & synthetic proxy specifications
├── models/
│   └── .gitkeep                        # Auto-generated model checkpoint directory
├── pages/                              # Multi-page Streamlit modules
│   ├── 1_dashboard.py                  # Page 1: Executive KPI Overview & Summary Map
│   ├── 2_prospectivity.py              # Page 2: Satellite Prospectivity & Spectral SHAP
│   ├── 3_production.py                 # Page 3: Daily Production Forecaster & Regressor
│   ├── 4_risk_recommendations.py       # Page 4: Prescriptive Operational Risk Engine
│   ├── 5_data_explorer.py              # Page 5: Datasets, Distributions & CSV Export
│   ├── 6_model_performance.py          # Page 6: AI/ML Benchmarks, ROC & Metrics
│   ├── 7_architecture.py               # Page 7: Flowcharts, Pipeline & Literature
│   ├── 8_whatif_simulator.py           # Page 8: Digital Twin "What-If" Scenario Simulator
│   └── 9_exploration_ops_link.py       # Page 9: Exploration-to-Operations Strategic Link
├── src/                                # Core logic & algorithmic backend
│   ├── data_loader.py                  # Geological & operational data synthesis & caching
│   ├── prospectivity_model.py          # Ensemble classifier (RF+XGB+NB), SMOTE & SHAP
│   ├── production_model.py             # Operational regressor & multi-class risk classifier
│   ├── risk_engine.py                  # Rule-based operational dispatch recommendation engine
│   └── visualization.py               # Folium GIS & Plotly dark-theme visualizers
├── app.py                              # Main application entry point & router
├── requirements.txt                    # Production Python dependencies
├── YOUTUBE_DEMO_SCRIPT.md              # 4-5 minute page-by-page screen recording script
└── README.md                           # Comprehensive documentation
```

---

## 🏛️ Team & Hackathon Details

- **Event:** Smart India Hackathon (SIH) 2026
- **Problem Statement ID:** #26009
- **Domain:** Space Technology / AI & ML in Mining
- **Supported Organization:** MOIL Limited (Ministry of Steel, Govt. of India)
- **Repository:** [https://github.com/Sanil-Samir-Mhatre/Minex_26009](https://github.com/Sanil-Samir-Mhatre/Minex_26009)
