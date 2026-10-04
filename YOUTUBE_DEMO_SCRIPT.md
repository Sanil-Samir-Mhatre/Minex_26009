# 🎬 MINEX: YouTube Video Recording & Demonstration Script
### Video Duration: ~4:30 – 5:00 Minutes
### Problem Statement: SIH 2026 #26009 — MOIL Ltd. (Ministry of Steel)
> **Goal:** High-impact, professional screen-recording presentation demonstrating every page of the application with clear visual cues and exact spoken voiceover dialogue.

---

## ⏱️ Video Timeline Overview
| Timestamp | Segment / Page | Focus |
|---|---|---|
| **0:00 – 0:30** | **Intro & Executive Brief** | Problem Statement #26009, MOIL's dual challenge, Dual-Pipeline Concept |
| **0:30 – 1:05** | **Page 1: Executive Dashboard** | High-level KPIs, GIS mine locations, fleet status, quick shortfall metrics |
| **1:05 – 1:45** | **Page 2: Manganese Prospectivity** | Space Tech, Sentinel-2 spectral indices, Top-K drill targets, SHAP explainability |
| **1:45 – 2:20** | **Page 3: Production Shortfall Forecaster** | Operational ML regressor, planned vs predicted, interactive shortfall sliders |
| **2:20 – 2:55** | **Page 4: Real-Time Risk & Mitigation Engine** | Automated risk tiers, root-cause diagnostics, actionable dispatch guidance |
| **2:55 – 3:25** | **Page 5: Data Explorer & Transparency Hub** | Datasets, feature distributions, data fidelity, CSV export capability |
| **3:25 – 3:55** | **Page 6: Model Performance & Benchmarking** | ROC-AUC, confusion matrix, regression metrics (R², RMSE), validation rigor |
| **3:55 – 4:25** | **Pages 7, 8 & 9: What-If Simulator & Value Chain** | Operational digital twin, weather shock testing, exploration-ops synergy |
| **4:25 – 4:45** | **Conclusion & Future Scope** | Hyperspectral sensors, 3D inversion, SCADA IoT integration, closing |

---

## 🎙️ Detailed Page-by-Page Script & Visual Cues

---

### [0:00 – 0:30] — Introduction & Platform Overview
* **🖥️ Screen Action:** Start on the main **Home page** (`app.py`). Show the title banner, platform architecture cards, and system status indicators.
* **🗣️ Spoken Voiceover:**
  > "Hello everyone! Welcome to the demonstration of **MINEX**, our AI and Space Technology-driven Mining Intelligence Platform developed for **Smart India Hackathon 2026, Problem Statement #26009 for MOIL Limited, Ministry of Steel**.
  >
  > MOIL is India's largest producer of manganese ore. Their primary operational hurdles are two-fold: first, **exploring vast geographical areas** to locate high-potential manganese reserves; and second, **overcoming daily production shortfalls** caused by equipment breakdowns, severe monsoon rains, and operational delays.
  >
  > MINEX unites **Spaceborne Satellite Imagery** and **Ground Operational Telemetry** into one seamless, interpretable platform. Let’s dive straight into the features!"

---

### [0:30 – 1:05] — Page 1: Executive Overview Dashboard
* **🖥️ Screen Action:** Click on **`01 Dashboard`** in the left sidebar. Scroll smoothly down the page showing the summary metric cards, the interactive GIS map of Central India (Nagpur–Bhandara–Balaghat belt), and the daily shortfall alert table.
* **🗣️ Spoken Voiceover:**
  > "Here on Page 1 is the **Executive Dashboard**, designed specifically for mine managers and directors.
  >
  > At a glance, we see today's key performance indicators: daily planned versus actual tonnage, fleet availability, weather alerts, and the count of active operational shortfalls.
  >
  > On the interactive map, you can see all key MOIL mining locations—including Dongri Buzurg, Gumgaon, Mansar, and Balaghat. Clicking on any mine displays live operational status, current production capacity, and weather conditions. Below the map, the platform immediately flags which mines are running at risk today so leadership can act without delay."

---

### [1:05 – 1:45] — Page 2: Manganese Prospectivity (Space Technology)
* **🖥️ Screen Action:** Click on **`02 Prospectivity`**. Show the satellite spectral filters. Toggle between **Prospectivity Score Heatmap** and **Spectral Mineral Ratios** (Iron Oxide, Clay, SWIR). Scroll down to show the **Top-K Exploration Target Table** and the **SHAP Feature Importance Plot**.
* **🗣️ Spoken Voiceover:**
  > "Moving to Page 2: **Manganese Prospectivity Mapping**, our core space-technology module.
  >
  > Rather than surveying thousands of square kilometers blindly, we ingest Sentinel-2 multispectral bands and compute diagnostic mineral indices: such as the **Iron Oxide ratio (B4/B2)**, **Clay Alteration index (B11/B12)**, and **Bare Soil Index**.
  >
  > Our soft-voting ensemble model combines Random Forest, XGBoost, and Naive Bayes with structural fault density to generate a continuous **Prospectivity Score from 0 to 1**.
  >
  > Down below, the system automatically ranks the **Top High-Priority Exploration Targets**, giving exact GPS coordinates and uncertainty estimates to guide exploratory core drilling. And with integrated **SHAP explainability**, geologists can see exactly which spectral and structural features drove each prediction."

---

### [1:45 – 2:20] — Page 3: Production Shortfall Forecaster
* **🖥️ Screen Action:** Click on **`03 Production`**. Show the production trend chart (Planned vs. Predicted actual tonnage). Move the interactive sliders on the sidebar or page (change Equipment Availability % or Downtime Hours) and watch the predicted production and shortfall risk update live!
* **🗣️ Spoken Voiceover:**
  > "Next, Page 3 tackles the operational side: the **Production Shortfall Forecaster**.
  >
  > Mining output is highly dynamic. Our XGBoost Regressor predicts the exact daily tonnage output based on equipment availability, planned targets, blasting schedules, and rainfall.
  >
  > Watch what happens when I adjust the equipment downtime slider: as machine availability drops, our model instantly recalculates the forecast, projecting the expected tonnage and switching the shortfall risk tier from Low to High. This gives shift supervisors an early-warning horizon 24 to 48 hours before an actual production deficit occurs."

---

### [2:20 – 2:55] — Page 4: Real-Time Risk & Mitigation Engine
* **🖥️ Screen Action:** Click on **`04 Risk & Recommendations`**. Highlight the color-coded risk cards (Critical, High, Medium, Low). Click to expand one of the "Critical" or "High" risk alerts to showcase the automated root cause and prescriptive recommendations.
* **🗣️ Spoken Voiceover:**
  > "Predicting a shortfall is helpful, but solving it is what creates real value. That is the job of Page 4: our **Prescriptive Risk & Mitigation Engine**.
  >
  > The platform continuously screens all active mines and classifies risk into four distinct categories: Low, Medium, High, and Critical.
  >
  > When an alert triggers—for instance, here at Dongri Buzurg during a heavy monsoon storm—the engine doesn't just display a warning; it provides **Actionable Prescriptive Steps**: such as deploying auxiliary dewatering pumps to bench levels, rerouting haulers to stabilized haul roads, and reallocating standby excavators. This directly mitigates downtime on the ground."

---

### [2:55 – 3:25] — Page 5: Data Explorer & Transparency Hub
* **🖥️ Screen Action:** Click on **`05 Data Explorer`**. Switch between tabs: *Operational Data*, *Geological & Satellite Features*, and *Weather Records*. Click on the **Download CSV** button to show that all processed datasets are readily accessible.
* **🗣️ Spoken Voiceover:**
  > "Page 5 is our **Data Explorer Hub**, emphasizing full data transparency and auditability.
  >
  > Users can inspect every row of our operational records, satellite reflectance values, and ERA5 weather indicators. 
  >
  > We provide distribution histograms, correlation matrices, and filtering controls across all parameters. Mine engineers can immediately export filtered subsets as CSV files for their local reporting or offline analysis."

---

### [3:25 – 3:55] — Page 6: Model Performance & Benchmarks
* **🖥️ Screen Action:** Click on **`06 Model Performance`**. Show the **ROC-AUC curve**, the **Confusion Matrix**, and the **Regression Residuals chart** ($R^2$, MAE, RMSE metrics).
* **🗣️ Spoken Voiceover:**
  > "On Page 6, we provide comprehensive **ML Diagnostics and Validation**.
  >
  > We benchmark our prospectivity ensemble across 5-fold cross-validation, achieving an exceptional **ROC-AUC of over 0.94**, enhanced by SMOTE class balancing to handle deposit sparsity.
  >
  > For our production regressor, we evaluate Mean Absolute Error, Root Mean Squared Error, and an **R-squared exceeding 0.90**, ensuring that forecasts remain dependable under shifting seasonal and mechanical conditions."

---

### [3:55 – 4:25] — Pages 7, 8 & 9: Architecture, What-If Simulator & Value Chain
* **🖥️ Screen Action:** Quickly navigate through **`08 What-If Simulator`** and **`09 Exploration-Ops Link`**. On Page 8, drag a slider to test a sudden "60mm Monsoon Rainfall Shock" and show the financial impact. On Page 9, show the link connecting new prospective targets to future mine life extension.
* **🗣️ Spoken Voiceover:**
  > "Our platform also features advanced decision modules:
  >
  > In **Page 8: What-If Scenario Simulator**, mine managers can run stress tests—such as simulating an unexpected 60-millimeter monsoon cloudburst or a 3-hour blasting delay—to see the immediate financial and tonnage shortfall impact.
  >
  > And on **Page 9: Exploration-to-Operations Link**, we bridge the macro exploration findings with long-term mine planning, calculating how newly identified satellite prospects replenish depleting pit reserves and extend the operating life of MOIL's processing plants."

---

### [4:25 – 4:45] — Conclusion & Future Scope
* **🖥️ Screen Action:** Return to the **Dashboard** or **Architecture Page (`07 Architecture`)**. Show the clear system flowchart.
* **🗣️ Spoken Voiceover:**
  > "In conclusion, **MINEX** provides MOIL with a closed-loop digital twin: space technology identifies *where future reserves lie*, while operational AI prevents *where today's production might fail*.
  >
  > In our future roadmap, we plan to ingest high-resolution spaceborne hyperspectral data like EnMAP and PRISMA, integrate live IoT CAN-bus sensors from haul trucks, and automate 3D subsurface deposit modeling.
  >
  > Thank you for watching!"

---

## 💡 Quick Tips for the Recording:
1. **Resolution:** Record at 1080p (1920x1080) for sharp text and map rendering.
2. **Theme:** The dark theme is already configured in `.streamlit/config.toml`—it looks modern and professional on video.
3. **Cursor:** Enable mouse cursor highlighting in your recording software (OBS Studio, Loom, or Windows Game Bar: `Win + G`).
4. **Pacing:** Speak at a steady, conversational pace; pause for half a second when switching pages so viewers can absorb the visuals.
