# HeatWave Guardian

**HeatWave Guardian** is a personalized physiological heat-stress assessment, forecasting, and disaster intelligence platform engineered for outdoor workers, industrial enterprises, and municipal authorities in India. 

Developed for **Smart India Hackathon (SIH 2026) — Problem Statement ID 26083: Extreme Heatwave Early Warning and Human Thermal Stress Index**, the platform combines live atmospheric observations, the **Liljegren et al. (2008)** Wet-Bulb Globe Temperature (WBGT) thermodynamic model, **India Meteorological Department (IMD)** regional threshold criteria, and transparent physiological vulnerability adjustments to generate actionable hourly work-rest schedules and area-wide resilience intelligence.

---

## Table of Contents
- [1. Problem Statement & Mission](#1-problem-statement--mission)
- [2. Feature Set (Features A–M)](#2-feature-set-features-am)
- [3. Core Architecture & Pipeline](#3-core-architecture--pipeline)
- [4. Data Honesty & Classification](#4-data-honesty--classification)
- [5. Technology Stack](#5-technology-stack)
- [6. Installation & Setup](#6-installation--setup)
- [7. Local Development & Serving](#7-local-development--serving)
- [8. Environment Variables](#8-environment-variables)
- [9. REST API Endpoints](#9-rest-api-endpoints)
- [10. Production Data Reliability, Resilience & External Data Audit](#10-production-data-reliability-resilience--external-data-audit)
- [11. Testing & Verification](#11-testing--verification)
- [12. Production Cloud Deployment](#12-production-cloud-deployment)
- [13. Scientific Limitations & Disclaimers](#13-scientific-limitations--disclaimers)

---

## 1. Problem Statement & Mission

**SIH 2026 Problem Statement ID 26083**: Extreme Heatwave Early Warning and Human Thermal Stress Index.

Extreme heat represents an existential threat to outdoor laborers, construction workers, agricultural workers, delivery executives, and factory personnel across India. Traditional heat indices (such as dry-bulb air temperature or simplified heat indices) fail because:
1. They ignore the dramatic cooling impairment caused by high relative humidity.
2. They omit solar radiative heating and wind convective cooling.
3. They apply arbitrary single thresholds across diverse climates (e.g. desert vs. coast vs. hills).
4. They treat all individuals identically, ignoring age, lack of heat adaptation, heavy metabolic exertion, and personal protective equipment (PPE).

**HeatWave Guardian** addresses these systemic challenges through a unified biometeorological pipeline where **"40°C isn't equally dangerous to everyone."**

---

## 2. Feature Set (Features A–M)

### Feature A & H: Expanded Nationwide Geocoding & Location Coverage
- Dynamic geocoding autocomplete across all Indian cities, districts, and towns via Open-Meteo Geocoding.
- In-memory TTL caching layer (`services/geo_service.py`) to eliminate redundant upstream calls.
- Strict India-only filtering (`country_code == 'IN'`) with population-based administrative ranking and coordinate boundary validation (`-90..90`, `-180..180`).
- Seamless backward compatibility with preset cities (Delhi, Hyderabad, Mumbai, Chennai, Kolkata, Pune, Ahmedabad, Nagpur, Jaisalmer, Bhubaneswar, Khammam).

### Feature B: Live 5-Day Personalized Heat-Risk Forecast (`/forecast`)
- Fetches 5-day hourly atmospheric projections (temperature, relative humidity, 10m wind speed, solar radiation).
- Evaluates every forecast day through the exact core pipeline: daily maximum temperature ($T_{\max}$), peak Liljegren WBGT, IMD classification, risk fusion, personal vulnerability escalation, and ISO 7243 work/rest splits.
- Interactive dual-axis **Chart.js** trajectory displaying outdoor WBGT (°C) alongside Personalized Risk Severity (1–5).
- 5-Day interactive cards with day-selector drawer and comprehensive chronological breakdown table.

### Feature C: Historical Validation Dashboard (`/validation`)
- Retrospective evaluation of the catastrophic **2015 Andhra Pradesh & Telangana heatwave** (15 May – 01 June 2015) across 18 consecutive days.
- Interactive station selector: **Khammam** (12/18 days flagged Heatwave+) and **Hyderabad** (0/18 days flagged Heatwave).
- Comparative multi-series Chart.js timeline showing computed ERA5 values vs. IMD qualifying thresholds (Plains: 40°C Heat Day, 42°C Heatwave, 44°C Severe Heatwave).
- Transparent limitation disclosure: explains the published **ERA5 reanalysis coarse grid (~25km) cold bias over India**, which smoothed localized station spikes (e.g. Hyderabad station 46.0°C smoothed to 41.6°C).

### Feature D: Institute-Level Heat-Risk Dashboard (`/institute`)
- Multi-site occupational monitoring for colleges, manufacturing plants, infrastructure projects, and logistics hubs.
- Real-time aggregation: Total Sites, Extreme Risk, High Risk, Moderate Risk, and Active Workforce Count.
- Site task table evaluating live WBGT, IMD status, personal task demands, PPE restriction, and ISO 7243 work-rest cycles.
- Site Deep-Dive Modal revealing transparent calculation drivers and shift supervisor safety checklist.

### Feature E: Authority & Municipal Heat-Risk Map (`/authority`)
- Geospatial risk surveillance powered by an interactive **Leaflet map** with color-coded risk markers across major Indian municipal centers.
- Interactive popup inspection cards displaying live ambient temperature, Liljegren WBGT, IMD status, and demographic exposure proxies.
- National Disaster Management Authority (NDMA) Heat Action Plan (HAP) standard operating matrix (Red Emergency, Orange Warning, Yellow Watch, Green Normal).
- Area filter: view All Centers, Red/Orange Alerts, or Extreme Risk zones.

### Feature F: Experimental Mortality Risk Index (`/api/mortality-risk`)
- Modular proxy population heat-stress index on a 0.0 to 10.0 scale (`services/mortality_service.py`).
- Factors: WBGT severity (0–4.5 pts), IMD regional departure (0–2.5 pts), cumulative heat duration (0–1.5 pts), and demographic exposure (0–1.5 pts).
- **Strict Data Honesty**: Clearly titled *"Experimental Mortality Risk Indicator (Proxy Index)"* with mandatory disclaimer: *"NOT a clinically or epidemiologically validated mortality prediction."* Never fabricates individual mortality counts.

### Feature G: Cooling-Station Siting Recommender (`/cooling-stations`)
- Algorithmic decision-support module evaluating candidate urban transit corridors, wholesale markets, and informal labor zones (`services/cooling_recommender.py`).
- Explainable scoring: $Score = f(\text{WBGT Heat Load [45%]} + \text{Hospital Distance [25%]} + \text{Worker Density [20%]} + \text{Canopy Deficit [10%]})$ on a 0–100 scale.
- Ranked priority list with suggested infrastructure (Emergency Misting Canopy, Shaded Water Piao, Mobile Tanker Dispatch) and transparent rationale.

### Feature I: Demographic Data Integration Architecture (`services/demographic_service.py`)
- Clean adapter interface with structured schema: population, density, elderly %, children %, outdoor worker %, and informal housing %.
- Guaranteed `is_demo_data: True` labeling on all proxy values to maintain absolute data honesty until official Census/NFHS APIs are connected.

### Feature J: Government Heat Action Plan (HAP) Integration (`services/hap_service.py`)
- Extensible integration framework modelled on NDMA guidelines and the Ahmedabad Municipal Corporation Heat Action Plan.
- Maps environmental risk to standard color codes (Red, Orange, Yellow, Green), departmental response duties, and public health advisories.

### Feature K: SMS / WhatsApp Alert Integration (`services/alert_service.py`)
- Emergency notification engine supporting SMS and WhatsApp delivery.
- Twilio provider adapter configured strictly via environment variables (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`).
- **Simulation Mode**: If credentials are not provided, the application runs without crashing, generates formatted alerts, logs them to an in-memory audit trail with status `"SIMULATED"`, and displays an informative simulation badge in the UI.
- Cooldown mechanism: enforces a 60-minute window to prevent duplicate spamming.

### Feature L: What-If Scenario Simulator (`/what-if`)
- Deterministic occupational intervention simulator comparing baseline vs. proposed operational scenarios under identical live atmospheric conditions (`/api/simulate-scenario`).
- Interactive adjustments for Workload Intensity, Recovery Setting (Shade / AC / Direct Sun), Clothing/PPE, and Heat Adaptation.
- Transparent side-by-side comparison: Action Level shift (e.g. Very High -> Moderate), ISO 7243 work/rest split delta, and causal physiological drivers changed without misleading fake percentages.
- Preset operational scenarios: *Construction + Shade Canopy*, *Factory + AC Breakroom*, *PPE -> Breathable Cotton*.

### Feature M: Product UI/UX Design System & Information Architecture
- **Unified Navigation Modes**: My Risk (`/`), 5-Day Forecast (`/forecast`), What-If (`/what-if`), Worksite Dashboard (`/worksite` & `/institute`), Cooling Stations (`/cooling-stations`), Authority Map (`/authority`), Historical Validation (`/validation`), and Scientific Methodology (`/methodology`).
- **Instant Decision Answers (3-Second Rule)**: RiskSummary hero component answering *Where am I?*, *What is the Action Level?*, *Why?*, *What should I do?*, *When is the next peak?*, and *How confident is the result?*.
- **Authoritative Design Tokens**: High-contrast, WCAG-compliant status badges (`--risk-low`, `--risk-moderate`, `--risk-high`, `--risk-very-high`, `--risk-extreme`), structured card elevations, and keyboard accessibility focus rings.
- **Explainable Multi-Tier Intelligence**: Level 1 (Plain-English Operational Summary), Level 2 (Detailed Drivers & Triggered Decision Rules), Level 3 (Technical Biometeorological Audit Trace JSON).
- **⚡ Presentation Mode**: One-click toggle in the header enabling instant scenario presets.

### Feature M: Comprehensive Automated & Browser Regression Testing
- 100% test pass rate across engine physics, personalization rules, expanded endpoints, modular services, and edge cases.
- Live browser-tested user flows with zero console errors.

### Feature N: Work/Rest + Exposure Intervention Engine (ISO 7243 & Occupational Decision Support)
- **Primary Product Principle**: Rather than merely stating *"Risk = High"*, HeatWave Guardian answers *"Given this environmental heat level and this work situation, what operational precautions should be taken?"*
- **Explicit 5-Layer Exposure Context**:
  1. *Environmental Heat Level*: Liljegren WBGT, dry-bulb air temperature, solar irradiance, 10m wind speed, direct sun exposure.
  2. *Work Context*: Metabolic intensity (Light ~150W, Moderate ~250W, Heavy ~400W, Very Heavy >500W), continuous vs. intermittent pattern, shift duration.
  3. *Worker Context*: Acclimatization state (Acclimatized, Partially Acclimatized, Unacclimatized).
  4. *Clothing / PPE Context*: Clothing Adjustment Values (CAVs), fabric permeability, impermeable vapor-barrier PPE detection (+11°C effective thermal load).
  5. *Recovery Environment Context*: Air-conditioned indoor, shaded outdoor, direct sun (compromised), warm unconditioned indoor, or unknown setting.
- **Decision-Support Recommendation Levels**:
  - `Normal precautions`
  - `Enhanced precautions`
  - `Frequent recovery needed`
  - `High-exposure controls required`
  - `Suspend/defer strenuous exposure pending site assessment`
  - `Site-specific assessment required` (triggered when environmental metrics are missing or incomplete)
- **Categorized Operational Controls**: Transparent, traceable precautions classified into 7 operational domains:
  `HYDRATION`, `SCHEDULING`, `RECOVERY`, `WORKLOAD`, `ACCLIMATIZATION`, `MONITORING`, and `CLOTHING_PPE`.
- **Forecast Diurnal Peak-Risk Windows**: Algorithmic evaluation of 5-day hourly forecast curves to identify lower-burden morning/evening windows vs. peak midday hazard hours, offering concrete schedule adjustment advice.
- **Safety Language Integrity**: Formulated strictly as screening decision support; cleanly avoids prohibited terms ("guaranteed safe", "safe work/rest ratio", "drink exactly X litres", "prevents heat illness").

### Feature O: Explainable Risk Engine + Decision Timeline (Deterministic Rule-Based Intelligence)
- **Primary Product Principle**: HeatWave Guardian never leaves the user blind to *"Risk = High"*. It answers: *WHY is it this level? WHAT is driving it? WHAT is protecting? WHAT can change? WHAT happens during the timeline? WHY was each control recommended? WHAT information is uncertain?*
- **Deterministic 6-Stage Decision Trace (`build_decision_trace`)**:
  Captures `1. Inputs -> 2. Derived Conditions (Liljegren WBGT) -> 3. Environmental Level (IMD Criteria) -> 4. Exposure Context -> 5. Action Level -> 6. Recommended Controls` with explicit triggered rule IDs citing ISO 7243:2017, ACGIH TLVs, and IMD criteria.
- **Decision Factor Classification (`classify_decision_factors`)**:
  Separates variables into `primary_drivers` (material determinants), `contributing_factors` (secondary load), `protective_factors` (physiological defenses), `uncertainty_limitations` (plain-English data caveats), and `actionable_levers` (modifiable scenario parameters).
- **Counterfactual Factor Sensitivities (`evaluate_factor_sensitivities`)**:
  Tests 5 levers (Workload, Acclimatization, Recovery Environment, Clothing, Solar Exposure) under identical weather. Explicitly declares when modifying a factor does *not* alter the category under extreme ambient conditions, explaining why the ambient thermal barrier dominates.
- **Operational Decision Timeline (5 Diurnal Shift Windows)**:
  Groups 24-hour diurnal progression into 5 operational shift periods: *Early Morning (05:00–08:00)*, *Morning Build-up (08:00–11:00)*, *Peak Diurnal Hazard (11:00–16:00)*, *Late Afternoon (16:00–19:00)*, and *Night & Rest (19:00–05:00)*.
- **Multi-Day Decision Story & Day-to-Day Transitions**:
  Formulates daily operational narratives and tracks structured day-to-day transitions (`ESCALATING`, `IMPROVING`, `STABLE`).
- **Traceable Controls & Tiered Explanations**:
  Enriches recommendations with `trigger`, `rule`, `source`, `confidence`, and `limitation`. Provides 3 explanation depths: *Quick (Level 1)*, *Detailed (Level 2)*, and *Technical / Audit (Level 3)*.

### Feature P: Product UI/UX System & Actionable Information Architecture
- **Primary Product Principle**: The interface makes a complex biometeorological engine feel simple, trustworthy, actionable, and understandable within seconds—avoiding generic weather widgets, sci-fi gimmicks, or cliché flame graphics.
- **Immediate Operational Clarity**: The top of the risk results immediately answers 6 core questions before technical metrics:
  1. *WHERE AM I?* (Current city, region, dynamic coordinates, and freshness status).
  2. *WHAT IS THE CURRENT HEAT ACTION LEVEL?* (Dominant visual risk badge with accessible text and contrast).
  3. *WHY?* (Plain-English synthesis of Liljegren WBGT, metabolic workload, acclimatization, and PPE).
  4. *WHAT SHOULD I DO?* (Actionable ISO 7243 work-rest split, recovery setting, hydration protocol).
  5. *WHEN IS THE NEXT PEAK?* (Diurnal peak-risk exposure window, e.g., 12:00–16:00).
  6. *HOW CONFIDENT IS THE RESULT?* (Data quality score, input verification, and freshness status).
- **Expandable Explainability ("Why this Risk?")**:
  A tiered disclosure accordion allowing non-technical operators to understand primary drivers immediately, while technical reviewers can inspect the full mathematical decision trace, counterfactual factor sensitivities, and standard citations.
- **Design System & Aesthetics**:
  Curated color palette using Hanken Grotesk and JetBrains Mono typography, 8pt spacing grid, responsive card hierarchy, accessible contrast (WCAG 2.1 AA), and zero reliance on color alone.
- **Zero Horizontal Overflow Mobile-First Layout**:
  Responsive reflow tested down to 390px mobile viewports with zero horizontal scrolling or clipping.

### Feature Q: Production Data Reliability & Atmospheric Resilience Pipeline
- **Centralized Atmospheric Client (`services/weather_service.py`)**:
  Production-grade client handling Open-Meteo current conditions and multi-day hourly forecasts.
- **Resilience Engineering**:
  - *Explicit Timeouts*: 8s for current weather, 10s for multi-day forecast; eliminates indefinite socket hangs.
  - *Controlled Transient Retry*: Maximum 1 retry with exponential backoff on connection drops and upstream 5xx errors; strictly rejects 4xx client errors without retry.
  - *Physical & Physiological Range Validation*: Enforces valid atmospheric boundaries (-10°C to 65°C, 0–100% RH, non-negative wind and solar radiation); rejects corrupted or out-of-range sensor readings safely.
  - *Strict Structural Validation*: Confirms presence of all required hourly series; rejects incomplete forecast responses (<5 distinct days) without synthetic fabrication.
  - *Freshness Tracking*: Classifies observations into `FRESH` (<=1h old), `STALE` (>1h old with visible UI warning badge), or `UNKNOWN`.
  - *Zero Synthetic Fallback for Live Queries*: Completely eliminates silent substitution of hardcoded fallback weather (e.g. 32°C/55% RH) during live assessments; fails safely and transparently with informative 502 error envelopes.
  - *Distributed Tracing*: Assigns and propagates unique correlation IDs (`X-Request-ID`) across all requests and error responses.
### Feature R: Intervention Intelligence Engine (Step 6 — Explainable Action Plan & Decision Support)
- **Primary Product Principle**: Rather than merely broadcasting alerts or predicting temperatures, HeatWave Guardian answers: *"Given the predicted heat conditions and exposure context, what controllable factors could reduce exposure or move the task away from the highest-risk period?"*
- **Occupational Control Hierarchy Organization**:
  1. *Elimination / Avoid Peak Exposure (Level 1)*: Identifies candidate morning/evening windows from hourly NWP forecasts that bypass diurnal peak WBGT windows.
  2. *Engineering / Task Modification (Level 2)*: Recommends metabolic pacing, task rotation, and mechanical lifting aids for heavy tasks (>350W).
  3. *Environmental Engineering Controls (Level 3)*: Deploys portable sunshading, airflow/misting fans, and task relocation to shaded staging areas.
  4. *Recovery Facility Access (Level 4)*: Mandates air-conditioned or ventilated shaded rest shelters; integrates with `CoolingRecommenderService` to surface proximate municipal cooling hubs.
  5. *Administrative Controls & Escalation (Level 5)*: Formulates ISO 7243 hourly work/rest pacing, cumulative shift caps (<=6h), peer buddy surveillance, and supervisor review thresholds.
  6. *PPE Considerations & Microclimate (Level 6)*: Flags vapor-barrier and heavy garments (+2°C to +11°C effective thermal load); deploys microclimate cooling vests.
- **Strict Safety PPE Guardrail**: The engine strictly prohibits instructing or authorizing the removal of mandatory safety PPE (e.g. chemical suits, respirators, fall protection, hard hats). Instead, it shortens continuous work cycles and mandates cooled recovery intervals.
- **Role-Specific Decision Support**:
  - *Worker View*: Simple, direct, personal precautions (hydration intervals, peer buddy checks, hourly pacing, no PPE removal).
  - *Supervisor / Worksite View*: Tactical workforce management, shift scheduling options, mandatory work/rest split, water station verification, and escalation triggers.
  - *Municipal Authority View*: Strategic disaster planning, public gathering timing adjustments, cooling center deployment, and labor safety field inspections.
- **Unified Decision Timeline**:
  Tracks `TIME → FORECAST → ML LOCALIZED CONDITION → WBGT → RISK → PEAK PERIOD → INTERVENTION → STATUS` across 5 operational states: `ROUTINE MONITORING`, `ENHANCED CONTROLS`, `MODIFY / RESTRICT EXPOSURE`, `SUPERVISOR REVIEW`, `AUTHORITY REVIEW` (strictly never called "guaranteed safe").
- **What-If Simulation Link**: One-click simulation connecting candidate shift windows or proposed controls directly to `ScenarioSimulator`, generating transparent baseline vs. counterfactual comparison deltas labeled clearly as `SIMULATED SCENARIO`.
- **Structured Alert Object Preparation**: Emits standardized alert payloads with full model provenance (`engine_version`, `Liljegren (2008)`, `ISO 7243:2017`, `NDMA Heat Action Plan 2024`) ready for downstream notification dispatch.
### Feature S: Role-Based, Multilingual & Low-Connectivity Alert Intelligence (Step 7)
- **Role Model Matrix**:
  - *Worker*: Current/forecast risk, peak window, hydration intervals, work/rest pacing, early heat-exhaustion symptom recognition.
  - *Supervisor*: Affected tasks, rotational shifts, buddy checks, cooling station access, work restriction enforcement during peak hours.
  - *Site Manager*: Site-wide workforce exposure index, cold water tanker logistics, operational stand-down for heavy outdoor work, audit trail.
  - *Municipal Authority*: Monitored zones/wards, IMD departure, HAP alert level (Yellow/Orange/Red), public health warnings, cooling shelter activation.
  - *Healthcare Facility*: Thermal load timing, surge preparedness signal, IV fluid & ORS reserves, emergency cooling beds. **STRICT GUARDRAIL**: Operational readiness indicator only; strictly prohibits unvalidated hospital admission counts or mortality projections.
- **Controlled Multilingual Engine**:
  - High-fidelity, reviewed templates for **English (`en`)**, **Hindi (`hi`)**, and **Marathi (`mr`)**.
  - **Zero LLM translation hallucination** for safety-critical instructions.
  - Strict preservation of critical numbers, physical units (°C, min), and ISO timestamps across all languages.
  - Standardized structure: `[RISK]`, `[LOCATION]`, `[PEAK WINDOW]`, `[ACTION]`, `[CONFIDENCE]`, `[DATA TIMESTAMP]`.
- **Low-Connectivity Compact SMS Standard**:
  - Ultra-compact format ($\le 160$ characters) for 2G, GSM 7-bit, or low-bandwidth environments:
    `[HWG ALERT] {LOC}: {RISK} (WBGT {wbgt}C). Peak {peak}. {action}. Conf:{conf}. {time}`
  - Retains all essential actionable metadata without truncation.
- **Human-in-the-Loop (HITL) Governance**:
  - Lifecycle: `DRAFT` $\rightarrow$ `PENDING_APPROVAL` $\rightarrow$ `APPROVED` $\rightarrow$ `DISPATCHED` $\rightarrow$ `AUDIT` (or `REJECTED`).
  - High-impact authority and site-wide suspension alerts default to requiring human approval.
  - Real-time approval and rejection APIs (`POST /api/alerts/<id>/approve`, `POST /api/alerts/<id>/reject`).
- **Consent, Quiet Hours & Privacy Protection**:
  - Granular recipient opt-in/opt-out preferences model.
  - Quiet hours window suppression (e.g. 22:00–06:00) with automatic bypass for critical/extreme emergency thresholds.
  - Strict privacy protection: phone numbers and emails masked in logs (`+91 98*** ***10`), SHA-256 fingerprinting for duplicate cooldown tracking.
- **Data Honesty & Simulation Dispatch Guardrail**:
  - Retains full provenance: source data timestamp, Liljegren (2008) physics, IMD standard, confidence score.
  - Counterfactual simulations (`is_simulated: true`) are flagged `[SIMULATED SCENARIO]` and strictly blocked from live telecom dispatch.
- **Interactive UI Command Center**:
  - Dedicated `/alerts` operations dashboard with live preview sandbox, language selector, compact SMS character counter, and HITL authorization controls.

---

## 3. Core Architecture & Pipeline

### Pipeline Flow: OBSERVE → MODEL → PERSONALIZE → EXPLAIN → PREDICT → ACT → VALIDATE

All endpoints and client interfaces plug directly into the established single source of truth risk pipeline:

```
[OBSERVE] Live Atmospheric Observations (Ta, RH, Wind 10m, Solar Shortwave via Open-Meteo)
    │
    ▼
[MODEL — Environmental Heat Stress]
    ├─► Liljegren et al. (2008) WBGT (Psychrometric Tnwb + Energy Balance Tg)
    │     └─► WBGT Band: Safe (<27°C) | Caution (27–29.3°C) | Ext. Caution (29.4–31°C) | Danger (31.1–32.1°C) | Ext. Danger (>32.1°C)
    ├─► IMD Regional Threshold Criteria (Plains 40°C | Coastal 37°C | Hilly 30°C)
    │     └─► IMD Status: Normal (<0°C) | Heat Day (0–2°C) | Heatwave (2–4°C) | Severe Heatwave (>=4°C)
    └─► 20-Cell Categorical Risk Fusion Matrix (Lookup table without double-counting)
          └─► Environmental Level: Low | Moderate | High | Very High | Extreme
    │
    ▼
[PERSONALIZE — Operational Exposure & Individual Context]
    ├─► Operational Exposure Layer (ISO 7243:2017 & ACGIH)
    │     ├─ Metabolic Workload: Light (~150W) | Moderate (~240W) | Heavy (~360W) | Very Heavy (~480W)
    │     ├─ Clothing Adjustment Values (CAVs): Light (0°C) | Heavy (+2°C) | PPE (+6°C effective WBGT)
    │     └─ Solar Microclimate: Direct sun vs. shaded/indoor work
    ├─► Acclimatization as a First-Class Factor (NIOSH 2016 / OSHA Guidance)
    │     ├─ Acclimatized (7–14 days regular exposure) | Partially Acclimatized (returning worker) | Unacclimatized (new worker)
    │     └─ Progressive Pacing Schedules (New worker 5-day: 20% to 100%; Returning worker 4-day: 50% to 100%)
    └─► Physiological Vulnerability Context (Age extremes: <18 youth and >=60 older adult thermoregulatory strain)
          └─► Overall Action Level: Low | Moderate | High | Very High | Extreme
    │
    ▼
[EXPLAIN & ACT — Structured Intelligence & Interventions]
    ├─► Traceable Drivers: Environmental drivers, operational drivers, vulnerability drivers, protective factors
    ├─► Qualitative Confidence Engine: High | Medium | Limited (completeness, freshness, physical plausibility)
    ├─► Work/Rest Decision Support: ISO 7243 hourly screening guidance
    └─► Occupational Hydration Protocols: Non-prescriptive, continuous potable water access + electrolyte guidelines
```

---

## 4. Data Honesty & Classification

In strict compliance with project data honesty principles:

| Component | Status | Data Source / Protocol |
| :--- | :--- | :--- |
| **Real-Time Weather Observations** | **REAL / LIVE** | Open-Meteo Public API (HTTPS, no API key required, zero synthetic fallbacks). |
| **5-Day Atmospheric Projections** | **REAL / LIVE** | Open-Meteo 5-day hourly atmospheric models. |
| **Nationwide Geocoding** | **REAL / LIVE** | Open-Meteo Geocoding API filtered strictly to India with TTL caching. |
| **WBGT Thermodynamic Equations** | **REAL / FORMULA** | Liljegren et al. (2008) Newton-Raphson heat balance & Stull (2011) wet bulb. |
| **Regional IMD Thresholds** | **REAL / STANDARD** | India Meteorological Department official departure criteria. |
| **Historical Validation Benchmark**| **REAL / HISTORICAL**| 2015 AP/Telangana Heatwave (ECMWF ERA5 Reanalysis vs. IMD ground records). |
| **Demographic Exposure Indicators** | **DEMO DATA** | Modular proxy dataset; clearly tagged `is_demo_data: true` in UI and JSON. |
| **Mortality Risk Index** | **EXPERIMENTAL PROXY**| Experimental population severity index (0–10 scale); includes non-medical disclaimer. |
| **Cooling-Station Siting** | **DECISION SUPPORT** | Multi-criteria spatial model; labelled as advisory decision-support planning aid. |
| **SMS / WhatsApp Alerts** | **CREDENTIALS / SIM** | Live via Twilio if env vars exist; otherwise graceful **Simulation Mode** audit log. |

---

## 5. Technology Stack

- **Backend Framework**: Python 3.10+ / Flask 3.1
- **WSGI Production Servers**:
  - `gunicorn` (Linux cloud platforms: Render, Railway, Ubuntu VPS)
  - `waitress` (Windows production serving & cross-platform local serving)
- **Frontend Architecture**: Semantic HTML5, Vanilla JavaScript (ES6+), Tailwind CSS CDN
- **Data Visualizations**: Chart.js (Dual-axis forecast trajectory & historical comparison)
- **Geospatial Mapping**: Leaflet 1.9.4 & OpenStreetMap standard raster tiles (CDN-based, zero API key required)
- **External Atmospheric APIs**: Open-Meteo Weather & Geocoding APIs (Public, HTTPS)
- **Messaging Adapter**: Twilio REST API (with automatic Simulation Mode fallback)

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10 or newer
- `pip` (Python package manager)
- `git`

### Step 1: Clone Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/heatwave-guardian.git
cd heatwave-guardian
```

### Step 2: Create and Activate Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Local Development & Serving

### Option A: Flask Development Server
```bash
python app.py
```
Access in your browser at:
```
http://127.0.0.1:5000/
```

### Option B: Production Waitress Server (Windows & Local Production)
```bash
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Option C: Production Gunicorn Server (Linux / macOS)
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## 8. Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Flask Mode & Network
FLASK_ENV=production
FLASK_DEBUG=0
HOST=0.0.0.0
PORT=5000
SECRET_KEY=replace-with-a-secure-random-32-byte-hex-string

# SMS / WhatsApp Alerts (Optional — If omitted, Simulation Mode is active)
# TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TWILIO_AUTH_TOKEN=your_auth_token_here
# TWILIO_PHONE_NUMBER=+1234567890
```

---

## 9. REST API Endpoints

### 1. Location Autocomplete
- `GET /api/search-location?q=<query>`
- Returns deduplicated Indian places prioritized by population with inferred region.

### 2. Check Personal Heat Risk
- `POST /api/check-risk`
- Payload: `location`, `latitude`, `longitude`, `age_group`, `experience`, `work_intensity`, `clothing`, `day_in_heat` (optional), `is_sun_exposed` (optional).
- Primary Structured Output (Phase 6 Model):
  - `environmental_level`: Baseline environmental risk (`Low`, `Moderate`, `High`, `Very High`, `Extreme`)
  - `operational_exposure_level`: Activity & clothing heat burden (`Low`, `Moderate`, `High`, `Very High`)
  - `overall_action_level`: Final operational action tier (`Low`, `Moderate`, `High`, `Very High`, `Extreme`)
  - `drivers`: Traceable environmental, operational, and vulnerability drivers
  - `protective_factors`: Mitigating factors (acclimatization, light clothing, shaded recovery)
  - `recommended_actions`: Contextual work/rest pacing and non-prescriptive hydration protocols
  - `confidence`: Qualitative confidence assessment (`level`: High/Medium/Limited, `score`, `reasons`, `completeness_pct`)
  - `acclimatization_context`: Status, protocol note, and NIOSH progressive pacing schedule
  - `work_rest_guidance`: Hourly work/rest screening minutes and instructions
  - `hydration_guidance`: Practical occupational drinking protocol (non-prescriptive)
  - Full backward compatibility: `risk_category`, `base_risk`, `personal_risk`, `wbgt_c`, `wbgt_band`, `thermal_stress`, `summary`, `work_minutes`, `rest_minutes`, `instruction`, `severity`, `risk_color`.

### 3. 5-Day Atmospheric Forecast & Hourly Risk Intelligence
- `GET` or `POST /api/forecast`
- Returns array of 5 sequential daily forecast evaluations enriched with:
  - `hourly_profile`: Complete 24-hour sequence of ISO 7243 risk profiles, hourly WBGT, and work/rest splits
  - `peak_risk_window`: Peak thermal stress period (e.g. `12:00–15:00`), peak hour, elevated risk boundaries, and environmental drivers
  - `lower_exposure_windows`: Data-driven lower-exposure daylight windows for scheduling strenuous tasks
  - `confidence`: Qualitative confidence level and explicit forecast uncertainty notice
  - Backward compatibility: Preserves standard 5-day list response contract.
- Optional query parameter: `view=intelligence` returns an executive summary object with `current_action_level`, `next_peak_window`, and `forecast_days`.

### 4. Operational What-If Exposure Simulator
- `POST /api/simulate-scenario`
- Evaluates baseline work conditions against modified operational scenarios under identical forecast heat stress.
- Powered strictly by the core biometeorological engine (`assess_full_risk_profile`).
- Explains physiological and ergonomic drivers causally (metabolic Watts delta, evaporative resistance, radiant flux) without arbitrary percentages or false clinical claims.
- Explicitly flags when user context was unprovided and resolved to `DEFAULT_EXPOSURE_SCENARIO`.
- Payload format:
  ```json
  {
    "baseline": {
      "workload": "heavy",
      "clothing": "heavy",
      "recovery_environment": "shade",
      "is_sun_exposed": true
    },
    "scenario": {
      "workload": "light",
      "clothing": "light",
      "recovery_environment": "cool_indoor",
      "is_sun_exposed": false
    },
    "weather_data": {
      "temp_c": 38.0,
      "wbgt_c": 32.5,
      "rh_pct": 50.0,
      "wind_kph": 10.0,
      "solar_wm2": 650.0
    }
  }
  ```
- Output provides comparative action levels, work/rest splits, causal explanations, protective factor deltas, and occupational safety disclaimers.

### 5. Institute Micro-Sites Evaluation
- `GET /api/institute/sites?institute_id=<id>`
- Evaluates live risk for all operational sites in a selected campus/enterprise.

### 5. Authority Geospatial Risk Layer
- `GET /api/authority/zones`
- Returns GeoJSON FeatureCollection of monitored Indian cities with live WBGT, IMD status, HAP alert level, and demographic proxies.

### 6. Cooling-Station Siting Recommender
- `GET /api/cooling-stations/recommendations?city=<optional_city>`
- Returns ranked candidate zones based on thermal load, hospital distance, and worker density.

### 7. Role-Based & Multilingual Alert Intelligence (Step 7)
- `POST /api/alerts/generate` or `POST /api/alerts/evaluate`:
  - Synthesizes structured alert objects across 5 roles (`worker`, `supervisor`, `site_manager`, `authority`, `healthcare_facility`).
  - Inputs: `location`, `risk`, `wbgt_c`, `temp_c`, `roles`, `language` (`en`, `hi`, `mr`), `channel`, `is_simulated`.
- `POST /api/alerts/preview`:
  - Real-time preview of standard structured layout and low-connectivity compact SMS (<160 chars).
- `GET /api/alerts`:
  - Retrieves stored alerts with filtering by `role`, `status`, `approval_state`, `urgency`, and `limit`.
- `GET /api/alerts/<alert_id>`:
  - Retrieves single standardized alert with chronological audit trail and model provenance.
- `POST /api/alerts/<alert_id>/approve`:
  - Human-in-the-loop authorization for high-impact alerts (`PENDING` -> `APPROVED`).
- `POST /api/alerts/<alert_id>/reject`:
  - Human-in-the-loop cancellation with documented reason (`PENDING` -> `REJECTED`).
- `POST /api/alerts/<alert_id>/dispatch`:
  - Dispatches approved alert with consent, quiet hours, cooldown, and simulation checks.
- `POST /api/alerts/send`:
  - Backward-compatible direct dispatch endpoint (SMS/WhatsApp/Dashboard/Simulation).
- `GET /api/alerts/preferences` & `POST /api/alerts/preferences`:
  - Recipient consent management, opt-in/opt-out status, quiet hours window, role, and preferred language.
- `GET /api/alerts/history`:
  - Returns telecom provider configuration and audit dispatch trail.

### 9. Experimental Mortality Risk Index
- `GET /api/mortality-risk?wbgt_c=32.5&temp_c=42.0&imd_status=Heatwave&location=Delhi`
- Returns proxy population score (0–10 scale), contributing factors, and medical disclaimer.

### 10. Government HAP Guidelines
- `GET /api/hap-guidelines?risk_category=High&imd_status=Heatwave&wbgt_c=31.5`
- Returns standardized NDMA/Ahmedabad HAP alert level and departmental response instructions.

### 11. Deterministic Decision Trace Audit Endpoint
- `GET /api/decision-trace?case_id=<case_id>` or `POST /api/decision-trace`
- Direct inspection endpoint for auditing the complete deterministic decision trace:
  `INPUTS -> DERIVED CONDITIONS -> ENVIRONMENTAL LEVEL -> EXPOSURE CONTEXT -> ACTION LEVEL -> RECOMMENDED CONTROLS`.
- Supports historical landmark presets (`khammam_2015_peak`, `hyderabad_2015_peak`) or custom evaluated scenarios.
- Returns 6 pipeline stages, triggered rules with standard citations, material determinants, and raw audit JSON.

### 12. Operational Health & Tracing Endpoint
- `GET /health` or `GET /api/health`
- Returns operational service health without secret leakage:
  - System status (`healthy`), version, UTC timestamp, environment.
  - Dependent service status: `weather_service` (Open-Meteo), `geocoding_service`, `decision_engine`.
  - Request tracing: returns unique correlation ID (`X-Request-ID`) in payload and HTTP headers.

---

## 10. Production Data Reliability, Resilience & External Data Audit

HeatWave Guardian operates on a strict production reliability principle: **Fail Safely and Transparently**. It will never crash from an upstream provider outage, silently invent missing weather, display stale observations as current, or substitute synthetic/demo weather into a live risk calculation.

```
REAL DATA -> VALIDATE -> NORMALIZE -> ASSESS QUALITY -> CALCULATE -> EXPLAIN LIMITATIONS -> FAIL SAFELY
```

### Complete External Data Audit (Phase 1)

| Dependency / Source | Endpoint & Purpose | Required Fields | Optional Fields | Units & Plausible Bounds | Rate Limits & Auth | Timeout & Retry Policy | Fallback Behavior & Freshness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Open-Meteo Weather API** | `https://api.open-meteo.com/v1/forecast`<br>Current surface observations | `temperature_2m`<br>`relative_humidity_2m`<br>`wind_speed_10m` | `shortwave_radiation_instant` | °C [-10, 65]<br>% [0, 100]<br>m/s [0, 100]<br>W/m² [0, 1500] | 10,000 calls/day<br>~600 calls/min<br>No API key required<br>Attribution required | **Current**: 8s timeout<br>Max 1 retry (0.3s backoff)<br>No retry on 4xx | **ZERO synthetic fallback**<br>Raises `WeatherServiceUnavailableError`<br>Returns HTTP 502 with friendly JSON<br>Freshness TTL: 3600s (1 hour) |
| **Open-Meteo Forecast API** | `https://api.open-meteo.com/v1/forecast`<br>5-day hourly forecast series | `hourly.time`<br>`hourly.temperature_2m`<br>`hourly.relative_humidity_2m`<br>`hourly.wind_speed_10m` | `hourly.shortwave_radiation` | °C [-10, 65]<br>% [0, 100]<br>m/s [0, 100]<br>W/m² [0, 1500] | 10,000 calls/day<br>No API key required | **Forecast**: 10s timeout<br>Max 1 retry (0.4s backoff)<br>No retry on 4xx | Rejects incomplete days (< 5)<br>Deduplicates duplicate timestamps<br>Freshness TTL: 14400s (4 hours) |
| **Open-Meteo Geocoding API** | `https://geocoding-api.open-meteo.com/v1/search`<br>Nationwide location resolution | `name`<br>`latitude`<br>`longitude` | `admin1`<br>`country_code`<br>`population` | Lat [-90, 90]<br>Lon [-180, 180]<br>Strict India filter (`IN`) | ~600 calls/min<br>No API key required | **Geocoding**: 10s timeout<br>1 retry on failure | Fallback to curated Indian city presets (`CITIES` in `engine.py`) if upstream fails |
| **ECMWF ERA5 Reanalysis** | Copernicus Climate Data Store (`services/dataset_registry.py`) | 2m temp, dewpoint, 10m u/v wind components, downward solar | N/A | °C, %, m/s, W/m² | Immutable reanalysis archive | In-memory archive (no runtime HTTP calls) | Complete continuous coverage<br>Documented 2.0°C–4.4°C pre-monsoon cold bias disclosed |
| **IMD Station Bulletins** | IMD National Data Centre (`services/dataset_registry.py`) | Max temperature, departure from normal, heatwave flag | N/A | °C, category | Point surface observatory ground-truth | In-memory archive (no runtime HTTP calls) | Official reference ground stations (Khammam, Hyderabad Begumpet) |
| **OpenStreetMap Tile Provider** | `https://tile.openstreetmap.org/{z}/{x}/{y}.png`<br>Raster map display | Map coordinates | Zoom levels (0–19) | Standard spherical Mercator | Free open-source<br>Zero API key required<br>Attribution required | Client-side Leaflet fetch | `tileerror` handled gracefully with user notice; risk calculations remain 100% operational |
| **Cooling-Station Siting Repo** | `services/cooling_recommender.py`<br>Urban transit corridors | Candidate zone coordinates, facility distance, worker density | Canopy % | Distance (km), %, score (0–100) | Local decision-support engine | Internal computation | Clear advisory disclaimer: planning aid, not executive directive |
| **Demographic Proxy Service** | `services/demographic_service.py`<br>Urban district vulnerability | Population, elderly %, children %, outdoor worker %, housing % | Density | %, count | Local mock registry | Internal computation | Guaranteed `is_demo_data: True` labeling to preserve scientific honesty |

### Centralized Service Architecture (Phase 2)
External data access is strictly consolidated into dedicated service modules:
- `WeatherService` (`services/weather_service.py`): Fetching, validating, normalizing, freshness evaluation, and caching for atmospheric observations and NWP forecasts.
- `GeoService` (`services/geo_service.py`): Geocoding validation, Indian locality ranking, duplicate deduplication, and caching.
- `DatasetRegistry` (`services/dataset_registry.py`): Provenance, validation eligibility, and metadata tracking for historical datasets.
- `CoolingRecommenderService` (`services/cooling_recommender.py`): Multi-criteria spatial siting recommendations.
- `DemographicService` (`services/demographic_service.py`): Demographic proxy indicators with explicit demo-mode disclosure.

### Request Timeouts & Retry Policy (Phases 3 & 4)
- **Timeouts**: Every external HTTP request enforces an explicit timeout (8s for current weather, 10s for forecast, 10s for geocoding). No request can hang the server indefinitely.
- **Controlled Retries**: Max 1 retry with exponential backoff (0.3s–0.4s) for transient network timeouts or HTTP 5xx server errors.
- **Strictly Non-Retryable**: Deterministic 4xx client errors (e.g. 400 Bad Request) and invalid inputs raise `WeatherServiceValidationError` immediately without retrying, preventing retry storms.

### Response Validation & Physical Plausibility (Phase 5)
Every external payload is strictly validated before use:
- Geographic coordinates: numeric, latitude in `[-90.0, 90.0]`, longitude in `[-180.0, 180.0]`.
- Temperature: `[-10.0°C, 65.0°C]` (covers extreme Indian desert highs to Himalayan foothills).
- Relative Humidity: `[0.0%, 100.0%]`.
- Wind Speed: `[0.0 m/s, 100.0 m/s]`.
- Solar Radiation: `[0.0 W/m², 1500.0 W/m²]`.
- Forecast Series: enforces minimum 5-day series length, verifies equal array lengths across variables, and deduplicates repeated timestamps.

### Unit Normalization & Timezone Hardening (Phases 6 & 7)
- **Internal Standard**: All atmospheric metrics are converted once at the service boundary to Celsius (°C), Percentage (%), Meters/second (m/s), and Watts/square meter (W/m²).
- **Timezone Precision**: Open-Meteo responses with `"timezone": "auto"` provide `"utc_offset_seconds"` (e.g. `19800` for India Standard Time, UTC+05:30). The freshness evaluator parses this offset to construct timezone-aware `datetime` objects, eliminating false staleness and timezone skew across IST and UTC.

### Stale Data Detection & Confidence Pipeline (Phases 8 & 13)
- Observations carry explicit freshness states: `FRESH` (age $\le$ 3600s), `STALE` (age $>$ 3600s), or `UNKNOWN`.
- When data is `STALE`, `evaluate_data_quality()` automatically adds a warning flag, adds an explanation reason, and degrades qualitative confidence (downgrades `High` to `Medium` or `Limited`, applying a 0.20 score penalty).
- Stale data is prominently displayed with an amber warning badge in the UI.

### Missing Data Policy & Zero Synthetic Fallback (Phases 9 & 10)
- **Required Missing Data**: `temperature_2m`, `relative_humidity_2m`, and `wind_speed_10m` are mandatory. If missing, the request fails safely with `WeatherServiceValidationError`—no fabricated defaults are inserted.
- **Optional Missing Data**: Solar radiation defaults safely to clear-sky/0.0 W/m² with documented limitations in the confidence trace.
- **Zero Synthetic Fallback**: Live user queries to `/api/check-risk` and `/api/forecast` never substitute synthetic/demo weather. If external providers are unavailable, the endpoints return a transparent HTTP 502 with a friendly explanation (`"Weather data is temporarily unavailable. Please try again."`). Demo data exists strictly for test fixtures and clearly labeled demo modes.

### Safe Failure Modes & Graceful Degradation (Phases 11 & 12)
The application handles 12 distinct failure modes without crashing:
1. *Weather API unavailable* &rarr; returns HTTP 502 with sanitized user message and correlation ID.
2. *Geocoder unavailable* &rarr; falls back to curated Indian city presets.
3. *Forecast unavailable* &rarr; returns HTTP 502; current assessment remains independently operational.
4. *Partial weather response* &rarr; rejected by validator; user is notified of incomplete observations.
5. *Invalid location* &rarr; returns HTTP 400 with actionable error.
6. *Map provider unavailable* &rarr; Leaflet `tileerror` triggers fallback notice; all risk analytics remain accessible.
7. *Cooling station data empty* &rarr; empty advisory state rendered without breaking the page.
8. *Demographic source unavailable* &rarr; environmental risk calculations continue with demo limitation disclosure.
9. *Validation dataset unavailable* &rarr; historical viewer displays clear unavailable state.
10. *Network timeout* &rarr; caught, sanitized, and returned as HTTP 502 without stack traces.
11. *Rate limit (HTTP 429)* &rarr; rejected safely without infinite retry loops.
12. *Malformed API response* &rarr; validation error raised and converted to HTTP 502.

### Caching Architecture & Rate-Limit Protection (Phases 17 & 18)
- In-memory TTL caches isolate queries by rounded coordinate keys `(round(lat, 3), round(lon, 3))` (~100m grid cell resolution):
  - Current weather: 10 minutes cache (600s).
  - Forecast series: 30 minutes cache (1800s).
  - Geocoding results: 1 hour cache (3600s).
- Frontend search input implements a 260ms debounce timer to prevent uncontrolled API traffic during typing.
- Caching status is exposed via `"is_cached": true` in metadata and source transparency blocks.

### Security, Error Observability & Health Checks (Phases 19–25)
- **Zero Secrets Committed**: Environment variables (`SECRET_KEY`, `PORT`, `FLASK_ENV`) manage configuration.
- **Request Tracing**: A unique correlation ID (`g.request_id`, header `X-Request-ID`) is attached to all incoming requests, server logs, and outgoing responses.
- **Health Check Endpoint**: `/health` and `/api/health` report service status (`healthy`), version (`1.0.0`), build ID (`v1.0.0-prod`), environment, uptime in seconds, and sub-service operational flags without leaking secrets or---

## 10. Real-World Evidence, Dataset Registry & Automated Backtesting Architecture

HeatWave Guardian includes an empirical evaluation and automated backtesting framework (`services/dataset_registry.py`, `services/validation_engine.py`, `run_backtest.py`) engineered to validate biometeorological environmental heat metrics, IMD regional criteria fidelity, NWP forecast accuracy, and ISO 7243 operational work-rest schedules without fabricating data or retrofitting thresholds.

### 1. Dataset Registry & Provenance (`services/dataset_registry.py`)
Every historical dataset is recorded with strict cryptographic metadata:
- **Provider & Source**: ECMWF Copernicus ERA5 reanalysis, IMD station bulletins, Open-Meteo NWP Forecast Archive.
- **Coverage & Resolution**: Coordinates, bounding boxes, temporal periods (2015, 2016, 2023, 2024), and hourly/daily resolutions.
- **Strict Data Separation**:
  - `REAL_OBSERVATION` & `REANALYSIS` are eligible for empirical validation.
  - `SYNTHETIC_DEMO` datasets (`demo_synthetic_sample_2024`) are strictly quarantined (`validation_eligibility=False`) and blocked from entering official benchmark metrics.

### 2. Multi-Climate Historical Benchmark Cases (70 Case-Days)
Evaluates 7 distinct historical cases across 7 contrasting Indian bioclimatic zones:
1. **Khammam (May 2015)**: Hot & Dry Inland Plains (Telangana) — 18 days of extreme inland heat (Station peak 48.0°C).
2. **Hyderabad (May 2015)**: Deccan Plateau / Semi-Arid (Telangana) — 18 days evaluating reanalysis spatial smoothing and cold bias (-1.55°C).
3. **Delhi NCR (May 2024)**: Northern Inland Plains — 7 days of 45–48°C extreme summer heat with NWP 24h forecast verification.
4. **Mumbai (Apr 2023)**: Hot-Humid Coastal (Maharashtra) — 7 days evaluating compound humidity (35°C with 80% RH yielding severe WBGT 31.8°C).
5. **Ahmedabad (May 2016)**: Western Arid & Semi-Arid (Gujarat) — 7 days validating NDMA / AMC Heat Action Plan Red Alert triggers.
6. **Kolkata (Apr 2024)**: Eastern Humid Gangetic Plains (West Bengal) — 7 days of oppressive pre-monsoon moisture stagnation.
7. **Bengaluru (Jan 2023) [Negative Control]**: Southern Plateau (>900m elevation) — 6 days of temperate background weather (27–29°C) validating zero false heatwave alarms.

### 3. Automated Backtesting CLI Runner (`run_backtest.py`)
Run automated multi-event backtesting directly from the command line:
```bash
# Run complete backtest across all 7 historical cases and export JSON report
python run_backtest.py

# Run backtest for specific cases
python run_backtest.py --case delhi_2024_extreme mumbai_2023_coastal_humid

# Export tabular CSV summary for external analysis
python run_backtest.py --format csv --out validation_report.csv
```

### 4. 4-Way Model Comparison Summary
Quantifies where model outputs differ across evaluated cases (70 days):
| Model Approach | Flagged Days | Flagged % | Key Behavioral Distinction |
| :--- | :---: | :---: | :--- |
| **Temperature-Only ($T \ge 40^\circ\text{C}$)** | 43 | 61.4% | Fails to detect high-humidity coastal thermal stress (Mumbai 35°C / 80% RH). |
| **IMD Regional Criteria** | 31 | 44.3% | Adjusts for regional climatology (Plains 40°C, Coastal 37°C, Hills 30°C). |
| **Liljegren WBGT ($\text{WBGT} \ge 29.4^\circ\text{C}$)** | 63 | 90.0% | Captures evaporative deficit, wind convection, and solar radiation flux. |
| **Integrated HeatWave Guardian** | 50 | 71.4% | Fuses biometeorology with ISO 7243 metabolic workload, PPE, and acclimatization. |

### 5. Component Ablation Regression (7 Atmospheric & Operational Layers)
Ablating individual components across 80 standardized operational scenarios:
- **Relative Humidity**: 75.0% scenario impact (demonstrates humidity is essential to thermal burden).
- **Metabolic Workload**: 80.0% scenario impact (work intensity governs physical strain).
- **Acclimatization State**: 80.0% scenario impact (unacclimatized workers require heightened pacing).
- **Clothing / PPE**: 80.0% scenario impact (impermeable suits compound metabolic heat retention).
- **Recovery Environment**: 53.8% scenario impact (cool indoor breakrooms restore thermal reserves).
- **Solar Radiative Flux**: 50.0% scenario impact (direct sunlight elevates effective WBGT by 2–4°C).
- **Wind Convective Cooling**: 23.8% scenario impact (air movement aids evaporative heat dissipation).

### 6. Decision Monotonicity & Consistency Audit
- **Workload Monotonicity**: PASSED (0 violations — heavier workload never decreases caution level).
- **Acclimatization Monotonicity**: PASSED (0 violations — unacclimatized state never decreases caution level).
- **Recovery Monotonicity**: PASSED (0 violations — AC $\le$ Shaded $\le$ Direct Sun caution ordering).
- **Clothing/PPE Monotonicity**: PASSED (0 violations — Light $\le$ Heavy $\le$ Impermeable caution ordering).
- **Irrelevant Variable Invariance**: PASSED (display strings do not alter physical calculations).

### 7. Structured Error Taxonomy (26 Disagreements Classified)
- **Spatial Mismatch (12)**: Coarse ~25km ERA5 reanalysis blunting localized urban station spikes (e.g. Hyderabad).
- **Category Boundary Shift (11)**: Borderline temperatures near threshold boundaries (e.g. 43.8°C vs 44.0°C IMD cutoff).
- **Model Limitation (3)**: Lack of localized microclimatic sensor adjustments without on-site telemetry.

---

## 10. MapLibre GL JS Spatial Visualization Architecture

The Authority / Municipal Heat-Risk Map (`/authority`) utilizes **MapLibre GL JS (v4.7.1)** while preserving the lightweight Flask + vanilla JavaScript + Tailwind CSS architecture (strictly avoiding heavyweight frontend framework migrations).

### Why MapLibre GL JS?
1. **High-Performance WebGL Rendering**: Eliminates DOM marker overhead by utilizing GPU-accelerated canvas layers for hundreds of microclimate grid cells and municipal centers.
2. **Native Layer Filtering**: Real-time filter state transitions (`All`, `High+`, `Very High+`, `Extreme`) execute directly on the GPU via `map.setFilter()` with zero DOM teardown or network latency.
3. **Data-Driven Styling**: Symbol, circle, and polygon properties (`circle-radius`, `circle-color`, `fill-color`, `line-color`) are driven directly from backend GeoJSON properties matching core risk color palettes (`#2f9e44`, `#f2a900`, `#e8590c`, `#c1121f`, `#6a040f`).
4. **Keyless, Open-Source Basemap**: Fully independent of commercial tile tokens or paid API keys, eliminating "API KEY REQUIRED" failures and credential leaks.

### Basemap Provider & Attribution
- **Provider**: OpenStreetMap standard raster tile servers via centralized configuration in `static/map_config.js` (`MAP_CONFIG.getMapLibreStyle()`).
- **Required Attribution**: `&copy; OpenStreetMap contributors` rendered persistently on the map canvas.
- **Zero Credentials**: 100% keyless open-source implementation.

### GeoJSON Spatial Layers Architecture
The map integrates two native GeoJSON sources:
1. **`authority-zones` (Points)**:
   - Data source: `GET /api/authority/zones`
   - Layers: `zones-selected-halo` (outer highlight ring), `zones-heat-circle` (stepped radius 10–16px, data-driven fill matching NDMA alert levels).
2. **`spatial-grid` (Polygons)**:
   - Data source: `GET /api/authority/grid` (or `?city=<city>`)
   - Layers: `grid-cells-fill` (~1km closed polygons with 0.45 fill opacity), `grid-cells-line` (dashed boundary), `grid-cells-highlight` (selected cell boundary).

### Pipeline Flow: ML Downscaling vs. Deterministic WBGT
```
Coarse NWP / ERA5 Weather (~25km) + Urban Morphology Features
                      ↓
     ML Random Forest Regressor (v1.0.0-rf)
                      ↓
  Localized In-Situ Temperature (T_local^ML = T_bg + ΔT_UHI)
                      ↓
     Deterministic Liljegren Physical Equations
                      ↓
  Wet-Bulb Globe Temperature (WBGT) & ISO 7243 Pacing
                      ↓
        NDMA Heat Action Plan Risk Matrix
                      ↓
  MapLibre GL JS GPU-Accelerated Layers & Popups
```
*Scientific Safety Guarantee*: The frontend strictly renders the precomputed backend assessments. ML refines localized environmental temperature estimates only; all WBGT and risk classifications remain deterministic physics.

---

## 11. Testing & Verification

HeatWave Guardian includes **805 automated unit, integration, biometeorological audit, failure injection, reliability, performance, and backtesting tests (100% passing across 34 test modules)**:

```bash
# Run complete test suite (all 805 tests)
pytest -q

# Run dedicated MapLibre GL JS Authority Map verification suite
pytest test_maplibre_authority.py -v

# Run the dedicated Real-World Evidence & Automated Backtesting verification suite
pytest test_validation_system.py -v

# Run the dedicated Production Reliability & Resilience verification suite
pytest test_production_reliability.py -v

# Run the dedicated Failure Injection test suite
pytest test_failure_injection.py -v
```

| Test Suite | Tests | Key Areas Covered |
| :--- | :---: | :---: |
| `test_production_reliability.py` | **81 / 81** | Timeouts, retry backoff, physical bounds (-10..65°C), unit normalization, zero synthetic fallback, correlation IDs, health checks |
| `test_validation_system.py` | **76 / 76** | Dataset registry, case registry, backtest engine, environmental error metrics (MAE/RMSE/bias), IMD fidelity, forecast backtesting, peak window detection, component ablation, decision monotonicity, error taxonomy, data leakage prevention, regression |
| `test_role_alert_intelligence.py` | **53 / 53** | Step 7 Role-Based Alert Intelligence & 5-Tier Escalation Hierarchy (Worker -> Supervisor -> Site Manager -> Authority -> System Admin): multilingual templates (EN, HI, MR), compact SMS (<160 chars), HITL approval & rejection workflows, quiet hours suppression & critical bypass, consent enforcement, data honesty & simulation dispatch guardrail, and REST API suite |
| `test_forecast_simulator.py` | **52 / 52** | Hourly risk profiles, Liljegren WBGT, peak detection, flat curves, lower windows |
| `test_scientific_benchmarks.py` | **33 / 33** | Liljegren et al. (2008) vs ISO 7243 benchmark parity, solar elevation angle, psychrometric dew point equations |
| `test_intervention_intelligence_engine.py` | **32 / 32** | Step 6 Intervention Intelligence Engine: 7 categories (A–G), control hierarchy (Levels 1–6), PPE safety guardrail, candidate shift windows, role views (Worker, Supervisor, Authority), What-If simulation connection, unified decision timeline, structured alert objects, API endpoints contract |
| `test_exposure_intervention_engine.py` | **31 / 31** | ISO 7243 work/rest screening, 5-layer exposure context, intervention catalog, safety phrasing |
| `test_expanded_features.py` | **31 / 31** | Geocoding India filtering, upstream 502 safety, demographic integration, HAP matrix, alert dispatch |
| `test_core_engine_audit.py` | **31 / 31** | Liljegren WBGT physics, IMD thresholds, fusion matrix, data quality confidence |
| `test_authority_map.py` | **29 / 29** | Authority zones GeoJSON, municipal risk surveillance, data binding |
| `test_thermal_stress.py` | **24 / 24** | WBGT boundary thresholds, 3-concept separation, hierarchy |
| `test_ml_hyperlocal.py` | **24 / 24** | Hyperlocal RF model inference, spatial morphology features, confidence scoring |
| `test_weather_availability_state.py` | **23 / 23** | Upstream provider availability states, circuit breaker behavior, stale cache fallback |
| `test_explainable_risk_engine.py` | **19 / 19** | Decision trace pipeline, factor sensitivities, NO_CHANGE thermal barrier, 5 diurnal periods, day stories, API audit |
| `test_ui_ux_system.py` | **19 / 19** | Modern UI/UX design tokens, glassmorphic layout, interactive tabs, mobile responsive reflow |
| `test_personalization.py` | **18 / 18** | Age groups, acclimatization, clothing, metabolic workload escalation |
| `test_maplibre_authority.py` | **16 / 16** | MapLibre GL JS markup, OSM keyless basemap style, GeoJSON Points/Polygons, data-driven styling, 3-tier inspection, GPU filtering |
| `test_ml_spatial_grid.py` | **15 / 15** | ~1km spatial microclimate grid, Liljegren WBGT recalculation, driver attribution |
| `test_ml_provenance_scientific.py` | **15 / 15** | Scientific provenance tracking, dataset integrity, feature attribution reproducibility |
| `test_what_if_simulator.py` | **14 / 14** | Interactive scenario modification, decision timeline, ISO 7243 work/rest delta |
| `test_ml_forecast_peak_uncertainty.py` | **14 / 14** | 5-day ML forecast, diurnal peak windows, 100-tree RF uncertainty spread |
| `test_failure_injection.py` | **13 / 13** | Provider timeout injection, transient 500 retry recovery, 400 no-retry verification, malformed JSON, out-of-bounds rejection, stale timestamp confidence penalty |
| `test_services.py` | **14 / 14** | Demographic proxy, geo service, institute sites, cooling recommender |
| `test_new_endpoints.py` | **12 / 12** | API endpoints for decision trace, factor sensitivities, and timeline narratives |
| `test_thermo_and_batch_ml.py` | **11 / 11** | High-throughput batch ML downscaling, thermodynamics verification |
| `test_engine.py` | **8 / 8** | Foundational biometeorological equations & Liljegren physics |
| `test_map_integration.py` | **6 / 6** | Map configuration, MapLibre & Leaflet coexistence, coordinates validation |
| `tests/performance/*` & `tests/*` | **104 / 104** | API load test, forecast load, ML load, spatial grid load, soak test, cross-module deeplinks |
| **Total Automated Tests** | **805 / 805** | **100% Pass Rate Across Complete Test Harness** |

### Automated Browser UI/UX QA (Playwright)
- **Desktop Viewport (1280x800)**: Verified Homepage (`/`), Forecast (`/forecast`), What-If (`/what-if`), Worksite (`/worksite`), Authority (`/authority`), Cooling Stations (`/cooling-stations`), Validation (`/validation`), and Methodology (`/methodology`).
- **Mobile Viewport (390x844)**: Verified zero horizontal scroll overflow (`scrollWidth === clientWidth === 390px`), clean vertical reflow of cards, forms, and timeline.
- **Console Errors**: 0 JavaScript console errors across all screens.

---

## 12. Production Cloud Deployment

### Deployment on Render (1-Click)
1. Push repository to GitHub.
2. Log in at [render.com](https://render.com/) &rarr; **New +** &rarr; **Web Service**.
3. Select your repository.
4. Render automatically detects configuration via the included `render.yaml` Blueprint:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Under Environment, provide `SECRET_KEY`.
6. Deploy service.

### Deployment on Linux VPS (Ubuntu / Nginx / Systemd)
1. Clone repo to `/var/www/heatwave-guardian`.
2. Set up virtual environment and install requirements:
   ```bash
   python3 -m venv venv
   ./venv/bin/pip install -r requirements.txt
   ```
3. Run Gunicorn bound to `0.0.0.0:5000`:
   ```bash
   ./venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 app:app
   ```
4. Proxy through Nginx with SSL via Let's Encrypt Certbot.

---

## 13. Scientific Limitations & Disclaimers

1. **Non-Medical Decision Support Positioning**: HeatWave Guardian is an occupational biometeorological decision-support and exposure-screening tool. It does NOT diagnose medical conditions, predict individual clinical probability of illness, or prescribe personalized fluid medication.
2. **Hydration Guidance Principle**: Hydration guidance is non-prescriptive. The platform does not prescribe universal volumetric quotas (e.g. "drink exactly X litres/hr"); workers should maintain continuous access to cool potable water, drink regularly in small quantities, and follow site safety and individual physician instructions (especially those with renal, cardiovascular, or fluid-restricted medical conditions).
3. **Acclimatization Pacing**: Acclimatization recommendations are occupational administrative guidelines based on NIOSH (2016) / OSHA protocols to guide work-pacing schedules. They do not measure clinical physiological adaptation.
4. **Live Atmospheric Feed**: Real-time assessments require active network connectivity to Open-Meteo. If provider connectivity drops, the system returns an explicit HTTP 502 error rather than fabricating synthetic observations.
5. **Urban Microclimates & ERA5 Smoothing**: Coarse reanalysis (~25km grid resolution) averages out localized microclimatic hot spots by ~2–4°C compared to ground station thermometers. In-situ thermal conditions may vary under direct solar exposure or within dense built environments.
