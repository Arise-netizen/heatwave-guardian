"""
HeatWave Guardian — Dataset Registry & Provenance Architecture
==============================================================
Manages structured metadata, provenance, and validation eligibility for
all real-world atmospheric observations, reanalysis archives, ground station
bulletins, numerical forecast series, and isolated demo/synthetic test fixtures.

Features:
- Strict separation: REAL_OBSERVATION vs REANALYSIS vs REFERENCE_BULLETIN vs FORECAST_ARCHIVE vs DEMO_SYNTHETIC
- Traceable provenance: Provider, retrieval dates, geographic/temporal bounds, spatial/temporal resolution
- Scientific limitations disclosure and missing-data policies
- Immutable metadata validation and test fixture isolation
"""

import copy
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone


class DatasetType:
    REAL_OBSERVATION = "REAL_OBSERVATION"
    REANALYSIS = "REANALYSIS"
    REFERENCE_BULLETIN = "REFERENCE_BULLETIN"
    FORECAST_ARCHIVE = "FORECAST_ARCHIVE"
    DEMO_SYNTHETIC = "DEMO_SYNTHETIC"

    ALL_TYPES = [
        REAL_OBSERVATION,
        REANALYSIS,
        REFERENCE_BULLETIN,
        FORECAST_ARCHIVE,
        DEMO_SYNTHETIC,
    ]


# Baseline immutable registry dictionary
_DEFAULT_DATASETS: Dict[str, Dict[str, Any]] = {
    "era5_ap_telangana_2015": {
        "dataset_id": "era5_ap_telangana_2015",
        "name": "ECMWF ERA5 Reanalysis — AP & Telangana 2015 Heatwave",
        "source": "European Centre for Medium-Range Weather Forecasts (ECMWF) Copernicus Climate Change Service",
        "url_or_reference": "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels",
        "provider": "ECMWF",
        "retrieval_date": "2024-03-15",
        "retrieved_at": "2024-03-15",
        "data_type": DatasetType.REANALYSIS,
        "coverage": {
            "geographic": "Andhra Pradesh & Telangana (Khammam 17.25°N, 80.15°E; Hyderabad 17.38°N, 78.49°E)",
            "temporal_start": "2015-05-15",
            "temporal_end": "2015-06-01",
            "total_days": 18,
        },
        "resolution": {
            "spatial": "0.25° (~25 km) grid cell",
            "temporal": "Hourly and Daily Maximum",
        },
        "variables": [
            {"name": "2m_temperature", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "2m_dewpoint_temperature", "unit": "°C", "standard_name": "dew_point_temperature"},
            {"name": "10m_u_wind_component", "unit": "m/s", "standard_name": "eastward_wind"},
            {"name": "10m_v_wind_component", "unit": "m/s", "standard_name": "northward_wind"},
            {"name": "surface_solar_radiation_downwards", "unit": "J/m²", "standard_name": "surface_downwelling_shortwave_flux_in_air"},
        ],
        "units": {
            "temperature": "Celsius (°C)",
            "relative_humidity": "Percentage (%)",
            "wind_speed": "Meters per second (m/s)",
            "solar_radiation": "Watts per square meter (W/m²)",
        },
        "transformations": [
            "Relative humidity derived via Magnus-Tetens approximation from dry-bulb and dewpoint temperatures.",
            "Wind speed derived via vector magnitude sqrt(u² + v²).",
            "Solar radiation flux converted from hourly accumulated joules to instant W/m².",
        ],
        "missing_data_policy": "ERA5 reanalysis provides complete continuous coverage; no missing values in analyzed series.",
        "limitations": [
            "Coarse spatial resolution (~25 km) systematically smooths intense localized station spikes.",
            "Documented cold bias of 2.0°C to 4.4°C over Indian subcontinent epicenters during pre-monsoon heat extremes.",
        ],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_station_records_2015": {
        "dataset_id": "imd_station_records_2015",
        "name": "IMD Ground Station Daily Bulletins — 2015 Heatwave",
        "source": "India Meteorological Department (IMD) National Data Centre, Pune",
        "url_or_reference": "https://mausam.imd.gov.in / Daily Weather Reports & Climate Diagnostics",
        "provider": "India Meteorological Department",
        "retrieval_date": "2024-04-10",
        "retrieved_at": "2024-04-10",
        "data_type": DatasetType.REFERENCE_BULLETIN,
        "coverage": {
            "geographic": "Khammam Surface Observatory & Hyderabad Begumpet Airport Observatory",
            "temporal_start": "2015-05-15",
            "temporal_end": "2015-06-01",
            "total_days": 18,
        },
        "resolution": {
            "spatial": "Point ground station observatories (Stevenson screen 1.5m)",
            "temporal": "Daily Maximum / Minimum & 08:30 / 17:30 IST synoptic records",
        },
        "variables": [
            {"name": "maximum_temperature", "unit": "°C", "standard_name": "daily_max_air_temperature"},
            {"name": "departure_from_normal", "unit": "°C", "standard_name": "temperature_anomaly"},
            {"name": "imd_heatwave_warning", "unit": "category", "standard_name": "official_heatwave_declaration"},
        ],
        "units": {
            "temperature": "Celsius (°C)",
            "departure": "Celsius anomaly (°C)",
        },
        "transformations": [
            "Categorized into IMD criteria (Heat Day, Heatwave, Severe Heatwave) according to official Plains thresholds."
        ],
        "missing_data_policy": "Station ground truths taken strictly from certified IMD daily bulletins.",
        "limitations": [
            "Point measurements reflect Stevenson screen conditions and do not measure direct solar radiation or outdoor WBGT directly.",
        ],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_delhi_heatwave_2024": {
        "dataset_id": "imd_delhi_heatwave_2024",
        "name": "Delhi NCR Northern Plains Extreme Heatwave (May 2024)",
        "source": "IMD Regional Meteorological Centre, New Delhi & Surface Station Network",
        "url_or_reference": "https://rmcnewdelhi.imd.gov.in",
        "provider": "India Meteorological Department / Open-Meteo Atmospheric Archive",
        "retrieval_date": "2024-07-01",
        "retrieved_at": "2024-07-01",
        "data_type": DatasetType.REAL_OBSERVATION,
        "coverage": {
            "geographic": "Delhi NCR (Safdarjung & Mungeshpur 28.61°N, 77.21°E)",
            "temporal_start": "2024-05-25",
            "temporal_end": "2024-05-31",
            "total_days": 7,
        },
        "resolution": {
            "spatial": "Station / 0.1° high-resolution reanalysis",
            "temporal": "Hourly",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
            {"name": "shortwave_radiation", "unit": "W/m²", "standard_name": "surface_solar_radiation"},
        ],
        "units": {
            "temperature": "°C",
            "relative_humidity": "%",
            "wind_speed": "m/s",
            "solar_radiation": "W/m²",
        },
        "transformations": ["Liljegren WBGT evaluated across hourly series."],
        "missing_data_policy": "Verified complete 7-day hourly series.",
        "limitations": ["Extreme hot-dry inland conditions with very low afternoon RH (15–25%)."],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_mumbai_coastal_2023": {
        "dataset_id": "imd_mumbai_coastal_2023",
        "name": "Mumbai Coastal Hot-Humid Thermal Stress Event (May 2023)",
        "source": "IMD Santacruz & Colaba Observatories / Atmospheric Archive",
        "url_or_reference": "https://imdmumbai.gov.in",
        "provider": "India Meteorological Department / Open-Meteo Archive",
        "retrieval_date": "2024-05-10",
        "retrieved_at": "2024-05-10",
        "data_type": DatasetType.REAL_OBSERVATION,
        "coverage": {
            "geographic": "Mumbai Coastal Zone (19.07°N, 72.88°E)",
            "temporal_start": "2023-05-10",
            "temporal_end": "2023-05-16",
            "total_days": 7,
        },
        "resolution": {
            "spatial": "Station / 0.1° coastal grid",
            "temporal": "Hourly",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
            {"name": "shortwave_radiation", "unit": "W/m²", "standard_name": "surface_solar_radiation"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%", "wind_speed": "m/s", "solar_radiation": "W/m²"},
        "transformations": ["Evaluated through IMD Coastal Criteria (37°C threshold) and Liljegren WBGT."],
        "missing_data_policy": "Continuous validated 7-day coastal record.",
        "limitations": [
            "Moderate dry-bulb temperature (34–36°C) paired with extreme humidity (70–85%) creates high WBGT despite no standard plains heatwave declaration.",
        ],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_ahmedabad_arid_2016": {
        "dataset_id": "imd_ahmedabad_arid_2016",
        "name": "Ahmedabad Landmark Arid Heatwave (May 2016)",
        "source": "Ahmedabad Municipal Corporation (AMC) Heat Action Plan Archive & IMD",
        "url_or_reference": "https://ahmedabadcity.gov.in / AMC HAP 2016 Evaluation Report",
        "provider": "IMD / AMC Disaster Management Cell",
        "retrieval_date": "2024-02-20",
        "retrieved_at": "2024-02-20",
        "data_type": DatasetType.REAL_OBSERVATION,
        "coverage": {
            "geographic": "Ahmedabad Urban Area (23.02°N, 72.57°E)",
            "temporal_start": "2016-05-17",
            "temporal_end": "2016-05-22",
            "total_days": 6,
        },
        "resolution": {
            "spatial": "Urban Station & NWP Grid",
            "temporal": "Daily peak & Hourly breakdown",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%", "wind_speed": "m/s"},
        "transformations": ["AMC HAP Red Alert criteria (>45.0°C) cross-compared with ISO 7243 rest guidelines."],
        "missing_data_policy": "Complete landmark heat records.",
        "limitations": ["Arid western climate with dry desert wind (Loo)."],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_kolkata_humid_2024": {
        "dataset_id": "imd_kolkata_humid_2024",
        "name": "Kolkata Gangetic Humid Heatwave (April 2024)",
        "source": "IMD Regional Meteorological Centre, Kolkata",
        "url_or_reference": "https://imdkolkata.gov.in",
        "provider": "India Meteorological Department / Open-Meteo Archive",
        "retrieval_date": "2024-05-02",
        "retrieved_at": "2024-05-02",
        "data_type": DatasetType.REAL_OBSERVATION,
        "coverage": {
            "geographic": "Kolkata Urban & Gangetic Basin (22.57°N, 88.36°E)",
            "temporal_start": "2024-04-24",
            "temporal_end": "2024-04-30",
            "total_days": 7,
        },
        "resolution": {
            "spatial": "0.1° regional grid",
            "temporal": "Hourly",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
            {"name": "shortwave_radiation", "unit": "W/m²", "standard_name": "surface_solar_radiation"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%", "wind_speed": "m/s", "solar_radiation": "W/m²"},
        "transformations": ["Plains IMD criteria with high moisture compounding."],
        "missing_data_policy": "Continuous observation record.",
        "limitations": ["High diurnal humidity variation."],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "imd_bangalore_control_2023": {
        "dataset_id": "imd_bangalore_control_2023",
        "name": "Bengaluru Southern Plateau Non-Heat Control (Jan 2023)",
        "source": "IMD Bengaluru Observatory & Atmospheric Archive",
        "url_or_reference": "https://imd.gov.in",
        "provider": "India Meteorological Department / Open-Meteo Archive",
        "retrieval_date": "2024-03-01",
        "retrieved_at": "2024-03-01",
        "data_type": DatasetType.REAL_OBSERVATION,
        "coverage": {
            "geographic": "Bengaluru (12.97°N, 77.59°E, Elevation 920m)",
            "temporal_start": "2023-01-10",
            "temporal_end": "2023-01-16",
            "total_days": 7,
        },
        "resolution": {
            "spatial": "Station / 0.1° grid",
            "temporal": "Hourly",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%", "wind_speed": "m/s"},
        "transformations": ["Negative control validation (evaluates false alarm rates during non-heatwave periods)."],
        "missing_data_policy": "Continuous winter benchmark record.",
        "limitations": ["Mild plateau climate; temperatures 24–28°C; zero heatwave expectation."],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "openmeteo_nwp_forecast_archive": {
        "dataset_id": "openmeteo_nwp_forecast_archive",
        "name": "Open-Meteo Numerical Weather Prediction (NWP) Historical Forecast Runs",
        "source": "Open-Meteo NWP Historical Model Run Archive (ECMWF IFS / DWD ICON)",
        "url_or_reference": "https://open-meteo.com/en/docs/historical-forecast-api",
        "provider": "Open-Meteo GmbH",
        "retrieval_date": "2024-08-15",
        "retrieved_at": "2024-08-15",
        "data_type": DatasetType.FORECAST_ARCHIVE,
        "coverage": {
            "geographic": "India Multi-Station Array (Delhi, Hyderabad, Mumbai, Khammam)",
            "temporal_start": "2024-05-01",
            "temporal_end": "2024-05-31",
            "total_days": 31,
        },
        "resolution": {
            "spatial": "0.1° (~11 km)",
            "temporal": "Hourly 24h–72h lead time forecasts",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
            {"name": "wind_speed_10m", "unit": "m/s", "standard_name": "wind_speed"},
            {"name": "shortwave_radiation", "unit": "W/m²", "standard_name": "surface_solar_radiation"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%", "wind_speed": "m/s", "solar_radiation": "W/m²"},
        "transformations": ["Forecast vs observed peak timing difference and thermal stress error evaluation."],
        "missing_data_policy": "Archived numerical model outputs.",
        "limitations": [
            "Forecast skill diminishes at +72h to +120h lead times; localized convective clouds create radiation forecast errors.",
        ],
        "validation_eligibility": True,
        "is_real_data": True,
    },

    "demo_synthetic_sample_2024": {
        "dataset_id": "demo_synthetic_sample_2024",
        "name": "Synthetic Test Fixture — Demo Heat Profile",
        "source": "Synthetically Generated Test Fixture (Non-Real)",
        "url_or_reference": "N/A - Internal Test Harness Only",
        "provider": "HeatWave Guardian Test Harness",
        "retrieval_date": "2024-01-01",
        "retrieved_at": "2024-01-01",
        "data_type": DatasetType.DEMO_SYNTHETIC,
        "coverage": {
            "geographic": "Simulated Generic Grid (15.00°N, 75.00°E)",
            "temporal_start": "2024-01-01",
            "temporal_end": "2024-01-03",
            "total_days": 3,
        },
        "resolution": {
            "spatial": "Point Simulation",
            "temporal": "Hourly Simulated",
        },
        "variables": [
            {"name": "temperature_2m", "unit": "°C", "standard_name": "air_temperature"},
            {"name": "relative_humidity_2m", "unit": "%", "standard_name": "relative_humidity"},
        ],
        "units": {"temperature": "°C", "relative_humidity": "%"},
        "transformations": ["Artificially generated sine-wave diurnal cycle for test assertion only."],
        "missing_data_policy": "Full continuous synthetic stream.",
        "limitations": [
            "STRICTLY FOR TEST FIXTURING. NOT REAL EVIDENCE. MUST NEVER ENTER SCIENTIFIC VALIDATION.",
        ],
        "validation_eligibility": False,
        "is_real_data": False,
    },
}


class DatasetRegistry:
    """Registry of documented historical and validation datasets."""

    _REGISTRY: Dict[str, Dict[str, Any]] = copy.deepcopy(_DEFAULT_DATASETS)

    @classmethod
    def reset_registry(cls):
        """Reset registry to pristine default configuration."""
        cls._REGISTRY = copy.deepcopy(_DEFAULT_DATASETS)

    @classmethod
    def _normalize_dataset(cls, ds: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dataset dictionary to expose both nested and canonical flat aliases."""
        res = ds.copy()
        cov = res.get("coverage", {})
        res["geographic_coverage"] = cov.get("geographic", "")
        res["temporal_coverage"] = f"{cov.get('temporal_start', '')} to {cov.get('temporal_end', '')}"
        res["temporal_resolution"] = res.get("resolution", {}).get("temporal", "")
        res["spatial_resolution"] = res.get("resolution", {}).get("spatial", "")
        
        vars_raw = res.get("variables", [])
        res["variables_available"] = [v["name"] if isinstance(v, dict) else str(v) for v in vars_raw]
        res["known_limitations"] = res.get("limitations", [])
        
        # Canonical observed vs forecast indicator
        dt = res.get("data_type", "")
        if dt in [DatasetType.REAL_OBSERVATION, DatasetType.REANALYSIS, DatasetType.REFERENCE_BULLETIN]:
            res["observed_vs_forecast"] = "observed"
        elif dt == DatasetType.FORECAST_ARCHIVE:
            res["observed_vs_forecast"] = "forecast"
        elif dt == DatasetType.DEMO_SYNTHETIC:
            res["observed_vs_forecast"] = "demo"
        else:
            res["observed_vs_forecast"] = str(dt).lower()

        if "retrieved_at" not in res and "retrieval_date" in res:
            res["retrieved_at"] = res["retrieval_date"]

        return res

    @classmethod
    def get_dataset(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset metadata by ID."""
        ds = cls._REGISTRY.get(dataset_id)
        return cls._normalize_dataset(ds) if ds else None

    @classmethod
    def list_datasets(cls, data_type: Optional[str] = None, real_only: bool = False) -> List[Dict[str, Any]]:
        """List registered datasets, optionally filtered by dataset type and real-only flag."""
        res = []
        for d in cls._REGISTRY.values():
            if real_only and not d.get("is_real_data", False):
                continue
            if data_type and d["data_type"] != data_type:
                continue
            res.append(cls._normalize_dataset(d))
        return res

    @classmethod
    def get_real_datasets(cls) -> List[Dict[str, Any]]:
        """Retrieve all real atmospheric observation and reanalysis datasets."""
        return cls.list_datasets(real_only=True)

    @classmethod
    def get_demo_datasets(cls) -> List[Dict[str, Any]]:
        """Retrieve isolated test fixture and demo datasets."""
        return cls.list_datasets(data_type=DatasetType.DEMO_SYNTHETIC)

    @classmethod
    def register_dataset(cls, metadata: Dict[str, Any]) -> str:
        """Register a new validation dataset with strict schema validation."""
        required = [
            "dataset_id", "name", "source", "provider", "retrieval_date",
            "data_type", "coverage", "resolution", "variables", "units"
        ]
        for f in required:
            if f not in metadata or metadata[f] is None or metadata[f] == "":
                raise ValueError(f"Missing required dataset metadata field: '{f}'")

        if metadata["data_type"] not in DatasetType.ALL_TYPES:
            raise ValueError(f"Invalid dataset data_type: '{metadata['data_type']}'. Must be one of {DatasetType.ALL_TYPES}")

        cov = metadata.get("coverage")
        if not isinstance(cov, dict) or not cov.get("temporal_start") or not cov.get("temporal_end"):
            raise ValueError("Dataset 'coverage' must be a dict specifying at least 'temporal_start' and 'temporal_end'")

        if not isinstance(metadata.get("variables"), list) or len(metadata["variables"]) == 0:
            raise ValueError("Dataset 'variables' must be a non-empty list of variable definitions")

        ds_id = metadata["dataset_id"]
        # Default flags if omitted
        meta = metadata.copy()
        if "is_real_data" not in meta:
            meta["is_real_data"] = meta["data_type"] != DatasetType.DEMO_SYNTHETIC
        if "validation_eligibility" not in meta:
            meta["validation_eligibility"] = meta["is_real_data"]
        if "retrieved_at" not in meta:
            meta["retrieved_at"] = meta.get("retrieval_date")
        if "limitations" not in meta:
            meta["limitations"] = meta.get("known_limitations", [])

        cls._REGISTRY[ds_id] = meta
        return ds_id

    @classmethod
    def validate_metadata(cls, ds_id_or_dict: Union[str, Dict[str, Any]]) -> bool:
        """Check whether a dataset has complete and valid provenance metadata."""
        if isinstance(ds_id_or_dict, str):
            ds = cls.get_dataset(ds_id_or_dict)
        elif isinstance(ds_id_or_dict, dict):
            ds = ds_id_or_dict
        else:
            return False

        if not ds:
            return False

        cov = ds.get("coverage", {})
        return bool(
            ds.get("dataset_id")
            and ds.get("name")
            and ds.get("source")
            and ds.get("provider")
            and (ds.get("retrieval_date") or ds.get("retrieved_at"))
            and ds.get("data_type") in DatasetType.ALL_TYPES
            and isinstance(cov, dict)
            and cov.get("temporal_start")
            and cov.get("temporal_end")
            and ds.get("variables")
            and isinstance(ds.get("variables"), list)
            and len(ds.get("variables")) > 0
            and ds.get("units")
        )

    @classmethod
    def delete_dataset(cls, dataset_id: str) -> bool:
        """Delete a dataset from registry (used for test cleanup)."""
        if dataset_id in cls._REGISTRY:
            del cls._REGISTRY[dataset_id]
            return True
        return False

    @classmethod
    def get_dataset_provenance_summary(cls) -> Dict[str, Any]:
        """Summary of provenance, source separation, and coverage across all registered datasets."""
        all_ds = cls.list_datasets()
        real_ds = [d for d in all_ds if d.get("is_real_data", False)]
        demo_ds = [d for d in all_ds if not d.get("is_real_data", False)]
        
        return {
            "total_datasets": len(all_ds),
            "real_datasets_count": len(real_ds),
            "demo_datasets_count": len(demo_ds),
            "providers": list(set(d.get("provider", "Unknown") for d in all_ds)),
            "data_types": list(set(d.get("data_type", "Unknown") for d in all_ds)),
            "datasets": all_ds,
        }
