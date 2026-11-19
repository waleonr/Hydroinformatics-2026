# Hydroinformatics – Course Assignment Repository

This repository contains the code, notebooks, and analyses developed for the **Hydroinformatics** and **Advanced Hydroinformatics** courses at the Universidad Nacional de Colombia. The material focuses on applying mathematical, statistical, and computational tools to the analysis and modeling of hydrosystems, with case studies in Colombian basins (Orinoco, Meta, and the Bogotá Savanna).

<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/6feb60b9-cf66-4f96-b36b-9b31a14806e4" />

---

## Repository Organization

The repository is organized by course and then by task. Each task folder contains the notebooks or scripts, the assignment statement (`.ppt` / `.pptx`), and, where applicable, results and reports.

```
Hydroinformatics-2026/
├── Hydroinformatics/          # Hydroinformatics course (Python / Jupyter)
│   ├── Visualization/         # Task 1
│   ├── Visualization2/        # Task 2
│   ├── MultiGauss/            # Task 3
│   ├── Fourier/               # Task 4
│   ├── Holy week/             # Task 5
│   ├── Multifractal/          # Task 7
│   └── Project/               # Final project
└── Advanced/                  # Advanced Hydroinformatics course
    ├── Task1/                 # ID3 decision trees (Python)
    ├── Task3/                 # Fuzzy systems, ANNs and ANFIS (MATLAB)
    └── Project/               # Final project: land cover classification
```

### Hydroinformatics

| Folder | Notebook | Content |
|---|---|---|
| `Visualization` | `assignment1.ipynb` | Reading and visualizing hydrological data |
| `Visualization2` | `assignment2.ipynb` | Visualization of gridded data (NetCDF) with `xarray` |
| `MultiGauss` | `assignment3.ipynb` | 2D histograms, 3D multivariate Gaussian distributions, and statistical moments |
| `Fourier` | `assignment4.ipynb` | Discrete Fourier transform and power spectra of time series |
| `Holy week` | `assignment5.ipynb` | Hurst phenomenon and exponent, Noah and Joseph effects, mutual information between precipitation, relative humidity, and streamflow |
| `Multifractal` | `Task7.ipynb` | Fractal river basins, chaos, and multifractal analysis |
| `Project` | `Project.ipynb` | Morphometric and hydrological pipeline per basin: DEM from Google Earth Engine, drainage network with WhiteboxTools, Horton's laws (Rb, RL, RA), Strahler order, and multifractal spectrum (Δα, f(α)max). Summary in `morfometria_resumen.xlsx` |

### Advanced Hydroinformatics

| Folder | Files | Content |
|---|---|---|
| `Task1` | `parte1_id3_rutina_familiar.ipynb`, `parte2_id3_riesgo_hidrico_orinoquia.ipynb` | ID3 algorithm implemented from scratch (entropy and information gain), applied to a toy example and to water-risk classification in Orinoquía subbasins under the IPCC AR6 framework (Risk = f(Hazard, Exposure, Vulnerability)) |
| `Task3` | `Punto1`–`Punto7`, `tabla_patrones_todos.csv` | Literature review (Points 1–2); Mamdani/Sugeno fuzzy inference systems (Points 3–4), single-layer and multilayer neural networks (Points 5–6), and ANFIS (Point 7) for crop water-stress risk in maize at C.I. Tibaitatá (FAO-56). Each point includes its MATLAB script and PDF report |
| `Project` | `Meta_LandCover_v6_clean.ipynb`, `results_v6/` | Land cover classification in the Meta River basin with Sentinel-2, Sentinel-1, and MapBiomas Colombia: Random Forest, 3D-2D CNN, Vision Transformer, hybrid models, and ensembles. Includes metrics, accuracy assessment, change maps (2024–2025), and GeoTIFF outputs |

---

## Topics Covered

The tasks explore key topics in hydroinformatics and computational hydrology, including:

- Statistical characterization of hydrological variables
- Probability distributions and statistical moments
- Information theory and entropy-based metrics
- Time series analysis and spectral methods
- Wavelet analysis
- Differential equation modeling
- Stochastic processes and random fields
- Power laws and fractal analysis
- Multifractal theory
- Nonlinear dynamics and chaotic behavior in time series
- Inverse problems and model calibration
- Optimization techniques
- Graph theory and complex networks
- Intelligent systems: decision trees, fuzzy logic, neural networks, neuro-fuzzy systems, and deep learning applied to hydrological and remote sensing data

---

## Requirements

**Python (Jupyter notebooks)**

- Core: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `statsmodels`, `sympy`
- Gridded data: `xarray`, `rioxarray`, `netCDF4`, `dask`
- Geospatial: `geopandas`, `rasterio`, `fiona`, `shapely`, `pyproj`, `whitebox`, `matplotlib-scalebar`
- Google Earth Engine: `earthengine-api`, `geemap` (requires an authenticated GEE account)
- Machine learning: `scikit-learn`, `tensorflow` / `keras`, `joblib`

**MATLAB** (Advanced/Task3)

- MATLAB R2019b or later with the **Fuzzy Logic Toolbox**
- Scripts are organized in sections (`%%`) and can be opened as Live Scripts

---

## Data Management

Some datasets used in the analyses are too large to be stored in the repository. Therefore:

- Large input datasets are excluded from version control
- Intermediate outputs are only partially stored
- Scripts and notebooks are provided so that the analyses can be reproduced with the original data (e.g., data accessed through Google Earth Engine)

---

## Purpose

This repository serves as:

- A structured record of course tasks
- A collection of computational experiments in hydroinformatics
- A reference implementation of statistical, mathematical, and machine learning methods applied to hydrosystems

---

## Authors

- **William Alfonso León Rueda** – <waleonr@unal.edu.co>
- **Douglas Andrés Gómez Latorre** – <doagomezla@unal.edu.co>

Research interests include:

- Hydroinformatics
- Spatial modeling
- Climate change vulnerability and resilience
- Machine learning for environmental systems
