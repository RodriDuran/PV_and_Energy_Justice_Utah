# utah-ejpi

Heating and cooling energy of the dwellings of every census tract, built from
modeled dwellings (ResStock, EnergyPlus) linked to the housing counts of the
American Community Survey. It is the first block of a study of a just energy
transition for heating and cooling (energy poverty, rooftop solar and a justice
index per census tract). The package name is provisional.

The study is run for Utah first. Every method is chosen to work for any state of the
United States; the parts validated only in Utah are flagged in the documentation of
each function.

> **Status.** Work in progress. The energy block (what each tract needs to heat and
> cool its dwellings to a normative standard, how much more than it uses today, and
> the electricity of a heat pump that would supply it) is implemented, tested and run
> for Utah. Energy poverty, rooftop solar and the justice index are not implemented.
> No license has been chosen yet.

## What it estimates

For each 2020 census tract (key `GEOID`, 11 characters):

1. **Housing stock.** Which modeled dwellings stand for each type of dwelling of the
   tract (tenure, year built, building type), with weights fitted to the Census
   counts.
2. **Energy used today.** Annual and hourly (8,760 hours) useful heating and cooling
   energy, with the climate of the tract.
3. **Energy needed and gap.** Each modeled dwelling is simulated again with a
   normative thermostat (at least 18 °C and at most 26 °C while someone is at home
   and awake, with 3 hours of pre-conditioning). The gap is the normative energy
   minus the energy of the current thermostat.
4. **Heat pump.** Electricity of a heat pump that supplies the useful energy, from
   the ResStock heat pump package simulated again, or from a coefficient of
   performance curve by outdoor temperature for the dwellings that have no package.
5. **Uncertainty.** Every quantity comes as a set of simulations that resample the
   donor dwellings, with the error of transferring a dwelling from another place
   estimated from a test on a known population.

## Layout

```
src/utah_ejpi/   package modules (one per block of the study)
analysis/        numbered scripts that run the study for Utah by calling the package
tests/           pytest tests on small synthetic samples; no Internet needed
```

| Module | Content |
| --- | --- |
| `geography` | Master table of tracts and their link to ResStock geography |
| `housing_stock`, `resstock` | Housing stock of each tract; ResStock results in Census categories |
| `climate`, `energy_requirement`, `hourly` | Degree days, climate correction, hourly profiles |
| `uncertainty`, `gap_uncertainty` | Simulations by tract with the donors resampled |
| `comfort_gap`, `resimulation` | Normative thermostat, activity windows, edits of the EnergyPlus models |
| `heat_pump` | Efficiency curve of the heat pump and its electricity |
| `validation`, `utils` | Comparisons with independent data; shared helpers |
| `download` | Download of the raw data sources |

Analysis scripts, to be run from the project root in this order:

| Script | Output |
| --- | --- |
| `01_download_data.py` | Raw data enabled in `config.py` |
| `02_housing_stock.py` | Link tracts to ResStock, degree days |
| `03_energy_requirement.py` | Annual energy and hourly profiles of the tracts |
| `04_reweighting.py` | Dwelling weights fitted to several Census tables |
| `05_uncertainty.py` | Simulations of the energy of each tract |
| `06_resimulation.py` | Normative thermostat and simulation of every donor dwelling |
| `07_gap_by_tract.py` | Gap and heat pump electricity by tract, with intervals |
| `08_sensitivities.py` | Sensitivity to the thermostat parameters and to evaporative coolers |
| `09_gap_profiles.py` | Hourly profiles of the gap by tract |
| `10_climate_shift.py` | Physical test of the climate correction |
| `13_validation.py` | Validation against Census, survey and utility data |

## Installation

Python 3.10 or later.

```bash
pip install -e ".[dev]"
pytest
ruff check
```

Optional groups: `geo`, `climate`, `validation`. The tests use small synthetic
samples and never need Internet.

## Running the study

All settings are in `analysis/config.py`. Downloads and long computations are off by
default, each one a flag with a description of what it does, its approximate time
and its output. The simulation of every dwelling with EnergyPlus (`RUN_RESIMULATION`)
needs EnergyPlus 24.2 installed and takes several hours on a workstation.
The raw data are never shipped with the package; the download functions fail with an
informative message if a source is not available.

## Data sources

All have national coverage. American Community Survey 5-year estimates 2020 to 2024
(with margins of error), Census TIGER/Line geography and 2010 to 2020 relationship
files, NREL ResStock 2025 release 1 (weather year 2018), PRISM climate grids, the
Residential Energy Consumption Survey 2020, EIA sales totals, LEAD tool (Department
of Energy) and the American Time Use Survey. The catalog of every variable, its
source, universe and verification state is in
`src/utah_ejpi/data/variable_catalog.csv`. References are in `references.bib`.

## Conventions

- Unit and key: census tract 2020, `GEOID`.
- American Community Survey: always with margin of error; coefficient of variation
  = margin / (1.645 × estimate); quality high below 12 %, medium 12 to 40 %, low
  above 40 %.
- Race is not used as a term of any index.
- Code, comments and docstrings in English, numpydoc style.

## Resumen en español

Paquete que estima, por cada tract censal de Estados Unidos (unidad parecida a un
radio censal), la energía de calefacción y refrigeración de las viviendas,
vinculando viviendas modeladas con ResStock (modelos de EnergyPlus del Departamento
de Energía) a los conteos de vivienda del Censo. Calcula la energía que se usa hoy,
la que haría falta para mantener 18 a 26 °C cuando hay alguien en casa, la brecha
entre ambas, la electricidad de una bomba de calor y la incertidumbre de cada
cifra. Se corre para Utah y se diseña para cualquier estado. Es el primer bloque de
un estudio de transición energética justa; pobreza energética, energía solar en
techos e índice de justicia aún no están implementados.
