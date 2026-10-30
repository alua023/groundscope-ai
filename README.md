# GroundScope AI

An AI tool for a first assessment of land plots in Almaty, Kazakhstan. The user clicks a point on a map and gets a suitability score with an explanation, plus an estimate of how strongly the city is developing around that point.

**Live demo:** https://groundscope-ai-4jwblqhhjvfetquhjmjvpc.streamlit.app/ (if the app was idle, it may need a minute to wake up)

> **Status:** early prototype. The suitability score is based on a temporary formula, not on expert labels. The development pressure layer is trained on real observed urban growth, but it is a first experiment. Neither is meant for real decisions.

## What it does

- Shows a map of Almaty with the city boundary and a switchable satellite layer
- On click, collects features for the selected point:
  - elevation (Open-Meteo elevation API)
  - terrain slope, calculated from elevations of neighbouring points (about 100 m step)
  - distance to the nearest road (OpenStreetMap)
  - building density within 300 m (OpenStreetMap)
  - distance to the nearest river, stream or canal (OpenStreetMap), used as a rough proxy for flood and mudflow exposure
- Predicts a suitability score from 0 to 100 with a Random Forest model
- Shows four tabs under the result:
  - **Why this score:** how much each feature raises or lowers the score compared with a typical plot
  - **What could be built:** rough rule-based hints for housing, commerce and warehouses (thresholds are assumptions, not a trained model)
  - **Development pressure:** low, medium or high pressure of new construction around the point, from the growth experiment below
  - **Signal to machinery (demo):** shows the format of a message that could be sent to equipment on site. No real equipment is connected
- Warns when a point is outside the city, outside the elevation range the model was trained on, on a slope steeper than 15% (treated as an unfavourable condition for construction in Almaty), or very close to a watercourse

## Growth experiment: where does the city develop?

The suitability score cannot be validated, because its labels come from a formula. To get real labels, the project uses the Global Human Settlement Layer (GHS-BUILT-S R2023A, 100 m grid, tile R4_C25).

**Setup**
- Candidates: 100 m cells that were almost empty in 2000 and lie within 3 km of built-up area (28,608 cells)
- Label: the cell became built-up (at least 30% of its area) by 2015 (1,810 cells, 6.3%)
- Features are taken only from the start of the period, so the model cannot see the answer: share of built-up area around the cell (500 m and 1 km), distance to built-up area, distance to the city centre, distance to watercourses, elevation and slope
- Validation: the city is split into 2 km blocks, and neighbouring cells never appear in training and testing at the same time

**Results (ROC AUC)**

| Model | AUC |
|---|---|
| Simple rule: closer to built-up area means more likely | 0.78 |
| Random Forest without terrain | 0.87 |
| Random Forest with elevation and slope | 0.89 |

Across the five parts of the city the AUC was 0.87 (±0.04) without terrain and 0.88 (±0.04) with terrain. Terrain improved the result in all five parts, but the gain is small. The most important features are the share of built-up area around a cell and elevation.

**How the layer is used in the app.** The model trained on 2000 to 2015 is applied to the 2020 state of the city. Free cells are divided into three equal groups (low, medium, high pressure). These classes are relative within Almaty and are not probabilities.

**Important:** development pressure shows where the city is expanding, not where building is safe. High pressure next to a steep slope or a river is exactly the case that needs a risk check.

## Project structure

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads the city boundary, roads, buildings and watercourses from OpenStreetMap into `data/` |
| `train_model.py` | Builds the training dataset (random points in Almaty) and trains the suitability model |
| `app.py` | Streamlit web app with the interactive map |
| `check_ghsl.py` | Reads the GHSL layers, clips them to Almaty and shows where the city grew |
| `elevation_grid.py` | Downloads elevations on a 500 m grid for the growth experiment |
| `growth_model.py` | The growth experiment: labels, features, spatial validation |
| `build_growth_layer.py` | Trains the final growth model and builds the layer used by the app |
| `01_site_analysis_mvp.ipynb` | Notebook with the first data exploration (early version of the pipeline) |
| `site_model.pkl`, `reference.json` | Trained suitability model and the "typical plot" used for explanations |
| `requirements.txt` | Dependencies for deployment |

## How to run

1. Install Python 3.11 or newer and create a virtual environment.
2. Install the dependencies:

```
pip install geopandas osmnx folium streamlit streamlit-folium scikit-learn pandas requests joblib matplotlib rasterio
```

3. Download the city data (once, takes a few minutes):

```
python prepare_data.py
```

4. Train the suitability model:

```
python train_model.py
```

5. Start the app:

```
streamlit run app.py
```

**Growth layer (optional).** Download GHS-BUILT-S R2023A, 100 m, Mollweide, tile R4_C25, for the epochs 2000, 2015 and 2020 from the GHSL website and unpack the `.tif` files into `data/ghsl/`. Then run:

```
python elevation_grid.py
python growth_model.py
python build_growth_layer.py
```

## Limitations

- The suitability score is calculated with a temporary formula based on elevation, road distance, building density, slope and distance to watercourses. The 15% slope threshold comes from construction restrictions described for Almaty; the size of the penalties is an assumption. The model learns this formula, so the explanation shows the logic of the model, not real soil or terrain properties.
- The model reproduces the temporary formula with a mean error of about 3 points on held-out points. This shows that the pipeline works, not that the score is accurate.
- Seismic zones, landslides, mudflows, groundwater, soil type and zoning rules are not included yet.
- Slope is approximate: the elevation source has a resolution of about 90 m, and the growth experiment uses a 500 m grid.
- Distance to watercourses is only a rough proxy for flood and mudflow exposure. It is not a hazard map and does not replace water protection zone checks.
- GHSL layers between the observed years are interpolated, and 2020 is partly extrapolated, so the growth results are approximate. The thresholds for the label (30% built-up, 3 km radius) were chosen by the author and their influence has not been tested yet.
- Development pressure classes are relative and are not probabilities. Growth does not mean suitability.
- The "What could be built" hints use simple rules with assumed thresholds. The real land use is defined by the master plan and zoning, which are not included.
- The tool gives a preliminary estimate only. It does not replace geological surveys or checking urban planning restrictions.

## Roadmap

- Add region-specific risk data for Kazakhstan (seismic zones, mudflow and landslide risk)
- Replace the temporary score with expert labels or land-value data
- Test the growth model on a later period (train on one period, check on the next) and check how the label thresholds affect the result
- Show the development pressure as a map layer, not only for the clicked point
- Recommendation of the most suitable land use with a trained model
- Extension to other cities in Kazakhstan

## Data sources

- OpenStreetMap contributors (city boundary, roads, buildings, watercourses)
- Open-Meteo elevation API
- GHS-BUILT-S R2023A, Global Human Settlement Layer, European Commission, Joint Research Centre. Reuse is authorised with acknowledgment of the source; please follow the citation guidance on the GHSL website.
- Esri World Imagery (satellite map layer)

## Author

Alua Madyarova, first-year student in AI and Machine Learning.
