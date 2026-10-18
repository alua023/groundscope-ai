# GroundScope AI

An AI tool for a first assessment of land plot suitability for construction. The user clicks a point on a map of Almaty, Kazakhstan, and gets a suitability score together with an explanation of what affects it.

> **Status:** early prototype (MVP v0.1). The score is based on a temporary formula, not on expert labels. It is meant for testing the pipeline, not for real decisions.

## What it does

- Shows a map of Almaty with the city boundary
- On click, collects features for the selected point:
  - elevation (Open-Meteo elevation API)
  - distance to the nearest road (OpenStreetMap)
  - building density within 300 m (OpenStreetMap)
- Predicts a suitability score from 0 to 100 with a Random Forest model
- Explains the result by showing how much each feature raises or lowers the score compared with a typical plot
- Warns when a point is outside the city or outside the elevation range the model was trained on

## Project structure

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads the city boundary, roads and buildings from OpenStreetMap into the `data/` folder |
| `train_model.py` | Builds the training dataset (random points in Almaty) and trains the Random Forest model |
| `app.py` | Streamlit web app with the interactive map |
| `01_site_analysis_mvp.ipynb` | Notebook with the first data exploration (early version of the pipeline) |
| `site_model.pkl` | Trained Random Forest model |

## How to run

1. Install Python 3.11 or newer and create a virtual environment.
2. Install the dependencies:

```
pip install geopandas osmnx folium streamlit streamlit-folium scikit-learn pandas requests joblib
```

3. Download the city data (once, takes a few minutes):

```
python prepare_data.py
```

4. Train the model (a few minutes, it requests elevations from a free API):

```
python train_model.py
```

5. Start the app:

```
streamlit run app.py
```

## Limitations

- The suitability score is calculated with a temporary formula based on elevation, road distance and building density. The model learns this formula, so the explanation shows the logic of the model, not real soil or terrain properties.
- Seismic zones, landslides, mudflows, soil type and zoning rules are not included yet.
- The training set is small: about 480 random points in Almaty below 1300 m.
- The model reproduces the temporary formula with a mean error of about 1.7 points on held-out points. This shows that the pipeline works, not that the score is accurate.
- The tool gives a preliminary estimate only. It does not replace geological surveys or checking urban planning restrictions.

## Roadmap

- Add region-specific risk data for Kazakhstan (seismic zones, mudflow and landslide risk)
- Replace the temporary score with expert labels or land-value data
- Larger training set and better validation
- Recommendation of the most suitable land use for a plot
- Extension to other cities in Kazakhstan

## Data sources

- OpenStreetMap contributors (city boundary, roads, buildings)
- Open-Meteo elevation API

## Author

Alua Madyarova, first-year student in AI and Machine Learning.
