"""Собирает датасет по Алматы и обучает модель оценки участка.

Запуск (после prepare_data.py):  python train_model.py

Признаки считаются так же, как в app.py: высота, расстояние до ближайшей дороги,
число зданий в радиусе 300 м и уклон рельефа. На выходе:
  - site_model.pkl            обученная модель
  - reference.json            "типичный участок" (медианы признаков) для объяснений
  - data/training_dataset.csv собранный датасет
"""
import json
import math
import time

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
import shapely
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CRS = 32643  # UTM zone 43N: подходит для Алматы
N_CANDIDATES = 1500  # случайных точек внутри границы города (часть отсеется по высоте)
MAX_ELEVATION = 1300  # выше начинаются горы, это не зона застройки
SLOPE_STEP_M = 100  # шаг в метрах для расчёта уклона (высоты берутся с шагом ~90 м)
SEED = 42
FEATURES = ["высота_м", "расст_до_дороги_м", "плотность_застройки", "уклон_%"]


def get_elevations(lats, lons, batch=50, pause=6.0):
    """Высоты по списку координат небольшими пачками, с ожиданием при лимите запросов."""
    result = []
    for i in range(0, len(lats), batch):
        lat_part = ",".join(f"{v:.6f}" for v in lats[i:i + batch])
        lon_part = ",".join(f"{v:.6f}" for v in lons[i:i + batch])
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_part}&longitude={lon_part}"

        for attempt in range(6):
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  сервис просит подождать (лимит запросов), жду {wait} с...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            raise RuntimeError("Сервис высот долго отказывает. Запустите скрипт позже.")

        result.extend(response.json()["elevation"])
        print(f"  высоты: {len(result)}/{len(lats)}")
        time.sleep(pause)
    return result


def neighbor_points(lat, lon, step=SLOPE_STEP_M):
    """Четыре соседние точки: север, юг, восток, запад."""
    dlat = step / 111320
    dlon = step / (111320 * math.cos(math.radians(lat)))
    return [(lat + dlat, lon), (lat - dlat, lon), (lat, lon + dlon), (lat, lon - dlon)]


def slope_percent(z_north, z_south, z_east, z_west, step=SLOPE_STEP_M):
    """Уклон в процентах по высотам четырёх соседних точек."""
    dz_dy = (z_north - z_south) / (2 * step)
    dz_dx = (z_east - z_west) / (2 * step)
    return 100 * math.hypot(dz_dx, dz_dy)


def suitability_score(row):
    """Условная формула score (временная замена экспертным меткам).

    Порог уклона 15% взят из ограничений на застройку в Алматы (склоны круче 15%
    относятся к неблагоприятным условиям). Размеры штрафов выбраны условно.
    """
    score = 100
    score -= min(row["расст_до_дороги_м"] / 10, 40)
    if row["высота_м"] > 1000:
        score -= (row["высота_м"] - 1000) / 10
    elif row["высота_м"] < 700:
        score -= (700 - row["высота_м"]) / 10
    if row["плотность_застройки"] < 30:
        score -= (30 - row["плотность_застройки"]) / 2
    elif row["плотность_застройки"] > 250:
        score -= (row["плотность_застройки"] - 250) / 5

    slope = row["уклон_%"]
    if slope > 15:
        score -= min(10 + (slope - 15) * 2, 40)
    elif slope > 5:
        score -= (slope - 5)
    return round(max(score, 0), 1)


if __name__ == "__main__":
    print("Загружаю данные города...")
    boundary = gpd.read_file("data/boundary.gpkg").to_crs(CRS).geometry.iloc[0]
    roads = gpd.read_file("data/roads.gpkg")
    buildings = gpd.read_file("data/buildings.gpkg")

    print(f"Генерирую {N_CANDIDATES} случайных точек и оставляю те, что внутри города...")
    rng = np.random.default_rng(SEED)
    minx, miny, maxx, maxy = boundary.bounds
    xs = rng.uniform(minx, maxx, N_CANDIDATES)
    ys = rng.uniform(miny, maxy, N_CANDIDATES)
    inside = shapely.contains_xy(boundary, xs, ys)
    points_proj = gpd.GeoSeries(gpd.points_from_xy(xs[inside], ys[inside]), crs=CRS)
    points_wgs = points_proj.to_crs(4326)
    lats = points_wgs.y.to_numpy()
    lons = points_wgs.x.to_numpy()
    print(f"Внутри города: {len(points_proj)} точек")

    print("Запрашиваю высоты точек...")
    elevations = get_elevations(lats, lons)

    keep = [i for i, e in enumerate(elevations) if e <= MAX_ELEVATION]
    print(f"Ниже {MAX_ELEVATION} м: {len(keep)} точек")

    print("Запрашиваю высоты соседних точек для расчёта уклона (самый долгий шаг)...")
    neighbor_lats, neighbor_lons = [], []
    for i in keep:
        for n_lat, n_lon in neighbor_points(lats[i], lons[i]):
            neighbor_lats.append(n_lat)
            neighbor_lons.append(n_lon)
    neighbor_elevations = get_elevations(neighbor_lats, neighbor_lons)

    print("Считаю остальные признаки...")
    rows = []
    for n, i in enumerate(keep):
        z_north, z_south, z_east, z_west = neighbor_elevations[4 * n:4 * n + 4]
        point = points_proj.iloc[i]
        road_dist = float(roads.distance(point).min())
        density = int(len(buildings.sindex.query(point.buffer(300), predicate="intersects")))
        rows.append({
            "lat": lats[i],
            "lon": lons[i],
            "высота_м": elevations[i],
            "расст_до_дороги_м": round(road_dist, 1),
            "плотность_застройки": density,
            "уклон_%": round(slope_percent(z_north, z_south, z_east, z_west), 1),
        })
        if (n + 1) % 100 == 0:
            print(f"  обработано {n + 1}/{len(keep)}")

    dataset = pd.DataFrame(rows)
    dataset["score_пригодности"] = dataset.apply(suitability_score, axis=1)
    dataset.to_csv("data/training_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"Датасет: {len(dataset)} точек в зоне застройки (ниже {MAX_ELEVATION} м)")
    print(f"Уклон: медиана {dataset['уклон_%'].median():.1f}%, "
          f"доля точек круче 15%: {(dataset['уклон_%'] > 15).mean() * 100:.0f}%")

    X = dataset[FEATURES]
    y = dataset["score_пригодности"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = RandomForestRegressor(n_estimators=200, random_state=SEED)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print()
    print(f"Средняя ошибка на тесте (MAE): {mean_absolute_error(y_test, predictions):.2f} баллов")
    print(f"R²: {r2_score(y_test, predictions):.2f}")
    print("Важно: score задан формулой от этих же признаков, поэтому высокий R² значит,")
    print("что модель выучила формулу, а не реальную пригодность участков.")

    # Финальная модель обучается на всех данных
    model.fit(X, y)
    joblib.dump(model, "site_model.pkl")

    reference = {col: round(float(dataset[col].median()), 1) for col in FEATURES}
    with open("reference.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)

    print()
    print("Важность признаков:")
    for name, value in sorted(zip(FEATURES, model.feature_importances_), key=lambda item: -item[1]):
        print(f"  {name}: {value:.2f}")
    print()
    print("Готово: site_model.pkl, reference.json и data/training_dataset.csv сохранены.")
