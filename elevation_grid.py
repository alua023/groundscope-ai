"""Скачивает высоты Алматы на сетке 500 м для эксперимента с ростом города.

Запуск:  python elevation_grid.py
Занимает около 10 минут (бесплатный сервис высот ограничивает частоту запросов).
Прогресс сохраняется: если скрипт прервать, при новом запуске он продолжит с того же места.
Чтобы начать заново:  python elevation_grid.py --refresh
Результат: data/elevation_grid.npz
"""
import os
import sys

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer

from growth_model import find_layer, load_clipped
from train_model import get_elevations

STEP = 5  # каждая 5-я ячейка по 100 м = сетка 500 м
BUFFER_M = 1500  # берём точки и чуть за границей города, чтобы уклон у края считался верно
CHUNK = 500  # после каждых 500 точек прогресс сохраняется на диск
OUT = "data/elevation_grid.npz"
PARTIAL = "data/elevation_grid_partial.npy"


def coarse_coordinates(shape, transform, step=STEP):
    """Координаты (в проекции слоя) узлов грубой сетки: каждая step-я ячейка."""
    rows, cols = shape
    n_rows, n_cols = (rows - 1) // step + 1, (cols - 1) // step + 1
    row_idx, col_idx = np.meshgrid(np.arange(n_rows) * step, np.arange(n_cols) * step, indexing="ij")
    xs = transform.c + (col_idx + 0.5) * transform.a
    ys = transform.f + (row_idx + 0.5) * transform.e
    return xs, ys


if __name__ == "__main__":
    if "--refresh" in sys.argv and os.path.exists(PARTIAL):
        os.remove(PARTIAL)

    boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    b0, transform, crs_wkt, geoms = load_clipped(find_layer(2000), boundary)
    xs, ys = coarse_coordinates(b0.shape, transform)

    area = shapely.union_all(geoms).buffer(BUFFER_M)
    need = shapely.contains_xy(area, xs, ys)
    lons, lats = Transformer.from_crs(crs_wkt, 4326, always_xy=True).transform(xs[need], ys[need])
    lats, lons = np.asarray(lats), np.asarray(lons)

    done = list(np.load(PARTIAL)) if os.path.exists(PARTIAL) else []
    if len(done) > len(lats):
        done = []
    print(f"Нужно точек: {len(lats)}, уже скачано: {len(done)}")

    while len(done) < len(lats):
        start = len(done)
        end = min(start + CHUNK, len(lats))
        done.extend(get_elevations(lats[start:end], lons[start:end]))
        np.save(PARTIAL, np.array(done))
        print(f"Всего скачано: {len(done)}/{len(lats)}")

    coarse = np.full(xs.shape, np.nan)
    coarse[need] = np.array(done)
    np.savez(OUT, coarse=coarse, step=STEP)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    values = np.array(done)
    print()
    print(f"Высоты: от {values.min():.0f} до {values.max():.0f} м, медиана {np.median(values):.0f} м")
    print(f"Сохранено: {OUT}")
