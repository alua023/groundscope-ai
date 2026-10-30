"""Строит слой "давление застройки" для приложения.

Модель учится на росте 2000→2015 (признаки на 2000 год) и применяется к состоянию 2020 года:
оценивает, какие сейчас почти пустые ячейки вероятнее всего будут застраиваться дальше.
Запуск:  python build_growth_layer.py   (после elevation_grid.py)
Результат: data/growth_layer.npz (небольшой файл, его же использует приложение)
"""
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier

from growth_model import (
    BUILT_MIN, CENTER, ELEVATION_GRID, SEED, START_MAX,
    features_and_labels, find_layer, load_clipped, terrain_from_coarse,
)

OUT = "data/growth_layer.npz"
# коды слоя: 0 вне зоны оценки, 1 низкое, 2 среднее, 3 высокое давление застройки,
# 4 уже застроено, 5 частично застроено


def assign_levels(b_now, candidate, proba):
    """Делит кандидатов на три равные группы по вероятности и помечает уже застроенные ячейки."""
    level = np.zeros(b_now.shape, dtype="uint8")
    level[(b_now >= START_MAX) & (b_now < BUILT_MIN)] = 5
    level[b_now >= BUILT_MIN] = 4
    q1, q2 = np.quantile(proba, [1 / 3, 2 / 3])
    classes = np.where(proba >= q2, 3, np.where(proba >= q1, 2, 1)).astype("uint8")
    level[candidate] = classes
    return level, (float(q1), float(q2))


if __name__ == "__main__":
    boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    b0, transform, crs_wkt, geoms = load_clipped(find_layer(2000), boundary)
    b1, _, _, _ = load_clipped(find_layer(2015), boundary)
    b2, _, _, _ = load_clipped(find_layer(2020), boundary)

    inside = rasterize([(g, 1) for g in geoms], out_shape=b0.shape, transform=transform, fill=0).astype(bool)
    water = gpd.read_file("data/waterways.gpkg").to_crs(crs_wkt)
    water_raster = rasterize(
        [(g, 1) for g in water.geometry], out_shape=b0.shape, transform=transform, fill=0, all_touched=True
    )
    rows, cols = np.indices(b0.shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    cx, cy = Transformer.from_crs(4326, crs_wkt, always_xy=True).transform(CENTER[1], CENTER[0])

    grid = np.load(ELEVATION_GRID)
    elevation, slope = terrain_from_coarse(grid["coarse"], int(grid["step"]), b0.shape)

    def build(base, later):
        X, y, _, candidate = features_and_labels(base, later, inside, water_raster, xs, ys, cx, cy)
        return np.column_stack([X, elevation[candidate], slope[candidate]]), y, candidate

    X_train, y_train, _ = build(b0, b1)
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED
    ).fit(X_train, y_train)
    print(f"Модель обучена на росте 2000→2015: {len(y_train)} ячеек, {y_train.sum()} стали застроенными")

    X_now, _, candidate_now = build(b2, b2)
    proba = model.predict_proba(X_now)[:, 1]
    level, thresholds = assign_levels(b2, candidate_now, proba)

    np.savez_compressed(
        OUT,
        level=level,
        transform=np.array([transform.c, transform.a, transform.f, transform.e]),
        crs=crs_wkt,
    )
    names = {0: "вне зоны оценки", 1: "низкое", 2: "среднее", 3: "высокое", 4: "уже застроено", 5: "частично застроено"}
    print("Ячеек по классам:")
    for code, name in names.items():
        print(f"  {name}: {int((level == code).sum())}")
    print(f"Сохранено: {OUT}")
