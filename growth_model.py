"""Эксперимент: прогноз появления новой застройки в Алматы по слоям GHSL.

Идея: берём ячейки 100x100 м, которые в 2000 году были почти пустыми, и смотрим, стали ли они
застроенными к 2015 году. Признаки берутся только на начало периода (2000), чтобы модель не
"подсматривала" ответ. Если есть файл data/elevation_grid.npz (его создаёт elevation_grid.py),
дополнительно проверяется, помогают ли высота и уклон рельефа.

Запуск:  python growth_model.py
Нужны: data/ghsl/ (слои E2000 и E2015, тайл R4_C25), data/boundary.gpkg, data/waterways.gpkg
Результат: цифры в консоли и картинка data/growth_check.png
"""
import glob
import os
import time

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.mask import mask
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELL = 100  # размер ячейки, м
BUILT_MIN = 3000  # м² застройки в ячейке (из 10 000), начиная с которых ячейка считается застроенной
START_MAX = 1000  # кандидаты: ячейки, почти пустые в 2000 году
NEAR_M = 3000  # рассматриваем ячейки не дальше 3 км от застройки 2000 года
BLOCK = 20  # размер пространственного блока для проверки: 20 ячеек = 2 км
CENTER = (43.238, 76.945)  # центр Алматы (широта, долгота), приблизительно
SEED = 42
ELEVATION_GRID = "data/elevation_grid.npz"
FEATURE_NAMES = [
    "расст_до_застройки_м",
    "доля_застройки_500м",
    "доля_застройки_1км",
    "расст_до_воды_м",
    "расст_до_центра_м",
]
TERRAIN_NAMES = ["высота_м", "уклон_%"]


def features_and_labels(b0, b1, inside, water_raster, xs, ys, cx, cy):
    """Признаки на начало периода и метка "стала застроенной". Возвращает данные только для кандидатов."""
    built0 = b0 >= BUILT_MIN
    dist_built = ndimage.distance_transform_edt(~built0) * CELL
    share_500 = ndimage.uniform_filter(b0 / 10000.0, size=11, mode="constant")
    share_1000 = ndimage.uniform_filter(b0 / 10000.0, size=21, mode="constant")
    dist_water = ndimage.distance_transform_edt(water_raster == 0) * CELL
    dist_center = np.hypot(xs - cx, ys - cy)

    candidate = inside & (b0 < START_MAX) & (dist_built <= NEAR_M)
    label = (b1 >= BUILT_MIN)

    maps = [dist_built, share_500, share_1000, dist_water, dist_center]
    X = np.column_stack([m[candidate] for m in maps])
    y = label[candidate].astype(int)
    rows, cols = np.indices(b0.shape)
    groups = ((rows // BLOCK) * 1000 + (cols // BLOCK))[candidate]
    return X, y, groups, candidate


def terrain_from_coarse(coarse, step, shape):
    """Высота и уклон (в процентах) для каждой ячейки 100 м по грубой сетке высот."""
    coarse = coarse.astype("float64")
    missing = np.isnan(coarse)
    if missing.any():
        nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        coarse = coarse[tuple(nearest)]
    grad_row, grad_col = np.gradient(coarse, step * CELL)  # метры высоты на метр расстояния
    slope_coarse = 100 * np.hypot(grad_row, grad_col)
    rows, cols = np.indices(shape)
    coords = np.array([rows / step, cols / step])
    elevation = ndimage.map_coordinates(coarse, coords, order=1, mode="nearest")
    slope = ndimage.map_coordinates(slope_coarse, coords, order=1, mode="nearest")
    return elevation, slope


def evaluate(X, y, groups):
    """Пространственная кросс-проверка: соседние ячейки не попадают одновременно в обучение и проверку."""
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    proba = cross_val_predict(model, X, y, cv=GroupKFold(n_splits=5), groups=groups, method="predict_proba")[:, 1]
    auc_model = roc_auc_score(y, proba)
    auc_distance_only = roc_auc_score(y, -X[:, 0])  # простое правило: чем ближе к застройке, тем вероятнее
    model.fit(X, y)
    return proba, auc_model, auc_distance_only, model.feature_importances_


def fold_aucs(X, y, groups, proba):
    """AUC отдельно по каждому из 5 пространственных блоков: показывает, насколько результат устойчив."""
    return [roc_auc_score(y[test], proba[test]) for _, test in GroupKFold(n_splits=5).split(X, y, groups)]


def load_clipped(path, boundary_wgs):
    with rasterio.open(path) as src:
        geoms = list(boundary_wgs.to_crs(src.crs.to_wkt()).geometry)
        data, transform = mask(src, geoms, crop=True, nodata=0)
        arr = data[0].astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = 0
        arr[arr < 0] = 0
        return arr, transform, src.crs.to_wkt(), geoms


def find_layer(epoch):
    files = glob.glob(f"data/ghsl/**/*E{epoch}_*R4_C25*.tif", recursive=True)
    if not files:
        raise SystemExit(f"Не найден слой за {epoch} год в data/ghsl/")
    return files[0]


if __name__ == "__main__":
    boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    b0, transform, crs_wkt, geoms = load_clipped(find_layer(2000), boundary)
    b1, _, _, _ = load_clipped(find_layer(2015), boundary)
    if b0.shape != b1.shape:
        raise SystemExit("Слои разного размера: проверьте, что оба из тайла R4_C25.")

    inside = rasterize([(g, 1) for g in geoms], out_shape=b0.shape, transform=transform, fill=0).astype(bool)
    water = gpd.read_file("data/waterways.gpkg").to_crs(crs_wkt)
    water_raster = rasterize(
        [(g, 1) for g in water.geometry], out_shape=b0.shape, transform=transform, fill=0, all_touched=True
    )

    rows, cols = np.indices(b0.shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    cx, cy = Transformer.from_crs(4326, crs_wkt, always_xy=True).transform(CENTER[1], CENTER[0])

    X, y, groups, candidate = features_and_labels(b0, b1, inside, water_raster, xs, ys, cx, cy)
    print(f"Кандидатов (почти пустых ячеек в 2000 году рядом с застройкой): {len(y)}")
    print(f"Из них стали застроенными к 2015 году: {y.sum()} ({y.mean() * 100:.1f}%)")

    proba, auc_model, auc_distance, importances = evaluate(X, y, groups)
    names = FEATURE_NAMES
    folds_base = fold_aucs(X, y, groups, proba)
    print()
    print(f"Простое правило (чем ближе к застройке, тем вероятнее), AUC: {auc_distance:.2f}")
    print(f"Модель без рельефа, AUC: {auc_model:.2f}")

    if os.path.exists(ELEVATION_GRID):
        grid = np.load(ELEVATION_GRID)
        elevation, slope = terrain_from_coarse(grid["coarse"], int(grid["step"]), b0.shape)
        X_terrain = np.column_stack([X, elevation[candidate], slope[candidate]])
        proba, auc_terrain, _, importances = evaluate(X_terrain, y, groups)
        names = FEATURE_NAMES + TERRAIN_NAMES
        print(f"Модель с рельефом (высота и уклон), AUC: {auc_terrain:.2f}")
        print(f"Разница от рельефа: {auc_terrain - auc_model:+.3f}")
        folds_terrain = fold_aucs(X_terrain, y, groups, proba)
        better = sum(t > b for t, b in zip(folds_terrain, folds_base))
        print(f"По 5 частям города: без рельефа {np.mean(folds_base):.2f} (разброс ±{np.std(folds_base):.2f}), "
              f"с рельефом {np.mean(folds_terrain):.2f} (разброс ±{np.std(folds_terrain):.2f})")
        print(f"Рельеф улучшил результат в {better} частях из 5")
    else:
        print("Файла data/elevation_grid.npz нет: сначала запустите python elevation_grid.py, чтобы проверить рельеф.")
    print("(AUC 0,5 = случайное угадывание, 1,0 = идеально)")
    print()
    print("Важность признаков (итоговая модель):")
    for name, value in sorted(zip(names, importances), key=lambda item: -item[1]):
        print(f"  {name}: {value:.2f}")

    prob_map = np.full(b0.shape, np.nan)
    prob_map[candidate] = proba
    actual = np.full(b0.shape, np.nan)
    actual[candidate] = y
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(b0, cmap="gray_r", vmin=0, vmax=10000)
    axes[0].set_title("Застройка 2000")
    axes[1].imshow(prob_map, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Предсказанная вероятность новой застройки")
    axes[2].imshow(actual, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Что реально построено к 2015")
    for ax in axes:
        ax.axis("off")
    picture = "data/growth_check.png"
    try:
        plt.savefig(picture, dpi=110, bbox_inches="tight")
    except OSError:
        # файл может быть открыт в просмотрщике и заблокирован: сохраняем под новым именем
        picture = f"data/growth_check_{int(time.time())}.png"
        plt.savefig(picture, dpi=110, bbox_inches="tight")
    print()
    print(f"Картинка сохранена: {picture}")
