"""Ещё два эксперимента с CNN.

1) Синтетическое расширение данных (аугментация): каждый снимок 21x21 поворачивается на 0, 90, 180
   и 270 градусов, часть снимков отражается. Сравниваем сеть на исходных и на расширенных данных
   (это вариант B задания: обогатить набор данных синтетическими примерами и сравнить результат).
2) Неопределённость (Monte Carlo Dropout): один и тот же снимок пропускается через сеть 30 раз
   со случайно выключаемыми нейронами. Большой разброс значит, что сеть не уверена.

Запуск:  python cnn_more.py     (около 15 минут; сначала должны быть выполнены cnn_growth.py и cnn_extra.py)
Результат: таблица в консоли, data/cnn_more_results.csv, data/cnn_uncertainty.png
"""
import csv

import geopandas as gpd
import matplotlib
import numpy as np
import torch
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn

from cnn_extra import train_model
from cnn_growth import build_channels, extract_patches, train_and_score
from growth_model import (
    CELL, CENTER, ELEVATION_GRID, features_and_labels, find_layer, load_clipped, terrain_from_coarse,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
DEPTH = 2
PASSES = 30


def augment(X, y):
    """Синтетические примеры: повороты на 0, 90, 180, 270 градусов; у поворотов 90 и 270 ещё и отражение."""
    xs, ys = [], []
    for k in range(4):
        rotated = np.rot90(X, k, axes=(2, 3))
        if k % 2 == 1:
            rotated = rotated[:, :, :, ::-1]
        xs.append(np.ascontiguousarray(rotated))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def mc_predict(model, X, passes=PASSES):
    """Среднее и разброс вероятностей при случайно выключаемых нейронах."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # оставляем Dropout включённым
    runs = []
    with torch.no_grad():
        for _ in range(passes):
            parts = [torch.sigmoid(model(torch.from_numpy(X[i:i + 1024]))).numpy() for i in range(0, len(X), 1024)]
            runs.append(np.concatenate(parts))
    runs = np.stack(runs)
    return runs.mean(axis=0), runs.std(axis=0)


def split_by_uncertainty(y, p_mean, p_std):
    """AUC отдельно для более уверенной и менее уверенной половины ячеек."""
    certain = p_std <= np.median(p_std)
    return roc_auc_score(y[certain], p_mean[certain]), roc_auc_score(y[~certain], p_mean[~certain])


if __name__ == "__main__":
    boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    b0, transform, crs_wkt, geoms = load_clipped(find_layer(2000), boundary)
    b1, _, _, _ = load_clipped(find_layer(2015), boundary)
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
    dist_water = ndimage.distance_transform_edt(water_raster == 0) * CELL
    _, y, groups, candidate = features_and_labels(b0, b1, inside, water_raster, xs, ys, cx, cy)
    ch = build_channels(b0, elevation, slope, dist_water, inside)
    X = extract_patches([ch["built"], ch["water"]], candidate)
    splits = list(GroupKFold(n_splits=5).split(X, y, groups))

    # 1) исходные данные против расширенных
    auc_orig, auc_aug = [], []
    for fold, (train, test) in enumerate(splits, start=1):
        auc_orig.append(roc_auc_score(y[test], train_and_score(X[train], y[train], X[test], DEPTH, SEED)))
        X_aug, y_aug = augment(X[train], y[train])
        auc_aug.append(roc_auc_score(y[test], train_and_score(X_aug, y_aug, X[test], DEPTH, SEED)))
        print(f"  часть {fold}/5: исходные {auc_orig[-1]:.3f}, с аугментацией {auc_aug[-1]:.3f}")

    # 2) неопределённость
    p_mean = np.zeros(len(y), dtype="float32")
    p_std = np.zeros(len(y), dtype="float32")
    for fold, (train, test) in enumerate(splits, start=1):
        model = train_model(X[train], y[train], DEPTH, SEED)
        p_mean[test], p_std[test] = mc_predict(model, X[test])
        print(f"  неопределённость, часть {fold}/5 готова")
    auc_all = roc_auc_score(y, p_mean)
    auc_certain, auc_uncertain = split_by_uncertainty(y, p_mean, p_std)

    print("\nИтог")
    print(f"  Исходные данные: AUC {np.mean(auc_orig):.3f} (по частям ±{np.std(auc_orig):.3f})")
    print(f"  С аугментацией:  AUC {np.mean(auc_aug):.3f} (по частям ±{np.std(auc_aug):.3f})")
    print(f"  Улучшение в {sum(a > o for a, o in zip(auc_aug, auc_orig))} частях из 5")
    print(f"  MC Dropout, все ячейки: AUC {auc_all:.3f}")
    print(f"  Более уверенная половина: AUC {auc_certain:.3f}; менее уверенная: AUC {auc_uncertain:.3f}")

    with open("data/cnn_more_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["эксперимент", "AUC", "разброс по частям"])
        writer.writerow(["CNN, исходные данные", round(np.mean(auc_orig), 3), round(np.std(auc_orig), 3)])
        writer.writerow(["CNN, с аугментацией", round(np.mean(auc_aug), 3), round(np.std(auc_aug), 3)])
        writer.writerow(["MC Dropout, уверенная половина", round(auc_certain, 3), ""])
        writer.writerow(["MC Dropout, неуверенная половина", round(auc_uncertain, 3), ""])

    mean_map = np.full(b0.shape, np.nan)
    mean_map[candidate] = p_mean
    std_map = np.full(b0.shape, np.nan)
    std_map[candidate] = p_std
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mean_map, cmap="viridis")
    axes[0].set_title("Средняя вероятность роста (CNN)")
    axes[1].imshow(std_map, cmap="magma")
    axes[1].set_title("Неопределённость (разброс при MC Dropout):\nсветлее значит сеть менее уверена")
    for ax in axes:
        ax.axis("off")
    plt.savefig("data/cnn_uncertainty.png", dpi=110, bbox_inches="tight")
    print("\nСохранено: data/cnn_more_results.csv, data/cnn_uncertainty.png")
