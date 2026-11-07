"""Индекс Морана: группируется ли рост застройки в пространстве?

Значение около 0 значит, что ячейки с ростом разбросаны случайно, около 1 значит, что они сгруппированы.
Если рост сильно сгруппирован, случайное разбиение на обучение и проверку даёт завышенный результат,
поэтому мы проверяем модели по пространственным блокам.

Запуск:  python morans_i.py     (меньше минуты)
"""
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage

from growth_model import BUILT_MIN, CELL, NEAR_M, START_MAX, find_layer, load_clipped

KERNEL = np.ones((3, 3))
KERNEL[1, 1] = 0  # соседи по 8 направлениям


def morans_i(values, mask):
    """Индекс Морана для значений на сетке (только ячейки внутри маски)."""
    m = mask.astype(float)
    z = np.where(mask, values - values[mask].mean(), 0.0)
    neighbours = ndimage.convolve(z, KERNEL, mode="constant")
    weight_sum = (m * ndimage.convolve(m, KERNEL, mode="constant")).sum()
    return (m.sum() / weight_sum) * (z * neighbours).sum() / (z ** 2).sum()


def permutation_p(values, mask, observed, n=99, seed=42):
    """Доля случайных перемешиваний, давших индекс не меньше наблюдаемого."""
    rng = np.random.default_rng(seed)
    shuffled_values = values[mask].copy()
    not_smaller = 0
    for _ in range(n):
        rng.shuffle(shuffled_values)
        shuffled = np.zeros(values.shape)
        shuffled[mask] = shuffled_values
        if morans_i(shuffled, mask) >= observed:
            not_smaller += 1
    return (not_smaller + 1) / (n + 1)


if __name__ == "__main__":
    boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    b0, transform, _, geoms = load_clipped(find_layer(2000), boundary)
    b1, _, _, _ = load_clipped(find_layer(2015), boundary)
    inside = rasterize([(g, 1) for g in geoms], out_shape=b0.shape, transform=transform, fill=0).astype(bool)
    dist_built = ndimage.distance_transform_edt(~(b0 >= BUILT_MIN)) * CELL
    candidate = inside & (b0 < START_MAX) & (dist_built <= NEAR_M)
    label = (b1 >= BUILT_MIN).astype(float)

    observed = morans_i(label, candidate)
    p_value = permutation_p(label, candidate, observed)
    print(f"Ячеек-кандидатов: {int(candidate.sum())}, доля застроенных к 2015 году: {label[candidate].mean() * 100:.1f}%")
    print(f"Индекс Морана для роста застройки: {observed:.3f}")
    print(f"Псевдо-p при 99 перемешиваниях: {p_value:.2f} (чем меньше, тем надёжнее группировка)")
    print("Около 0: рост разбросан случайно. Чем ближе к 1, тем сильнее группировка.")
