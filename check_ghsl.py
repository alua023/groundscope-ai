"""Проверяет слои GHSL (застроенная площадь) для Алматы и показывает, где рос город.

Перед запуском положите скачанные файлы .tif (по эпохам 2000, 2014, 2018) в папку data/ghsl/.
Запуск:  python check_ghsl.py
Результат: цифры в консоли и картинка data/ghsl_check.png
"""
import glob
import os
import re

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.mask import mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDER = "data/ghsl"
PICTURE = "data/ghsl_check.png"

files = sorted(glob.glob(os.path.join(FOLDER, "**", "*.tif"), recursive=True))
if not files:
    raise SystemExit(f"В папке {FOLDER} нет файлов .tif. Скачайте слои GHS-BUILT-S и распакуйте их туда.")

boundary = gpd.read_file("data/boundary.gpkg").to_crs(4326)
layers = {}  # эпоха -> массив застроенной площади (м² на ячейку)
cell_area_km2 = None

for path in files:
    match = re.search(r"_E(\d{4})_", os.path.basename(path))
    if not match:
        print(f"Пропускаю {os.path.basename(path)}: в имени не найден год (E2000, E2014 и т. п.)")
        continue
    epoch = int(match.group(1))
    with rasterio.open(path) as src:
        shapes = [geom for geom in boundary.to_crs(src.crs.to_wkt()).geometry]
        try:
            clipped, transform = mask(src, shapes, crop=True, nodata=0)
        except ValueError:
            print(f"Файл {os.path.basename(path)} не покрывает Алматы. Нужен другой тайл.")
            continue
        data = clipped[0].astype("float64")
        if src.nodata is not None:
            data[data == src.nodata] = 0
        data[data < 0] = 0
        layers[epoch] = data
        cell_area_km2 = abs(transform.a * transform.e) / 1e6
        print(f"Эпоха {epoch}: размер {data.shape[1]}x{data.shape[0]} ячеек, CRS {src.crs}")

if len(layers) < 2:
    raise SystemExit("Нужно минимум две эпохи (например, 2000 и 2014). Проверьте файлы выше.")

epochs = sorted(layers)
print()
for epoch in epochs:
    print(f"Застроенная площадь в Алматы, {epoch}: {layers[epoch].sum() / 1e6:.1f} км²")

first, last = epochs[0], epochs[-1]
growth = layers[last] - layers[first]
grew = (growth > 0).sum()
print()
print(f"Рост {first}→{last}: +{growth.clip(min=0).sum() / 1e6:.1f} км² застройки")
print(f"Ячеек 100x100 м, где застройка выросла: {grew} из {int((layers[last] > 0).sum())} застроенных к {last} году")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(layers[first], cmap="gray_r", vmin=0, vmax=10000)
axes[0].set_title(f"Застройка {first}")
axes[1].imshow(layers[last], cmap="gray_r", vmin=0, vmax=10000)
axes[1].set_title(f"Застройка {last}")
axes[2].imshow(growth.clip(min=0), cmap="Reds", vmin=0, vmax=5000)
axes[2].set_title(f"Новая застройка {first}-{last}")
for ax in axes:
    ax.axis("off")
plt.savefig(PICTURE, dpi=110, bbox_inches="tight")
print()
print(f"Картинка сохранена: {PICTURE}. Откройте её и посмотрите, совпадают ли красные области с районами роста города.")
