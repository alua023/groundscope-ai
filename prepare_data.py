"""Скачивает данные Алматы из OpenStreetMap и сохраняет их в папку data/.

Запускается один раз:  python prepare_data.py
Уже скачанные файлы пропускаются, поэтому повторный запуск докачивает только недостающее.
После этого приложение (app.py) работает быстро и без повторных загрузок.
"""
import os

import geopandas as gpd
import osmnx as ox

CITY = "Almaty, Kazakhstan"
CRS = 32643  # UTM zone 43N: подходит для Алматы (76.9° в.д.)

os.makedirs("data", exist_ok=True)


def already_done(path):
    if os.path.exists(path):
        print(f"  {path} уже есть, пропускаю")
        return True
    return False


print("1/4 Границы города...")
if not already_done("data/boundary.gpkg"):
    boundary = ox.geocode_to_gdf(CITY)[["geometry"]]
    boundary.to_file("data/boundary.gpkg", driver="GPKG")

print("2/4 Дороги...")
if not already_done("data/roads.gpkg"):
    graph = ox.graph_from_place(CITY, network_type="drive")
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    roads = gpd.GeoDataFrame(geometry=roads.geometry.values, crs=roads.crs).to_crs(CRS)
    roads.to_file("data/roads.gpkg", driver="GPKG")

print("3/4 Здания (долгий шаг, может занять несколько минут)...")
if not already_done("data/buildings.gpkg"):
    buildings = ox.features_from_place(CITY, tags={"building": True})
    centroids = buildings.to_crs(CRS).geometry.centroid
    buildings_points = gpd.GeoDataFrame(geometry=centroids.values, crs=CRS)
    buildings_points.to_file("data/buildings.gpkg", driver="GPKG")

print("4/4 Реки, ручьи и каналы...")
if not already_done("data/waterways.gpkg"):
    water = ox.features_from_place(CITY, tags={"waterway": ["river", "stream", "canal"]})
    water = water[water.geom_type.isin(["LineString", "MultiLineString"])]
    water = gpd.GeoDataFrame(geometry=water.geometry.values, crs=water.crs).to_crs(CRS)
    water.to_file("data/waterways.gpkg", driver="GPKG")
    print(f"  найдено водотоков: {len(water)}")

print("Готово! Данные сохранены в папке data/")
