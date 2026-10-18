"""Скачивает данные Алматы из OpenStreetMap и сохраняет их в папку data/.

Запускается один раз:  python prepare_data.py
После этого приложение (app.py) работает быстро и без повторных загрузок.
"""
import os

import geopandas as gpd
import osmnx as ox

CITY = "Almaty, Kazakhstan"
CRS = 32643  # UTM zone 43N: подходит для Алматы (76.9° в.д.)

os.makedirs("data", exist_ok=True)

print("1/3 Границы города...")
boundary = ox.geocode_to_gdf(CITY)[["geometry"]]
boundary.to_file("data/boundary.gpkg", driver="GPKG")

print("2/3 Дороги...")
graph = ox.graph_from_place(CITY, network_type="drive")
roads = ox.graph_to_gdfs(graph, nodes=False, edges=True)
roads = gpd.GeoDataFrame(geometry=roads.geometry.values, crs=roads.crs).to_crs(CRS)
roads.to_file("data/roads.gpkg", driver="GPKG")

print("3/3 Здания (самый долгий шаг, может занять несколько минут)...")
buildings = ox.features_from_place(CITY, tags={"building": True})
centroids = buildings.to_crs(CRS).geometry.centroid
buildings_points = gpd.GeoDataFrame(geometry=centroids.values, crs=CRS)
buildings_points.to_file("data/buildings.gpkg", driver="GPKG")

print(f"Готово! Дорог: {len(roads)}, зданий: {len(buildings_points)}.")
print("Данные сохранены в папке data/")
