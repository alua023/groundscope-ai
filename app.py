import json
import math
import os

import folium
import joblib
import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

CRS = 32643  # UTM zone 43N: подходит для Алматы
MAX_TRAIN_ELEVATION = 1300  # модель обучена на участках ниже этой высоты
SLOPE_STEP_M = 100  # шаг в метрах для расчёта уклона (как в train_model.py)
SLOPE_LIMIT = 15  # склоны круче 15% относятся к неблагоприятным условиям застройки
WATER_NEAR_M = 50  # ближе этого расстояния до водотока показываем предупреждение

FEATURES = ["высота_м", "расст_до_дороги_м", "плотность_застройки", "уклон_%", "расст_до_воды_м"]
LABELS = {
    "высота_м": "Высота",
    "расст_до_дороги_м": "Расстояние до дороги",
    "плотность_застройки": "Плотность застройки",
    "уклон_%": "Уклон рельефа",
    "расст_до_воды_м": "Расстояние до водотока",
}

# "Типичный участок": медианы обучающей выборки.
# Файл reference.json создаёт train_model.py; если его нет, берутся значения по умолчанию.
DEFAULT_REFERENCE = {
    "высота_м": 754.5,
    "расст_до_дороги_м": 40.6,
    "плотность_застройки": 84.5,
    "уклон_%": 3.0,
    "расст_до_воды_м": 400.0,
}


def load_reference():
    reference = DEFAULT_REFERENCE.copy()
    if os.path.exists("reference.json"):
        with open("reference.json", encoding="utf-8") as f:
            reference.update(json.load(f))
    return reference


REFERENCE = load_reference()

st.set_page_config(page_title="GroundScope AI", page_icon="🏗️")
st.title("🏗️ GroundScope AI: оценка участка")
st.write("Кликните по карте Алматы, чтобы получить предварительную оценку пригодности участка под застройку.")

with st.sidebar:
    st.header("О проекте")
    st.write("Предварительная оценка пригодности участков Алматы под застройку на открытых данных.")
    st.caption(
        "Слои карты (схема и спутник) переключаются в правом верхнем углу карты. "
        "Красный круг показывает радиус 300 м, в котором считается плотность застройки."
    )


@st.cache_resource
def load_resources():
    model = joblib.load("site_model.pkl")
    boundary_wgs = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    boundary_proj = boundary_wgs.to_crs(CRS)
    roads = gpd.read_file("data/roads.gpkg")
    buildings = gpd.read_file("data/buildings.gpkg")
    waterways = gpd.read_file("data/waterways.gpkg")
    return model, boundary_wgs, boundary_proj.geometry.iloc[0], roads, buildings, waterways


try:
    model, boundary_wgs, city_polygon, roads, buildings, waterways = load_resources()
except Exception:
    st.error("Не найдены данные города. Сначала запустите в терминале: python prepare_data.py")
    st.stop()


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


def get_elevation_and_slope(lat, lon):
    """Одним запросом получает высоту точки и четырёх соседних, считает уклон."""
    points = [(lat, lon)] + neighbor_points(lat, lon)
    lat_part = ",".join(f"{p[0]:.6f}" for p in points)
    lon_part = ",".join(f"{p[1]:.6f}" for p in points)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_part}&longitude={lon_part}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    z_center, z_north, z_south, z_east, z_west = response.json()["elevation"]
    return z_center, slope_percent(z_north, z_south, z_east, z_west)


def explain(model, features):
    """Насколько каждый признак меняет оценку по сравнению с типичным значением."""
    base = float(model.predict(features)[0])
    effects = {}
    for col in features.columns:
        typical = features.copy()
        typical[col] = REFERENCE[col]
        effects[LABELS[col]] = base - float(model.predict(typical)[0])
    return effects


# --- Карта ---
if "point" not in st.session_state:
    st.session_state.point = (43.2220, 76.8512)

city_map = folium.Map(location=[43.25, 76.92], zoom_start=11)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Спутник",
).add_to(city_map)
folium.GeoJson(
    boundary_wgs,
    name="Граница города",
    style_function=lambda _: {"fillOpacity": 0.05, "weight": 2},
).add_to(city_map)
folium.Circle(
    st.session_state.point, radius=300, color="#e4572e", fill=False, tooltip="Радиус 300 м"
).add_to(city_map)
folium.Marker(st.session_state.point, tooltip="Выбранный участок").add_to(city_map)
folium.LayerControl().add_to(city_map)

map_data = st_folium(
    city_map,
    height=450,
    use_container_width=True,
    returned_objects=["last_clicked"],
)

clicked = map_data.get("last_clicked") if map_data else None
if clicked:
    new_point = (clicked["lat"], clicked["lng"])
    if new_point != st.session_state.point:
        st.session_state.point = new_point
        st.rerun()

lat, lon = st.session_state.point
st.caption(f"Выбранная точка: {lat:.5f}, {lon:.5f}")

# --- Анализ выбранной точки ---
point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(CRS).iloc[0]

if not city_polygon.contains(point):
    st.error("Эта точка находится за пределами Алматы. Кликните внутри границы города.")
    st.stop()

with st.spinner("Анализируем участок..."):
    try:
        elevation, slope = get_elevation_and_slope(lat, lon)
    except Exception:
        st.error("Не удалось получить высоту. Проверьте интернет и кликните по карте ещё раз.")
        st.stop()

    road_dist = float(roads.distance(point).min())
    water_dist = float(waterways.distance(point).min())
    nearby = buildings.sindex.query(point.buffer(300), predicate="intersects")
    density = int(len(nearby))

    features = pd.DataFrame(
        [[elevation, road_dist, density, slope, water_dist]],
        columns=FEATURES,
    )
    score = float(model.predict(features)[0])
    effects = explain(model, features)

st.metric("Предварительный score пригодности", f"{score:.1f} / 100")

row1 = st.columns(3)
row1[0].metric("Высота", f"{elevation:.0f} м")
row1[1].metric("До дороги", f"{road_dist:.0f} м")
row1[2].metric("Зданий в 300 м", density)
row2 = st.columns(3)
row2[0].metric("Уклон", f"{slope:.1f} %")
row2[1].metric("До водотока", f"{water_dist:.0f} м")

warnings = []
if elevation > MAX_TRAIN_ELEVATION:
    warnings.append(
        "Высота выше диапазона, на котором обучена модель (до 1300 м). "
        "Оценка может быть ненадёжной."
    )
if slope > SLOPE_LIMIT:
    warnings.append(
        f"Уклон больше {SLOPE_LIMIT}%: склоны круче этого значения относятся "
        "к неблагоприятным условиям застройки в Алматы."
    )
if water_dist < WATER_NEAR_M:
    warnings.append(
        f"Участок ближе {WATER_NEAR_M} м к реке, ручью или каналу: возможен риск подтопления "
        "или селевого воздействия. Это приближённый признак по данным OpenStreetMap, "
        "водоохранные зоны нужно проверять отдельно."
    )

for text in warnings:
    st.warning(text)
if not warnings:
    if score > 80:
        st.success("Участок хорошо подходит под застройку по этим признакам.")
    elif score > 50:
        st.warning("Участок требует дополнительной оценки.")
    else:
        st.error("У участка есть серьёзные ограничения по этим признакам.")

def suggest_uses(road_dist, density, slope, water_dist, elevation):
    """Ориентировочные правила (пороги условные): какое использование участка выглядит разумным."""
    blocked = []
    if slope > SLOPE_LIMIT:
        blocked.append("крутой склон")
    if water_dist < WATER_NEAR_M:
        blocked.append("близость к водотоку")
    if elevation > MAX_TRAIN_ELEVATION:
        blocked.append("большая высота")
    if blocked:
        why = "мешает: " + ", ".join(blocked)
        return [(use, "не рекомендуется без специальных решений", why)
                for use in ("Жильё", "Коммерция", "Склад, логистика")]

    results = []
    if road_dist < 200 and 30 <= density <= 250:
        results.append(("Жильё", "подходит", "есть подъезд и соседняя застройка"))
    else:
        results.append(("Жильё", "возможно", "проверьте подъезд и инфраструктуру рядом"))
    if road_dist < 100 and density >= 80:
        results.append(("Коммерция", "подходит", "рядом дорога и плотная застройка"))
    else:
        results.append(("Коммерция", "слабо подходит", "нужны близкая дорога и плотное окружение"))
    if road_dist < 150 and density < 60 and slope < 5:
        results.append(("Склад, логистика", "подходит", "ровный участок у дороги на окраине"))
    else:
        results.append(("Склад, логистика", "возможно", "лучше ровный участок у дороги с редкой застройкой"))
    return results


tab_why, tab_use, tab_signal = st.tabs(
    ["Почему такая оценка", "Что можно построить", "Сигнал технике (демо)"]
)

with tab_why:
    chart = pd.DataFrame({"Вклад (баллы)": effects})
    st.bar_chart(chart, horizontal=True, sort="-Вклад (баллы)")
    notable = [(name, value) for name, value in effects.items() if abs(value) >= 0.5]
    if notable:
        for name, value in sorted(notable, key=lambda item: item[1]):
            direction = "повышает" if value > 0 else "понижает"
            st.write(f"- **{name}** {direction} оценку на {abs(value):.1f} балла по сравнению с типичным участком")
    else:
        st.write("Все признаки близки к типичному участку, поэтому сильных отклонений нет.")

with tab_use:
    for use, verdict, reason in suggest_uses(road_dist, density, slope, water_dist, elevation):
        st.write(f"**{use}:** {verdict}. {reason}.")
    st.caption(
        "Это простые правила с условными порогами, а не обученная модель. Реальное назначение "
        "участка определяют генплан и зонирование, их данных в прототипе пока нет."
    )

with tab_signal:
    ready = score >= 80 and not warnings
    st.write(
        "Идея: если участок прошёл проверку, система может сообщить технике на площадке, что "
        "работы можно начинать. Здесь только демонстрация формата сообщения: реальная техника "
        "не подключена."
    )
    if ready:
        st.success("Статус: участок готов к застройке по этим признакам.")
    else:
        st.info("Статус: участок не прошёл проверку, сигнал не отправляется.")
    if st.button("Сформировать сигнал (демо)", disabled=not ready):
        st.json({
            "type": "site_approved",
            "demo": True,
            "site": {"lat": round(lat, 5), "lon": round(lon, 5)},
            "score": round(score, 1),
            "checks": {"slope_ok": slope <= SLOPE_LIMIT, "water_ok": water_dist >= WATER_NEAR_M},
        })
        st.caption(
            "Такое сообщение можно отправить на учебного робота по Wi-Fi. К настоящей технике "
            "подключать систему нельзя без сертификации безопасности."
        )

st.caption(
    "Это предварительная оценка на открытых данных. Она не заменяет геологические "
    "изыскания и проверку градостроительных ограничений. Score пока рассчитан по "
    "условной формуле, а не по экспертным меткам, поэтому объяснение показывает "
    "логику модели, а не реальные свойства грунта. Уклон считается по цифровой модели "
    "рельефа с шагом около 90 м и даёт только приближённую оценку. Расстояние до водотока "
    "не заменяет карту селевой опасности."
)
