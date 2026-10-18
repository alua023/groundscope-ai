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

# "Типичный участок": медианы обучающей выборки (88 точек Алматы)
REFERENCE = {
    "высота_м": 754.5,
    "расст_до_дороги_м": 40.6,
    "плотность_застройки": 84.5,
}
LABELS = {
    "высота_м": "Высота",
    "расст_до_дороги_м": "Расстояние до дороги",
    "плотность_застройки": "Плотность застройки",
}

st.set_page_config(page_title="GroundScope AI", page_icon="🏗️")
st.title("🏗️ GroundScope AI: оценка участка")
st.write("Кликните по карте Алматы, чтобы получить предварительную оценку пригодности участка под застройку.")


@st.cache_resource
def load_resources():
    model = joblib.load("site_model.pkl")
    boundary_wgs = gpd.read_file("data/boundary.gpkg").to_crs(4326)
    boundary_proj = boundary_wgs.to_crs(CRS)
    roads = gpd.read_file("data/roads.gpkg")
    buildings = gpd.read_file("data/buildings.gpkg")
    return model, boundary_wgs, boundary_proj.geometry.iloc[0], roads, buildings


try:
    model, boundary_wgs, city_polygon, roads, buildings = load_resources()
except Exception:
    st.error("Не найдены данные города. Сначала запустите в терминале: python prepare_data.py")
    st.stop()


def get_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["elevation"][0]


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
folium.GeoJson(
    boundary_wgs,
    style_function=lambda _: {"fillOpacity": 0.05, "weight": 2},
).add_to(city_map)
folium.Marker(st.session_state.point, tooltip="Выбранный участок").add_to(city_map)

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
        elevation = get_elevation(lat, lon)
    except Exception:
        st.error("Не удалось получить высоту. Проверьте интернет и кликните по карте ещё раз.")
        st.stop()

    road_dist = float(roads.distance(point).min())
    nearby = buildings.sindex.query(point.buffer(300), predicate="intersects")
    density = int(len(nearby))

    features = pd.DataFrame(
        [[elevation, road_dist, density]],
        columns=["высота_м", "расст_до_дороги_м", "плотность_застройки"],
    )
    score = float(model.predict(features)[0])
    effects = explain(model, features)

st.metric("Предварительный score пригодности", f"{score:.1f} / 100")

col1, col2, col3 = st.columns(3)
col1.metric("Высота", f"{elevation:.0f} м")
col2.metric("До ближайшей дороги", f"{road_dist:.0f} м")
col3.metric("Зданий в радиусе 300 м", density)

if elevation > MAX_TRAIN_ELEVATION:
    st.warning(
        "Высота выше диапазона, на котором обучена модель (до 1300 м). "
        "Оценка может быть ненадёжной."
    )
elif score > 80:
    st.success("Участок хорошо подходит под застройку по этим признакам.")
elif score > 50:
    st.warning("Участок требует дополнительной оценки.")
else:
    st.error("У участка есть серьёзные ограничения по этим признакам.")

st.subheader("Почему такая оценка")
st.bar_chart(pd.DataFrame({"Вклад в оценку (баллы)": effects}))

notable = [(name, value) for name, value in effects.items() if abs(value) >= 0.5]
if notable:
    for name, value in sorted(notable, key=lambda item: item[1]):
        direction = "повышает" if value > 0 else "понижает"
        st.write(f"- **{name}** {direction} оценку на {abs(value):.1f} балла по сравнению с типичным участком")
else:
    st.write("Все признаки близки к типичному участку, поэтому сильных отклонений нет.")

st.caption(
    "Это предварительная оценка на открытых данных. Она не заменяет геологические "
    "изыскания и проверку градостроительных ограничений. Score пока рассчитан по "
    "условной формуле, а не по экспертным меткам, поэтому объяснение показывает "
    "логику модели, а не реальные свойства грунта."
)
