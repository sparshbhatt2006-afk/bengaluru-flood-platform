import streamlit as st
import pandas as pd
import cv2
import numpy as np
import folium
from streamlit_folium import st_folium
import os
import random

st.set_page_config(page_title="Bengaluru Flood Demo", layout="wide")
st.title("🌊 Bengaluru Urban Flood Risk & CV Platform")

PHOTOS_DIR = "data/photos/images"

if not os.path.exists(PHOTOS_DIR):
    st.error(f"Directory {PHOTOS_DIR} not found! Please check your folder structure.")
    st.stop()

# Get all images
image_files = [f for f in os.listdir(PHOTOS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

if not image_files:
    st.warning("No images found in `data/photos/images/`!")
    st.stop()

# Spread out all 401 images across realistic Bengaluru flood-prone hubs
bengaluru_hubs = [
    (12.9172, 77.6228),  # Silk Board
    (12.9352, 77.6245),  # Koramangala
    (12.9716, 77.5946),  # KR Circle / Central
    (13.0358, 77.5970),  # Hebbal
    (12.9784, 77.6408),  # Indiranagar
    (12.8452, 77.6602),  # Electronic City
]

random.seed(42)
data = []
for i, filename in enumerate(image_files):
    img_id = os.path.splitext(filename)[0]
    base_lat, base_lon = random.choice(bengaluru_hubs)
    lat_offset = random.uniform(-0.02, 0.02)
    lon_offset = random.uniform(-0.02, 0.02)

    risk_score = round(random.uniform(0.1, 0.95), 2)

    data.append({
        "id": img_id,
        "filename": filename,
        "lat": base_lat + lat_offset,
        "lon": base_lon + lon_offset,
        "predicted_risk": risk_score
    })

df = pd.DataFrame(data)


# OpenCV CV Water Checker
def check_flooding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False, 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_water = np.array([0, 0, 20])
    upper_water = np.array([180, 50, 160])
    mask = cv2.inRange(hsv, lower_water, upper_water)

    water_pixels = np.count_nonzero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    pct = round(float((water_pixels / total_pixels) * 100), 2)

    return pct > 8.0, pct


# Sidebar Selector
st.sidebar.header("📁 Browse Dataset (401 Images)")
selected_id = st.sidebar.selectbox("Choose Image ID", df["id"].tolist())
row = df[df["id"] == selected_id].iloc[0]

img_full_path = os.path.join(PHOTOS_DIR, row["filename"])
is_flooded, water_pct = check_flooding(img_full_path)

# Fusion Logic
pred_risk = row["predicted_risk"]
high_risk = pred_risk >= 0.6

if high_risk and is_flooded:
    status, color = "Confirmed Hotspot (Red)", "#E63946"
elif high_risk and not is_flooded:
    status, color = "Predicted, Not Observed (Orange)", "#F4A261"
elif not high_risk and is_flooded:
    status, color = "Unexpected Flooding / Model Gap (Purple)", "#9D4EDD"
else:
    status, color = "Clear (Green)", "#2A9D8F"

# Layout: Map on Left, Analysis Panel on Right
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("📍 Bengaluru Camera Observations Map")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="OpenStreetMap")

    # Plot clean scattered observation markers across the city
    for _, r in df.iterrows():
        is_selected = (r["id"] == selected_id)
        # Assign color based on mock risk or selection
        marker_color = color if is_selected else ("#E63946" if r["predicted_risk"] >= 0.6 else "#2A9D8F")
        marker_radius = 10 if is_selected else 5

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=marker_radius,
            color="white",
            weight=1,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.9,
            popup=f"ID: {r['id']} | Risk: {r['predicted_risk']}"
        ).add_to(m)

    st_folium(m, width="100%", height=550)

with col2:
    st.subheader("🔍 Analysis Panel")
    st.write(f"**Selected File:** {row['filename']}")
    st.write(f"**Fusion Status:** {status}")
    st.metric(label="CV Water Detected", value=f"{water_pct}%")
    st.metric(label="Model Risk Score", value=f"{pred_risk}")

    if os.path.exists(img_full_path):
        st.image(img_full_path, caption=f"Sample ({row['id']})")
    else:
        st.warning("Image not found.")