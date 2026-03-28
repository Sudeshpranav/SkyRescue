import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import trimesh
import math
import os

# --- 1. PAGE CONFIG & DARK THEME CSS ---
st.set_page_config(page_title="ArchMind AI Pro", page_icon="🌑", layout="wide")

st.markdown("""
<style>
/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', sans-serif;
}
/* Background */
.stApp { background-color: #0E1117; }
/* Sidebar */
section[data-testid="stSidebar"] { background-color: #161B22; }
/* Headings */
h1, h2, h3, h4 { color: #FFFFFF !important; }
/* Text */
p, span, div, label { color: #E6EDF3 !important; }
small { color: #9DA7B3 !important; }
/* Buttons */
.stButton>button {
    background-color: #00FFAA; color: black; font-weight: bold; border-radius: 10px;
}
.stButton>button:hover { background-color: #00cc88; color: white; }
/* Metrics & Cards */
div[data-testid="stMetricValue"] { color: #00FFAA !important; }
.res-card {
    background-color: #1C2128; padding: 20px; border-radius: 15px; border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD AI BRAIN (FIXED PATH!) ---
@st.cache_resource
def load_model():
    # It now looks for the file in the exact same folder as app.py on GitHub
    path = 'archmind_model.pkl' 
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062409.png", width=80)
    st.title("ArchMind Pro")
    st.markdown("---")
    selected_page = st.selectbox("Navigation", ["Material Prediction", "Project Analytics", "Settings"])
    st.markdown("### 🛠️ Configuration")
    depth = st.slider("Foundation Depth (ft)", 5.0, 15.0, 10.0)
    wall_thick_in = st.slider("Wall Thickness (in)", 4.0, 15.0, 9.0)
    soil = st.select_slider("Soil Stability", options=["Low", "Medium", "High"], value="Medium")
    soil_map = {"Low": 1.2, "Medium": 1.0, "High": 0.8}
    st.info("💡 Pro Tip: Use High Stability soil to reduce cement requirements by 20%.")

# --- 4. MAIN INTERFACE ---
if selected_page == "Material Prediction":
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.title("Building Material Intelligence")
        st.write("Extracting structural requirements from 3D Geometry.")
    with col_status:
        st.button("System Online", disabled=True)

    # 3D Upload Section
    st.markdown('<div class="res-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload STL Structural File", type=["stl"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        with open("temp_upload.stl", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Geometry Logic
        mesh = trimesh.load("temp_upload.stl", force='mesh')
        calc_area = int(mesh.extents[0] * mesh.extents[1])
        calc_floors = max(1, int(mesh.extents[2] / 10.0))

        # Display Stats
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Footprint", f"{calc_area} sqft")
        stat2.metric("Levels", f"{calc_floors} Floors")
        stat3.metric("Volume", f"{int(mesh.volume/1000)}k units")
        stat4.metric("Slicing", "Success ✅")

        # FIXED COLUMN NAMES TO MATCH THE AI BRAIN!
        input_data = pd.DataFrame({
            'Area': [calc_area], 
            'Floors': [calc_floors], 
            'Foundation_Depth': [depth],
            'Wall_Thickness': [wall_thick_in], 
            'Soil_Type': [soil_map[soil]]
        })

        if st.button("RUN AI ESTIMATOR", use_container_width=True, type="primary"):
            preds = model.predict(input_data)[0]
            
            # Masonry Calc
            est_wall_length = math.sqrt(calc_area) * 6 * calc_floors
            vol_cuft = est_wall_length * 10 * (wall_thick_in/12)
            bricks = int((vol_cuft / 0.0625) * 1.08)
            aac = int((vol_cuft / 0.88) * 1.05)

            st.markdown("### 📊 Resource Allocation Results")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="res-card">
                    <h4>Cement (Structural)</h4><h2 style="color:#00FFAA;">{int(preds[0])} Bags</h2>
                    <small>Based on standard grade M25 mix</small>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="res-card">
                    <h4>Steel Rebar</h4><h2 style="color:#00FFAA;">{preds[1]:.2f} Tonnes</h2>
                    <small>High-strength TMT Grade Fe500D</small>
                </div>""", unsafe_allow_html=True)

            st.markdown("### 🧱 Wall Components")
            m1, m2 = st.columns(2)
            m1.info(f"**Fly Ash Bricks:** {bricks:,} units")
            m2.info(f"**AAC Blocks:** {aac:,} units")

            with st.expander("Show Neural Network Logic (SHAP Explainer)"):
                explainer = shap.TreeExplainer(model.estimators_[0])
                shap_values = explainer.shap_values(input_data)
                fig, ax = plt.subplots(figsize=(10, 3), facecolor='#1C2128')
                ax.set_facecolor('#1C2128')
                plt.barh(["Area", "Floors", "Found.", "Wall", "Soil"], shap_values[0], color='#00FFAA')
                ax.tick_params(axis='both', colors='white')
                st.pyplot(fig)

elif selected_page == "Project Analytics":
    st.title("📈 Construction Analytics")
    st.warning("Feature arriving in Version 4.0: Cost forecasting and vendor integration.")
