import streamlit as st
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

import time
from piml_model import PhysicsInformedModel
from bohrium_connector import BohriumMDConnector

# --- Page Config ---
st.set_page_config(
    page_title="PIML Nanocarrier Uptake",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Medical-Tech Theme
st.markdown("""
    <style>
        .stApp {
            background-color: #f8fbfa;
            color: #2c3e50;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
            border-top: 4px solid #1abc9c;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stSlider > div > div > div > div {
            background-color: #1abc9c !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🧬 Physics-Informed Nanocarrier Uptake Simulator")
st.markdown("Predicts curcumin nanocarrier cellular interactions using XGBoost embedded with DLVO colloidal constraints.")

# --- Initialization ---
@st.cache_resource
def load_model():
    with st.spinner("Training Physics-Informed XGBoost Model on Synthetic DLVO bounds..."):
        return PhysicsInformedModel()

model = load_model()

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Formulation Parameters")
    
    selected_size = st.slider("Particle Size (nm)", min_value=50, max_value=500, value=200, step=5,
                              help="Optimal zone is typically 180-220 nm for endocytosis.")
    selected_zeta = st.slider("Zeta Potential (mV)", min_value=-100, max_value=50, value=-35, step=1,
                              help="Negative charge favors stability (-30 to -40). Low magnitude promotes clumping. Positive is often cytotoxic.")
    selected_pdi = st.slider("Polydispersity Index (PDI)", min_value=0.05, max_value=0.50, value=0.10, step=0.01)
    selected_coating = st.slider("Chitosan / Alginate Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                 help="1.0 = Pure Chitosan (Positive leaning), 0.0 = Pure Alginate (Negative leaning).")
    
    st.markdown("---")
    sim_btn = st.button("Simulate Dynamics 🚀", use_container_width=True, type="primary")

# --- Core Logic execution ---
safety_score, uptake_score, shap_data = model.predict(selected_size, selected_zeta, selected_pdi, selected_coating)

# Determine the physical state
if selected_zeta > 0:
    sim_state = "repel"
elif selected_zeta > -25:
    sim_state = "clumping"
elif selected_size < 180 or selected_size > 250:
    sim_state = "repel"
else:
    sim_state = "optimal"

# --- Main Layout ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h3>Safety Score</h3>
            <h1 style="color:{'#27ae60' if safety_score > 80 else '#e74c3c' if safety_score < 50 else '#f39c12'}">
                {safety_score:.1f}%
            </h1>
            <p>Predicted Cytotoxicity Avoidance</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h3>Uptake Efficiency</h3>
            <h1 style="color:{'#27ae60' if uptake_score > 75 else '#e74c3c' if uptake_score < 40 else '#f39c12'}">
                {uptake_score:.1f}%
            </h1>
            <p>Cellular Internalization</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    status_text = "Optimal Endocytosis" if sim_state == 'optimal' else "Severe Clumping Risk" if sim_state == 'clumping' else "Cytotoxic / High Repulsion"
    st.markdown(f"""
        <div class="metric-card">
            <h3>Colloidal State</h3>
            <h2 style="color:#2c3e50">{status_text}</h2>
            <p>Based on DLVO physical constraints</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_vis, col_data = st.columns([1.2, 1])

with col_vis:
    st.subheader("🔬 3D Interaction Visualizer (Three.js)")
    
    try:
        with open("threejs_sim.html", 'r') as f:
            html_content = f.read()
            html_content = html_content.replace("___SIM_STATE___", sim_state)
            # Injecting a timestamp makes the HTML string unique per rerun,
            # forcing Streamlit to refresh the iframe and replay the Three.js animation on every reading.
            html_content += f"\n<!-- REFRESH_TRIGGER: {time.time()} -->"
            components.html(html_content, height=450)
    except FileNotFoundError:
        st.error("Three.js simulation template not found.")

with col_data:
    st.subheader("📊 Model Explainability (SHAP)")
    st.caption("How each chemical parameter influenced the Safety Score.")
    
    # SHAP Waterfall via Plotly
    labels = ["Base Value"] + shap_data['features']
    measures = ["absolute"] + ["relative"] * len(shap_data['features'])
    values = [shap_data['base_value']] + list(shap_data['values'])
    
    fig = go.Figure(go.Waterfall(
        name = "SHAP",
        orientation = "v",
        measure = measures,
        x = labels,
        y = values,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color":"#2ecc71"}},
        decreasing = {"marker":{"color":"#e74c3c"}},
        totals = {"marker":{"color":"#34495e"}}
    ))
    
    fig.update_layout(
        title="Predictive Parameter Impact",
        waterfallgap=0.3,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Bohrium AI Grounding ---
st.subheader("🧬 Contextual Grounding (Bohrium AI MD Mock)")
with st.expander("View Molecular Dynamics Interaction Report", expanded=(sim_btn)):
    with st.spinner("Querying Molecular Dynamics data..."):
        report = BohriumMDConnector.get_md_analysis(selected_coating, selected_zeta)
        st.markdown(report)

import time

class BohriumMDConnector:
    """
    Mock connector simulating API calls to Bohrium AI for Molecular Dynamics
    (MD) data regarding Chitosan/Alginate coating interactions with lipid bilayers.
    """
    
    @staticmethod
    def get_md_analysis(chitosan_ratio, zeta_potential):
        """
        Returns mock molecular dynamics findings based on the coating formulation and zeta potential.
        """
        # Removed artificial latency to prevent UI freezing on every reading slider interaction
        pass

        
        # Determine the primary coating agent
        if chitosan_ratio > 0.6:
            coating = "Chitosan-dominant"
            charge_status = "highly protonated" if zeta_potential > 0 else "de-protonated (anomalous)"
        elif chitosan_ratio < 0.4:
            coating = "Alginate-dominant"
            charge_status = "carboxylate-rich"
        else:
            coating = "Balanced Chitosan-Alginate complex"
            charge_status = "polyelectrolyte complexed"
            
        # Generate scientific grounding text
        analysis = f"""
**Bohrium AI Molecular Dynamics Report**

Simulation Parameters:
- Formulation: {coating} 
- Charge Profile: {charge_status} ({zeta_potential} mV)

**Interaction with Lipid Bilayer (POPC/POPG model):**
"""
        
        if zeta_potential < -30:
            analysis += """
- The strong negative surface charge (-30 to -40 mV) creates ideal electrostatic repulsion between individual nanocarriers (DLVO theory).
- High structural stability in phosphate-buffered saline (PBS) environments.
- Steric hindrance from the polymer chains facilitates smooth energy-barrier crossing during endocytosis without membrane rupture.
"""
        elif zeta_potential > 10:
            analysis += """
- The positive charge promotes very strong electrostatic attraction to the negatively charged cell membrane.
- *Warning:* High risk of acute membrane depolarization and cytotoxicity. MD simulations show rapid pore formation and localized membrane damage.
"""
        else: # -30 to 10
            analysis += """
- Low magnitude charge limits colloidal stability.
- Van der Waals forces dominate over electrostatic repulsion.
- MD trajectories show high probability of nanocarrier aggregation (clumping) before reaching the lipid bilayer, significantly severely impeding cellular uptake.
"""
        
        return analysis
