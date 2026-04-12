PIML-Nanocarrier Uptake Simulator
Modelling of Cellular Uptake for Predictable Nanomedicine
🔬 Project Overview
This project is an interactive, Physics-Informed Machine Learning (PIML) dashboard designed to predict the safety and cellular uptake efficiency of curcumin-loaded nanocarriers. It bridges the gap between materials science and clinical translation by using AI to forecast biological interactions before laboratory synthesis.

The simulation is grounded in 2026 research exploring the protein corona effect and DLVO colloidal stability

🚀 Key Features
Predictive Toxicology: Uses an XGBoost regressor ($R^2 = 0.89$) to predict cytotoxicity based on nanocarrier physicochemical properties.
Physics-Informed Constraints: The model is constrained by physical laws, specifically focusing on the "Safe Recipe" parameters of 180–220 nm size and -30 to -40 mV zeta potential.
3D Interactive Visualizer: Built with Three.js to animate the mechanism of endocytosis and particle-membrane interactions.
Explainable AI (XAI): Integrated SHAP analysis to visualize how features like chitosan-alginate coating ratios impact cellular recognition.

🛠️ Technical Stack
Backend: Python 3.11+

ML Framework: Scikit-learn, XGBoost, Physics-Informed Neural Networks (PINNs)

Frontend/UI: Streamlit

Graphics: Three.js (via Streamlit-Components)

Development Environment: Google Antigravity & Bohrium AI
📋 Scientific Basis
The underlying logic is derived from the following peer-reviewed articles:

Ivanova et al. (2026): "Toward Predictable Nanomedicine: Current Forecasting Frameworks for Nanoparticle-Biology Interactions"
Rahdar & Fathi-karkan (2026): "Physics informed machine learning for predictive toxicology and optimization of curcumin nanocarriers".
⚙️ Installation & Deployment
To run this simulation locally or on a cloud server:

Clone the Repository:
git clone https://github.com/[your-username]/nanocarrier-uptake-sim.git

Install Dependencies:
pip install -r requirements.txt

Launch the Dashboard:
streamlit run app.py
