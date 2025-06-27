# app.py (Universal Business Simulator v3.0)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Universal Business Simulator",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        .stApp { background-color: #0f1116; }
        .stMetric { background-color: #1a1c24; border: 1px solid #2e3439; border-radius: 10px; padding: 20px; }
        .stMetric .st-ae { font-size: 1.1rem; color: #a1a1a1; }
        .stButton>button { border-radius: 10px; font-weight: bold; }
        .stExpander { border: 1px solid #2e3439 !important; border-radius: 10px !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
if 'variables' not in st.session_state:
    st.session_state.variables = []

# ==============================================================================
# 3. CORE SIMULATION ENGINE
# ==============================================================================
def run_monte_carlo(formula, variables, num_simulations):
    simulation_results = []
    for _ in range(num_simulations):
        # Create a dictionary for the current simulation's variable values
        local_scope = {}
        for var in variables:
            if var['dist'] == 'Normal':
                value = np.random.normal(var['param1'], var['param2'])
            elif var['dist'] == 'Uniform':
                value = np.random.uniform(var['param1'], var['param2'])
            elif var['dist'] == 'Constant':
                value = var['param1']
            local_scope[var['name']] = value
        
        # Safely evaluate the user's formula within the local_scope
        try:
            result = eval(formula, {"__builtins__": None}, local_scope)
            simulation_results.append(result)
        except Exception as e:
            st.error(f"Error evaluating formula: {e}. Please check variable names and formula syntax.")
            return None
            
    return np.array(simulation_results)

# ==============================================================================
# 4. UI LAYOUT
# ==============================================================================
st.title("🔮 Universal Business Simulator")
st.markdown("A platform to model, simulate, and understand any business scenario under uncertainty.")

# --- Sidebar for Model Building ---
with st.sidebar:
    st.image("https://i.imgur.com/vVw2G71.png", width=100)
    st.header("Model Builder")
    st.markdown("Define your inputs, model, and simulation settings.")
    st.divider()

    # --- Step 1: Define Input Variables ---
    with st.expander("Step 1: Define Input Variables", expanded=True):
        for i, var in enumerate(st.session_state.variables):
            c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
            var['name'] = c1.text_input("Variable Name", var['name'], key=f"name_{i}").replace(" ", "_")
            var['dist'] = c2.selectbox("Distribution", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
            
            if var['dist'] == "Normal":
                p1, p2 = st.columns(2)
                var['param1'] = p1.number_input("Mean (μ)", value=var['param1'], key=f"p1_{i}", format="%.2f")
                var['param2'] = p2.number_input("Std Dev (σ)", value=var['param2'], key=f"p2_{i}", format="%.2f")
            elif var['dist'] == "Uniform":
                p1, p2 = st.columns(2)
                var['param1'] = p1.number_input("Min", value=var['param1'], key=f"p1_{i}", format="%.2f")
                var['param2'] = p2.number_input("Max", value=var['param2'], key=f"p2_{i}", format="%.2f")
            else: # Constant
                var['param1'] = st.number_input("Value", value=var['param1'], key=f"p1_{i}", format="%.2f")
                var['param2'] = 0 # Not used

            if c3.button("❌", key=f"del_{i}", use_container_width=True):
                st.session_state.variables.pop(i)
                st.rerun()
            st.markdown("---")

        if st.button("Add Variable", use_container_width=True):
            st.session_state.variables.append({'name': f'Var_{len(st.session_state.variables)+1}', 'dist': 'Normal', 'param1': 100, 'param2': 10})
            st.rerun()

    # --- Step 2: Define the Model Formula ---
    with st.expander("Step 2: Define Your Model Formula", expanded=True):
        st.info("Use the variable names defined above. Example: `(Revenue - Costs) * Market_Share`")
        formula = st.text_area("Formula", "(Var_1 - Var_2) * 0.5", label_visibility="collapsed")
    
    # --- Step 3: Simulation Settings ---
    with st.expander("Step 3: Configure Simulation"):
        num_simulations = st.select_slider("Number of Simulations", [1000, 5000, 10000, 20000], value=10000)

    st.divider()
    run_button = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# --- Main Panel for Results ---
if run_button:
    if not st.session_state.variables:
        st.warning("Please add at least one input variable to run a simulation.")
    elif not formula:
        st.warning("Please enter a model formula.")
    else:
        results = run_monte_carlo(formula, st.session_state.variables, num_simulations)
        
        if results is not None:
            st.header("Simulation Results", anchor=False)
            
            # KPI Metrics
            mean_val, std_val = results.mean(), results.std()
            p5, p95 = np.percentile(results, 5), np.percentile(results, 95)
            
            col1, col2, col3 = st.columns(3, gap="large")
            col1.metric("Average Outcome", f"{mean_val:,.2f}", help="The mean of all simulation runs.")
            col2.metric("Std. Deviation (Risk)", f"{std_val:,.2f}", help="Measures the volatility or risk of the outcome. Higher is riskier.")
            col3.metric("90% Confidence Interval", f"{p5:,.2f} to {p95:,.2f}", help="We are 90% confident the outcome will fall within this range.")

            st.divider()

            # Visualization
            fig, ax = plt.subplots()
            sns.histplot(results, kde=True, ax=ax, color='#7E57C2', bins=50)
            ax.axvline(mean_val, color='#FDD835', linestyle='--', lw=2, label=f"Mean: {mean_val:,.2f}")
            ax.set_title("Distribution of Potential Outcomes", fontsize=16)
            ax.set_xlabel("Outcome Value")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("View Raw Simulation Data"):
                st.dataframe(pd.DataFrame(results, columns=['Outcome']))
else:
    st.info("Build your model in the sidebar and click **▶ Run Simulation** to see the future.")
