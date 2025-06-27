# app.py (v4.1 - Definitive Version, No Emojis)

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
    page_icon="◈",  # Professional, non-emoji icon
    layout="wide"
)

# This is the core of the UI: Lucide icon library and custom CSS
st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        /* Base Styling */
        .stApp { background-color: #0a0a0a; color: #e6e6e6; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }

        /* Icon Styling */
        i[data-lucide] {
            width: 20px;
            height: 20px;
            stroke-width: 2px;
            vertical-align: middle;
            margin-right: 0.5rem;
            color: #a1a1a1;
        }
        h1 > i[data-lucide] { width: 32px; height: 32px; }

        /* Metric Card Styling */
        .stMetric {
            background-color: #1a1a1a;
            border: 1px solid #2c2c2c;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        }
        .stMetric .st-ae { font-size: 1.1rem; color: #a1a1a1; }
        
        /* Button Styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #4a4a4a;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            border-color: #8A2BE2; /* A premium accent color */
            color: #8A2BE2;
        }

        /* Expander Styling */
        .stExpander {
            border: 1px solid #2c2c2c !important;
            border-radius: 10px !important;
            background-color: #1c1c1c;
        }
        
        /* Sidebar Styling */
        .st-emotion-cache-16txtl3 { background-color: #121212; }
        
        /* Delete button styling */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            color: #ff4b4b;
            border-color: #ff4b4b;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            color: #ffffff;
            background-color: #ff4b4b;
            border-color: #ff4b4b;
        }

    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
if 'variables' not in st.session_state:
    st.session_state.variables = [
        {'name': 'Revenue', 'dist': 'Normal', 'param1': 5000000, 'param2': 500000},
        {'name': 'COGS_Percent', 'dist': 'Uniform', 'param1': 25, 'param2': 30},
        {'name': 'OpEx', 'dist': 'Constant', 'param1': 1200000, 'param2': 0}
    ]

# ==============================================================================
# 3. CORE SIMULATION ENGINE (Unchanged)
# ==============================================================================
def run_monte_carlo(formula, variables, num_simulations):
    simulation_results = []
    for _ in range(num_simulations):
        local_scope = {}
        for var in variables:
            if var['dist'] == 'Normal': value = np.random.normal(var['param1'], var['param2'])
            elif var['dist'] == 'Uniform': value = np.random.uniform(var['param1'], var['param2'])
            else: value = var['param1']
            local_scope[var['name']] = value
        try:
            result = eval(formula, {"__builtins__": None}, local_scope)
            simulation_results.append(result)
        except Exception as e:
            st.error(f"Error evaluating formula: {e}. Check variable names and syntax.")
            return None
    return np.array(simulation_results)

# ==============================================================================
# 4. UI LAYOUT
# ==============================================================================
st.markdown("<h1><i data-lucide='brain-circuit'></i> Universal Business Simulator</h1>", unsafe_allow_html=True)
st.markdown("A platform to model, simulate, and understand any business scenario under uncertainty.")

with st.sidebar:
    st.markdown("<h2><i data-lucide='sliders-horizontal'></i> Model Builder</h2>", unsafe_allow_html=True)
    st.markdown("Define your inputs, model, and simulation settings.")
    st.divider()

    with st.expander("Step 1: Define Input Variables", expanded=True):
        for i, var in enumerate(st.session_state.variables):
            st.markdown(f"**Variable: `{var['name']}`**")
            c1, c2 = st.columns([0.8, 0.2]); var['name'] = c1.text_input("Variable Name", var['name'], key=f"name_{i}", label_visibility="collapsed").replace(" ", "_")
            if c2.button("X", key=f"del_{i}", use_container_width=True, help="Remove this variable"): st.session_state.variables.pop(i); st.rerun()
            var['dist'] = st.selectbox("Distribution Type", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
            if var['dist'] == "Normal": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Mean (μ)", value=var['param1'], key=f"p1_{i}", format="%.2f"); var['param2'] = p2.number_input("Std Dev (σ)", value=var['param2'], key=f"p2_{i}", format="%.2f")
            elif var['dist'] == "Uniform": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Min", value=var['param1'], key=f"p1_{i}", format="%.2f"); var['param2'] = p2.number_input("Max", value=var['param2'], key=f"p2_{i}", format="%.2f")
            else: var['param1'] = st.number_input("Value", value=var['param1'], key=f"p1_{i}", format="%.2f"); var['param2'] = 0
            st.markdown("---")
        if st.button("Add New Variable", use_container_width=True): st.session_state.variables.append({'name': f'NewVar_{len(st.session_state.variables)+1}', 'dist': 'Normal', 'param1': 100, 'param2': 10}); st.rerun()

    with st.expander("Step 2: Define Your Model Formula", expanded=True):
        st.info("Use variable names from Step 1. Example: (Revenue * (1 - COGS_Percent/100)) - OpEx")
        formula = st.text_area("Formula", "(Revenue * (1 - COGS_Percent/100)) - OpEx", label_visibility="collapsed")
    
    with st.expander("Step 3: Configure Simulation"):
        num_simulations = st.select_slider("Simulations", [1000, 10000, 20000, 50000, 100000], value=20000)

    st.divider()
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

if run_button:
    if not st.session_state.variables or not formula: st.warning("Please define at least one variable and a formula.")
    else:
        with st.spinner("Running thousands of scenarios..."): results = run_monte_carlo(formula, st.session_state.variables, num_simulations)
        if results is not None:
            st.markdown("<h2><i data-lucide='bar-chart-3'></i> Simulation Results</h2>", unsafe_allow_html=True)
            
            mean_val, std_val = results.mean(), results.std(); p5, p95 = np.percentile(results, 5), np.percentile(results, 95); prob_positive = (results > 0).mean() * 100
            
            col1, col2, col3 = st.columns(3, gap="large")
            col1.metric("Average Outcome", f"{mean_val:,.2f}", help="The mean of all simulation runs.")
            col2.metric("Risk (Std. Deviation)", f"{std_val:,.2f}", help="Measures the volatility of the outcome. Higher is riskier.")
            col3.metric("Probability of Positive Outcome", f"{prob_positive:.1f}%", help="The percentage of simulation runs that resulted in a value greater than zero.")

            st.divider()

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(results, kde=True, ax=ax, color='#8A2BE2', bins=50)
            ax.axvline(mean_val, color='#FDD835', linestyle='--', lw=2, label=f"Mean: {mean_val:,.2f}")
            ax.axvline(p5, color='#e57373', linestyle=':', lw=2, label=f"5th Percentile: {p5:,.2f}")
            ax.axvline(p95, color='#e57373', linestyle=':', lw=2, label=f"95th Percentile: {p95:,.2f}")
            ax.set_title("Distribution of Potential Outcomes", fontsize=16); ax.set_xlabel("Outcome Value"); ax.set_ylabel("Frequency"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            with st.expander("View Full Statistical Summary & Raw Data"): st.dataframe(pd.Series(results).describe().to_frame('Statistics'))
else:
    st.markdown("""
        <div style="background-color: #1a1a1a; border-radius: 12px; padding: 2rem; text-align: center; border: 1px solid #2c2c2c;">
            <i data-lucide="play-circle" style="width: 48px; height: 48px; color: #a1a1a1;"></i>
            <h3 style="margin-top: 1rem;">Welcome to the Universal Simulator</h3>
            <p style="color: #a1a1a1;">Build your model using the <b>Control Panel</b> on the left, then click <b>Run Simulation</b> to see the results.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
