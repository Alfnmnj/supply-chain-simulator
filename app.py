# app.py (Universal Business Simulator v3.1 - Premium UI/UX)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & AESTHETIC STYLING
# ==============================================================================
st.set_page_config(
    page_title="Strategic Simulation Platform",
    layout="wide"
)

# --- Inject custom CSS for a premium, modern UI ---
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        /* --- Base & General --- */
        .stApp {
            background-color: #0d1117; 
        }
        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        /* --- Metric Card Styling --- */
        .metric-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-card h5 { /* Icon and Title */
            margin-bottom: 15px;
            color: #c9d1d9;
            font-size: 1.1rem;
            font-weight: 500;
        }
        .metric-card h5 .bi { /* Icon specific */
            margin-right: 10px;
            font-size: 1.5rem;
            vertical-align: middle;
        }
        /* --- Button Styling --- */
        .stButton>button {
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #30363d;
        }
        .stButton>button[kind="primary"] {
            border: 1px solid #238636;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
if 'variables' not in st.session_state:
    st.session_state.variables = []

# ==============================================================================
# 3. CORE SIMULATION ENGINE (Unchanged - Pure Logic)
# ==============================================================================
def run_monte_carlo(formula, variables, num_simulations):
    simulation_results = []
    for _ in range(num_simulations):
        local_scope = {var['name']: (np.random.normal(var['param1'], var['param2']) if var['dist'] == 'Normal' else
                                     np.random.uniform(var['param1'], var['param2']) if var['dist'] == 'Uniform' else
                                     var['param1']) for var in variables}
        try:
            result = eval(formula, {"__builtins__": None, "np": np}, local_scope)
            simulation_results.append(result)
        except Exception as e:
            st.error(f"Error evaluating formula: {e}. Check syntax and variable names.", icon="🚨")
            return None
    return np.array(simulation_results)

# ==============================================================================
# 4. UI LAYOUT & COMPONENTS
# ==============================================================================
st.markdown('<h1 style="text-align: center;"><i class="bi bi-bar-chart-line-fill"></i> Strategic Simulation Platform</h1>', unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #8b949e;'>Model any business scenario under uncertainty to forecast outcomes and quantify risk.</h5>", unsafe_allow_html=True)
st.divider()

# --- Sidebar for Model Building ---
with st.sidebar:
    st.markdown("<h3><i class='bi bi-tools'></i> Model Builder</h3>", unsafe_allow_html=True)
    st.markdown("Define your inputs, model, and simulation settings.")
    st.divider()

    st.markdown('<h5><i class="bi bi-sliders"></i> 1. Define Input Variables</h5>', unsafe_allow_html=True)
    for i, var in enumerate(st.session_state.variables):
        with st.container():
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                var['name'] = st.text_input("Variable Name", var['name'], key=f"name_{i}", placeholder="e.g., Revenue_Growth").replace(" ", "_")
                var['dist'] = st.selectbox("Distribution", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
                if var['dist'] == "Normal": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Mean (μ)", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Std Dev (σ)", value=var['param2'], key=f"p2_{i}", min_value=0.0)
                elif var['dist'] == "Uniform": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Min", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Max", value=var['param2'], key=f"p2_{i}")
                else: var['param1'] = st.number_input("Value", value=var['param1'], key=f"p1_{i}"); var['param2'] = 0
            with c2:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button("🗑️", key=f"del_{i}", help="Remove this variable", use_container_width=True):
                    st.session_state.variables.pop(i); st.rerun()
            st.markdown("<hr style='margin:10px 0; border-color: #30363d;'>", unsafe_allow_html=True)

    if st.button("Add Variable", use_container_width=True):
        st.session_state.variables.append({'name': f'Variable_{len(st.session_state.variables)+1}', 'dist': 'Normal', 'param1': 100.0, 'param2': 10.0})
        st.rerun()
    
    st.divider()
    st.markdown('<h5><i class="bi bi-calculator-fill"></i> 2. Define Model Formula</h5>', unsafe_allow_html=True)
    st.info("Use variable names from above. Example: `(Revenue - Costs) * 0.2`", icon="ℹ️")
    formula = st.text_area("Formula", "(Variable_1 - 50) * 0.75", label_visibility="collapsed")
    
    st.divider()
    st.markdown('<h5><i class="bi bi-gear-fill"></i> 3. Configure Simulation</h5>', unsafe_allow_html=True)
    num_simulations = st.select_slider("Number of Simulations", [1000, 5000, 10000, 20000, 50000], value=10000)
    
    st.divider()
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

# --- Main Panel for Results ---
if run_button:
    if not st.session_state.variables: st.warning("Please add at least one input variable to run a simulation.", icon="⚠️")
    elif not formula: st.warning("Please enter a model formula to simulate.", icon="⚠️")
    else:
        results = run_monte_carlo(formula, st.session_state.variables, num_simulations)
        
        if results is not None:
            st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Simulation Dashboard</h3>", unsafe_allow_html=True)
            
            # KPI Metric Cards
            mean_val, std_val = results.mean(), results.std()
            p5, p95 = np.percentile(results, 5), np.percentile(results, 95)
            
            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                st.markdown(f'<div class="metric-card"><h5><i class="bi bi-bullseye"></i>Average Outcome</h5><h2>{mean_val:,.2f}</h2></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h5><i class="bi bi-lightning-charge-fill"></i>Risk (Std. Dev)</h5><h2>{std_val:,.2f}</h2></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h5><i class="bi bi-arrows-left-right"></i>90% Confidence Range</h5><h4>{p5:,.2f} — {p95:,.2f}</h4></div>', unsafe_allow_html=True)

            st.divider()

            # Visualization
            st.markdown("<h4><i class='bi bi-distribute-vertical'></i> Distribution of Potential Outcomes</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.histplot(results, kde=True, ax=ax, color='#7E57C2', bins=50)
            ax.axvline(mean_val, color='#FDD835', linestyle='--', lw=2, label=f"Mean: {mean_val:,.2f}")
            ax.set_title("Distribution of Potential Outcomes", fontsize=16, color='white', pad=20)
            ax.set_xlabel("Outcome Value", color='white'); ax.set_ylabel("Frequency", color='white')
            ax.tick_params(colors='white'); ax.spines['left'].set_color('white'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#30363d')
            st.pyplot(fig)

            with st.expander("View Raw Simulation Data"):
                st.dataframe(pd.DataFrame(results, columns=['Outcome']))
else:
    st.markdown("<div style='text-align: center; padding-top: 50px;'><h3 style='color: #8b949e;'>Build your model in the sidebar and click 'Run Simulation' to forecast the future.</h3></div>", unsafe_allow_html=True)
