# app.py (Strategic Simulation Platform v6.0 - Step-by-Step Model Builder)

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
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        .stApp { background-color: #0d1117; }
        [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
        .metric-card { background-color: #161b22; border: 1px solid #30363d; padding: 25px; border-radius: 10px; text-align: center; }
        .metric-card h5 { margin-bottom: 15px; color: #c9d1d9; font-size: 1.1rem; font-weight: 500; }
        .metric-card h5 .bi { margin-right: 10px; font-size: 1.5rem; vertical-align: middle; }
        .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #30363d; }
        .stButton>button[kind="primary"] { border: 1px solid #238636; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE & MODEL BUILDING LOGIC
# ==============================================================================
if 'mode' not in st.session_state: st.session_state.mode = "Beginner"
if 'variables' not in st.session_state: st.session_state.variables = []
if 'model_steps' not in st.session_state: st.session_state.model_steps = []
if 'start_variable' not in st.session_state: st.session_state.start_variable = None

def build_formula_from_steps():
    if not st.session_state.start_variable: return ""
    formula = st.session_state.start_variable
    for step in st.session_state.model_steps:
        op_map = {"Add": "+", "Subtract": "-", "Multiply by": "*", "Divide by": "/"}
        operator = op_map.get(step['op'])
        value = step['val'] if step['type'] == 'Variable' else str(step['val'])
        formula = f"({formula}) {operator} {value}"
    return formula

# ==============================================================================
# 3. CORE SIMULATION ENGINE (Unchanged)
# ==============================================================================
def run_monte_carlo(formula, variables, num_simulations):
    # This robust engine remains the same
    simulation_results = []
    for _ in range(num_simulations):
        local_scope = {var['name']: (np.random.normal(var['param1'], var['param2']) if var['dist'] == 'Normal' else
                                     np.random.uniform(var['param1'], var['param2']) if var['dist'] == 'Uniform' else
                                     var['param1']) for var in variables}
        try: result = eval(formula, {"__builtins__": None, "np": np}, local_scope); simulation_results.append(result)
        except Exception as e: st.error(f"Error evaluating formula: {e}", icon="🚨"); return None
    return np.array(simulation_results)

# ==============================================================================
# 4. UI LAYOUT & COMPONENTS
# ==============================================================================
st.markdown('<h1 style="text-align: center;"><i class="bi bi-bar-chart-line-fill"></i> Strategic Simulation Platform</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://i.imgur.com/vVw2G71.png", width=100)
    st.markdown("<h3><i class='bi bi-tools'></i> Control Panel</h3>", unsafe_allow_html=True)
    st.radio("Select Analysis Mode", ["Beginner", "Advanced"], key="mode", horizontal=True)
    st.divider()

    # --- BEGINNER MODE UI ---
    if st.session_state.mode == "Beginner":
        st.markdown("<h5><i class='bi bi-truck'></i> Supply Chain Risk Model</h5>", unsafe_allow_html=True)
        st.info("Adjust these high-level drivers to model your scenario.", icon="💡")
        base_cost = st.number_input("Average Cost per Chip ($)", 1.0, 500.0, 25.0, 1.0)
        stability = st.select_slider("Geopolitical Stability", ["Stable", "Uncertain", "High Risk", "Crisis"], "Stable")
        concentration = st.select_slider("Supplier Concentration", ["Diversified", "Consolidated", "Single Source"], "Single Source")
        
        stability_map = {"Stable": 10, "Uncertain": 40, "High Risk": 75, "Crisis": 100}
        concentration_map = {"Diversified": 10, "Consolidated": 60, "Single Source": 90}
        
        sim_vars = [
            {'name': 'Base_Unit_Cost', 'dist': 'Normal', 'param1': base_cost, 'param2': 0.5 + (concentration_map[concentration] / 100) * 4.0},
            {'name': 'Tariff_Rate', 'dist': 'Uniform', 'param1': 0.0, 'param2': (stability_map[stability] / 100) * 0.30},
            {'name': 'Disruption_Premium', 'dist': 'Uniform', 'param1': 0.0, 'param2': 5.0 + (stability_map[stability] / 100) * 75.0}
        ]
        sim_formula = "(Base_Unit_Cost * (1 + Tariff_Rate)) + Disruption_Premium"
        
    # --- ADVANCED MODE UI ---
    else:
        st.markdown('<h5><i class="bi bi-sliders"></i> 1. Define Input Variables</h5>', unsafe_allow_html=True)
        for i, var in enumerate(st.session_state.variables):
            # (UI for variable definition is unchanged)
            with st.container():
                c1,c2 = st.columns([0.85, 0.15]);
                with c1:
                    var['name'] = st.text_input("Variable Name", var['name'], key=f"name_{i}").replace(" ", "_"); var['dist'] = st.selectbox("Distribution", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
                    if var['dist'] == "Normal": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Mean (μ)", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Std Dev (σ)", value=var['param2'], key=f"p2_{i}", min_value=0.0)
                    elif var['dist'] == "Uniform": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Min", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Max", value=var['param2'], key=f"p2_{i}")
                    else: var['param1'] = st.number_input("Value", value=var['param1'], key=f"p1_{i}"); var['param2'] = 0
                with c2: st.write(""); st.write("");
                    if st.button("🗑️", key=f"del_{i}", help="Remove", use_container_width=True): st.session_state.variables.pop(i); st.rerun()
                st.markdown("<hr style='margin:10px 0; border-color: #30363d;'>", unsafe_allow_html=True)
        if st.button("Add Variable", use_container_width=True): st.session_state.variables.append({'name': f'Var_{len(st.session_state.variables)+1}', 'dist': 'Normal', 'param1': 100.0, 'param2': 10.0}); st.rerun()
        
        st.divider()
        st.markdown('<h5><i class="bi bi-diagram-3-fill"></i> 2. Build Your Model Step-by-Step</h5>', unsafe_allow_html=True)
        var_names = [v['name'] for v in st.session_state.variables]
        st.session_state.start_variable = st.selectbox("Start calculation with variable:", var_names, index=var_names.index(st.session_state.start_variable) if st.session_state.start_variable in var_names else 0)
        
        for i, step in enumerate(st.session_state.model_steps):
            op_col, type_col, val_col, del_col = st.columns([0.3, 0.25, 0.3, 0.15])
            step['op'] = op_col.selectbox("Then,", ["Add", "Subtract", "Multiply by", "Divide by"], key=f"op_{i}", label_visibility="collapsed")
            step['type'] = type_col.radio("With:", ["Variable", "Constant"], key=f"type_{i}", horizontal=True, label_visibility="collapsed")
            if step['type'] == 'Variable':
                step['val'] = val_col.selectbox("Select variable", var_names, key=f"val_var_{i}", label_visibility="collapsed")
            else:
                step['val'] = val_col.number_input("Enter value", value=step.get('val', 1.0), key=f"val_num_{i}", label_visibility="collapsed")
            if del_col.button("🗑️", key=f"del_step_{i}", use_container_width=True): st.session_state.model_steps.pop(i); st.rerun()

        if st.button("Add Step", use_container_width=True): st.session_state.model_steps.append({'op': 'Add', 'type': 'Constant', 'val': 10.0}); st.rerun()

        sim_formula = build_formula_from_steps()
        st.success(f"Generated Formula: `{sim_formula}`", icon="✅") if sim_formula else st.warning("Your model is not yet complete.", icon="⚠️")
        sim_vars = st.session_state.variables

    st.divider()
    st.markdown('<h5><i class="bi bi-gear-fill"></i> Simulation Settings</h5>', unsafe_allow_html=True)
    num_simulations = st.select_slider("Simulation Runs", [1000, 10000, 20000, 50000], value=10000)
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

# --- Main Panel for Results ---
if run_button:
    if not sim_vars or not sim_formula: st.warning("Please configure your model in the sidebar before running.", icon="⚠️")
    else:
        results = run_monte_carlo(sim_formula, sim_vars, num_simulations)
        if results is not None:
            # The entire results dashboard remains unchanged as it is robust and professional
            st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Simulation Dashboard</h3>", unsafe_allow_html=True)
            mean_val, std_val = results.mean(), results.std(); p5, p95 = np.percentile(results, 5), np.percentile(results, 95)
            col1, col2, col3 = st.columns(3, gap="large")
            with col1: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-bullseye"></i>Average Outcome</h5><h2>{mean_val:,.2f}</h2></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-lightning-charge-fill"></i>Risk (Std. Dev)</h5><h2>{std_val:,.2f}</h2></div>', unsafe_allow_html=True)
            with col3: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-arrows-left-right"></i>90% Confidence Range</h5><h4>{p5:,.2f} — {p95:,.2f}</h4></div>', unsafe_allow_html=True)
            st.divider()
            st.markdown("<h4><i class='bi bi-distribute-vertical'></i> Distribution of Potential Outcomes</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(); sns.histplot(results, kde=True, ax=ax, color='#7E57C2', bins=50)
            ax.axvline(mean_val, color='#FDD835', linestyle='--', lw=2, label=f"Mean: {mean_val:,.2f}")
            ax.set_title("Distribution of Potential Outcomes", fontsize=16, color='white', pad=20); ax.set_xlabel("Outcome Value", color='white'); ax.set_ylabel("Frequency", color='white')
            ax.tick_params(colors='white'); ax.spines['left'].set_color('white'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#30363d')
            st.pyplot(fig)
else:
    st.markdown("<div style='text-align: center; padding-top: 50px;'><h3 style='color: #8b949e;'>Configure your scenario in the sidebar and click 'Run Simulation' to begin.</h3></div>", unsafe_allow_html=True)
