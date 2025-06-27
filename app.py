# app.py (Strategic Simulation Platform - v6.2 FINAL, Verified)

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
if 'mode' not in st.session_state: st.session_state.mode = "Guided"
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
# 3. CORE SIMULATION ENGINE
# ==============================================================================
@st.cache_data
def run_monte_carlo(formula, variables, num_simulations):
    results = {}
    for var in variables:
        if var['dist'] == 'Normal': results[var['name']] = np.random.normal(var['param1'], var['param2'], num_simulations)
        elif var['dist'] == 'Uniform': results[var['name']] = np.random.uniform(var['param1'], var['param2'], num_simulations)
        else: results[var['name']] = var['param1']
    
    try:
        final_outcome = pd.eval(formula, local_dict=results)
        return final_outcome
    except Exception as e: st.error(f"Error evaluating formula: {e}", icon="🚨"); return None

# ==============================================================================
# 4. UI LAYOUT & COMPONENTS
# ==============================================================================
st.markdown('<h1 style="text-align: center;"><i class="bi bi-shield-check"></i> Strategic Simulation Platform</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://i.imgur.com/vVw2G71.png", width=100)
    st.markdown("<h3><i class='bi bi-tools'></i> Control Panel</h3>", unsafe_allow_html=True)
    st.radio("Select Analysis Mode", ["Guided", "Expert"], key="mode", horizontal=True)
    st.divider()

    # --- GUIDED MODE UI ---
    if st.session_state.mode == "Guided":
        st.markdown("<h5><i class='bi bi-truck'></i> Supply Chain Risk Scenario</h5>", unsafe_allow_html=True)
        st.info("Model the impact of geopolitical risk on a critical component.", icon="💡")
        base_cost = st.number_input("Average Component Cost ($)", 1.0, 500.0, 25.0, 1.0)
        annual_volume = st.number_input("Annual Unit Volume", 10000, 10000000, 1000000)
        st.markdown("<h6>Define Disruption Scenario</h6>", unsafe_allow_html=True)
        geo_risk = st.select_slider("Geopolitical Risk", ["Low", "Medium", "High", "Crisis"], "Low")
        supplier_base = st.select_slider("Supplier Base", ["Diversified", "Consolidated", "Single Source"], "Single Source")

    # --- EXPERT MODE UI ---
    else:
        st.markdown('<h5><i class="bi bi-sliders"></i> 1. Define Input Variables</h5>', unsafe_allow_html=True)
        for i, var in enumerate(st.session_state.variables):
            with st.container():
                c1,c2 = st.columns([0.85, 0.15])
                with c1:
                    var['name'] = st.text_input("Variable Name", var['name'], key=f"name_{i}").replace(" ", "_"); var['dist'] = st.selectbox("Distribution", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
                    if var['dist'] == "Normal": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Mean (μ)", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Std Dev (σ)", value=var['param2'], key=f"p2_{i}", min_value=0.0)
                    elif var['dist'] == "Uniform": p1, p2 = st.columns(2); var['param1'] = p1.number_input("Min", value=var['param1'], key=f"p1_{i}"); var['param2'] = p2.number_input("Max", value=var['param2'], key=f"p2_{i}")
                    else: var['param1'] = st.number_input("Value", value=var['param1'], key=f"p1_{i}"); var['param2'] = 0
                with c2: 
                    st.write(""); st.write("")
                    # ** THE FIX IS HERE. This line's indentation is now correct. **
                    if st.button("🗑️", key=f"del_{i}", help="Remove", use_container_width=True):
                        st.session_state.variables.pop(i)
                        st.rerun()
                st.markdown("<hr style='margin:10px 0; border-color: #30363d;'>", unsafe_allow_html=True)
        if st.button("Add Variable", use_container_width=True): st.session_state.variables.append({'name': f'Var_{len(st.session_state.variables)+1}', 'dist': 'Normal', 'param1': 100.0, 'param2': 10.0}); st.rerun()
        
        st.divider()
        st.markdown('<h5><i class="bi bi-diagram-3-fill"></i> 2. Build Model Step-by-Step</h5>', unsafe_allow_html=True)
        var_names = [v['name'] for v in st.session_state.variables]
        if not var_names: var_names = [None]
        st.session_state.start_variable = st.selectbox("Start with variable:", var_names, index=var_names.index(st.session_state.start_variable) if st.session_state.start_variable in var_names else 0)
        for i, step in enumerate(st.session_state.model_steps):
            op_col, type_col, val_col, del_col = st.columns([0.3, 0.25, 0.3, 0.15])
            step['op'] = op_col.selectbox("Then,", ["Add", "Subtract", "Multiply by", "Divide by"], key=f"op_{i}", label_visibility="collapsed")
            step['type'] = type_col.radio("With:", ["Variable", "Constant"], key=f"type_{i}", horizontal=True, label_visibility="collapsed")
            if step['type'] == 'Variable': step['val'] = val_col.selectbox("Select variable", var_names, key=f"val_var_{i}", label_visibility="collapsed")
            else: step['val'] = val_col.number_input("Enter value", value=step.get('val', 1.0), key=f"val_num_{i}", label_visibility="collapsed")
            if del_col.button("🗑️", key=f"del_step_{i}", use_container_width=True): st.session_state.model_steps.pop(i); st.rerun()
        if st.button("Add Step", use_container_width=True): st.session_state.model_steps.append({'op': 'Add', 'type': 'Constant', 'val': 10.0}); st.rerun()

    st.divider()
    num_simulations = st.select_slider("Simulation Runs", [1000, 10000, 20000, 50000], value=10000)
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

if run_button:
    # Set simulation parameters based on mode
    if st.session_state.mode == "Guided":
        risk_map = {"Low": 0, "Medium": 30, "High": 65, "Crisis": 100}
        conc_map = {"Diversified": 10, "Consolidated": 60, "Single Source": 90}
        
        # Scenario 1: Normal
        normal_vars = [
            {'name': 'Base_Cost', 'dist': 'Normal', 'param1': base_cost, 'param2': 0.5 + (conc_map[supplier_base] / 100) * 2.0},
            {'name': 'Tariff', 'dist': 'Constant', 'param1': 0.0, 'param2': 0},
            {'name': 'Supply_Cut_Premium', 'dist': 'Constant', 'param1': 0.0, 'param2': 0}
        ]
        
        # Scenario 2: Crisis
        crisis_vars = [
            {'name': 'Base_Cost', 'dist': 'Normal', 'param1': base_cost, 'param2': 1.0 + (conc_map[supplier_base] / 100) * 4.0},
            {'name': 'Tariff', 'dist': 'Uniform', 'param1': 0.0, 'param2': (risk_map[geo_risk] / 100) * 0.40},
            {'name': 'Supply_Cut_Premium', 'dist': 'Uniform', 'param1': 0.0, 'param2': (risk_map[geo_risk] / 100) * (base_cost * 1.5)}
        ]
        formula = "Base_Cost * (1 + Tariff) + Supply_Cut_Premium"
        
        with st.spinner("Running comparative simulations..."):
            normal_results = run_monte_carlo(formula, normal_vars, num_simulations)
            crisis_results = run_monte_carlo(formula, crisis_vars, num_simulations)

        if normal_results is not None and crisis_results is not None:
            # Display Guided Mode Dashboard
            st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Geopolitical Risk Impact Analysis</h3>", unsafe_allow_html=True)
            normal_cost, crisis_cost = normal_results.mean(), crisis_results.mean()
            stockout_threshold = base_cost * 1.5
            normal_risk, crisis_risk = np.mean(normal_results > stockout_threshold) * 100, np.mean(crisis_results > stockout_threshold) * 100
            
            col1, col2 = st.columns(2, gap="large")
            with col1: st.metric("Average Cost (Normal)", f"${normal_cost:,.2f}", f"{((crisis_cost-normal_cost)/normal_cost):.0%} increase in crisis")
            with col2: st.metric("Stockout Probability (Normal)", f"{normal_risk:.1f}%", f"{crisis_risk:.1f}% in crisis", delta_color="inverse")

            fig, ax = plt.subplots();
            sns.kdeplot(normal_results, ax=ax, fill=True, label="Normal Operations", color="#1E88E5"); sns.kdeplot(crisis_results, ax=ax, fill=True, label="Crisis Scenario", color="#D81B60")
            ax.set_title("Distribution of Potential Landed Costs", color='white'); ax.set_xlabel("Landed Cost per Unit ($)", color='white'); ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22'); ax.legend(); ax.grid(True, alpha=0.2)
            st.pyplot(fig)

            with st.expander("Show Auto-Generated Business Continuity Plan (BCP)", expanded=True):
                st.markdown(f"""<h4>BCP for Critical Component (Avg. Cost: ${base_cost})</h4><p>Based on a scenario with <strong>{geo_risk}</strong> geopolitical risk and a <strong>{supplier_base}</strong> supplier base.</p><h5><i class="bi bi-exclamation-triangle-fill" style="color: #ff4b4b;"></i> Threat Assessment</h5><p>The simulation indicates that in a crisis, the average landed cost per unit could rise from <b>${normal_cost:,.2f}</b> to <b>${crisis_cost:,.2f}</b>. The probability of a cost-prohibitive stockout event increases from a manageable <b>{normal_risk:.1f}%</b> to an unacceptable <b>{crisis_risk:.1f}%</b>.</p><h5><i class="bi bi-card-checklist"></i> Recommended Actions</h5><ol><li><b>Immediate (Buffer Stock):</b> Procure an additional 60-90 days of safety stock.</li><li><b>Medium-Term (Supplier Diversification):</b> Immediately initiate a program to qualify a secondary supplier in a different geopolitical region.</li><li><b>Long-Term (Design for Resilience):</b> Mandate that new product designs are qualified with at least two different, interchangeable critical components.</li></ol>""", unsafe_allow_html=True)

    else: # Expert Mode Display
        sim_formula = build_formula_from_steps()
        sim_vars = st.session_state.variables
        if not sim_vars or not sim_formula: st.warning("Please build your model in the sidebar before running.", icon="⚠️")
        else:
            results = run_monte_carlo(sim_formula, sim_vars, num_simulations)
            if results is not None:
                st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Expert Model Simulation Results</h3>", unsafe_allow_html=True)
                mean_val, std_val = results.mean(), results.std(); p5, p95 = np.percentile(results, 5), np.percentile(results, 95)
                col1, col2, col3 = st.columns(3, gap="large");
                with col1: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-bullseye"></i>Average Outcome</h5><h2>{mean_val:,.2f}</h2></div>', unsafe_allow_html=True)
                with col2: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-lightning-charge-fill"></i>Risk (Std. Dev)</h5><h2>{std_val:,.2f}</h2></div>', unsafe_allow_html=True)
                with col3: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-arrows-left-right"></i>90% Confidence Range</h5><h4>{p5:,.2f} — {p95:,.2f}</h4></div>', unsafe_allow_html=True)
                st.divider(); st.markdown("<h4><i class='bi bi-distribute-vertical'></i> Distribution of Potential Outcomes</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(); sns.histplot(results, kde=True, ax=ax, color='#7E57C2', bins=50)
                ax.axvline(mean_val, color='#FDD835', linestyle='--', lw=2, label=f"Mean: {mean_val:,.2f}")
                ax.set_title("Distribution of Potential Outcomes", fontsize=16, color='white', pad=20); ax.set_xlabel("Outcome Value", color='white'); ax.set_ylabel("Frequency", color='white')
                ax.tick_params(colors='white'); ax.spines['left'].set_color('white'); ax.spines['bottom'].set_color('white'); ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')
                fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#30363d')
                st.pyplot(fig)
else:
    st.markdown("<div style='text-align: center; padding-top: 50px;'><h3 style='color: #8b949e;'>Configure your scenario in the sidebar and click 'Run Simulation' to begin.</h3></div>", unsafe_allow_html=True)
    
