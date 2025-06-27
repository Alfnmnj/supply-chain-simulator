# app.py (Universal Strategic Simulation Platform v4.0)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Strategic Simulation Platform",
    page_icon="🧠",
    layout="wide"
)

# Professional UI/UX Styling
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        .stApp { background-color: #0f1116; }
        .stMetric { background-color: #1a1c24; border: 1px solid #2e3439; border-radius: 10px; padding: 20px; }
        .stMetric .st-ae { font-size: 1.1rem; color: #a1a1a1; }
        .stButton>button { border-radius: 10px; font-weight: bold; }
        .stExpander { border: 1px solid #2e3439 !important; border-radius: 10px !important; }
        div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL TEMPLATES & SESSION STATE
# ==============================================================================
TEMPLATES = {
    "Custom Model": {
        'variables': [{'name': 'Revenue', 'dist': 'Normal', 'p1': 500000, 'p2': 50000},
                      {'name': 'Costs', 'dist': 'Normal', 'p1': 300000, 'p2': 30000}],
        'formula': "Revenue - Costs"
    },
    "Finance: Net Profit Forecast": {
        'variables': [{'name': 'Sales_Volume', 'dist': 'Normal', 'p1': 10000, 'p2': 1500},
                      {'name': 'Price_per_Unit', 'dist': 'Uniform', 'p1': 150, 'p2': 165},
                      {'name': 'Variable_Cost_per_Unit', 'dist': 'Normal', 'p1': 80, 'p2': 5},
                      {'name': 'Fixed_Costs', 'dist': 'Constant', 'p1': 250000, 'p2': 0}],
        'formula': "(Sales_Volume * (Price_per_Unit - Variable_Cost_per_Unit)) - Fixed_Costs"
    },
    "Marketing: Campaign ROI": {
        'variables': [{'name': 'Ad_Spend', 'dist': 'Constant', 'p1': 50000, 'p2': 0},
                      {'name': 'Click_Through_Rate', 'dist': 'Uniform', 'p1': 0.015, 'p2': 0.03},
                      {'name': 'Conversion_Rate', 'dist': 'Normal', 'p1': 0.05, 'p2': 0.01},
                      {'name': 'Customer_Lifetime_Value', 'dist': 'Normal', 'p1': 2500, 'p2': 400}],
        'formula': "((Ad_Spend * Click_Through_Rate / 0.5) * Conversion_Rate * Customer_Lifetime_Value) - Ad_Spend"
    },
    "HR: Attrition Cost Savings": {
        'variables': [{'name': 'Employee_Count', 'dist': 'Constant', 'p1': 500, 'p2': 0},
                      {'name': 'Base_Attrition_Rate', 'dist': 'Uniform', 'p1': 0.12, 'p2': 0.18},
                      {'name': 'Attrition_Reduction_Factor', 'dist': 'Uniform', 'p1': 0.15, 'p2': 0.30},
                      {'name': 'Avg_Cost_to_Replace', 'dist': 'Constant', 'p1': 75000, 'p2': 0}],
        'formula': "Employee_Count * Base_Attrition_Rate * Attrition_Reduction_Factor * Avg_Cost_to_Replace"
    }
}

if 'variables' not in st.session_state:
    st.session_state.variables = TEMPLATES['Custom Model']['variables']
    st.session_state.formula = TEMPLATES['Custom Model']['formula']

# ==============================================================================
# 3. CORE SIMULATION ENGINE
# ==============================================================================
@st.cache_data
def run_monte_carlo(formula, variables, num_simulations):
    # (This robust engine remains unchanged)
    local_scope = {}
    for var in variables:
        if var['dist'] == 'Normal': local_scope[var['name']] = np.random.normal(var['p1'], var['p2'], num_simulations)
        elif var['dist'] == 'Uniform': local_scope[var['name']] = np.random.uniform(var['p1'], var['p2'], num_simulations)
        else: local_scope[var['name']] = var['p1']
    try:
        return eval(formula, {"__builtins__": {"np": np}}, local_scope)
    except Exception: return None

# ==============================================================================
# 4. UI LAYOUT
# ==============================================================================
st.title("🧠 Universal Strategic Simulator")
st.markdown("A platform to model, simulate, and compare business scenarios under uncertainty.")

# --- Sidebar for Model Building ---
with st.sidebar:
    st.image("https://i.imgur.com/vVw2G71.png", width=100)
    st.header("Model Configuration")
    
    # Template Loader
    selected_template = st.selectbox("Load Model Template", list(TEMPLATES.keys()))
    if st.button("Load Template"):
        st.session_state.variables = TEMPLATES[selected_template]['variables']
        st.session_state.formula = TEMPLATES[selected_template]['formula']
        st.rerun()

    st.divider()

    # Model Definition
    with st.expander("Model Definition", expanded=True):
        st.session_state.formula = st.text_area("Model Formula", st.session_state.formula, height=100)
        
        for i, var in enumerate(st.session_state.variables):
            st.markdown(f"**Variable: `{var['name']}`**")
            c1, c2 = st.columns(2)
            var['dist'] = c1.selectbox("Distribution", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
            if var['dist'] == "Normal": var['p1'] = c2.number_input("Mean (μ)", value=var['p1'], key=f"p1_{i}"); var['p2'] = c2.number_input("Std Dev (σ)", value=var['p2'], key=f"p2_{i}")
            elif var['dist'] == "Uniform": var['p1'] = c2.number_input("Min", value=var['p1'], key=f"p1_{i}"); var['p2'] = c2.number_input("Max", value=var['p2'], key=f"p2_{i}")
            else: var['p1'] = c2.number_input("Value", value=var['p1'], key=f"p1_{i}"); var['p2'] = 0
            st.markdown("---")
        
    num_simulations = st.select_slider("Simulation Runs", [1000, 10000, 20000, 50000], value=10000)
    run_button = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# --- Main Panel for Results ---
if run_button:
    base_vars = st.session_state.variables
    base_results = run_monte_carlo(st.session_state.formula, base_vars, num_simulations)

    if base_results is None:
        st.error("Formula evaluation failed. Check variable names and syntax.")
    else:
        st.header("Scenario Comparison Dashboard", anchor=False)
        
        # --- Create and run the comparative scenario ---
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Base Scenario", anchor=False)
            comp_vars = [v.copy() for v in base_vars] # Deep copy for modification
            with st.container(border=True):
                st.markdown("Modify variables below to create a comparative scenario.")
                for var in comp_vars:
                    if var['dist'] != 'Constant':
                        var['p1'] = st.slider(f"Change `{var['name']}` (Parameter 1)", float(var['p1'] * 0.8), float(var['p1'] * 1.2), float(var['p1']), key=f"comp_{var['name']}")
        
        comp_results = run_monte_carlo(st.session_state.formula, comp_vars, num_simulations)

        # --- Display KPIs ---
        with col1:
            mean_val, std_val, p5, p95 = base_results.mean(), base_results.std(), np.percentile(base_results, 5), np.percentile(base_results, 95)
            st.metric("Average Outcome", f"{mean_val:,.2f}")
            st.metric("Risk (Std. Deviation)", f"{std_val:,.2f}")
            st.metric("90% Confidence Range", f"{p5:,.2f} to {p95:,.2f}")

        with col2:
            st.subheader("Comparative Scenario", anchor=False)
            mean_val_c, std_val_c, p5_c, p95_c = comp_results.mean(), comp_results.std(), np.percentile(comp_results, 5), np.percentile(comp_results, 95)
            st.metric("Average Outcome", f"{mean_val_c:,.2f}", f"{mean_val_c-mean_val:,.2f}")
            st.metric("Risk (Std. Deviation)", f"{std_val_c:,.2f}", f"{std_val_c-std_val:,.2f}")
            st.metric("90% Confidence Range", f"{p5_c:,.2f} to {p95_c:,.2f}")

        st.divider()

        # --- Visualization ---
        st.subheader("Distribution of Potential Outcomes", anchor=False)
        fig, ax = plt.subplots()
        sns.kdeplot(base_results, ax=ax, fill=True, label='Base Scenario', color='#4A90E2', lw=2)
        sns.kdeplot(comp_results, ax=ax, fill=True, label='Comparative Scenario', color='#50E3C2', alpha=0.7, lw=2)
        ax.set_title("Scenario Outcome Comparison", fontsize=16); ax.set_xlabel("Outcome Value"); ax.set_ylabel("Probability Density"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        # --- Strategic Insights ---
        with st.expander("View Strategic Insights & Data", expanded=True):
            avg_change = (mean_val_c - mean_val) / abs(mean_val) * 100
            risk_change = (std_val_c - std_val) / abs(std_val) * 100
            insight_text = f"""
            The **Comparative Scenario** resulted in an average outcome of **{mean_val_c:,.2f}**, a change of **{avg_change:.1f}%** from the Base Scenario.
            The associated risk (standard deviation) changed by **{risk_change:.1f}%**.
            """
            if avg_change > 0 and risk_change < 5: insight_text += " This represents a highly favorable trade-off, achieving a better outcome with minimal additional risk."
            elif avg_change > 0 and risk_change > 5: insight_text += " While the average outcome improved, it came with a significant increase in volatility. Evaluate if this risk level is acceptable."
            elif avg_change < 0: insight_text += " The changes in this scenario led to a less favorable average outcome."
            
            st.markdown(f"<h5><i class='bi bi-lightbulb'></i> Key Insight</h5>", unsafe_allow_html=True)
            st.info(insight_text)

            st.markdown(f"<h5><i class='bi bi-table'></i> Raw Data Summary</h5>", unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                'Metric': ['Average Outcome', 'Risk (Std. Dev.)', '5th Percentile', '95th Percentile'],
                'Base Scenario': [mean_val, std_val, p5, p95],
                'Comparative Scenario': [mean_val_c, std_val_c, p5_c, p95_c]
            })
            st.dataframe(summary_df.style.format('{:,.2f}'), use_container_width=True)
else:
    st.info("Load a template or build a custom model in the sidebar, then click **▶ Run Simulation**.")
