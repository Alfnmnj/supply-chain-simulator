# app.py (Strategic Simulation Platform - v7.1 FINAL, Verified)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. PAGE CONFIGURATION & AESTHETIC STYLING
# ==============================================================================
st.set_page_config(page_title="Strategic Simulation Platform", layout="wide")
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
# 2. DATA & SESSION STATE
# ==============================================================================
@st.cache_data
def load_supplier_data():
    return pd.DataFrame({
        'Component': ['Mainboard Chipset', 'Mainboard Chipset', 'High-end CPU', 'High-end CPU'],
        'Supplier': ['Fab Taiwan', 'Fab USA', 'Fab USA (Primary)', 'Fab Malaysia'],
        'Country': ['Taiwan', 'USA', 'USA', 'Malaysia'],
        'Base Cost ($)': [25.0, 35.0, 150.0, 140.0],
        'Base Lead Time (days)': [75, 50, 90, 110],
        'Base Risk (%)': [5.0, 2.0, 2.0, 10.0], # Intrinsic operational risk
        'Is Primary': [True, False, True, False]
    })

supplier_df = load_supplier_data()

# --- Session State ---
if 'mode' not in st.session_state: st.session_state.mode = "Guided"
if 'variables' not in st.session_state: st.session_state.variables = []
if 'model_steps' not in st.session_state: st.session_state.model_steps = []
if 'start_variable' not in st.session_state: st.session_state.start_variable = None

# ==============================================================================
# 3. CORE SIMULATION ENGINES
# ==============================================================================
def run_supply_chain_simulation(strategy_suppliers, scenario):
    results = []
    for _, supplier in strategy_suppliers.iterrows():
        tariff = scenario['tariff'] if supplier['Country'] == scenario['country'] else 0
        supply_cut = scenario['supply_cut'] if supplier['Country'] == scenario['country'] else 0
        delay = scenario['delay'] if supplier['Country'] == scenario['country'] else 0
        impacted_cost = supplier['Base Cost ($)'] * (1 + tariff)
        impacted_lead_time = supplier['Base Lead Time (days)'] + delay
        impacted_risk = min(100, supplier['Base Risk (%)'] + (supply_cut * 100))
        results.append({'Sourcing %': supplier['Sourcing %'], 'Cost': impacted_cost, 'Lead Time': impacted_lead_time, 'Stockout Risk': impacted_risk})
    df = pd.DataFrame(results)
    return {'Avg Cost': np.average(df['Cost'], weights=df['Sourcing %']), 'Avg Lead Time': np.average(df['Lead Time'], weights=df['Sourcing %']), 'Avg Stockout Risk': np.average(df['Stockout Risk'], weights=df['Sourcing %'])}

@st.cache_data
def run_expert_mode_simulation(formula, variables, num_simulations):
    results = {var['name']: (np.random.normal(var['param1'], var['param2'], num_simulations) if var['dist'] == 'Normal' else
                             np.random.uniform(var['param1'], var['param2'], num_simulations) if var['dist'] == 'Uniform' else
                             var['param1']) for var in variables}
    try: return pd.eval(formula, local_dict=results)
    except Exception as e: st.error(f"Error: {e}", icon="🚨"); return None

def build_formula_from_steps():
    if not st.session_state.start_variable: return ""
    formula = st.session_state.start_variable
    for step in st.session_state.model_steps:
        op_map = {"Add": "+", "Subtract": "-", "Multiply by": "*", "Divide by": "/"}
        formula = f"({formula}) {op_map.get(step['op'])} {step['val'] if step['type'] == 'Variable' else str(step['val'])}"
    return formula

# ==============================================================================
# 4. UI LAYOUT & COMPONENTS
# ==============================================================================
st.markdown('<h1 style="text-align: center;"><i class="bi bi-shield-check"></i> Strategic Simulation Platform</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://i.imgur.com/vVw2G71.png", width=100)
    st.markdown("<h3><i class='bi bi-tools'></i> Control Panel</h3>", unsafe_allow_html=True)
    st.radio("Select Analysis Mode", ["Guided", "Expert"], key="mode", horizontal=True)
    st.divider()

    if st.session_state.mode == "Guided":
        st.markdown("<h5><i class='bi bi-truck'></i> Supply Chain Risk Simulator</h5>", unsafe_allow_html=True)
        selected_component = st.selectbox("1. Select Critical Component", supplier_df['Component'].unique())
        component_suppliers = supplier_df[supplier_df['Component'] == selected_component]
        primary_supplier = component_suppliers[component_suppliers['Is Primary']].iloc[0]
        secondary_supplier = component_suppliers[~component_suppliers['Is Primary']].iloc[0]
        st.markdown(f"<p style='font-size: 0.9rem;'>Primary: <b>{primary_supplier['Supplier']}</b> ({primary_supplier['Country']})<br>Alternate: <b>{secondary_supplier['Supplier']}</b> ({secondary_supplier['Country']})</p>", unsafe_allow_html=True)
        with st.expander("2. Define Disruption Scenario", expanded=True):
            disruption_country = st.selectbox("Disrupted Country", supplier_df['Country'].unique(), index=supplier_df['Country'].unique().tolist().index(primary_supplier['Country']))
            disruption_tariff = st.slider("Tariff Increase", 0, 100, 20, 5, format="%d%%") / 100.0
            disruption_supply_cut = st.slider("Supply Cut Probability", 0, 100, 50, 5, format="%d%%") / 100.0
            disruption_delay = st.slider("Logistics Delay (days)", 0, 60, 14, 1)
        with st.expander("3. Configure Resilient Strategy", expanded=True):
            resilient_mix = st.slider(f"Sourcing % from Alternate ({secondary_supplier['Country']})", 0, 100, 40, 5)

    else: # Expert Mode UI
        st.markdown('<h5><i class="bi bi-sliders"></i> 1. Define Variables</h5>', unsafe_allow_html=True)
        for i, var in enumerate(st.session_state.variables):
            with st.container():
                c1,c2 = st.columns([0.85, 0.15])
                with c1:
                    var['name'] = st.text_input("Name", var['name'], key=f"name_{i}").replace(" ", "_"); var['dist'] = st.selectbox("Dist", ["Normal", "Uniform", "Constant"], index=["Normal", "Uniform", "Constant"].index(var['dist']), key=f"dist_{i}")
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
        st.markdown('<h5><i class="bi bi-diagram-3-fill"></i> 2. Build Model</h5>', unsafe_allow_html=True)
        var_names = [v['name'] for v in st.session_state.variables] or [None]
        st.session_state.start_variable = st.selectbox("Start with:", var_names, index=var_names.index(st.session_state.start_variable) if st.session_state.start_variable in var_names else 0)
        for i, step in enumerate(st.session_state.model_steps):
            op_col, type_col, val_col, del_col = st.columns([0.3, 0.25, 0.3, 0.15])
            step['op'] = op_col.selectbox("Then,", ["Add", "Subtract", "Multiply by", "Divide by"], key=f"op_{i}", label_visibility="collapsed")
            step['type'] = type_col.radio("With:", ["Variable", "Constant"], key=f"type_{i}", horizontal=True, label_visibility="collapsed")
            if step['type'] == 'Variable': step['val'] = val_col.selectbox("Select", var_names, key=f"val_var_{i}", label_visibility="collapsed")
            else: step['val'] = val_col.number_input("Value", value=step.get('val', 1.0), key=f"val_num_{i}", label_visibility="collapsed")
            if del_col.button("🗑️", key=f"del_step_{i}", use_container_width=True): st.session_state.model_steps.pop(i); st.rerun()
        if st.button("Add Step", use_container_width=True): st.session_state.model_steps.append({'op': 'Add', 'type': 'Constant', 'val': 10.0}); st.rerun()

    st.divider()
    num_simulations = st.select_slider("Simulation Precision", [1000, 10000, 20000, 50000], value=10000)
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

if run_button:
    if st.session_state.mode == "Guided":
        baseline_suppliers = primary_supplier.to_frame().T; baseline_suppliers['Sourcing %'] = 100.0
        resilient_suppliers = pd.concat([primary_supplier.to_frame().T, secondary_supplier.to_frame().T])
        resilient_suppliers['Sourcing %'] = [100 - resilient_mix, resilient_mix]
        scenario = {'country': disruption_country, 'tariff': disruption_tariff, 'supply_cut': disruption_supply_cut, 'delay': disruption_delay}
        with st.spinner("Running comparative simulations..."):
            baseline_results = run_supply_chain_simulation(baseline_suppliers, scenario)
            resilient_results = run_supply_chain_simulation(resilient_suppliers, scenario)

        st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Geopolitical Risk Impact Analysis</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="large")
        with col1: st.metric("Avg. Landed Cost", f"${resilient_results['Avg Cost']:.2f}", f"{((resilient_results['Avg Cost'] - baseline_results['Avg Cost'])/baseline_results['Avg Cost']):.1%} vs Baseline", delta_color="inverse")
        with col2: st.metric("Avg. Lead Time", f"{resilient_results['Avg Lead Time']:.0f} days", f"{resilient_results['Avg Lead Time'] - baseline_results['Avg Lead Time']:.0f} days vs Baseline")
        with col3: st.metric("Avg. Stockout Risk", f"{resilient_results['Avg Stockout Risk']:.1f}%", f"{resilient_results['Avg Stockout Risk'] - baseline_results['Avg Stockout Risk']:.1f}% vs Baseline")
        
        df_plot = pd.DataFrame([baseline_results, resilient_results], index=['Baseline', 'Resilient (Diversified)']).reset_index()
        fig = px.bar(df_plot, x='index', y=['Avg Cost', 'Avg Lead Time', 'Avg Stockout Risk'], barmode='group', labels={'index': 'Sourcing Strategy', 'value': 'Impacted Value'}, title='<b>Strategy Comparison under Disruption Scenario</b>', facet_col='variable', color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_yaxes(matches=None, title_text=""); fig.update_xaxes(title_text="")
        fig.update_layout(title_font_size=20, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Actionable Recommendations & BCP", expanded=True):
            st.markdown(f"""<h4>Business Continuity Plan for: <b>{selected_component}</b></h4><p>Based on a simulated disruption in <strong>{disruption_country}</strong>.</p>
            <h5><i class="bi bi-exclamation-triangle-fill" style="color: #ff4b4b;"></i> Threat Assessment (Baseline Single-Source Strategy)</h5>
            <p>Our current single-source strategy is highly vulnerable. Under the simulated scenario, it results in an average landed cost of <b>${baseline_results['Avg Cost']:.2f}</b> and a stockout risk of <b>{baseline_results['Avg Stockout Risk']:.1f}%</b>.</p>
            <h5><i class="bi bi-shield-check" style="color: #28a745;"></i> Resilience Investment Analysis (Diversified Strategy)</h5>
            <p>By diversifying <b>{resilient_mix}%</b> of our supply to the alternate fab in <b>{secondary_supplier['Country']}</b>, we achieve a more resilient position:</p>
            <ul><li>The <b>Stockout Risk</b> is reduced to a more manageable <b>{resilient_results['Avg Stockout Risk']:.1f}%</b>.</li><li>The <b>Landed Cost</b> settles at <b>${resilient_results['Avg Cost']:.2f}</b> per unit. This cost differential represents the 'insurance premium' for supply chain security.</li><li>The <b>Lead Time</b> is stabilized at an average of <b>{resilient_results['Avg Lead Time']:.0f} days</b>.</li></ul>
            <h5><i class="bi bi-card-checklist"></i> Recommended Actions</h5><ol><li><b>Diversify Sourcing:</b> The simulation provides a clear data-driven case to immediately proceed with qualifying <b>{secondary_supplier['Supplier']}</b> and implementing the {100-resilient_mix}/{resilient_mix} sourcing strategy.</li><li><b>Build Safety Stock:</b> The increased lead time and stockout risk in a crisis justify an immediate increase in safety stock by at least {disruption_delay} days to buffer against initial logistics disruptions.</li></ol>
            """, unsafe_allow_html=True)

    else: # Expert Mode Display
        sim_formula = build_formula_from_steps()
        sim_vars = st.session_state.variables
        if not sim_vars or not sim_formula: st.warning("Please build your model in the sidebar.", icon="⚠️")
        else:
            results = run_expert_mode_simulation(sim_formula, sim_vars, num_simulations)
            if results is not None:
                st.markdown("<h3><i class='bi bi-clipboard-data-fill'></i> Expert Model Simulation Results</h3>", unsafe_allow_html=True)
                mean_val, std_val = results.mean(), results.std(); p5, p95 = np.percentile(results, 5), np.percentile(results, 95)
                col1, col2, col3 = st.columns(3, gap="large");
                with col1: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-bullseye"></i>Average Outcome</h5><h2>{mean_val:,.2f}</h2></div>', unsafe_allow_html=True)
                with col2: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-lightning-charge-fill"></i>Risk (Std. Dev)</h5><h2>{std_val:,.2f}</h2></div>', unsafe_allow_html=True)
                with col3: st.markdown(f'<div class="metric-card"><h5><i class="bi bi-arrows-left-right"></i>90% Confidence Range</h5><h4>{p5:,.2f} — {p95:,.2f}</h4></div>', unsafe_allow_html=True)
                st.divider(); st.markdown("<h4><i class='bi bi-distribute-vertical'></i> Distribution of Potential Outcomes</h4>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Histogram(x=results, nbinsx=50, marker_color='#7E57C2')])
                fig.add_vline(x=mean_val, line_dash="dash", line_color="#FDD835", annotation_text=f"Mean: {mean_val:.2f}")
                fig.update_layout(title_text='Distribution of Potential Outcomes', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("<div style='text-align: center; padding-top: 50px;'><h3 style='color: #8b949e;'>Configure your scenario in the sidebar and click 'Run Simulation' to begin.</h3></div>", unsafe_allow_html=True)
    
