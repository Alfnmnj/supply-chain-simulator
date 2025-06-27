# app.py (Definitive Version - Fulfilling the Original Prompt)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Supply Chain Resilience Simulator", page_icon="◈", layout="wide")

st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        .stApp { background-color: #0a0a0a; color: #e6e6e6; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        i[data-lucide] { width: 20px; height: 20px; stroke-width: 2px; vertical-align: middle; margin-right: 0.5rem; color: #a1a1a1; }
        h1 > i[data-lucide] { width: 32px; height: 32px; }
        .stMetric { background-color: #1a1a1a; border: 1px solid #2c2c2c; border-radius: 12px; padding: 20px; }
        .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #4a4a4a; transition: all 0.2s ease-in-out; }
        .stButton>button:hover { border-color: #007bff; color: #007bff; }
        .st-emotion-cache-16txtl3 { background-color: #121212; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADING & CACHING
# ==============================================================================
@st.cache_data
def load_data():
    master_data = {
        'Component': ['Mainboard chipset', 'Mainboard chipset', 'Mainboard chipset', 'High-end CPU die', 'FPGA (Stratix-class)', 'FPGA (Stratix-class)', 'Power management IC (PMIC)', 'Power management IC (PMIC)'],
        'Supplier': ['TSMC', 'Intel Fab (Ohio)', 'Si-Bharat Fab', 'Intel Fab (own)', 'Intel Foundry', 'Samsung', 'GlobalFoundries', 'ASE'],
        'Country': ['Taiwan', 'USA', 'India', 'USA', 'USA', 'South Korea', 'USA', 'Malaysia'],
        'Unit Cost ($)': [25.00, 32.00, 27.50, 150.00, 120.00, 115.00, 8.00, 7.50],
        'Avg Lead Time (days)': [75, 50, 60, 90, 112, 120, 130, 145],
        'Primary Supplier': [True, False, False, True, True, False, True, False]
    }
    df = pd.DataFrame(master_data)
    stockout_inputs = {'Safety stock days': {'baseline': 30}, 'Lead time distribution': {'std': 15}, 'Forecast error (σ/μ)': {'volatile': 0.35}}
    return df, stockout_inputs

master_df, stockout_inputs = load_data()

# ==============================================================================
# 3. CORE SIMULATION ENGINE
# ==============================================================================
def generate_dynamic_strategies(component, primary_supplier, alt_supplier_name, sourcing_split, df):
    strategies = {}
    if primary_supplier.empty: return strategies
    primary_df = primary_supplier.copy(); primary_df['Sourcing %'] = 100.0
    strategies['Baseline (Single Source)'] = primary_df
    if alt_supplier_name:
        alt_supplier_df = df[(df['Component'] == component) & (df['Supplier'] == alt_supplier_name)].copy()
        if not alt_supplier_df.empty:
            diversified_df = pd.concat([primary_df, alt_supplier_df]); diversified_df['Sourcing %'] = [sourcing_split, 100 - sourcing_split]
            strategies[f"Resilient ({sourcing_split}/{100-sourcing_split})"] = diversified_df
    return strategies

@st.cache_data
def monte_carlo_stockout_simulation(avg_lead_time, std_dev_lead_time, logistics_delay_days, supply_cut_prob):
    avg_daily_demand=1000; std_dev_demand=avg_daily_demand*stockout_inputs['Forecast error (σ/μ)']['volatile']
    safety_stock=avg_daily_demand*stockout_inputs['Safety stock days']['baseline']; reorder_point=(avg_daily_demand*avg_lead_time)+safety_stock
    all_sim_service_levels=[]
    for _ in range(2000):
        inventory,stockout_days,order_placed=reorder_point,0,False; order_pipeline={}
        for day in range(1,366):
            if day in order_pipeline: inventory+=order_pipeline.pop(day); order_placed=False
            demand=max(0,np.random.normal(avg_daily_demand,std_dev_demand))
            if inventory>=demand: inventory-=demand
            else: inventory,stockout_days=0,stockout_days+1
            if inventory<=reorder_point and not order_placed:
                if random.random()>supply_cut_prob:
                    disrupted_lead_time=int(np.random.normal(avg_lead_time,std_dev_lead_time)+logistics_delay_days)
                    arrival_day=day+max(1,disrupted_lead_time); order_pipeline[arrival_day]=reorder_point; order_placed=True
        all_sim_service_levels.append((365-stockout_days)/365)
    return np.mean(all_sim_service_levels)

def run_full_simulation(strategy_name, strategy_df, scenario):
    results=[]
    for _, supplier in strategy_df.iterrows():
        base_cost, base_avg_lead_time=supplier['Unit Cost ($)'], supplier['Avg Lead Time (days)']
        impacted_cost, logistics_delay, supply_cut=base_cost, 0, 0.0
        if supplier['Country']==scenario['country']:
            impacted_cost*=(1+scenario['tariff_increase']); logistics_delay,supply_cut=scenario['logistics_delay'],scenario['supply_cut']
        service_level=monte_carlo_stockout_simulation(base_avg_lead_time, stockout_inputs['Lead time distribution']['std'], logistics_delay, supply_cut)
        results.append({'Supplier':supplier['Supplier'], 'Country':supplier['Country'], 'Sourcing %':supplier['Sourcing %'], 'Final Cost ($)':impacted_cost, 'Final Lead Time (days)':base_avg_lead_time+logistics_delay, 'Stockout Risk (%)':(1-service_level)*100})
    df_results=pd.DataFrame(results)
    summary={'Strategy':strategy_name, 'Weighted Avg Cost ($)':np.average(df_results['Final Cost ($)'],weights=df_results['Sourcing %']), 'Weighted Avg Lead Time (days)':np.average(df_results['Final Lead Time (days)'],weights=df_results['Sourcing %']), 'Weighted Avg Stockout Risk (%)':np.average(df_results['Stockout Risk (%)'],weights=df_results['Sourcing %'])}
    return summary, df_results

# ==============================================================================
# 4. APPLICATION UI LAYOUT
# ==============================================================================
st.markdown("<h1><i data-lucide='shield-alert'></i> Supply Chain Resilience Simulator</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2><i data-lucide='sliders-horizontal'></i> Control Panel</h2>", unsafe_allow_html=True); st.divider()
    st.subheader("1. Sourcing Strategy"); selected_component = st.selectbox("Component to Analyze", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Primary Supplier']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Primary Supplier'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique()), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)

    st.subheader("2. Disruption Scenario"); country_to_disrupt = st.selectbox("Country to Disrupt", master_df['Country'].unique(), index=1)
    tariff_percent=st.slider("Import Tariff (%)", 0, 100, 25, 5); supply_cut_percent=st.slider("Supply Cut Probability (%)", 0, 100, 60, 5)
    logistics_delay_days=st.slider("Logistics Delay (days)", 0, 90, 21, 3)

    st.divider(); st.subheader("3. Financial Context"); annual_volume=st.number_input("Annual Unit Volume", 1000, 10000000, 1000000, 10000)
    line_down_cost=st.number_input("Cost of Halt per Day ($)", 10000, 5000000, 500000, 10000)
    
    st.divider(); run_button = st.button("Run Simulation", use_container_width=True, type="primary")

if run_button:
    if primary_supplier.empty: st.error(f"No primary supplier for {selected_component}.")
    else:
        strategies = generate_dynamic_strategies(selected_component, primary_supplier, alt_supplier_name, sourcing_split, master_df)
        all_results, _ = [], {};
        with st.spinner("Running thousands of scenarios..."):
            for name, df in strategies.items():
                summary, _ = run_full_simulation(name, df, {'country': country_to_disrupt, 'tariff_increase': tariff_percent/100.0, 'supply_cut': supply_cut_percent/100.0, 'logistics_delay': logistics_delay_days})
                all_results.append(summary)
        results_df = pd.DataFrame(all_results).set_index('Strategy')

        st.markdown("<h2><i data-lucide='bar-chart-3'></i> Executive Dashboard</h2>", unsafe_allow_html=True)
        
        baseline_kpis = results_df.loc['Baseline (Single Source)']
        resilient_kpis = results_df.loc[results_df.index != 'Baseline (Single Source)'].iloc[0] if len(results_df) > 1 else baseline_kpis

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline Stockout Risk", f"{baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f}%")
        col2.metric("Resilient Stockout Risk", f"{resilient_kpis['Weighted Avg Stockout Risk (%)']:.1f}%", delta=f"{resilient_kpis['Weighted Avg Stockout Risk (%)'] - baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f} pts", delta_color="inverse")
        col3.metric("Resilient Annual Cost", f"${(resilient_kpis['Weighted Avg Cost ($)'] * annual_volume)/1e6:.2f}M", delta=f"{((resilient_kpis['Weighted Avg Cost ($)']-baseline_kpis['Weighted Avg Cost ($)'])/baseline_kpis['Weighted Avg Cost ($)'])*100:.1f}% vs Baseline")
        cost_of_risk = (baseline_kpis['Weighted Avg Stockout Risk (%)'] / 100) * (annual_volume / 365) * line_down_cost * 30
        col4.metric("Baseline Monthly Cost of Risk", f"${cost_of_risk/1e6:.2f}M", help="Potential financial loss from stockouts in the baseline strategy over a 30-day period.")

        st.divider()
        st.subheader("Detailed KPI Comparison")
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        with chart_col1:
            fig = px.bar(results_df, y='Weighted Avg Stockout Risk (%)', text_auto='.1f', title="Stockout Risk (%)", template='plotly_dark')
            fig.update_traces(marker_color='#dc3545')
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            fig = px.bar(results_df, y='Weighted Avg Cost ($)', text_auto='.2f', title="Landed Cost ($)", template='plotly_dark')
            fig.update_traces(marker_color='#ffc107')
            st.plotly_chart(fig, use_container_width=True)
        with chart_col3:
            fig = px.bar(results_df, y='Weighted Avg Lead Time (days)', text_auto='.0f', title="Lead Time (days)", template='plotly_dark')
            fig.update_traces(marker_color='#007bff')
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Business Continuity Plan (BCP)", expanded=True):
            st.markdown(f"""<h4><i data-lucide="file-text"></i> BCP for: {selected_component}</h4><hr><h5><i data-lucide="alert-triangle" style="color: #ff4b4b;"></i> 1. Threat Assessment</h5><p>Our <b>Baseline</b> strategy for the <b>{selected_component}</b> faces a <b>{baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f}% stockout risk</b> under the simulated disruption in <b>{country_to_disrupt}</b>.</p><h5><i data-lucide="shield-check" style="color: #28a745;"></i> 2. Recommended Mitigation Strategy</h5><p>The <b>{resilient_kpis.name}</b> is the recommended approach, reducing risk to <b>{resilient_kpis['Weighted Avg Stockout Risk (%)']:.1f}%</b>.</p><h5><i data-lucide="list-checks"></i> 3. Immediate Actions</h5><ol><li><b>Secure Buffer Stock:</b> Procure 60-90 days of safety stock for the <b>{selected_component}</b>.</li><li><b>Initiate Diversification:</b> Form a task force to qualify and contract with <b>{alt_supplier_name}</b>, allocating {100-sourcing_split}% of volume.</li></ol>""", unsafe_allow_html=True)
else:
    st.info("Configure your strategy and disruption scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
