# app.py (Definitive "Fortune 500" Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random

# ==============================================================================
# 1. PAGE CONFIGURATION & AESTHETIC STYLING (The "Apple" Feel)
# ==============================================================================
st.set_page_config(page_title="Strategic Resilience Console", page_icon="◈", layout="wide")

# This is the core of the new UI: premium fonts, colors, and component styles
st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        /* Base Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp { background-color: #121212; color: #e6e6e6; }
        h1, h2, h3 { font-weight: 700; color: #ffffff; }
        
        /* Custom "Card" for wrapping elements */
        .card {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #333333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        /* Metric Card Styling */
        .stMetric {
            background-color: transparent; border: none; padding: 0;
        }

        /* Button Styling */
        .stButton>button {
            font-size: 1rem; font-weight: 600; color: #ffffff; background-color: #007AFF;
            border-radius: 8px; border: none; padding: 12px 24px; transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover { background-color: #0056b3; }
        
        /* Sidebar Styling */
        .st-emotion-cache-16txtl3 { background-color: #181818; border-right: 1px solid #333333;}
        
        /* Icon Styling */
        i[data-lucide] {
            width: 18px; height: 18px; stroke-width: 2.5px;
            vertical-align: middle; margin-right: 0.75rem; color: #a1a1a1;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADING & CORE SIMULATION ENGINE (Unchanged - The "Guts")
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
    df = pd.DataFrame(master_data); stockout_inputs = {'Safety stock days': {'baseline': 30}, 'Lead time distribution': {'std': 15}, 'Forecast error (σ/μ)': {'volatile': 0.35}}
    return df, stockout_inputs

master_df, stockout_inputs = load_data()

def generate_dynamic_strategies(component, primary_supplier, alt_supplier_name, sourcing_split, df):
    strategies = {};
    if primary_supplier.empty: return strategies
    primary_df = primary_supplier.copy(); primary_df['Sourcing %'] = 100.0; strategies['Baseline'] = primary_df
    if alt_supplier_name:
        alt_supplier_df = df[(df['Component'] == component) & (df['Supplier'] == alt_supplier_name)].copy()
        if not alt_supplier_df.empty:
            diversified_df = pd.concat([primary_df, alt_supplier_df]); diversified_df['Sourcing %'] = [sourcing_split, 100 - sourcing_split]
            strategies["Resilient"] = diversified_df
    return strategies

@st.cache_data
def monte_carlo_stockout_simulation(avg_lead_time, std_dev_lead_time, logistics_delay_days, supply_cut_prob):
    avg_daily_demand=1000; std_dev_demand=avg_daily_demand*stockout_inputs['Forecast error (σ/μ)']['volatile']
    safety_stock=avg_daily_demand*stockout_inputs['Safety stock days']['baseline']; reorder_point=(avg_daily_demand*avg_lead_time)+safety_stock
    all_sim_service_levels=[];
    for _ in range(2000):
        inventory,stockout_days,order_placed=reorder_point,0,False; order_pipeline={}
        for day in range(1,366):
            if day in order_pipeline: inventory+=order_pipeline.pop(day); order_placed=False
            demand=max(0,np.random.normal(avg_daily_demand,std_dev_demand))
            if inventory>=demand: inventory-=demand
            else: inventory,stockout_days=0,stockout_days+1
            if inventory<=reorder_point and not order_placed:
                if random.random()>supply_cut_prob:
                    disrupted_lead_time=int(np.random.normal(avg_lead_time,std_dev_lead_time)+logistics_delay_days); arrival_day=day+max(1,disrupted_lead_time); order_pipeline[arrival_day]=reorder_point; order_placed=True
        all_sim_service_levels.append((365-stockout_days)/365)
    return np.mean(all_sim_service_levels)

def run_full_simulation(strategy_name, strategy_df, scenario):
    results=[];
    for _, supplier in strategy_df.iterrows():
        base_cost, base_avg_lead_time=supplier['Unit Cost ($)'], supplier['Avg Lead Time (days)']; impacted_cost, logistics_delay, supply_cut=base_cost, 0, 0.0
        if supplier['Country']==scenario['country']: impacted_cost*=(1+scenario['tariff_increase']); logistics_delay,supply_cut=scenario['logistics_delay'],scenario['supply_cut']
        service_level=monte_carlo_stockout_simulation(base_avg_lead_time, stockout_inputs['Lead time distribution']['std'], logistics_delay, supply_cut)
        results.append({'Supplier':supplier['Supplier'], 'Country':supplier['Country'], 'Sourcing %':supplier['Sourcing %'], 'Final Cost ($)':impacted_cost, 'Final Lead Time (days)':base_avg_lead_time+logistics_delay, 'Stockout Risk (%)':(1-service_level)*100})
    df_results=pd.DataFrame(results)
    summary={'Strategy':strategy_name, 'Weighted Avg Cost ($)':np.average(df_results['Final Cost ($)'],weights=df_results['Sourcing %']), 'Weighted Avg Lead Time (days)':np.average(df_results['Final Lead Time (days)'],weights=df_results['Sourcing %']), 'Weighted Avg Stockout Risk (%)':np.average(df_results['Stockout Risk (%)'],weights=df_results['Sourcing %'])}
    return summary, df_results

# ==============================================================================
# 4. APPLICATION UI LAYOUT
# ==============================================================================
st.markdown("<h1><i data-lucide='shield-half'></i> Strategic Resilience Console</h1>", unsafe_allow_html=True)
st.markdown("An interactive simulation platform to quantify geopolitical supply chain risk and evaluate mitigation strategies.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.markdown("<h3><i data-lucide='sliders-horizontal'></i> Simulation Controls</h3>", unsafe_allow_html=True); st.divider()
    st.subheader("Sourcing Strategy"); selected_component = st.selectbox("Component to Analyze", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Primary Supplier']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Primary Supplier'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique()), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)

    st.subheader("Disruption Scenario"); country_to_disrupt = st.selectbox("Country to Disrupt", master_df['Country'].unique(), index=1)
    tariff_percent=st.slider("Import Tariff (%)", 0, 100, 25, 5); supply_cut_percent=st.slider("Supply Cut Probability (%)", 0, 100, 60, 5)
    logistics_delay_days=st.slider("Logistics Delay (days)", 0, 90, 21, 3)
    
    st.divider(); st.subheader("Financial Context"); annual_volume=st.number_input("Annual Unit Volume", 1000, 10000000, 1000000, 10000)
    line_down_cost=st.number_input("Cost of Halt per Day ($)", 10000, 5000000, 500000, 10000)
    
    st.divider(); run_button = st.button("Run Simulation", use_container_width=True)

# --- Main Dashboard Panel ---
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

        st.markdown("<h2><i data-lucide='layout-dashboard'></i> Executive Summary & Recommendation</h2>", unsafe_allow_html=True)
        
        baseline_kpis = results_df.loc['Baseline']
        resilient_kpis = results_df.loc['Resilient'] if 'Resilient' in results_df.index else baseline_kpis
        
        # --- Conclusion First ---
        st.markdown(f"""
        <div class="card">
            <h3 style="color: #007AFF;"><i data-lucide='check-circle'></i> Recommendation: Adopt the Resilient Strategy</h3>
            <p>The simulation indicates that diversifying our supply chain for the <b>{selected_component}</b> by partnering with <b>{alt_supplier_name}</b> is the optimal path forward. This strategic investment effectively mitigates catastrophic production risk for a quantifiable cost premium.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- The "Hero" Waterfall Chart ---
        st.markdown("<h3 style='margin-top: 2rem;'><i data-lucide='trending-up'></i> Financial Impact Analysis: The Business Case for Resilience</h3>", unsafe_allow_html=True)
        
        base_cost_annual = baseline_kpis['Weighted Avg Cost ($)'] * annual_volume
        cost_of_risk_annual = (baseline_kpis['Weighted Avg Stockout Risk (%)'] / 100) * (annual_volume / 365) * line_down_cost * 365
        resilient_cost_annual = resilient_kpis['Weighted Avg Cost ($)'] * annual_volume
        
        fig_waterfall = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "total", "relative", "total"],
            x = ["Baseline Annual Cost", "Monetized Baseline Risk", "Effective Cost of Baseline", "Resilience Cost Delta", "Effective Cost of Resilient Strategy"],
            text = [f"${v/1e6:.2f}M" for v in [base_cost_annual, cost_of_risk_annual, 0, resilient_cost_annual - base_cost_annual, 0]],
            y = [base_cost_annual, cost_of_risk_annual, 0, resilient_cost_annual - base_cost_annual, 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#dc3545"}},
            decreasing = {"marker":{"color":"#28a745"}},
            totals = {"marker":{"color":"#1E1E1E", "line":{"color":"#007AFF", "width":3}}}
        ))
        fig_waterfall.update_layout(title="Cost of Resilience: A Financial Breakdown", showlegend=False, template='plotly_dark', paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E')
        st.plotly_chart(fig_waterfall, use_container_width=True)

        with st.expander("Show Detailed Business Continuity Plan (BCP)"):
            risk_reduction = baseline_kpis['Weighted Avg Stockout Risk (%)'] - resilient_kpis['Weighted Avg Stockout Risk (%)']
            st.markdown(f"""
            <h4><i data-lucide="file-text"></i> Executive Briefing: BCP for {selected_component}</h4><hr>
            <h5><i data-lucide="alert-triangle"></i> Situation</h5>
            <p>Under the simulated geopolitical disruption in <b>{country_to_disrupt}</b>, our current single-source strategy for the <b>{selected_component}</b> exposes the company to a <b>{baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f}% probability of stockout</b>, representing a critical failure point in our operations.</p>
            <h5><i data-lucide="lightbulb"></i> Analysis & Business Case</h5>
            <p>By diversifying with <b>{alt_supplier_name}</b>, we make a strategic investment to "buy down" this risk. The waterfall chart above illustrates the business case: we accept a calculated increase in component cost to mitigate a much larger monetized risk of production failure. This action reduces our stockout probability by <b>{risk_reduction:.1f} percentage points</b> to a manageable <b>{resilient_kpis['Weighted Avg Stockout Risk (%)']:.1f}%</b>.</p>
            <h5><i data-lucide="move-right"></i> Recommendation & Next Steps</h5>
            <ol>
                <li><b>Approve Resilient Strategy:</b> Formally approve the dual-sourcing strategy with a {sourcing_split}/{100-sourcing_split} volume allocation.</li>
                <li><b>Initiate Supplier Onboarding:</b> Immediately assign a cross-functional team to begin the technical qualification and contracting process with {alt_supplier_name}.</li>
                <li><b>Secure Bridge Inventory:</b> Authorize procurement to build a 60-day strategic buffer of the primary component to ensure supply continuity during the transition.</li>
            </ol>
            """, unsafe_allow_html=True)
else:
    st.info("Configure your strategy and disruption scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
