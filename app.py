# app.py (Definitive Version with Geopolitical Scenario Builder)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64
from datetime import datetime
import random

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Strategic Resilience Console", page_icon="◈", layout="wide")

st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        h1, h2, h3 { font-weight: 700; color: #FFFFFF; }
        .card { background-color: #161B22; border-radius: 12px; padding: 25px; border: 1px solid #30363D; }
        .stMetric { background-color: transparent; border: none; padding: 0; }
        .stButton>button {
            font-size: 1rem; font-weight: 600; color: #FFFFFF; background-color: #007AFF;
            border-radius: 8px; border: none; padding: 12px 24px; transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover { background-color: #0056b3; }
        .st-emotion-cache-16txtl3 { background-color: #0D1117; border-right: 1px solid #30363D; }
        i[data-lucide] {
            width: 18px; height: 18px; stroke-width: 2.5px;
            vertical-align: -0.125em; margin-right: 0.75rem; color: #8B949E;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA & SIMULATION ENGINE
# ==============================================================================
@st.cache_data
def load_data():
    data = {
        'Component': ['Mainboard chipset', 'Mainboard chipset', 'Mainboard chipset', 'High-end CPU die', 'FPGA (Stratix-class)', 'FPGA (Stratix-class)', 'Power management IC (PMIC)', 'Power management IC (PMIC)'],
        'Supplier': ['TSMC', 'Intel Fab (Ohio)', 'Si-Bharat Fab', 'Intel Fab (own)', 'Intel Foundry', 'Samsung', 'GlobalFoundries', 'ASE'],
        'Country': ['Taiwan', 'USA', 'India', 'USA', 'USA', 'South Korea', 'USA', 'Malaysia'],
        'Base Cost ($)': [25.00, 32.00, 27.50, 150.00, 120.00, 115.00, 8.00, 7.50],
        'Base Lead Time (days)': [75, 50, 60, 90, 112, 120, 130, 145],
        'Is Primary': [True, False, False, True, True, False, True, False]
    }
    df = pd.DataFrame(data); stockout_inputs = {'Safety stock days': {'baseline': 30}, 'Lead time distribution': {'std': 15}, 'Forecast error (σ/μ)': {'volatile': 0.35}}
    return df, stockout_inputs

master_df, stockout_inputs = load_data()

# --- Pre-defined Geopolitical Event Templates ---
EVENT_TEMPLATES = {
    "Normal Operations": {'export_ban_country': 'None', 'export_ban_percent': 0, 'tariff_country': 'None', 'tariff_percent': 0, 'transit_delay': 0, 'supplier_shutdown_prob': 0.0},
    "Taiwan Strait Conflict": {'export_ban_country': 'Taiwan', 'export_ban_percent': 75, 'tariff_country': 'Taiwan', 'tariff_percent': 25, 'transit_delay': 30, 'supplier_shutdown_prob': 0.5},
    "Global Logistics Crisis": {'export_ban_country': 'None', 'export_ban_percent': 0, 'tariff_country': 'None', 'tariff_percent': 5, 'transit_delay': 45, 'supplier_shutdown_prob': 0.1},
    "Korean Peninsula Tensions": {'export_ban_country': 'South Korea', 'export_ban_percent': 40, 'tariff_country': 'South Korea', 'tariff_percent': 15, 'transit_delay': 10, 'supplier_shutdown_prob': 0.3}
}

# (Simulation functions remain unchanged)
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
def monte_carlo_stockout_simulation(base_lt, transit_delay, supply_cut_prob):
    n_simulations = 1000; avg_daily_demand = 100; safety_stock = avg_daily_demand * 30
    lt_dist = np.random.normal(base_lt + transit_delay, base_lt * 0.1, n_simulations)
    stockout_events = 0
    for lead_time in lt_dist:
        if np.random.rand() < supply_cut_prob: stockout_events += 1; continue
        demand_during_lt = np.random.normal(avg_daily_demand * lead_time, avg_daily_demand * lead_time * 0.2)
        if safety_stock < (demand_during_lt - (avg_daily_demand * lead_time)): stockout_events += 1
    return stockout_events / n_simulations

def run_full_simulation(strategy_df, scenario):
    total_cost, total_lt, total_risk = 0, 0, 0
    for _, row in strategy_df.iterrows():
        sourcing_pct = row['Sourcing %'] / 100; cost = row['Base Cost ($)']
        if row['Country'] == scenario['tariff_country']: cost *= (1 + scenario['tariff_percent'] / 100)
        supply_cut_prob = scenario['supplier_shutdown_prob'];
        if row['Country'] == scenario['export_ban_country']: supply_cut_prob = max(supply_cut_prob, scenario['export_ban_percent'] / 100)
        stockout_risk = monte_carlo_stockout_simulation(row['Base Lead Time (days)'], scenario['transit_delay'], supply_cut_prob)
        total_cost += cost * sourcing_pct; total_lt += row['Base Lead Time (days)'] * sourcing_pct; total_risk += stockout_risk * sourcing_pct
    return {'Cost': total_cost, 'Lead Time': total_lt, 'Stockout Risk': total_risk * 100}

# ==============================================================================
# 3. SIDEBAR / CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("<h3><i data-lucide='sliders-horizontal'></i> Simulation Controls</h3>", unsafe_allow_html=True); st.divider()
    st.subheader("Sourcing Strategy"); selected_component = st.selectbox("Critical Component", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Is Primary']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Is Primary'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique()), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)
    
    st.divider()
    st.subheader("Geopolitical Scenario Builder")
    
    def on_template_change():
        st.session_state.update(EVENT_TEMPLATES[st.session_state.event_template])

    selected_event = st.selectbox("Select Event Template", options=EVENT_TEMPLATES.keys(), key="event_template", on_change=on_template_change)

    st.markdown("---")
    st.markdown("_Customize the selected event's parameters below:_")
    
    scenario = {
        'export_ban_country': st.selectbox("Export Ban from", ["None", "China", "Taiwan", "South Korea"], key='export_ban_country'),
        'export_ban_percent': st.slider("Export Ban Intensity (%)", 0, 100, key='export_ban_percent'),
        'tariff_country': st.selectbox("Tariff on", ["None", "China", "Taiwan", "South Korea"], key='tariff_country'),
        'tariff_percent': st.slider("Tariff Increase (%)", 0, 100, key='tariff_percent'),
        'transit_delay': st.slider("Transit Delay (days)", 0, 45, key='transit_delay'),
        'supplier_shutdown_prob': st.slider("Supplier Shutdown Probability", 0.0, 1.0, key='supplier_shutdown_prob')
    }
    
    st.divider(); run_button = st.button("Run Simulation", use_container_width=True)

# ==============================================================================
# 4. MAIN DASHBOARD
# ==============================================================================
if run_button:
    if primary_supplier.empty: st.error(f"No primary supplier defined for {selected_component}.")
    else:
        strategies = generate_dynamic_strategies(selected_component, primary_supplier, alt_supplier_name, sourcing_split, master_df)
        with st.spinner(f"Simulating '{selected_event}' scenario..."):
            results = {name: run_full_simulation(df, scenario) for name, df in strategies.items()}
        results_df = pd.DataFrame(results).T

        st.markdown("<h2><i data-lucide='layout-dashboard'></i> Executive Dashboard</h2>", unsafe_allow_html=True)
        
        baseline_kpis = results_df.loc['Baseline']
        resilient_kpis = results_df.loc['Resilient'] if 'Resilient' in results_df.index else baseline_kpis
        
        col1, col2, col3 = st.columns(3)
        risk_reduction = baseline_kpis['Stockout Risk'] - resilient_kpis['Stockout Risk']
        cost_increase_pct = ((resilient_kpis['Cost'] - baseline_kpis['Cost']) / baseline_kpis['Cost']) * 100 if baseline_kpis['Cost'] > 0 else 0
        with col1: st.metric("Baseline Stockout Risk", f"{baseline_kpis['Stockout Risk']:.1f}%")
        with col2: st.metric("Resilient Strategy Risk", f"{resilient_kpis['Stockout Risk']:.1f}%", f"{-risk_reduction:.1f} pts")
        with col3: st.metric("Cost of Resilience", f"{cost_increase_pct:+.1f}%")
        
        # --- The Hero Visual: Risk vs Cost Quadrant ---
        st.markdown("<h3 style='margin-top: 2rem;'><i data-lucide='compass'></i> Strategy Positioning: Risk vs. Cost</h3>", unsafe_allow_html=True)
        results_df['Strategy'] = results_df.index
        fig_matrix = px.scatter(results_df, x="Cost", y="Stockout Risk", size="Lead Time", color="Strategy",
                                title=f"Strategy Comparison under '{selected_event}' Scenario", template="plotly_dark", size_max=40,
                                labels={"Cost": "Landed Cost per Unit ($)", "Stockout Risk": "Stockout Risk (%)"})
        if 'Resilient' in results_df.index:
            fig_matrix.add_annotation(x=resilient_kpis['Cost'], y=resilient_kpis['Stockout Risk'],
                                      ax=baseline_kpis['Cost'], ay=baseline_kpis['Stockout Risk'],
                                      text="Journey to Resilience", arrowhead=2, arrowwidth=2, arrowcolor="#007AFF", font=dict(color="#007AFF"))
        st.plotly_chart(fig_matrix, use_container_width=True)

        with st.expander("Show Executive Briefing & Business Continuity Plan", expanded=True):
            st.markdown(f"""
            <h4><i data-lucide="file-text"></i> Executive Briefing: BCP for {selected_component}</h4><hr>
            <h5><i data-lucide="alert-triangle"></i> 1. Situation Analysis</h5>
            <p>Under the simulated <b>'{selected_event}'</b> scenario, our current <b>Baseline (Single Source)</b> strategy for the <b>{selected_component}</b> faces a <b>{baseline_kpis['Stockout Risk']:.1f}% probability of stockout</b>. This represents a critical and unacceptable threat to our production continuity and revenue targets.</p>
            
            <h5><i data-lucide="dollar-sign"></i> 2. The Business Case for Resilience</h5>
            <p>We recommend a strategic investment in supply chain resilience by diversifying our sourcing to include <b>{alt_supplier_name}</b>. This action has a clear and compelling business case: for a calculated <b>{cost_increase_pct:.1f}%</b> increase in component cost, we reduce our catastrophic risk exposure by <b>{risk_reduction:.1f} percentage points</b>, achieving a stable state at <b>{resilient_kpis['Stockout Risk']:.1f}% risk</b>.</p>
            
            <h5><i data-lucide="move-right"></i> 3. Recommended Action Plan</h5>
            <ol>
                <li><b>Approve Resilient Strategy:</b> Formally approve the dual-sourcing strategy with a {sourcing_split}/{100-sourcing_split} volume allocation for the <b>{selected_component}</b>.</li>
                <li><b>Initiate Supplier Onboarding:</b> Immediately assign a cross-functional team (Engineering, Procurement, Quality) to begin the technical qualification and contracting process with <b>{alt_supplier_name}</b>.</li>
                <li><b>Secure Bridge Inventory:</b> Authorize procurement to build a 60-day strategic buffer of the primary component to ensure supply continuity during the transition period.</li>
            </ol>
            """, unsafe_allow_html=True)
else:
    st.info("Configure your sourcing strategy and a geopolitical scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
