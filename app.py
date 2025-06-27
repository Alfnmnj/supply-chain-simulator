# app.py (Strategic Risk Dashboard for Indian Electronics)

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
# 1. PAGE CONFIGURATION & AESTHETIC STYLING
# ==============================================================================
st.set_page_config(page_title="Strategic Risk Dashboard", page_icon="◈", layout="wide")

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
# 2. DATA LOADING & CORE SIMULATION ENGINE
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
def monte_carlo_stockout_simulation(base_lt, transit_delay, supply_cut_prob, demand_shock_pct, inventory_buffer_days):
    n_simulations = 1000
    lt_dist = np.random.normal(base_lt + transit_delay, base_lt * 0.1, n_simulations)
    avg_daily_demand = 100 * (1 + demand_shock_pct / 100)
    safety_stock = avg_daily_demand * inventory_buffer_days
    
    stockout_events = 0
    for lead_time in lt_dist:
        if np.random.rand() < supply_cut_prob:
            stockout_events += 1
            continue
        demand_during_lt = np.random.normal(avg_daily_demand * lead_time, avg_daily_demand * lead_time * stockout_inputs['Forecast error (σ/μ)']['volatile'])
        if safety_stock < (demand_during_lt - (avg_daily_demand * lead_time)):
            stockout_events += 1
    return stockout_events / n_simulations

def run_full_simulation(strategy_df, scenario):
    total_cost, total_lt, total_risk, total_suppliers = 0, 0, 0, len(strategy_df)
    for _, row in strategy_df.iterrows():
        sourcing_pct = row['Sourcing %'] / 100
        cost = row['Base Cost ($)']
        if row['Country'] == scenario['tariff_country']:
            cost *= (1 + scenario['tariff_percent'] / 100)
        
        supply_cut_prob = scenario['supplier_shutdown_prob']
        if row['Country'] == scenario['export_ban_country']:
            supply_cut_prob = max(supply_cut_prob, scenario['export_ban_percent'] / 100)

        stockout_risk = monte_carlo_stockout_simulation(row['Base Lead Time (days)'], scenario['transit_delay'], supply_cut_prob, scenario['demand_shock'], scenario['inventory_buffer'])
        
        total_cost += cost * sourcing_pct
        total_lt += row['Base Lead Time (days)'] * sourcing_pct
        total_risk += stockout_risk * sourcing_pct
    return {'Cost': total_cost, 'Lead Time': total_lt, 'Stockout Risk': total_risk * 100, 'Supplier Count': total_suppliers}

def calculate_resilience_score(risk_pct, cost, lead_time, base_cost, base_lt):
    risk_score = (1 - min(risk_pct / 100, 1)) * 60
    cost_score = (1 - min((cost - base_cost) / base_cost if base_cost > 0 else 0, 1)) * 20
    lt_score = (1 - min((lead_time - base_lt) / base_lt if base_lt > 0 else 0, 1)) * 20
    return max(0, risk_score + cost_score + lt_score)

def generate_pdf_report(results_df, scenario, component):
    pdf = FPDF()
    pdf.add_page(); pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, f'Strategic Risk Briefing: {component}', 0, 1, 'C'); pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Disruption Scenario Parameters', 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in scenario.items(): pdf.multi_cell(0, 5, f'{key.replace("_", " ").title()}: {value}')
    
    pdf.ln(5); pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Strategy Impact Analysis', 0, 1)
    pdf.set_font('Arial', 'B', 10); col_widths = [45, 25, 30, 35, 35]; header = ['Strategy'] + list(results_df.columns);
    for i, h in enumerate(header): pdf.cell(col_widths[i], 7, h, 1, 0, 'C');
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for index, row in results_df.iterrows():
        pdf.cell(col_widths[0], 6, index, 1)
        pdf.cell(col_widths[1], 6, f"${row['Cost']:.2f}", 1); pdf.cell(col_widths[2], 6, f"{row['Lead Time']:.0f} days", 1)
        pdf.cell(col_widths[3], 6, f"{row['Stockout Risk']:.1f}%", 1); pdf.cell(col_widths[4], 6, f"{row['Resilience Score']:.0f}/100", 1);
        pdf.ln()

    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
# 4. APPLICATION UI LAYOUT
# ==============================================================================
st.markdown("<h1><i data-lucide='shield-half'></i> Strategic Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("An interactive BCP tool for the Indian Electronics Supply Chain.")

with st.sidebar:
    st.markdown("<h3><i data-lucide='sliders-horizontal'></i> Simulation Controls</h3>", unsafe_allow_html=True); st.divider()
    
    st.subheader("Sourcing Strategy"); selected_component = st.selectbox("Critical Component", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Is Primary']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Is Primary'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique()), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)
    
    st.subheader("Geopolitical & Market Scenario")
    scenario = {
        'export_ban_country': st.selectbox("Export Ban from", ["None", "China", "Taiwan", "South Korea"], index=2),
        'export_ban_percent': st.slider("Export Ban Intensity (%)", 0, 100, 50),
        'tariff_country': st.selectbox("Tariff on", ["None", "China", "Taiwan", "South Korea"], index=2),
        'tariff_percent': st.slider("Tariff Increase (%)", 0, 100, 20),
        'transit_delay': st.slider("Transit Delay (days)", 0, 45, 14),
        'supplier_shutdown_prob': st.slider("Supplier Shutdown Probability", 0.0, 1.0, 0.1),
        'demand_shock': st.slider("Demand Growth Shock (%)", -50, 100, 0),
        'inventory_buffer': st.slider("Inventory Buffer (days)", 0, 120, 30)
    }
    
    st.divider(); run_button = st.button("Run Simulation", use_container_width=True)

if run_button:
    if primary_supplier.empty: st.error(f"No primary supplier defined for {selected_component}.")
    else:
        strategies = generate_dynamic_strategies(selected_component, primary_supplier, alt_supplier_name, sourcing_split, master_df)
        results = {name: run_full_simulation(df, scenario) for name, df in strategies.items()}
        results_df = pd.DataFrame(results).T
        results_df['Resilience Score'] = results_df.apply(lambda row: calculate_resilience_score(row['Stockout Risk'], row['Cost'], row['Lead Time'], results_df.loc['Baseline', 'Cost'], results_df.loc['Baseline', 'Lead Time']), axis=1)

        st.markdown("<h2><i data-lucide='layout-dashboard'></i> Executive Summary</h2>", unsafe_allow_html=True)
        
        baseline_kpis = results_df.loc['Baseline']
        resilient_kpis = results_df.loc['Resilient'] if 'Resilient' in results_df.index else baseline_kpis
        
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=baseline_kpis['Stockout Risk'], title={'text': "Baseline Stockout Risk"},
                gauge={'axis': {'range': [None, 100]}, 'steps': [{'range': [0,10], 'color': '#28a745'},{'range': [10,30], 'color': '#ffc107'},{'range': [30,100], 'color': '#dc3545'}], 'bar': {'color': 'white'}}))
            fig_gauge.update_layout(paper_bgcolor="#161B22", font={'color': "white"}, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            st.markdown(f"<div class='card'><h3>Resilience Score</h3><p style='font-size: 2.5rem; font-weight: bold; color: #007AFF;'>{resilient_kpis['Resilience Score']:.0f}<span style='font-size: 1rem; color: #8B949E;'>/100</span></p><p>An integrated measure of risk, cost, and lead time performance.</p></div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🌪️ Sensitivity Analysis", "💧 Cost Waterfall", "📈 Strategy Matrix", "📄 BCP Report"])

        with tab1:
            st.subheader("Tornado Chart: Cost Sensitivity")
            base_cost = resilient_kpis['Cost']
            sens_data = []
            drivers = {'Tariff +10%': {'key': 'tariff_percent', 'delta': 10}, 'Delay +7 days': {'key': 'transit_delay', 'delta': 7}, 'Demand Shock +20%': {'key': 'demand_shock', 'delta': 20}}
            for name, d in drivers.items():
                temp_scenario = scenario.copy(); temp_scenario[d['key']] += d['delta']
                cost_after = run_full_simulation(strategies['Resilient'], temp_scenario)['Cost']
                sens_data.append({'Driver': name, 'Impact': cost_after - base_cost})
            sens_df = pd.DataFrame(sens_data).sort_values(by='Impact')
            fig_tornado = px.bar(sens_df, x='Impact', y='Driver', orientation='h', title="Impact of Risk Drivers on Resilient Strategy Cost", template="plotly_dark", text_auto='.2f')
            st.plotly_chart(fig_tornado, use_container_width=True)

        with tab2:
            st.subheader("Financial Breakdown: The Cost of Resilience")
            cost_of_risk = (baseline_kpis['Stockout Risk'] / 100) * (scenario['inventory_buffer'] * 1000) # Simplified
            fig_waterfall = go.Figure(go.Waterfall(
                measure = ["absolute", "relative", "total", "relative", "absolute"],
                x = ["Baseline Cost", "Monetized Risk", "Total Risk Exposure", "Resilience Investment", "Final Resilient Cost"],
                y = [baseline_kpis['Cost'], cost_of_risk, 0, resilient_kpis['Cost'] - baseline_kpis['Cost'], 0],
                totals = {"marker":{"color":"#8B949E"}}
            ))
            fig_waterfall.update_layout(title="Cost Analysis: Baseline vs. Resilient Strategy", template="plotly_dark", paper_bgcolor="#161B22", plot_bgcolor="#161B22")
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with tab3:
            st.subheader("Strategy Matrix: Cost vs. Risk vs. Lead Time")
            results_df['Strategy'] = results_df.index
            fig_matrix = px.scatter(results_df, x="Cost", y="Stockout Risk", size="Lead Time", color="Strategy",
                                    title="Strategy Comparison Quadrant", template="plotly_dark", size_max=40,
                                    labels={"Cost": "Weighted Average Cost ($)", "Stockout Risk": "Stockout Risk (%)"})
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with tab4:
            st.subheader("Executive Briefing & Business Continuity Plan")
            st.download_button("📄 Download Full Report (PDF)", generate_pdf_report(results_df.reset_index(), scenario, selected_component), file_name=f"BCP_Report_{selected_component}.pdf", mime="application/pdf", use_container_width=True)
            st.markdown(f"""
            <div class='card'>
                <h4><i data-lucide="file-text"></i> BCP for: {selected_component}</h4><hr>
                <h5><i data-lucide="alert-triangle"></i> 1. Situation Analysis</h5>
                <p>Our current <b>Baseline (Single Source)</b> strategy for the <b>{selected_component}</b> faces a <b>{baseline_kpis['Stockout Risk']:.1f}% stockout risk</b> under the simulated disruption, posing an existential threat to production continuity.</p>
                <h5><i data-lucide="dollar-sign"></i> 2. The Business Case for Resilience</h5>
                <p>We recommend a strategic investment in diversifying our supply chain to include <b>{alt_supplier_name}</b>. This action has a clear and compelling business case: for a calculated increase in component cost, we reduce our catastrophic risk exposure and achieve a superior <b>Resilience Score of {resilient_kpis['Resilience Score']:.0f}/100</b>.</p>
                <h5><i data-lucide="move-right"></i> 3. Recommended Action Plan</h5>
                <ol>
                    <li><b>Approve Resilient Strategy:</b> Formally approve the dual-sourcing strategy with a {sourcing_split}/{100-sourcing_split} volume allocation.</li>
                    <li><b>Initiate Supplier Onboarding:</b> Immediately assign a cross-functional team to begin the technical qualification and contracting process with <b>{alt_supplier_name}</b>.</li>
                    <li><b>Optimize Inventory:</b> Adjust safety stock levels to the new, more stable lead times of the resilient network.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Configure your strategy and disruption scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
