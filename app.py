# app.py (Definitive Version with Memorandum Generator)

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

EVENT_TEMPLATES = {
    "Normal Operations": {'export_ban_country': 'None', 'export_ban_percent': 0, 'tariff_country': 'None', 'tariff_percent': 0, 'transit_delay': 0, 'supplier_shutdown_prob': 0.0},
    "Taiwan Strait Conflict": {'export_ban_country': 'Taiwan', 'export_ban_percent': 75, 'tariff_country': 'Taiwan', 'tariff_percent': 25, 'transit_delay': 30, 'supplier_shutdown_prob': 0.5},
    "Global Logistics Crisis": {'export_ban_country': 'None', 'export_ban_percent': 0, 'tariff_country': 'None', 'tariff_percent': 5, 'transit_delay': 45, 'supplier_shutdown_prob': 0.1},
    "Korean Peninsula Tensions": {'export_ban_country': 'South Korea', 'export_ban_percent': 40, 'tariff_country': 'South Korea', 'tariff_percent': 15, 'transit_delay': 10, 'supplier_shutdown_prob': 0.3}
}

# (Core simulation functions remain unchanged and robust)
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
    n_simulations = 1000; avg_daily_demand = 100 * (1 + demand_shock_pct / 100); safety_stock = avg_daily_demand * inventory_buffer_days
    lt_dist = np.random.normal(base_lt + transit_delay, base_lt * 0.1, n_simulations)
    stockout_events = 0
    for lead_time in lt_dist:
        if np.random.rand() < supply_cut_prob: stockout_events += 1; continue
        demand_during_lt = np.random.normal(avg_daily_demand * lead_time, avg_daily_demand * lead_time * stockout_inputs['Forecast error (σ/μ)']['volatile'])
        if safety_stock < (demand_during_lt - (avg_daily_demand * lead_time)): stockout_events += 1
    return stockout_events / n_simulations

def run_full_simulation(strategy_df, scenario):
    total_cost, total_lt, total_risk = 0, 0, 0
    for _, row in strategy_df.iterrows():
        sourcing_pct = row['Sourcing %'] / 100; cost = row['Base Cost ($)']
        if row['Country'] == scenario['tariff_country']: cost *= (1 + scenario['tariff_percent'] / 100)
        supply_cut_prob = scenario['supplier_shutdown_prob'];
        if row['Country'] == scenario['export_ban_country']: supply_cut_prob = max(supply_cut_prob, scenario['export_ban_percent'] / 100)
        stockout_risk = monte_carlo_stockout_simulation(row['Base Lead Time (days)'], scenario['transit_delay'], supply_cut_prob, scenario['demand_shock'], scenario['inventory_buffer'])
        total_cost += cost * sourcing_pct; total_lt += row['Base Lead Time (days)'] * sourcing_pct; total_risk += stockout_risk * sourcing_pct
    return {'Cost': total_cost, 'Lead Time': total_lt, 'Stockout Risk': total_risk * 100}

# ==============================================================================
# 3. MEMORANDUM GENERATION ENGINE
# ==============================================================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'CONFIDENTIAL: STRATEGIC RISK BRIEFING', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_memorandum_pdf(results_df, scenario, component, alt_supplier, split):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)

    # Memo Header
    pdf.cell(0, 6, f"TO: Executive Leadership Committee (CEO, CFO, COO)", 0, 1)
    pdf.cell(0, 6, f"FROM: Supply Chain Strategy Department", 0, 1)
    pdf.cell(0, 6, f"DATE: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, f"SUBJECT: Quantified Risk Analysis & BCP for {component}", 0, 1)
    pdf.ln(10)

    # Executive Summary
    baseline_kpis = results_df.loc['Baseline']
    resilient_kpis = results_df.loc['Resilient'] if 'Resilient' in results_df.index else baseline_kpis
    risk_reduction = baseline_kpis['Stockout Risk'] - resilient_kpis['Stockout Risk']
    cost_increase_pct = ((resilient_kpis['Cost'] - baseline_kpis['Cost']) / baseline_kpis['Cost']) * 100 if baseline_kpis['Cost'] > 0 else 0

    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "1. Executive Summary", 0, 1)
    pdf.set_font('Arial', '', 11)
    summary_text = f"This briefing addresses the critical risk to our supply of the {component} arising from the simulated '{scenario['name']}' scenario. Our current single-sourcing strategy faces a {baseline_kpis['Stockout Risk']:.1f}% stockout probability, an unacceptable threat to production. We recommend a dual-sourcing strategy with {alt_supplier}. For a calculated {cost_increase_pct:.1f}% increase in component cost, we can reduce our stockout risk by {risk_reduction:.1f} percentage points to a manageable {resilient_kpis['Stockout Risk']:.1f}%."
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(5)

    # Data-Driven Analysis
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "2. Data-Driven Analysis", 0, 1)
    pdf.set_font('Arial', 'I', 10); pdf.cell(0, 6, f"Scenario Simulated: {scenario['name']}", 0, 1); pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10); col_widths = [60, 40, 40, 40];
    header = ['Strategy', 'Landed Cost ($)', 'Lead Time (days)', 'Stockout Risk (%)']
    for i, h in enumerate(header): pdf.cell(col_widths[i], 7, h, 1, 0, 'C');
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for index, row in results_df.iterrows():
        pdf.cell(col_widths[0], 6, str(index), 1); pdf.cell(col_widths[1], 6, f"${row['Cost']:.2f}", 1); pdf.cell(col_widths[2], 6, f"{row['Lead Time']:.0f}", 1); pdf.cell(col_widths[3], 6, f"{row['Stockout Risk']:.1f}%", 1); pdf.ln()
    pdf.ln(5)

    # Recommended Action Plan
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, "3. Recommended Action Plan (BCP)", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, "Based on the compelling data from the simulation, we propose the following phased BCP:")
    pdf.set_font('Arial', 'B', 11); pdf.cell(0, 8, "Phase 1: Immediate Actions (0-3 Months)", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f"  1. Approve Resilient Strategy: Formally approve the dual-sourcing strategy with {alt_supplier} at a {split}/{100-split}% volume allocation.")
    pdf.multi_cell(0, 6, f"  2. Initiate Supplier Onboarding: Immediately assign a cross-functional team to begin the technical qualification and contracting process with {alt_supplier}.")
    pdf.multi_cell(0, 6, f"  3. Secure Bridge Inventory: Authorize procurement to build a 60-day strategic buffer of the primary component to ensure supply continuity.")
    
    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
# 4. SIDEBAR / CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("<h3><i data-lucide='sliders-horizontal'></i> Simulation Controls</h3>", unsafe_allow_html=True); st.divider()
    st.subheader("Sourcing Strategy"); selected_component = st.selectbox("Critical Component", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Is Primary']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Is Primary'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique()), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)
    
    st.divider(); st.subheader("Geopolitical Scenario Builder")
    def on_template_change(): st.session_state.update(EVENT_TEMPLATES[st.session_state.event_template])
    selected_event = st.selectbox("Select Event Template", options=EVENT_TEMPLATES.keys(), key="event_template", on_change=on_template_change, help="Select a pre-configured event to start with.")
    st.markdown("_Fine-tune the parameters of the selected event below:_")
    scenario_params = {
        'name': selected_event, 'export_ban_country': st.selectbox("Export Ban from", ["None", "China", "Taiwan", "South Korea"], key='export_ban_country'), 'export_ban_percent': st.slider("Export Ban Intensity (%)", 0, 100, key='export_ban_percent'),
        'tariff_country': st.selectbox("Tariff on", ["None", "China", "Taiwan", "South Korea"], key='tariff_country'), 'tariff_percent': st.slider("Tariff Increase (%)", 0, 100, key='tariff_percent'),
        'transit_delay': st.slider("Transit Delay (days)", 0, 45, key='transit_delay'), 'supplier_shutdown_prob': st.slider("Supplier Shutdown Probability", 0.0, 1.0, key='supplier_shutdown_prob'),
    }
    st.divider(); run_button = st.button("Run Simulation", use_container_width=True)

# ==============================================================================
# 5. MAIN DASHBOARD
# ==============================================================================
if run_button:
    if primary_supplier.empty: st.error(f"No primary supplier defined for {selected_component}.")
    else:
        strategies = generate_dynamic_strategies(selected_component, primary_supplier, alt_supplier_name, sourcing_split, master_df)
        with st.spinner(f"Simulating '{selected_event}' scenario..."): results = {name: run_full_simulation(df, scenario_params) for name, df in strategies.items()}
        results_df = pd.DataFrame(results).T

        st.markdown("<h2><i data-lucide='layout-dashboard'></i> Executive Dashboard</h2>", unsafe_allow_html=True)
        baseline_kpis = results_df.loc['Baseline']
        is_resilient_simulated = 'Resilient' in results_df.index
        resilient_kpis = results_df.loc['Resilient'] if is_resilient_simulated else baseline_kpis

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        kpi_cols = st.columns(3)
        risk_reduction = baseline_kpis['Stockout Risk'] - resilient_kpis['Stockout Risk'] if is_resilient_simulated else 0
        cost_increase_pct = ((resilient_kpis['Cost'] - baseline_kpis['Cost']) / baseline_kpis['Cost']) * 100 if is_resilient_simulated and baseline_kpis['Cost'] > 0 else 0
        kpi_cols[0].metric("Baseline Stockout Risk", f"{baseline_kpis['Stockout Risk']:.1f}%")
        kpi_cols[1].metric("Resilient Strategy Risk", f"{resilient_kpis['Stockout Risk']:.1f}%", f"{-risk_reduction:.1f} pts" if is_resilient_simulated else "N/A")
        kpi_cols[2].metric("Cost of Resilience", f"{cost_increase_pct:+.1f}%" if is_resilient_simulated else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Visuals and BCP ---
        if is_resilient_simulated:
            st.markdown("<h3 style='margin-top: 2rem;'><i data-lucide='compass'></i> Strategy Positioning: Risk vs. Cost</h3>", unsafe_allow_html=True)
            results_df['Strategy'] = results_df.index
            fig_matrix = px.scatter(results_df, x="Cost", y="Stockout Risk", size="Lead Time", color="Strategy",
                                    color_discrete_map={"Baseline": "#dc3545", "Resilient": "#007AFF"},
                                    title=f"Strategy Comparison under '{selected_event}' Scenario", template="plotly_dark", size_max=40,
                                    labels={"Cost": "Landed Cost per Unit ($)", "Stockout Risk": "Stockout Risk (%)"})
            fig_matrix.add_annotation(x=resilient_kpis['Cost'], y=resilient_kpis['Stockout Risk'], ax=baseline_kpis['Cost'], ay=baseline_kpis['Stockout Risk'],
                                      text="Journey to Resilience", arrowhead=2, arrowwidth=2, arrowcolor="#007AFF", font=dict(color="#007AFF"))
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.markdown("<h3 style='margin-top: 2rem;'><i data-lucide='file-text'></i> Executive Briefing & BCP</h3>", unsafe_allow_html=True)
            st.download_button("Generate Executive Memorandum", 
                               generate_memorandum_pdf(results_df, scenario_params, selected_component, alt_supplier_name, sourcing_split),
                               file_name=f"BCP_Memo_{selected_component}.pdf", 
                               mime="application/pdf", use_container_width=True)
        else:
            st.warning("Please select an alternative supplier in the sidebar to generate a resilient strategy and a full comparative analysis.")
else:
    st.info("Configure your sourcing strategy and a geopolitical scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
