# app.py (Intel Geopolitical Risk Simulator)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64
from datetime import datetime

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="Intel Supply Chain Risk Simulator", page_icon="intel_icon.png", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .stMetric { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Intel Geopolitical Risk & Resilience Simulator")

# ==============================================================================
# 2. DATA LOADING & MASTER TABLES
# ==============================================================================
@st.cache_data
def load_data():
    data = {
        'Component': ['Meteor Lake CPU Die', 'Meteor Lake CPU Die', 'Granite Rapids Chipset', 'Granite Rapids Chipset', 'Granite Rapids Chipset', 'Agilex-M FPGA', 'Agilex-M FPGA'],
        'Supplier': ['Intel Fab 4', 'TSMC', 'Intel Fab 18', 'TSMC', 'Intel OSAT India', 'Intel Foundry', 'Samsung'],
        'Country': ['USA', 'Taiwan', 'USA', 'Taiwan', 'India', 'USA', 'South Korea'],
        'Base Cost ($)': [120, 110, 45, 55, 12, 180, 175],
        'Base Lead Time (days)': [80, 95, 60, 75, 25, 110, 120],
        'Is Primary': [True, False, True, False, False, True, False]
    }
    return pd.DataFrame(data)

master_df = load_data()

# ==============================================================================
# 3. CORE SIMULATION ENGINE (MODULAR FUNCTIONS)
# ==============================================================================

def simulate_lead_time(base_lt, delay_days, n_simulations=1000):
    """Models lead time extension with stochasticity."""
    lt_std_dev = base_lt * 0.1 # Assume 10% variability
    disrupted_lt_mean = base_lt + delay_days
    return np.random.normal(disrupted_lt_mean, lt_std_dev, n_simulations)

def calculate_costs(base_cost, country, scenario):
    """Simulates cost increases from tariffs and penalties."""
    cost = base_cost
    if country == scenario['tariff_country']:
        cost *= (1 + scenario['tariff_percent'] / 100)
    # Placeholder for other costs like air freight
    return cost

@st.cache_data
def monte_carlo_stockout_simulation(lead_time_distribution, supply_cut_prob):
    """Uses a Monte Carlo simulation to estimate stockout probability."""
    n_simulations = len(lead_time_distribution)
    safety_stock_days = 30
    avg_daily_demand = 100
    
    reorder_point = np.mean(lead_time_distribution) * avg_daily_demand + (safety_stock_days * avg_daily_demand)
    
    stockout_events = 0
    for i in range(n_simulations):
        # Simulate a disruption where an order might be cancelled
        if np.random.rand() < supply_cut_prob:
            stockout_events += 1
            continue
            
        inventory_at_reorder = reorder_point
        demand_during_lead_time = avg_daily_demand * lead_time_distribution[i]
        
        if inventory_at_reorder < demand_during_lead_time:
            stockout_events += 1
            
    return stockout_events / n_simulations

def calculate_resilience_score(risk_pct, cost_usd, lead_time, base_cost, base_lt):
    """Builds a custom resilience score from 0-100."""
    risk_score = (1 - min(risk_pct / 100, 1)) * 60  # 60% weight
    cost_score = (1 - min((cost_usd - base_cost) / base_cost, 1)) * 20 # 20% weight
    lt_score = (1 - min((lead_time - base_lt) / base_lt, 1)) * 20 # 20% weight
    return max(0, risk_score + cost_score + lt_score)

def run_simulation_for_strategy(strategy_df, scenario):
    """Orchestrator function to run all simulations for a given strategy."""
    total_cost, total_lt, total_risk = 0, 0, 0
    
    for _, row in strategy_df.iterrows():
        sourcing_pct = row['Sourcing Mix (%)'] / 100
        
        # Simulate KPIs for this supplier
        final_cost = calculate_costs(row['Base Cost ($)'], row['Country'], scenario)
        lead_time_dist = simulate_lead_time(row['Base Lead Time (days)'], scenario['transit_delay'])
        
        supply_cut_prob = 0
        if row['Country'] == scenario['supply_cut_country']:
            supply_cut_prob = scenario['supply_cut_percent'] / 100
        
        stockout_risk = monte_carlo_stockout_simulation(lead_time_dist, supply_cut_prob)
        
        # Weighted average
        total_cost += final_cost * sourcing_pct
        total_lt += np.mean(lead_time_dist) * sourcing_pct
        total_risk += stockout_risk * sourcing_pct
        
    resilience_score = calculate_resilience_score(
        total_risk * 100, total_cost, total_lt, 
        np.average(strategy_df['Base Cost ($)'], weights=strategy_df['Sourcing Mix (%)']/100),
        np.average(strategy_df['Base Lead Time (days)'], weights=strategy_df['Sourcing Mix (%)']/100)
    )

    return {
        'Total Cost ($)': total_cost,
        'WALT (days)': total_lt,
        'Stockout Risk (%)': total_risk * 100,
        'Resilience Score': resilience_score
    }
    
# ==============================================================================
# 4. PDF REPORT GENERATION
# ==============================================================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Intel Geopolitical Risk Simulation Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(results_df, scenario):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Scenario Parameters:', 0, 1)
    pdf.set_font('Arial', '', 10)
    for key, value in scenario.items():
        pdf.cell(0, 5, f'  - {key.replace("_", " ").title()}: {value}', 0, 1)
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, 'Strategy Comparison Results:', 0, 1)
    
    pdf.set_font('Arial', 'B', 8)
    # Table Header
    col_widths = [40, 30, 30, 35, 35]
    header = ['Strategy'] + list(results_df.columns)
    for i, h in enumerate(header):
        pdf.cell(col_widths[i], 7, h, 1, 0, 'C')
    pdf.ln()

    # Table Rows
    pdf.set_font('Arial', '', 8)
    for index, row in results_df.iterrows():
        pdf.cell(col_widths[0], 6, index, 1)
        pdf.cell(col_widths[1], 6, f"${row['Total Cost ($)']:.2f}", 1)
        pdf.cell(col_widths[2], 6, f"{row['WALT (days)']:.1f}", 1)
        pdf.cell(col_widths[3], 6, f"{row['Stockout Risk (%)']:.2f}%", 1)
        pdf.cell(col_widths[4], 6, f"{row['Resilience Score']:.1f}/100", 1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
# 5. SIDEBAR / USER INPUTS
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Intel_logo_%282020%2C_dark_blue%29.svg/1280px-Intel_logo_%282020%2C_dark_blue%29.svg.png", use_column_width=True)
    st.header("Simulation Controls")
    
    st.subheader("Component Selection")
    selected_component = st.selectbox("Select Component", master_df['Component'].unique())
    
    st.subheader("Geopolitical Scenario")
    scenario = {
        'tariff_country': st.selectbox("Tariff on Country", ["None", "Taiwan", "South Korea"], index=1),
        'tariff_percent': st.slider("Tariff %", 0, 100, 20),
        'supply_cut_country': st.selectbox("Supply Cut from Country", ["None", "Taiwan", "South Korea"], index=2),
        'supply_cut_percent': st.slider("Supply Cut %", 0, 100, 50),
        'transit_delay': st.slider("Global Transit Delay (days)", 0, 30, 12),
        'shutdown_prob': st.slider("Supplier Shutdown Probability", 0.0, 1.0, 0.1)
    }
    
    st.subheader("Resilient Strategy Sourcing Mix")
    component_suppliers = master_df[master_df['Component'] == selected_component]
    primary_supplier = component_suppliers[component_suppliers['Is Primary']]
    alt_suppliers = component_suppliers[~component_suppliers['Is Primary']]
    
    resilient_mix = {}
    total_pct = 100
    if not primary_supplier.empty:
        primary_name = primary_supplier['Supplier'].iloc[0]
        resilient_mix[primary_name] = st.slider(f"Mix % for {primary_name} (Primary)", 0, 100, 60)
        total_pct -= resilient_mix[primary_name]
    
    if not alt_suppliers.empty:
        alt_supplier_1_name = alt_suppliers['Supplier'].iloc[0]
        resilient_mix[alt_supplier_1_name] = st.slider(f"Mix % for {alt_supplier_1_name}", 0, total_pct, 30)
        total_pct -= resilient_mix[alt_supplier_1_name]
        
        if len(alt_suppliers) > 1:
            alt_supplier_2_name = alt_suppliers['Supplier'].iloc[1]
            resilient_mix[alt_supplier_2_name] = total_pct # The rest
            st.info(f"{alt_supplier_2_name} will be allocated the remaining {total_pct}%.")

# ==============================================================================
# 6. MAIN DASHBOARD / VISUALIZATION
# ==============================================================================

# --- Define Strategies Based on Input ---
baseline_strategy_df = primary_supplier.copy()
baseline_strategy_df['Sourcing Mix (%)'] = 100

resilient_strategy_df = component_suppliers[component_suppliers['Supplier'].isin(resilient_mix.keys())].copy()
resilient_strategy_df['Sourcing Mix (%)'] = resilient_strategy_df['Supplier'].map(resilient_mix)

# --- Run Simulation ---
baseline_results = run_simulation_for_strategy(baseline_strategy_df, scenario)
resilient_results = run_simulation_for_strategy(resilient_strategy_df, scenario)

results_df = pd.DataFrame([baseline_results, resilient_results], index=['Baseline', 'Resilient'])

st.header("Executive Summary Dashboard")

# --- KPI Metrics & Gauge ---
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Baseline Resilience Score", f"{results_df.loc['Baseline', 'Resilience Score']:.1f}/100")
    st.metric("Resilient Strategy Score", f"{results_df.loc['Resilient', 'Resilience Score']:.1f}/100", 
              delta=f"{results_df.loc['Resilient', 'Resilience Score'] - results_df.loc['Baseline', 'Resilience Score']:.1f} pts")
    
    pdf_data = generate_pdf_report(results_df, scenario)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name=f'Risk_Report_{datetime.now().strftime("%Y%m%d")}.pdf',
        mime='application/octet-stream',
        use_container_width=True
    )

with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = results_df.loc['Resilient', 'Stockout Risk (%)'],
        title = {'text': "Resilient Strategy Stockout Risk (%)"},
        delta = {'reference': results_df.loc['Baseline', 'Stockout Risk (%)']},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [{'range': [0, 10], 'color': "green"}, {'range': [10, 30], 'color': "yellow"}, {'range': [30, 100], 'color': "red"}],
                 'bar': {'color': "white"}}))
    fig_gauge.update_layout(paper_bgcolor = "#0E1117", font = {'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Create Tabs for Different Visuals ---
tab1, tab2, tab3 = st.tabs(["📊 Financial & Risk Analysis", "🌪️ Sensitivity Analysis", "🗂️ Detailed Data"])

with tab1:
    st.subheader("Financial Impact: The Cost of Resilience")
    cost_baseline = results_df.loc['Baseline', 'Total Cost ($)']
    cost_resilient = results_df.loc['Resilient', 'Total Cost ($)']
    
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Resilience Cost",
        orientation = "v",
        measure = ["absolute", "relative", "total"],
        x = ["Baseline Cost", "Resilience Investment", "Final Resilient Cost"],
        text = [f"${cost_baseline:.2f}", f"${cost_resilient - cost_baseline:.2f}", f"${cost_resilient:.2f}"],
        y = [cost_baseline, cost_resilient - cost_baseline, cost_resilient],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        totals = {"marker":{"color":"#007bff"}}
    ))
    fig_waterfall.update_layout(title="Cost Breakdown: Baseline vs. Resilient Strategy", template="plotly_dark")
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
with tab2:
    st.subheader("Sensitivity Tornado Chart")
    st.info("How do different risk drivers impact the total cost of the Resilient Strategy?")

    sens_data = []
    base_cost = results_df.loc['Resilient', 'Total Cost ($)']
    
    # Sensitivity to Tariff
    temp_scenario = scenario.copy(); temp_scenario['tariff_percent'] += 10
    cost_after_tariff = run_simulation_for_strategy(resilient_strategy_df, temp_scenario)['Total Cost ($)']
    sens_data.append({'Driver': 'Tariff +10%', 'Impact': cost_after_tariff - base_cost})

    # Sensitivity to Delay
    temp_scenario = scenario.copy(); temp_scenario['transit_delay'] += 7
    cost_after_delay = run_simulation_for_strategy(resilient_strategy_df, temp_scenario)['Total Cost ($)']
    sens_data.append({'Driver': 'Delay +7 days', 'Impact': cost_after_delay - base_cost})

    # Sensitivity to Supply Cut
    temp_scenario = scenario.copy(); temp_scenario['supply_cut_percent'] += 10
    cost_after_cut = run_simulation_for_strategy(resilient_strategy_df, temp_scenario)['Total Cost ($)']
    sens_data.append({'Driver': 'Supply Cut +10%', 'Impact': cost_after_cut - base_cost})
    
    sens_df = pd.DataFrame(sens_data).sort_values(by='Impact')
    fig_tornado = px.bar(sens_df, x='Impact', y='Driver', orientation='h', title="Impact of Risk Drivers on Total Cost", template="plotly_dark")
    st.plotly_chart(fig_tornado, use_container_width=True)

    st.subheader("Risk Heatmap")
    heatmap_data = []
    tariff_range = np.linspace(0, 50, 5)
    supply_cut_range = np.linspace(0, 50, 5)
    
    for t in tariff_range:
        row = []
        for sc in supply_cut_range:
            temp_scenario = scenario.copy()
            temp_scenario['tariff_percent'] = t
            temp_scenario['supply_cut_percent'] = sc
            risk = run_simulation_for_strategy(resilient_strategy_df, temp_scenario)['Stockout Risk (%)']
            row.append(risk)
        heatmap_data.append(row)
    
    fig_heatmap = px.imshow(heatmap_data, 
                            labels=dict(x="Supply Cut %", y="Tariff %", color="Stockout Risk %"),
                            x=[f"{x:.0f}" for x in supply_cut_range],
                            y=[f"{y:.0f}" for y in tariff_range],
                            title="Stockout Risk under Varying Tariffs & Supply Cuts (Resilient Strategy)",
                            template="plotly_dark")
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Detailed Comparison Table")
    st.dataframe(results_df.style.format({
        "Total Cost ($)": "${:,.2f}",
        "WALT (days)": "{:.1f}",
        "Stockout Risk (%)": "{:.2f}%",
        "Resilience Score": "{:.1f}"
    }).background_gradient(cmap='RdYlGn', subset=['Resilience Score'])
    .background_gradient(cmap='Reds', subset=['Stockout Risk (%)']))
