# app.py (Definitive Version with Working Memorandum)

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
        h1, h2, h3, h4, h5 { font-weight: 700; color: #FFFFFF; }
        .card { background-color: #161B22; border-radius: 12px; padding: 25px; border: 1px solid #30363D; }
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
# 3. MEMORANDUM GENERATION ENGINE (REBUILT & ROBUST)
# ==============================================================================
class PDF(FPDF):
    def header(self): self.set_font('Helvetica', 'B', 12); self.cell(0, 10, 'CONFIDENTIAL: STRATEGIC RISK BRIEFING', 0, 1, 'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Helvetica', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def section_title(self, title): self.set_font('Helvetica', 'B', 12); self.cell(0, 10, title, 0, 1, 'L'); self.ln(2)
    def section_body(self, text): self.set_font('Helvetica', '', 11); self.multi_cell(190, 6, text)
    def phased_plan(self, phase, items):
        self.set_font('Helvetica', 'B', 11); self.cell(0, 8, phase, 0, 1)
        self.set_font('Helvetica', '', 11)
        for item in items: self.multi_cell(190, 6, f"  - {item}")

def generate_memorandum_pdf(results_df, scenario, component, primary_supplier_name, alt_supplier_name, split):
    pdf = PDF(); pdf.add_page()
    
    pdf.set_font('Helvetica', '', 11); pdf.cell(0, 6, "TO: Executive Leadership Committee (CEO, CFO, COO)", 0, 1); pdf.cell(0, 6, "FROM: Supply Chain Strategy Department", 0, 1); pdf.cell(0, 6, f"DATE: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.set_font('Helvetica', 'B', 11); pdf.cell(0, 6, f"SUBJECT: Urgent: Quantified Risk Analysis and Proposed BCP for {component}", 0, 1); pdf.ln(8)
    
    baseline_kpis = results_df.loc['Baseline']
    resilient_kpis = results_df.loc['Resilient']
    risk_reduction = baseline_kpis['Stockout Risk'] - resilient_kpis['Stockout Risk']
    cost_increase_pct = ((resilient_kpis['Cost'] - baseline_kpis['Cost']) / baseline_kpis['Cost']) * 100 if baseline_kpis['Cost'] > 0 else 0

    pdf.section_title("1. Executive Summary")
    summary_text = f"This briefing outlines the quantifiable risk to our supply of the critical {component} arising from the simulated '{scenario['name']}' scenario. Our current single-sourcing strategy faces a {baseline_kpis['Stockout Risk']:.1f}% stockout probability. We recommend a resilient dual-sourcing strategy with {alt_supplier_name}. For a calculated {cost_increase_pct:.1f}% increase in component cost, we reduce our stockout risk by over {risk_reduction:.1f} percentage points to a manageable {resilient_kpis['Stockout Risk']:.1f}%."
    pdf.section_body(summary_text); pdf.ln(5)

    pdf.section_title("2. Analysis of Simulation Results"); pdf.set_font('Helvetica', 'B', 10);
    col_widths = [60, 40, 40, 40]; header = ['Metric', 'Baseline Strategy', 'Resilient Strategy', 'Improvement']
    for i, h in enumerate(header): pdf.cell(col_widths[i], 7, h, 1, 0, 'C');
    pdf.ln(); pdf.set_font('Helvetica', '', 10)
    kpi_data = [
        ["Stockout Risk (%)", f"{baseline_kpis['Stockout Risk']:.1f}%", f"{resilient_kpis['Stockout Risk']:.1f}%", f"{-risk_reduction:.1f} pts"],
        ["Landed Cost ($/Unit)", f"${baseline_kpis['Cost']:.2f}", f"${resilient_kpis['Cost']:.2f}", f"+{cost_increase_pct:.1f}%"],
        ["Lead Time (days)", f"{baseline_kpis['Lead Time']:.0f}", f"{resilient_kpis['Lead Time']:.0f}", f"{resilient_kpis['Lead Time'] - baseline_kpis['Lead Time']:.0f} days"]
    ]
    for row in kpi_data:
        for i, item in enumerate(row): pdf.cell(col_widths[i], 6, item, 1)
        pdf.ln()
    pdf.ln(5)

    pdf.section_title("3. Proposed Business Continuity Plan (BCP)")
    pdf.phased_plan("Phase 1: Immediate Action (0-3 Months)", [
        "Form Task Force: Immediately stand up a dedicated, cross-functional 'Resilience Task Force'.",
        f"Secure Bridge Inventory: Authorize immediate procurement to increase on-hand safety stock of the {component} by 60 days.",
        f"Initiate Supplier Onboarding: Begin the formal technical and quality qualification process with {alt_supplier_name}."
    ])
    pdf.phased_plan("Phase 2: Transition & Implementation (3-9 Months)", [
        "Achieve Qualification: Complete all necessary quality and engineering approvals.",
        f"Dual-Source Ramp-Up: Gradually shift production volume to achieve the targeted {split}/{100-split}% sourcing split."
    ])
    
    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
# 4. SIDEBAR / CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("<h3><i data-lucide='sliders-horizontal'></i> Simulation Controls</h3>", unsafe_allow_html=True); st.divider()
    st.subheader("Sourcing Strategy"); selected_component = st.selectbox("Critical Component", master_df['Component'].unique())
    primary_supplier = master_df[(master_df['Component']==selected_component) & master_df['Is Primary']]
    alt_suppliers = master_df[(master_df['Component']==selected_component) & (~master_df['Is Primary'])]
    alt_supplier_name, sourcing_split = (st.selectbox("Alternative Supplier", alt_suppliers['Supplier'].unique(), index=0, placeholder="Select to enable resilience modeling..."), st.slider("Primary Supplier Sourcing %", 0, 100, 60, 5)) if not alt_suppliers.empty else (None, 100)
    
    st.divider(); st.subheader("Geopolitical Scenario Builder")
    def on_template_change(): st.session_state.update(EVENT_TEMPLATES[st.session_state.event_template])
    selected_event = st.selectbox("Select Event Template", options=EVENT_TEMPLATES.keys(), key="event_template", on_change=on_template_change)
    st.markdown("_Fine-tune parameters below:_")
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
        
        # ... (Gauge and KPI card code remains the same) ...

        tab_list = ["📊 Strategic Overview", "💰 Financial Analysis", "🌪️ Sensitivity Analysis", "📄 BCP & Memorandum"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_list)

        with tab1:
            # ... (Quadrant Chart code remains the same) ...
        
        with tab2:
            # ... (Waterfall chart code remains the same) ...
            
        with tab3:
            # ... (Tornado & Heatmap code remains the same) ...

        with tab4:
            st.subheader("Executive Briefing & Business Continuity Plan")
            if is_resilient_simulated:
                primary_supplier_name = primary_supplier['Supplier'].iloc[0]
                st.download_button("Generate & Download Memorandum", 
                                   generate_memorandum_pdf(results_df, scenario_params, selected_component, primary_supplier_name, alt_supplier_name, sourcing_split),
                                   file_name=f"BCP_Memo_{selected_component}_{datetime.now().strftime('%Y%m%d')}.pdf", 
                                   mime="application/pdf")
                
                # On-Screen Memorandum
                st.markdown(f"<div class='card' style='margin-top: 1rem;'>", unsafe_allow_html=True)
                st.markdown(f"<h4><i data-lucide='file-text'></i> MEMORANDUM</h4><hr>", unsafe_allow_html=True)
                st.markdown(f"""
                    <p><b>TO:</b> Executive Leadership Committee (CEO, CFO, COO)<br>
                    <b>FROM:</b> Supply Chain Strategy Department<br>
                    <b>DATE:</b> {datetime.now().strftime('%Y-%m-%d')}<br>
                    <b>SUBJECT:</b> Urgent: Quantified Risk Analysis and Proposed BCP for {selected_component}</p>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # On-Screen BCP
                st.markdown(f"<div class='card' style='margin-top: 1rem;'>", unsafe_allow_html=True)
                risk_reduction = baseline_kpis['Stockout Risk'] - resilient_kpis['Stockout Risk']
                cost_increase_pct = ((resilient_kpis['Cost'] - baseline_kpis['Cost']) / baseline_kpis['Cost']) * 100 if baseline_kpis['Cost'] > 0 else 0
                st.markdown(f"""
                    <h4><i data-lucide='shield-check'></i> Business Continuity Plan (BCP)</h4><hr>
                    <h5>1. Executive Summary</h5>
                    <p>This briefing outlines the quantifiable risk to our supply of the critical <b>{selected_component}</b> arising from the simulated <b>'{selected_event}'</b> scenario. Our current single-sourcing strategy faces a <b>{baseline_kpis['Stockout Risk']:.1f}% stockout probability</b>. We recommend a resilient dual-sourcing strategy with <b>{alt_supplier_name}</b>. For a calculated <b>{cost_increase_pct:.1f}%</b> increase in component cost, we reduce stockout risk by <b>{risk_reduction:.1f} percentage points</b> to a manageable <b>{resilient_kpis['Stockout Risk']:.1f}%</b>.</p>
                    <h5>2. Proposed Action Plan</h5>
                    <p><b>Phase 1: Immediate Action (0-3 Months)</b></p>
                    <ol>
                        <li><b>Form Task Force:</b> Immediately stand up a dedicated, cross-functional "Resilience Task Force".</li>
                        <li><b>Secure Bridge Inventory:</b> Authorize procurement to increase on-hand safety stock of the <b>{selected_component}</b> by 60 days.</li>
                        <li><b>Initiate Supplier Onboarding:</b> Begin the formal qualification process with <b>{alt_supplier_name}</b> for the <b>{selected_component}</b>.</li>
                    </ol>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("A full Business Continuity Plan requires a resilient strategy to be simulated. Please add an alternative supplier in the sidebar.")
else:
    st.info("Configure your sourcing strategy and a geopolitical scenario in the sidebar, then click 'Run Simulation'.")

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
