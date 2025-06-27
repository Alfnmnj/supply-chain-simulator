# app.py (Definitive Supply Chain Resilience Simulator)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Supply Chain Resilience Simulator",
    page_icon="◈",
    layout="wide"
)

# Inject custom CSS with the Lucide icon library for a premium, industry-grade UI/UX
st.markdown("""
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        .stApp { background-color: #0a0a0a; color: #e6e6e6; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        i[data-lucide] { width: 20px; height: 20px; stroke-width: 2px; vertical-align: middle; margin-right: 0.5rem; color: #a1a1a1; }
        h1 > i[data-lucide] { width: 32px; height: 32px; }
        .stMetric { background-color: #1a1a1a; border: 1px solid #2c2c2c; border-radius: 12px; padding: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); }
        .stMetric .st-ae { font-size: 1.1rem; color: #a1a1a1; }
        .stButton>button { border-radius: 8px; font-weight: bold; border: 1px solid #4a4a4a; transition: all 0.2s ease-in-out; }
        .stButton>button:hover { border-color: #007bff; color: #007bff; }
        .stExpander { border: 1px solid #2c2c2c !important; border-radius: 10px !important; background-color: #1c1c1c; }
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
def generate_strategies(selected_component, df):
    strategies = {}
    component_suppliers = df[df['Component'] == selected_component]
    primary_supplier = component_suppliers[component_suppliers['Primary Supplier']].copy()
    if not primary_supplier.empty:
        primary_supplier['Sourcing %'] = 100.0
        strategies['Baseline (Single Source)'] = primary_supplier

    secondary_supplier = component_suppliers[(~component_suppliers['Primary Supplier']) & (component_suppliers['Country'] != 'India')].head(1)
    if not primary_supplier.empty and not secondary_supplier.empty:
        dual_source_df = pd.concat([primary_supplier, secondary_supplier])
        dual_source_df['Sourcing %'] = [60.0, 40.0]
        strategies['Dual-Source (Global)'] = dual_source_df

    indian_supplier = component_suppliers[component_suppliers['Country'] == 'India'].head(1)
    if not primary_supplier.empty and not indian_supplier.empty:
        onshore_df = pd.concat([primary_supplier, indian_supplier])
        onshore_df['Sourcing %'] = [50.0, 50.0]
        strategies['Onshore (Indo-Global Mix)'] = onshore_df
    return strategies

@st.cache_data
def monte_carlo_stockout_simulation(avg_lead_time, std_dev_lead_time, logistics_delay_days, supply_cut_prob):
    avg_daily_demand = 1000; forecast_error = stockout_inputs['Forecast error (σ/μ)']['volatile']
    std_dev_demand = avg_daily_demand * forecast_error; safety_stock = avg_daily_demand * stockout_inputs['Safety stock days']['baseline']
    reorder_point = (avg_daily_demand * avg_lead_time) + safety_stock; all_sim_service_levels = []
    for _ in range(2000): # Reduced iterations for faster web response, still statistically significant
        inventory, stockout_days, order_placed = reorder_point, 0, False; order_pipeline = {}
        for day in range(1, 365 + 1):
            if day in order_pipeline: inventory += order_pipeline.pop(day); order_placed = False
            demand = max(0, np.random.normal(avg_daily_demand, std_dev_demand))
            if inventory >= demand: inventory -= demand
            else: inventory, stockout_days = 0, stockout_days + 1
            if inventory <= reorder_point and not order_placed:
                if random.random() > supply_cut_prob:
                    disrupted_lead_time = int(np.random.normal(avg_lead_time, std_dev_lead_time) + logistics_delay_days)
                    arrival_day = day + max(1, disrupted_lead_time); order_pipeline[arrival_day] = reorder_point; order_placed = True
        all_sim_service_levels.append((365 - stockout_days) / 365)
    return np.mean(all_sim_service_levels)

def run_full_simulation(strategy_name, strategy_df, scenario):
    results = []
    for _, supplier in strategy_df.iterrows():
        base_cost, base_avg_lead_time = supplier['Unit Cost ($)'], supplier['Avg Lead Time (days)']
        impacted_cost, logistics_delay, supply_cut = base_cost, 0, 0.0
        if supplier['Country'] == scenario['country']:
            impacted_cost *= (1 + scenario['tariff_increase']); logistics_delay, supply_cut = scenario['logistics_delay'], scenario['supply_cut']
        service_level = monte_carlo_stockout_simulation(base_avg_lead_time, stockout_inputs['Lead time distribution']['std'], logistics_delay, supply_cut)
        results.append({'Supplier': supplier['Supplier'], 'Country': supplier['Country'], 'Sourcing %': supplier['Sourcing %'], 'Final Cost ($)': impacted_cost, 'Final Lead Time (days)': base_avg_lead_time + logistics_delay, 'Stockout Risk (%)': (1 - service_level) * 100})
    df_results = pd.DataFrame(results)
    summary = {'Strategy': strategy_name, 'Weighted Avg Cost ($)': np.average(df_results['Final Cost ($)'], weights=df_results['Sourcing %']), 'Weighted Avg Lead Time (days)': np.average(df_results['Final Lead Time (days)'], weights=df_results['Sourcing %']), 'Weighted Avg Stockout Risk (%)': np.average(df_results['Stockout Risk (%)'], weights=df_results['Sourcing %'])}
    return summary, df_results

# ==============================================================================
# 4. APPLICATION UI LAYOUT
# ==============================================================================
st.markdown("<h1><i data-lucide='shield-alert'></i> Supply Chain Resilience Simulator</h1>", unsafe_allow_html=True)
st.markdown("A strategic tool for Indian Electronics Manufacturers to quantify geopolitical risks and evaluate business continuity plans.")

with st.sidebar:
    st.markdown("<h2><i data-lucide='sliders-horizontal'></i> Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("Configure the simulation parameters below.")
    st.divider()
    
    st.subheader("1. Component Selection")
    selected_component = st.selectbox("Component to Analyze", master_df['Component'].unique(), label_visibility="collapsed")

    st.subheader("2. Geopolitical Disruption Scenario")
    country_to_disrupt = st.selectbox("Select Country to Disrupt", master_df['Country'].unique(), index=1)
    tariff_percent = st.slider("Import Tariff Increase (%)", 0, 100, 25, 5)
    supply_cut_percent = st.slider("Supply Cut / Export Ban Probability (%)", 0, 100, 60, 5)
    logistics_delay_days = st.slider("Logistics & Port Delay (days)", 0, 90, 21, 3)

    st.divider()
    st.subheader("3. Business & Financial Context")
    annual_volume = st.number_input("Annual Unit Volume", 1000, 10000000, 1000000, 10000, format="%d")
    line_down_cost = st.number_input("Cost of Production Halt per Day ($)", 10000, 5000000, 500000, 10000, format="%d")
    
    st.divider()
    run_button = st.button("Run Simulation", use_container_width=True, type="primary")

if run_button:
    strategies = generate_strategies(selected_component, master_df)
    
    if not strategies:
        st.error(f"No sourcing strategies could be generated for '{selected_component}'. Please check the master data.")
    else:
        all_results, details_by_strategy = [], {}
        with st.spinner("Running thousands of supply chain scenarios..."):
            for name, df in strategies.items():
                summary, details = run_full_simulation(name, df, {'country': country_to_disrupt, 'tariff_increase': tariff_percent/100.0, 'supply_cut': supply_cut_percent/100.0, 'logistics_delay': logistics_delay_days})
                all_results.append(summary); details_by_strategy[name] = details
        results_df = pd.DataFrame(all_results).set_index('Strategy')

        st.markdown("<h2><i data-lucide='bar-chart-3'></i> Executive Dashboard</h2>", unsafe_allow_html=True)
        st.markdown(f"##### Analysis for **{selected_component}** under a simulated disruption in **{country_to_disrupt}**.")

        baseline_kpis = results_df.loc['Baseline (Single Source)']
        best_resilient_strategy_name = results_df['Weighted Avg Stockout Risk (%)'].idxmin()
        best_resilient_kpis = results_df.loc[best_resilient_strategy_name]
        cost_of_risk = (baseline_kpis['Weighted Avg Stockout Risk (%)'] / 100) * (annual_volume / 365) * line_down_cost * 30

        col1, col2, col3 = st.columns(3, gap="large")
        with col1: st.metric(label="Baseline Stockout Risk", value=f"{baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f}%", delta=f"{best_resilient_kpis['Weighted Avg Stockout Risk (%)']:.1f}% in Best Strategy", delta_color="inverse")
        with col2: total_annual_cost = baseline_kpis['Weighted Avg Cost ($)'] * annual_volume; st.metric(label="Baseline Total Annual Cost", value=f"${total_annual_cost/1e6:.2f}M", delta=f"${(best_resilient_kpis['Weighted Avg Cost ($)'] * annual_volume)/1e6:.2f}M in Best Strategy", delta_color="inverse")
        with col3: st.metric(label="Est. Monthly Cost of Risk", value=f"${cost_of_risk/1e6:.2f}M", help="Potential financial loss from stockouts in the baseline strategy over a 30-day period.", delta="High Exposure", delta_color="off")
        st.divider()

        col1_main, col2_main = st.columns([0.6, 0.4], gap="large")
        with col1_main:
            st.subheader("Strategy Risk Profile"); fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(data=results_df.reset_index(), x='Strategy', y='Weighted Avg Stockout Risk (%)', ax=ax, palette='plasma')
            ax.set_title(f"Stockout Risk by Sourcing Strategy", fontsize=14, pad=20); ax.set_ylabel("Stockout Risk (%)"); ax.set_xlabel(None); ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)
        with col2_main:
            st.subheader("Strategy Details"); tabs = st.tabs([f"{k}" for k in details_by_strategy.keys()])
            for tab, (strategy_name, details) in zip(tabs, details_by_strategy.items()):
                with tab: st.dataframe(details.style.format({'Sourcing %': '{:.0f}%', 'Final Cost ($)': '${:.2f}', 'Final Lead Time (days)': '{:.0f}', 'Stockout Risk (%)': '{:.2f}%'}), use_container_width=True)

        with st.expander("Show Business Continuity Plan & Recommendations", expanded=True):
            st.markdown(f"""
            <h4><i data-lucide="file-text"></i> Business Continuity Plan: {selected_component}</h4>
            <p>Based on a simulated disruption in <strong>{country_to_disrupt}</strong>.</p><hr style='margin-top: 0; margin-bottom: 1rem;'>
            <h5><i data-lucide="alert-triangle" style="color: #ff4b4b;"></i>  1. Threat Assessment</h5>
            <p>The current <b>Baseline (Single Source)</b> strategy faces a <b>{baseline_kpis['Weighted Avg Stockout Risk (%)']:.1f}% stockout risk</b>. This is a critical vulnerability with a potential monthly financial impact (Cost of Risk) of approximately <b>${cost_of_risk/1e6:.2f} million</b>.</p>
            <h5><i data-lucide="shield-check" style="color: #28a745;"></i>  2. Recommended Mitigation Strategy: `{best_resilient_strategy_name}`</h5>
            <p>Diversification is essential. The <b>`{best_resilient_strategy_name}`</b> strategy is the most effective at mitigating risk:</p>
            <ul><li><b>Reduces Stockout Risk</b> to a manageable <b>{best_resilient_kpis['Weighted Avg Stockout Risk (%)']:.1f}%</b>.</li><li><b>Total Annual Cost</b> is estimated at <b>${(best_resilient_kpis['Weighted Avg Cost ($)'] * annual_volume)/1e6:.2f} million</b>. This cost increase is a necessary premium for supply chain insurance.</li><li><b>Stabilizes Lead Time</b> to an average of <b>{best_resilient_kpis['Weighted Avg Lead Time (days)']:.0f} days</b>.</li></ul>
            <h5><i data-lucide="list-checks"></i>  3. Immediate Actions (BCP Phase 1)</h5>
            <ol><li><b>Secure Buffer Stock:</b> Immediately procure an additional 60-90 days of safety stock for the <b>{selected_component}</b> from the primary supplier to provide an operational buffer.</li><li><b>Initiate Diversification:</b> Form a cross-functional task force to begin qualification and contracting of the suppliers identified in the recommended strategy, prioritizing speed to market.</li><li><b>Onshoring Engagement:</b> If an Indian supplier was modeled, begin strategic engagement to align with the India Semiconductor Mission and secure long-term, domestic supply.</li></ol>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="background-color: #1a1a1a; border-radius: 12px; padding: 2rem; text-align: center; border: 1px solid #2c2c2c;">
            <i data-lucide="play-circle" style="width: 48px; height: 48px; color: #a1a1a1;"></i>
            <h3 style="margin-top: 1rem;">Ready to Analyze Your Supply Chain Risk</h3>
            <p style="color: #a1a1a1;">Configure your component and disruption scenario in the <b>Control Panel</b> on the left, then click <b>Run Simulation</b>.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<script>lucide.createIcons();</script>", unsafe_allow_html=True)
