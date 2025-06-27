# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==============================================================================
# Main App Configuration
# ==============================================================================
st.set_page_config(
    page_title="Supply Chain Risk Simulator",
    page_icon="⛓️",
    layout="wide"
)

st.title("⛓️ Geopolitical Supply Chain Risk Simulator")
st.markdown("This tool simulates the impact of geopolitical disruptions on a critical component's cost, lead time, and stockout risk.")

# ==============================================================================
# Data Loading and Caching (Use @st.cache_data to speed up the app)
# ==============================================================================
@st.cache_data
def load_data():
    # Supplier Data
    supplier_data = {
        'Component': ['Mainboard chipset'], 'Supplier(s)': ['TSMC'], 'Country': ['Taiwan'],
        'Unit Cost ($)': [25.00], 'Avg_Lead_Time': [75]
    }
    df_supplier = pd.DataFrame(supplier_data)
    
    # Stockout Inputs
    stockout_inputs = {
        'Safety stock days': {'baseline': 30},
        'Lead time distribution': {'std': 15},
        'Forecast error (σ/μ)': {'volatile': 0.35}
    }
    return df_supplier, stockout_inputs

df_supplier, stockout_inputs = load_data()


# ==============================================================================
# Define Sourcing Strategies
# ==============================================================================
def get_strategies():
    baseline_strategy_df = df_supplier.copy()
    baseline_strategy_df['Strategy'] = 'Baseline (Single Source)'
    baseline_strategy_df['Sourcing %'] = 100.0

    resilient_strategy_df = baseline_strategy_df.copy()
    resilient_strategy_df['Strategy'] = 'Resilient (Dual Source)'
    resilient_strategy_df['Sourcing %'] = 60.0

    new_supplier = {
        'Component': 'Mainboard chipset', 'Supplier(s)': 'Intel Fab (Ohio)', 'Country': 'USA',
        'Unit Cost ($)': 32.00, 'Avg_Lead_Time': 50,
        'Strategy': 'Resilient (Dual Source)', 'Sourcing %': 40.0
    }
    resilient_strategy_df = pd.concat([resilient_strategy_df, pd.DataFrame([new_supplier])], ignore_index=True)
    return baseline_strategy_df, resilient_strategy_df

baseline_strategy_df, resilient_strategy_df = get_strategies()


# ==============================================================================
# Simulation Engine (NO CHANGES NEEDED HERE)
# ==============================================================================
@st.cache_data
def monte_carlo_stockout_simulation(
    avg_lead_time, std_dev_lead_time, 
    logistics_delay_days, supply_cut_prob,
    sim_days=365, num_simulations=1000):
    
    avg_daily_demand = 1000
    forecast_error = stockout_inputs['Forecast error (σ/μ)']['volatile']
    std_dev_demand = avg_daily_demand * forecast_error
    safety_stock = avg_daily_demand * stockout_inputs['Safety stock days']['baseline']
    reorder_point = (avg_daily_demand * avg_lead_time) + safety_stock
    all_sim_service_levels = []

    for _ in range(num_simulations):
        inventory, stockout_days, order_placed = reorder_point, 0, False
        order_pipeline = {}
        for day in range(1, sim_days + 1):
            if day in order_pipeline:
                inventory += order_pipeline.pop(day)
                order_placed = False
            demand = max(0, np.random.normal(avg_daily_demand, std_dev_demand))
            if inventory >= demand:
                inventory -= demand
            else:
                inventory, stockout_days = 0, stockout_days + 1
            if inventory <= reorder_point and not order_placed:
                if random.random() > supply_cut_prob:
                    disrupted_lead_time = int(np.random.normal(avg_lead_time, std_dev_lead_time) + logistics_delay_days)
                    arrival_day = day + max(1, disrupted_lead_time)
                    order_pipeline[arrival_day] = reorder_point
                    order_placed = True
        all_sim_service_levels.append((sim_days - stockout_days) / sim_days)
    return np.mean(all_sim_service_levels)


def run_full_simulation(strategy_df, scenario):
    results = []
    for _, supplier in strategy_df.iterrows():
        base_cost, base_avg_lead_time = supplier['Unit Cost ($)'], supplier['Avg_Lead_Time']
        impacted_cost, logistics_delay, supply_cut = base_cost, 0, 0.0
        if supplier['Country'] == scenario['country']:
            impacted_cost *= (1 + scenario['tariff_increase'])
            logistics_delay, supply_cut = scenario['logistics_delay'], scenario['supply_cut']
        
        service_level = monte_carlo_stockout_simulation(
            avg_lead_time=base_avg_lead_time, std_dev_lead_time=stockout_inputs['Lead time distribution']['std'],
            logistics_delay_days=logistics_delay, supply_cut_prob=supply_cut
        )
        results.append({
            'Supplier': supplier['Supplier(s)'], 'Sourcing %': supplier['Sourcing %'],
            'Total Landed Cost ($)': impacted_cost, 'Total Lead Time (days)': base_avg_lead_time + logistics_delay,
            'Stockout Risk (%)': (1 - service_level) * 100
        })
    df_results = pd.DataFrame(results)
    return {
        'Strategy': strategy_df['Strategy'].iloc[0],
        'Weighted Avg Cost ($)': np.average(df_results['Total Landed Cost ($)'], weights=df_results['Sourcing %']),
        'Weighted Avg Lead Time (days)': np.average(df_results['Total Lead Time (days)'], weights=df_results['Sourcing %']),
        'Weighted Avg Stockout Risk (%)': np.average(df_results['Stockout Risk (%)'], weights=df_results['Sourcing %'])
    }

# ==============================================================================
# Sidebar for User Inputs
# ==============================================================================
st.sidebar.header("Disruption Scenario Inputs")
scenario_name = st.sidebar.text_input("Scenario Name", "Taiwan Geopolitical Crisis")
country_to_disrupt = st.sidebar.selectbox("Select Country to Disrupt", ["Taiwan", "South Korea", "Malaysia", "USA"])
tariff_percent = st.sidebar.slider("Tariff Increase (%)", 0, 100, 20, 5)
supply_cut_percent = st.sidebar.slider("Supply Cut Probability (%)", 0, 100, 50, 5)
logistics_delay_days = st.sidebar.slider("Logistics Delay (days)", 0, 60, 14, 1)

# Create the scenario dictionary from the UI inputs
interactive_scenario = {
    'name': scenario_name, 'country': country_to_disrupt,
    'tariff_increase': tariff_percent / 100.0, 'supply_cut': supply_cut_percent / 100.0,
    'logistics_delay': logistics_delay_days
}

# ==============================================================================
# Main Panel for Displaying Results
# ==============================================================================
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running Monte Carlo simulations... This may take a moment."):
        baseline_results = run_full_simulation(baseline_strategy_df, interactive_scenario)
        resilient_results = run_full_simulation(resilient_strategy_df, interactive_scenario)
        final_results_df = pd.DataFrame([baseline_results, resilient_results])

    st.header("Simulation Results")
    st.subheader("Executive Summary: Scenario Impact Comparison")
    st.dataframe(final_results_df.round(2))

    # --- Visualization ---
    st.subheader("Visual Comparison of Strategies")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Cost
    sns.barplot(data=final_results_df, x='Strategy', y='Weighted Avg Cost ($)', ax=axes[0], palette='Reds_r')
    axes[0].set_title('Total Landed Cost per Unit')
    
    # Plot 2: Lead Time
    sns.barplot(data=final_results_df, x='Strategy', y='Weighted Avg Lead Time (days)', ax=axes[1], palette='Blues_r')
    axes[1].set_title('Weighted Average Lead Time')
    
    # Plot 3: Stockout Risk
    sns.barplot(data=final_results_df, x='Strategy', y='Weighted Avg Stockout Risk (%)', ax=axes[2], palette='Greens_r')
    axes[2].set_title('Probability of Stockout')
    
    plt.tight_layout()
    st.pyplot(fig)

    # --- Recommendations ---
    st.header("Actionable Recommendations")
    baseline_risk = final_results_df.loc[0, 'Weighted Avg Stockout Risk (%)']
    resilient_risk = final_results_df.loc[1, 'Weighted Avg Stockout Risk (%)']
    cost_increase = ((final_results_df.loc[1, 'Weighted Avg Cost ($)'] - final_results_df.loc[0, 'Weighted Avg Cost ($)']) / final_results_df.loc[0, 'Weighted Avg Cost ($)']) * 100
    
    st.markdown(f"""
    - **Baseline Strategy Vulnerability:** The single-source strategy faces a **{baseline_risk:.2f}%** probability of stockout under this scenario, which is a critical risk.
    - **Resilience Strategy Effectiveness:** By dual-sourcing, the stockout risk is dramatically lowered to **{resilient_risk:.2f}%**.
    - **The Cost of Insurance:** Achieving this resilience requires a **{cost_increase:.2f}%** increase in the weighted average cost per unit. This is the premium paid to avoid costly line-down events.
    
    **Conclusion:** The simulation strongly supports diversifying the supply chain for the Mainboard Chipset to mitigate significant geopolitical risks, even at a higher unit cost.
    """)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")