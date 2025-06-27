# platform.py

import streamlit as st

st.set_page_config(
    page_title="Strategic Simulation Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Welcome to the Strategic Simulation Platform")
st.markdown("""
    This is your central hub for data-driven strategic planning. Use the navigation sidebar on the left to select a simulation module.
""")

st.subheader("Available Modules:")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 🔮 Universal Business Simulator")
        st.info("""
            A flexible, domain-agnostic platform. Define your own variables and formulas to model any business problem—from finance and marketing to HR and operations.
        """)

with col2:
    with st.container(border=True):
        st.markdown("#### 🛡️ Supply Chain Risk Analyzer")
        st.info("""
            A purpose-built application to quantify geopolitical risks for your critical components. Compare sourcing strategies and build a robust business continuity plan.
        """)

st.sidebar.success("Select a simulation module above.")