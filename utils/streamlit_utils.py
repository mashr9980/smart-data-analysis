import streamlit as st

def set_streamlit_page_config():
    """
    Configure Streamlit page settings for improved appearance.
    """
    st.set_page_config(
        page_title="Smart Data Analysis with LLaMA",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS to improve visualization layout
    st.markdown("""
        <style>
        .css-1r6slb0 {
            max-width: 900px;
            margin: auto;
        }
        .stPlotlyChart, .stplot {
            margin: auto;
            display: block;
            max-width: 700px;
        }
        .st-emotion-cache-13ln4jf {
            max-width: 900px;
            margin: auto;
        }
        .main .block-container {
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            text-align: center;
        }
        h3 {
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)