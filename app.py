"""MLperformance - entry point. Run with:  streamlit run app.py"""

import streamlit as st

from mlperformance import __version__

st.set_page_config(page_title="MLperformance", page_icon="📈", layout="wide")

train = st.Page("app_pages/train.py", title="New experiment", icon="🧪", default=True)
results = st.Page("app_pages/results.py", title="Results", icon="📊")

st.sidebar.markdown(f"**MLperformance** v{__version__}")
st.sidebar.caption("Find the best machine learning algorithm for your data.")
st.navigation([train, results]).run()
