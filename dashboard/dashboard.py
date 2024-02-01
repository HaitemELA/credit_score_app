import streamlit as st
import streamlit.components.v1 as components
from IPython import display
import numpy as np
from functools import lru_cache
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import time
import base64
import pandas as pd
import pickle
import tkinter
import mplleaflet
import logging
import os


#from streamlit_shap import st_shap
import shap
#from shap.plots import waterfall

from io import BytesIO
import matplotlib.pyplot as plt

import sys

sys.path.append(r'C:\Users\Imtech\Desktop\DATA_SCIENTIST\PORJET_7\project\utils')
from helper_functions import Plot_waterfall

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

@st.cache_data
def horizontal_table(data):
    # Create a horizontal table using Plotly
    fig = px.bar(data, orientation='h')

    # Convert the Plotly figure to HTML
    table_html = fig.to_html(full_html=False)

    # Use Streamlit components to display the HTML
    components.html(table_html, height=400)

@st.cache_data
def call_explainer():
    return Plot_waterfall()

@st.cache_data
def slider():
    top_features_options = list(range(1, 11))
    return st.selectbox("Select top features (1-10)", options=top_features_options, index=4)



def main():

    st.set_page_config(
    page_title="Your App Title",
    page_icon="📊",
    layout="wide"
    )

    exp_0, exp_1 = Plot_waterfall()

        # Get client ID from the user
    with st.sidebar:
        SK_ID_CURR = st.text_input("Enter loan request ID:")
        # Convert the input to numpy.int64
        try:
            SK_ID_CURR = np.int64(SK_ID_CURR)
        except ValueError:
            st.sidebar.warning("Invalid input. Please enter a valid integer for the loan request ID.")
            st.stop()  # Stop further execution if input is invalid

    age = st.slider('How old are you?', 0, 130, 25)
    st.write("I'm ", age, 'years old')




if __name__ == "__main__":
    main()