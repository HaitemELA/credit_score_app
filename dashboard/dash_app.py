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
from plotly.subplots import make_subplots

import ast

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

# Initialize session_state attributes
if "top_features" not in st.session_state:
    st.session_state.top_features = None
if "fig" not in st.session_state:
    st.session_state.fig = None

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)
    
@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def horizontal_table(data):
    # Create a horizontal table using Plotly
    fig = px.bar(data, orientation='h')

    # Convert the Plotly figure to HTML
    table_html = fig.to_html(full_html=False)

    # Use Streamlit components to display the HTML
    components.html(table_html, height=400)

#@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def waterfall_process(decision, cluster, idx, max_display):
    st.write('waterfall_process')
    exp_0, exp_1, feature_importance, features_test_clients = Plot_waterfall()
    top_features_names = feature_importance.head(30).col_name.tolist()
    sorted_client_features_values = features_test_clients[idx][:30]
    sorted_client_features_values = sorted_client_features_values.reshape(1, -1)
    sorted_client_features_values = sorted_client_features_values[0].tolist()
    sorted_client_features_values.insert(0, int(cluster))
    sorted_client_features_values.insert(0, int(decision))

    sorted_client_features_values = np.array(sorted_client_features_values).reshape(1, -1)



    if decision == 0:
        exp = exp_0
    elif decision == 1:
        exp = exp_1
    else:
        print('failure to charge the explainer')
        
    return shap.plots.waterfall(exp[idx], max_display=max_display), top_features_names, sorted_client_features_values



@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def run_api(SK_ID_CURR):
    api_url = "http://localhost:8000/predict"  # Update with the correct port
    json_data = json.dumps({"data": {"SK_ID_CURR": str(SK_ID_CURR)}})
    return requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def get_clients_data(selected_features):
    api_url = "http://localhost:8000/all_clients"  # Get all clients data
    selected_features= ['TARGET', 'Cluster'] + selected_features

    json_data = json.dumps({"data": {"Features": str(selected_features)}})
    response = requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        print('***********************************')
        print('response_data', type(response_data))
        print('***********************************')


        # Access the 'clients_data' key in the response
        df_clients = pd.DataFrame(response_data['clients_data'])
        df_clients = df_clients.reindex(columns=selected_features)

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}\n{response.text}")

    return df_clients

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def fetch_client_info( SK_ID_CURR):
    try:
        # Call the API to get the Scores
        api_url = "http://localhost:8000/client"  # Update with the correct port
        json_data = json.dumps({"data": {"SK_ID_CURR": str(SK_ID_CURR)}})
        response = requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            # Access the JSON content of the response
            json_response = response.json()

            # Create a dictionary to hold the data
            client_data = {
                'Gender': json_response['Gender'],
                'Pronoun': json_response['Pronoun'],
                'Age': json_response['Age'],
                'Family Status': json_response['Family Status'],
                'Housing Type': json_response['Housing Type'],
                'Income Total': json_response['Income Total'],
                'Income Type': json_response['Income Type'],
                'Employed since': json_response['Employed since'],
                'Occupation Type': json_response['Occupation Type'],
                'Credit Amount': json_response['Credit Amount'],
                'Annuity Amount': json_response['Annuity Amount']
            }

            return client_data

    except Exception as e:
        return print('error in fetch_client_info', str(e))

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def global_feature_importance_bar_chart(sorted_features):
    st.write("Global Feature Importance:")
        # Create a bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=sorted_features['Global Importance'],
        y=sorted_features['Feature'],
        orientation='h'
    ))

    fig.update_layout(
        title_text="Top 10 Global Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        width=600,
    )

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def create_credit_score_gauge(predicted_prob):
    try:
        # Convert probability to a score between 0 and 100
        credit_score = 100 - int(predicted_prob[0] * 100)
        Threshold = 37.37

        # Define labels for likely to pay or not
        pay_labels = ["Not Likely to Pay", "Likely to Pay"]
        pay_label = pay_labels[int(credit_score >= Threshold)]

        # Define ranking labels with four levels
        ranking_labels = ["Needs work", "Needs some work", "Good", "Excellent"]
        ranking = ranking_labels[min(3, credit_score // 25)]  # Adjusted based on four levels

        # Create a gauge chart
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 0.5], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickmode': 'array', 'tickvals': [0, 25, 50, 75, 100]},
                'bar': {'color': 'purple'},
                'steps': [
                    {'range': [0, 25], 'color': '#fafa6e'},
                    {'range': [25, 50], 'color': '#59c187'},
                    {'range': [50, 75], 'color': '#208f8b'},
                    {'range': [75, 100], 'color': '#053061'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': Threshold,
                }
            }
        ))

        # Update the plot with the final title
        fig.update_layout(
            title_text=f"Credit Score",
            title_x=0.2,
            title_y=1,
            title_font_size=24,  # Increased font size
            width=1000,  # Increased width
            height=300,  # Increased height
            margin=dict(l=10, r=10, b=10, t=10)  # Adjusted margin
        )

        for i in range(0, credit_score + 1, 2):
            fig.update_traces(value=i)
            time.sleep(0.01)

        # Add ranking and likelihood annotations in the middle of the plot
        annotation_text = f"<b>{ranking}</b><br>{pay_label}"
        fig.add_annotation(
            x=0.185,
            y=-0.04,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=16),
        )

        return fig


    except Exception as e:
        return print('error in create_credit_score_gauge', str(e))

#@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def shap_values_plot(shap_values, top_features=3):
    # If SHAP values are one-dimensional, no need to index
    if len(shap_values[0].shape) == 1:
        top_shap_values = shap_values
    else:
        # Concatenate SHAP values for all instances
        shap_values_concatenated = np.concatenate(shap_values, axis=0)
        
        # Calculate the absolute sum of SHAP values for each feature
        feature_importance = np.abs(shap_values_concatenated).mean(axis=0)
        
        # Get the indices of the top features
        top_feature_indices = np.argsort(feature_importance)[::-1][:top_features]
        
        # Extract the SHAP values and features for the top features
        top_shap_values = [shap_set[:, top_feature_indices] for shap_set in shap_values]

    # Create a bar plot for each set of top SHAP values
    for i, shap_set in enumerate(top_shap_values):
        shap_fig = go.Figure(go.Bar(x=shap_set, orientation='h'))
        shap_fig.update_layout(title_text=f'Top {top_features} Reasons - Set {i + 1}',
                               title_x=0.5,
                               xaxis_title='SHAP Values',
                               yaxis_title='Features',
                               height=400,
                               width=600,
                               xaxis=dict(title_text='SHAP Values', tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=[-1, -0.5, 0, 0.5, 1]),
                               )

        # Display the plot using st.plotly_chart()
        st.plotly_chart(shap_fig)

#@st.cache_data()
def distri_plot(merged_df, feature):
    # Plot position of the client's value
    fig_value_position = go.Figure()
    # Add histogram trace for the selected feature
    fig_value_position.add_trace(go.Histogram(x=merged_df[merged_df['TARGET'] == 0][feature], name=feature + ' destribution on class 0', nbinsx=100, histnorm='percent'))
    fig_value_position.add_trace(go.Histogram(x=merged_df[merged_df['TARGET'] == 1][feature], name=feature + ' destribution on class 1', nbinsx=100, histnorm='percent'))
    
    # Add vertical line for the client's value position
    client_value_position = merged_df[feature].iloc[-1]

    fig_value_position.add_shape(go.layout.Shape(type='line', x0=client_value_position, x1=client_value_position,
            y0=0, y1=1, yref='paper', line=dict(color='red', width=2, dash='dot'), name=f'Client {feature}'))

    # Add text annotation to describe the vertical line
    fig_value_position.add_annotation(go.layout.Annotation(text=f'Client {feature} Value', x=client_value_position,
            y=1.02, xref='x', yref='paper', showarrow=False, font=dict(size=10),))
   
    fig_value_position.update_layout(title_text='Distribution with Client\'s Value Position', 
                                     title=dict(x=0.15),  legend=dict(x=0, y=-0.2, orientation="h"))
    
    return fig_value_position


def main():
    st.set_page_config(
    page_title="P7_Client_Dashboard",
    page_icon="📊",
    layout="wide"
    )

    st.markdown("<h1 style='text-align: center;'>Client Dashboard</h1>", unsafe_allow_html=True)  

    # Get client ID from the user
    with st.sidebar:
        SK_ID_CURR = st.text_input("Enter loan request ID:")
        # Convert the input to numpy.int64
        try:
            SK_ID_CURR = np.int64(SK_ID_CURR)
        except ValueError:
            st.sidebar.warning("Invalid input. Please enter a valid integer for the loan request ID.")
            st.stop()  # Stop further execution if input is invalid

        # # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 2])

    # Call the API to get the Scores
    response = run_api(SK_ID_CURR)

    if response.status_code == 200:
        ## Fetch client Info
        client_data = fetch_client_info(SK_ID_CURR)
        st.session_state.client_data = client_data 

        with col1:
            container1 = st.container(border=True)
            container1.write("### Client Information")
            container1.write("**The client is a** {}, {} and {} is {}.".format(client_data['Gender'], client_data['Age'], client_data['Pronoun'], client_data['Family Status']))
            container1.write("**Occupation:** {} is a {}, {} since {}.".format(client_data['Pronoun'], client_data['Occupation Type'], client_data['Income Type'], client_data['Employed since']))

#   
        result = response.json()
        probability = result['probability']

        ## Display performance graph
        with col3:
            st.plotly_chart(create_credit_score_gauge(probability))        

        ## Create tabs
        tabs = st.tabs(["Client Features importance", "Features distribution", "Additional Client Information"])

        with tabs[0]:
            # Fetch client Info           
            st.session_state.top_features  = st.slider("Select top features (1-10)", min_value=1, max_value=10, value=5)
            cluster = np.int64(result["cluster"])
            decision = np.int64(result["decision"])
            idx = result['idx']#  
            ##waterfall_plot
            waterfall_plot, top_features_names, sorted_client_features_values = waterfall_process(decision, cluster, idx, st.session_state.top_features)
            st.pyplot(waterfall_plot)

        with tabs[1]:
            st.session_state.df_clients = get_clients_data(top_features_names)
            selected_features = st.multiselect('Select features to analyse', top_features_names)
             # Merge the two DataFrames
            features_client_df = pd.DataFrame(sorted_client_features_values, columns=st.session_state.df_clients.columns.tolist())
            merged_df = pd.concat([st.session_state.df_clients, features_client_df])
            merged_df['Cluster'] = merged_df['Cluster'].astype(int)

            for feature in selected_features:
                col11, col12 = st.columns([1, 1])
                with col11:
                    fig_value_position = distri_plot(merged_df, feature)
                    st.plotly_chart(fig_value_position)
                with col12:
                    fig_value_position = distri_plot(merged_df.iloc[np.where(merged_df.Cluster == cluster)[0]], feature)
                    st.plotly_chart(fig_value_position)
        with tabs[2]:
                client_data = pd.DataFrame({
                    "Details": ['Details'],
                    "Housing Type": [client_data['Housing Type']],
                    "Income Total": [client_data['Income Total']],
                    "Credit Amount": [client_data['Credit Amount']],
                    "Annuity Amount": [client_data['Annuity Amount']]
                })

                # Set "Housing Type" as the index
                client_data.set_index("Details", inplace=True)

                st.table(client_data)

    else:
        st.write(f"Error: Unable to get Scores from the API.")

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def pie_chart(top_features, SK_ID_CURR):
    # Display the top 3 features
    print("Top 3 features influencing the prediction:")
    st.write(top_features)

    # Create a pie chart using Plotly
    fig = px.pie(top_features, names='Feature', values='Importance',
                 title=f'Top 3 Features for Client {SK_ID_CURR}')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()