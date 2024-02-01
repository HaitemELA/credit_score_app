# utils/shap_analysis.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from helper_functions import load_model

#def perform_feature_selection(data, sorted_feature_importance, shap_values):
#    # Placeholder function, replace with your actual feature selection logic
#    # You might want to select features based on some criteria using the provided inputs.
#    # For demonstration purposes, this function returns the top N features.
#    num_features_to_select = 5
#    selected_features = sorted_feature_importance['Feature'].head(num_features_to_select).tolist()
#
#    return selected_features

def plot_and_save_summary(shap_values, features, class_label, output_dir='shap_plots'):
    os.makedirs(output_dir, exist_ok=True)

    shap.summary_plot(shap_values, features, plot_type='bar', show=False, color_bar=False, max_display=15)
    plt.title(f'SHAP Values Summary - Class {class_label}', fontsize=16)

    summary_path = f'shap_summary_class_{class_label}.png'
    plt.savefig(os.path.join(output_dir, summary_path))
    mlflow.log_artifact(summary_path)
    plt.show()

def shap_explainer(features):
    # SHAP values
    explainer= shap.TreeExplainer(load_model()[1])
    shap_values = explainer.shap_values(features)

    # Get the feature importance based on SHAP values
    #sorted_feature_importance = get_sorted_feature_importance(explainer, features.iloc[:num_instances, :])

    # Plot and log SHAP summary plots
    #plot_and_save_summary(shap_values, features.iloc[:num_instances, :], class_label=0)

    return shap_values #, sorted_feature_importance

def get_sorted_feature_importance(explainer, features):
    # Calculate SHAP values
    shap_values = explainer.shap_values(features)

    # Calculate mean absolute SHAP values for each feature
    mean_absolute_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame for feature importance
    sorted_feature_importance = pd.DataFrame({'Feature': features.columns, 'Importance': mean_absolute_shap_values})
    sorted_feature_importance = sorted_feature_importance.sort_values(by='Importance', ascending=False)

    return sorted_feature_importance
