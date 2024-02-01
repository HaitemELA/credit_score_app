# utils/helper_functions.py
import pandas as pd
import numpy as np
import pickle
import shap
from shap import TreeExplainer, Explanation
from sklearn.cluster import KMeans
import gc

def load_model(model_path='../model/best_model_Custom_F1_Score.pkl'):
    """
    Load the trained model from a given file path.

    Parameters:
    - model_path: str, path to the trained model file.

    Returns:
    - model: object, the loaded machine learning model.
    """
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        model, booster = pickle.load(model_file)
    gc.collect()
    
    return model, booster

def load_data(data_path='../DB/test_df_cleaned_sample.csv'):
    data = pd.read_csv(data_path, index_col='ID')
    gc.collect()

    return data

def load_raw_data(data_path='../DB/application_test.csv'):
    data_raw = pd.read_csv(data_path).fillna('')
    gc.collect()

    return data_raw

def load_train_data(data_path='../DB/train_df_cleaned.csv'):
    train_df = pd.read_csv(data_path).sample(n=3000)
    train_df, kmeansModel = Kmeans_add_cluster_column(train_df)
    gc.collect()

    return train_df, kmeansModel



def features_prediction(data, model_lgbm, SK_ID_CURR):
    print('SK_ID_CURR', type(SK_ID_CURR))
    try:
        ## Ensure the input features are in the correct order
        feature_order = model_lgbm.feature_name_

        ## Extract features from Data csv
        client_data = data.loc[np.where(data['SK_ID_CURR'] == SK_ID_CURR)].drop('SK_ID_CURR', axis = 1)
        idx = data.loc[np.where(data['SK_ID_CURR'] == SK_ID_CURR)].index.tolist()[0]
        local_features = [client_data[feature] for feature in feature_order]
        local_features = np.array(local_features).reshape(1, -1)
        
        # Decision
        client_decision = model_lgbm.predict(local_features)[0]
        # prediction probability
        client_probability = model_lgbm.predict_proba(local_features.reshape(1, -1))[0]

        # Global Features
        global_features = data.drop('SK_ID_CURR', axis = 1)
        #print('global_features', global_features)
        print('features_prediction3')

        gc.collect()

        return local_features, client_decision, client_probability, global_features, idx
    
    except Exception as e:
        gc.collect()
    
        print('error features_prediction' + str(e))
        return {'error': str(e)}
    

def shap_explainer(local_features, global_features, model_lgbm, idx, client_decision):
    try:
        # Load explainer
        explainer= shap.TreeExplainer(model_lgbm)

        # Global shape values
        global_shap_values = explainer.shap_values(global_features)
        mean_absolute_global_shap_values = np.abs(global_shap_values).mean(axis=1)

        # Local shap values
        local_shap_values = explainer.shap_values(local_features)

        # Plot and log SHAP summary plots
        feature_names = model_lgbm.feature_name_
        gc.collect()

        return global_shap_values, local_shap_values, feature_names, #Plot_waterfall_2(model_lgbm, global_features, idx, client_decision)

    except Exception as e:
        print('error shap_explainer:', str(e))
        gc.collect()

        return {'error': str(e)}
def Plot_waterfall():
    try:

        model_lgbm = load_model()[0]
        data = load_data().drop('SK_ID_CURR', axis=1)
        explainer = TreeExplainer(model_lgbm)
        # Calculate SHAP values for the selected instance
        shap_values = explainer.shap_values(data)

        mean_absolute_global_shap_values = np.abs(shap_values[0]).mean(0)
        feature_importance = pd.DataFrame({'col_name': data.columns.tolist(), 'feature_importance_vals': mean_absolute_global_shap_values.tolist()})
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

        # Create an Explanation object
        exp_0 = Explanation(
            values=shap_values[0],  #
            base_values=explainer.expected_value[0],
            data=data.values,
            feature_names=data.columns
        )

        exp_1 = Explanation(
            values=shap_values[1],  #
            base_values=explainer.expected_value[1],
            data=data.values,
            feature_names=data.columns
        )

        gc.collect()

        #return waterfall(exp[idx], max_display=max_features)
        return exp_0, exp_1, feature_importance, data[feature_importance.col_name.tolist()].values

    except Exception as e:
        gc.collect()

        print('Error in Plot_waterfall:', str(e))
        return {'error': str(e)}
    


def convert_numpy_to_lists(data):
    if isinstance(data, np.ndarray):
        gc.collect()

        return data.tolist()
    elif isinstance(data, list):
        gc.collect()

        return [convert_numpy_to_lists(item) for item in data]
    elif isinstance(data, dict):
        gc.collect()

        return {key: convert_numpy_to_lists(value) for key, value in data.items()}
    else:
        gc.collect()

        return data
    

def convert_lists_to_numpy(data):
    if isinstance(data, list):
        gc.collect()

        return np.array([convert_lists_to_numpy(item) for item in data])
    elif isinstance(data, dict):
        gc.collect()

        return {key: convert_lists_to_numpy(value) for key, value in data.items()}
    else:
        gc.collect()

        return data

def Kmeans_add_cluster_column(input_dataframe):
    """
    Method 1: Input dataframe - Output dataframe with the 'Cluster' column added to the end

    Parameters:
    - input_dataframe: The input dataframe
    - optimal_k: The optimal number of clusters

    Returns:
    - DataFrame: A new dataframe with the 'Cluster' column added
    """

    # Select features for clustering
    X = input_dataframe[input_dataframe.drop(columns=['SK_ID_CURR', 'TARGET', 'Unnamed: 0']).columns]


    # Determine the optimal number of clusters using the Elbow Method and second derivative
    inertias = []
    for k in range(3, 10):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    print('000000000000start00000000')
    
    # Calculate the second derivative
    #second_derivative = np.diff(np.diff(inertias))
    # Calculate the curvature (second derivative)
    curvature = np.diff(inertias, 2)
    
    # Find the index where the second derivative is maximized
    #optimal_k = np.argmax(second_derivative) + 1  # Adding 1 as we are skipping the first element in the second derivative
    optimal_k = np.argmax(curvature) + 3  # Adding 1 as we are skipping the first element in the second derivative
    print(optimal_k)

    # Fit KMeans with the optimal number of clusters
    kmeans_optimal = KMeans(n_clusters=optimal_k, n_init='auto')
    input_dataframe['Cluster'] = kmeans_optimal.fit_predict(X)
    input_dataframe['Cluster'] = input_dataframe['Cluster'].astype(int)

    print(optimal_k)
    print(input_dataframe['Cluster'].unique())
    print('000000000000finished00000000')

    del curvature, optimal_k, inertias, X
    gc.collect()

    return input_dataframe, kmeans_optimal

def find_similar_clients(input_dataframe, client_id, kmeans_optimal):
    """
    Method 2: Input the new dataframe and a client ID - Output the similar client dataframe

    Parameters:
    - input_dataframe: The dataframe with the 'Cluster' column
    - client_id: The client ID for finding similar clients

    Returns:
    - DataFrame: A new dataframe with clients similar to the input client
    """
    X = input_dataframe[input_dataframe.drop(columns=['SK_ID_CURR', 'Unnamed: 0', 'TARGET']).columns]
    input_client_cluster = input_dataframe[input_dataframe['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'Unnamed: 0', 'TARGET', 'Cluster'])

    input_dataframe['Cluster'] = int(kmeans_optimal.fit_predict(X))
    print(input_dataframe['Cluster'], "input_dataframe['Cluster']")

    # Select the input client's cluster
    print(int(input_client_cluster))
    # Select clients in the same cluster
    similar_clients = input_dataframe[input_dataframe['Cluster'] == int(input_client_cluster)]

    del X
    gc.collect()
    return similar_clients