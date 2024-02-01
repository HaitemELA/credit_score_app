# api_app.py

from flask import Flask, request, jsonify
import pandas as pd
import json
import pickle
import numpy as np
import sys
import base64
import ast
import gc



sys.path.append(r'C:\Users\Imtech\Desktop\DATA_SCIENTIST\PORJET_7\project\utils')
from helper_functions import load_model, load_data, load_raw_data, features_prediction, shap_explainer, load_train_data

app = Flask(__name__)

model = load_model()[0]  # Implement load_model based on your requirements
data = load_data()  # Implement load_data based on your requirements
raw_data = load_raw_data()
train_df = load_train_data()[0]
KmeansModel = load_train_data()[1]
#data = pd.DataFrame(data, index=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        rqst = request.get_json(force=True)

        ## Extract SK_ID_CURR from the JSON request
        SK_ID_CURR = np.int64(rqst['data']['SK_ID_CURR'])

        client_features, client_decision, client_probability, global_features, idx = features_prediction(data, model, SK_ID_CURR)
        # Encode SHAP values to base64
        global_shap_values, local_shap_values, feature_names = shap_explainer(client_features, global_features, model, idx, client_decision)

        serialized_global_shap_values = pickle.dumps(global_shap_values)
        base64_global_shap_values = base64.b64encode(serialized_global_shap_values).decode('utf-8')

        serialized_local_shap_values = pickle.dumps(local_shap_values)
        base64_local_shap_values = base64.b64encode(serialized_local_shap_values).decode('utf-8')

        print('clustering')
        print('data', data.loc[np.where(data['SK_ID_CURR'] == SK_ID_CURR)].drop('SK_ID_CURR', axis = 1))
        # Cluster
        cluster = KmeansModel.predict(data.loc[np.where(data['SK_ID_CURR'] == SK_ID_CURR)].drop('SK_ID_CURR', axis = 1))[0]
        print('cluster', str(cluster), type(str(cluster)))

        # Return the prediction as JSON
        response = {'decision': client_decision,
                    'probability': [client_probability[0], client_probability[1]],
                    'cluster': str(cluster),
                    'global_shap_values': base64_global_shap_values,
                    'local_shap_values': base64_local_shap_values,
                    'feature_names': feature_names,
                    'idx': idx                    
                    }
        #print('response[cluster]', response['cluster'], type(response['cluster']))
        #print('response[probability]', response['probability'], type(response['probability']))
        #print('response[cluster]', response[cluster], type(response[cluster]))
        #print('response', response)
        #gc.collect()

        return jsonify(response)

    except Exception as e:
        print('predict error: ' + str(e))
        gc.collect()
        return jsonify({'error': str(e)})

## Sample endpoint for getting SHAP values
@app.route('/shap_values', methods=['POST'])
def get_shap_values():
    try:
        # Get input data from the request
        rqst = request.get_json(force=True)
        SK_ID_CURR = np.int64(rqst['data']['SK_ID_CURR'])

        # Extract features for the given client
        client_data = data.loc[data['SK_ID_CURR'] == SK_ID_CURR].drop('SK_ID_CURR', axis=1)
        features = client_data.values[0]

        # Get SHAP values for the client
        shap_values = shap_explainer.shap_values(features)
        gc.collect()

        # Return the SHAP values as JSON
        return jsonify({'shap_values': shap_values.tolist()})
    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)})
#
    # Fetch client DATA
@app.route('/client', methods=['POST'])
def client_info():
    try:
        # Get input data from the request
        rqst = request.get_json(force=True)
        SK_ID_CURR = np.int64(rqst['data']['SK_ID_CURR'])

        # Extract client info
        client_data = raw_data.loc[raw_data['SK_ID_CURR'] == SK_ID_CURR].drop('SK_ID_CURR', axis=1)

        response = {
            'Gender': 'Man' if str(client_data.CODE_GENDER.values[0]) == 'M' else 'Woman',  # Convert to string
            'Pronoun': 'he' if str(client_data.CODE_GENDER.values[0]) == 'M' else 'she',  # Convert to string
            'Age': str(-int(client_data.DAYS_BIRTH.values[0] / 365.25)) + ' years old',  # Convert to years and integer
            'Family Status': str(client_data.NAME_FAMILY_STATUS.values[0]),  # Convert to string
            'Housing Type': str(client_data.NAME_HOUSING_TYPE.values[0]),  # Convert to string
            'Income Total': "{:} €".format(int(client_data.AMT_INCOME_TOTAL.values[0])),  # Format as currency
            'Income Type': str(client_data.NAME_INCOME_TYPE.values[0]),  # Convert to string
            'Employed since': str(-int(client_data.DAYS_EMPLOYED.values[0] / 365.25))  + ' years',  # Convert to years and integer
            'Occupation Type': str(client_data.OCCUPATION_TYPE.values[0]),  # Convert to string
            'Credit Amount': "{: } €".format(int(client_data.AMT_CREDIT.values[0])),  # Format as currency
            'Annuity Amount': "{: } €".format(int(client_data.AMT_ANNUITY.values[0]))  # Format as currency
        }

        gc.collect()

        return jsonify(response)

    except Exception as e:
        gc.collect()
        return jsonify({'error in client_info': str(e)})


@app.route('/all_clients', methods=['POST'])
def get_clients_data():
    try:
        # Get input data from the request
        rqst = request.get_json(force=True)
        features = rqst['data']['Features']

        #features_list = ['TARGET'] + ast.literal_eval(features)

        features_list = ast.literal_eval(features)
        print(features_list)


        missing_features = [feature for feature in features_list if feature not in train_df.columns]

        if missing_features:
            print(f"The following features are missing in train_df: {missing_features}")
        else:
            print("All features are present in train_df.")

        # Assuming train_df[features] is a pandas DataFrame
        clients_data = train_df.loc[:, features_list].to_dict(orient='records')
        response = {'clients_data': clients_data}

        gc.collect()

        return jsonify(response)

    except Exception as e:
        gc.collect()
        print('error get_clients_data', str(e))
        return jsonify({'error get_clients_data': str(e)})


## Sample endpoint for getting selected features
#@app.route('/selected_features', methods=['POST'])
#def get_selected_features():
#    try:
#        json_data = request.get_json()
#
#        # Check if required keys are present
#        if 'sorted_feature_importance' not in json_data or 'shap_values' not in json_data:
#            raise ValueError("Missing required keys in JSON data.")
#
#        sorted_feature_importance = json_data['sorted_feature_importance']
#        shap_values = json_data['shap_values']
#
#        # Perform feature selection
#        selected_features = perform_feature_selection(data, sorted_feature_importance, shap_values)
#
#        return jsonify({'selected_features': selected_features})
#
#    except Exception as e:
#        return jsonify({'error': str(e)})


if __name__ == '__main__':
    gc.collect()
    app.run(host='localhost', port=8000, debug=True)

