import os
import sys
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

directory_path = r'/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/GrowingSpheres'
sys.path.append(directory_path)

import growingspheres
from growingspheres import counterfactuals as cf

df = pd.read_csv('/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/df4_clean.csv', index_col=0)

X_test = pd.read_csv('/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/Test_set.csv', index_col=0)

file_path = '/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/lgb.pkl'
#file_path = 'xgb.pkl'

# Load the object from the pickle file
with open(file_path, 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_test)

'''generating CF'''

gs_cf = []
indices_gs_cf = []  # To store original indices for gs_cf

start_index = 0
end_index = len(X_test)
#end_index = 2

for i in range(start_index, end_index):

    instance = X_test.iloc[i].values.reshape(1, -1)

    CF = cf.CounterfactualExplanation(instance, model.predict, method='GS')
    CF.fit(n_in_layer=100, first_radius=1, dicrease_radius=1.2, sparse=True, verbose=True)
    cf_x = np.array(CF.enemy)
    print('cf_x shape: ', cf_x.shape)

    gs_cf.append(cf_x)
    num_rows = cf_x.shape[0]
    indices_gs_cf.extend([X_test.index[i]] * num_rows)  # Save the X_test index associated with gs_cf repeated as many times as the CFs found
    print(len(gs_cf)) # To see progress...

# Concatenate
gs_cf = np.concatenate(gs_cf, axis=0)

# Convert to np array
gs_cf = np.array(gs_cf)

# Reshape 
gs_cf = gs_cf.reshape(-1, gs_cf.shape[-1])

df_cf = pd.DataFrame(gs_cf, columns=X_test.columns, index=indices_gs_cf)

# Filter out zeros rows (not found CFs)
df_cf = df_cf[(df_cf != 0).any(axis=1)]

X_test_repeated = X_test.loc[list(df_cf.index)]
X_test_np = np.array(X_test_repeated)

# Validity of CFs generated

y_pred_instances = model.predict(X_test_repeated)
y_pred_cf = model.predict(df_cf)

differing_predictions = sum(y_pred_instances != y_pred_cf)

# Calculate the total number of predictions
total_predictions = len(y_pred_instances)

# Calculate the percentage of differing predictions
percentage_differing = (differing_predictions / total_predictions) * 100

print(f"Percentage of differing predictions: {percentage_differing:.2f}%")


file_path = 'explanations_GS.csv'
# file_path = 'explanations_GS_xgb.csv'
df_cf.to_csv(file_path, mode='w', index=True)

