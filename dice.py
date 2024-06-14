import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import json
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import dice_ml
from dice_ml.diverse_counterfactuals import _DiverseCFV1SchemaConstants, _DiverseCFV2SchemaConstants, json_converter, CounterfactualExamples

df = pd.read_csv('/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/df4_clean.csv', index_col=0)

X_test = pd.read_csv('/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/Test_set.csv', index_col=0)
y_test = pd.read_csv('/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/Train_set_y.csv', index_col=0)
# Import black box

file_path = '/Users/sam/Library/CloudStorage/OneDrive-Personale/Bank Customer Churn XAI Project/3) Iranian Churn/lgb.pkl'
#file_path = 'xgb.pkl'

# Load the object from the pickle file
with open(file_path, 'rb') as file:
    model = pickle.load(file)

"""DICE requires to pass continuous variables and categorical variables as lists"""

# cat_features_list = ['Gender', 'HasCrCard', 'IsActiveMember']
cat_features_list = ['Tariff Plan', 'Status']
target_variable = 'churn'
continuous_features = [col for col in df.columns if col not in cat_features_list and col != target_variable]

# DICE requires data in its specific format

d = dice_ml.Data(dataframe = df,
                 categorical_features = cat_features_list,
                 continuous_features = continuous_features,
                 outcome_name = 'churn')

# provide the trained black-box model to DiCE's model object
m = dice_ml.Model(model = model, backend = 'sklearn')

exp_random = dice_ml.Dice(d, m, method = "random")
exp_genetic = dice_ml.Dice(d, m, method = "genetic")

instances = X_test.copy()

# Set the default weight values
proximity_weight = 1.0
diversity_weight = 1.0
sparsity_weight = 1.0
categorical_penalty = 0.1

# Generate counterfactuals
dice_exp_genetic_many = exp_genetic.generate_counterfactuals(instances,
                                                          total_CFs=10,
                                                          desired_class='opposite',
                                                          sparsity_weight=sparsity_weight, #Larger this weight, less features are changed from the query_instance
                                                          proximity_weight=proximity_weight, #Higher values will prioritize CFs close to the query instance in terms of feature values
                                                          diversity_weight=diversity_weight, # # Higher values of diversity_weight will prioritize counterfactuals that are diverse from each other
                                                          categorical_penalty= categorical_penalty, # A weight to ensure that all levels of a categorical variable sums to 1
                                                          posthoc_sparsity_algorithm="linear", # Perform either linear or binary search, binary only when monotonic relationship and large values
                                                          verbose=False
                                                          )

# Load and access the counterfactual explanations
cf_json = dice_exp_genetic_many.to_json()
cf_dict = json.loads(cf_json)
cf_list = cf_dict['cfs_list']

# Process counterfactuals and categorize as found or not found
found_counterfactuals = [item for item in cf_list if isinstance(item, list)]
not_found_counterfactuals = [item for item in cf_list if item is None]

# Calculate the percentage of found counterfactuals
total_instances = len(instances)
total_found = len(found_counterfactuals)
counterfactuals_count = len(found_counterfactuals)
percentage_found = (counterfactuals_count / total_instances) * 100

print('counterfactuals found: ', counterfactuals_count)
print('percentage_found: ', percentage_found)

# repeat CFs index so that they can be matched with corresponding index in X_test
df_cf = pd.concat([
                    pd.DataFrame(np.array(cfs)[:,:-1],
                    index=[i,]*(len(cfs)),
                    columns=cf_dict['feature_names'])
            for i,cfs in zip(instances.index, cf_list) if isinstance(cfs, list)])

df_cf.head()

# Count occurrences of each index in the DataFrame
index_counts = df_cf.index.value_counts()

# Filter indexes repeated less than 10 times
less_than_10_times = index_counts[index_counts < 10]

# Count the number of indexes repeated less than 10 times
count_less_than_10 = len(less_than_10_times)

print(f"Number of indexes repeated less than 10 times: {count_less_than_10}")

file_path = 'explanations_DiCE_sparse.csv'
#file_path = 'explanations_DiCE_xgb.csv'

df_cf.to_csv(file_path, mode='w', index=True)

df_cf = pd.read_csv(file_path, index_col=0)


# Repeat indexes of X_test based on cf_df
X_test_aligned = X_test.loc[list(df_cf.index)]
X_test_aligned.shape

cf_array = np.array(df_cf)
X_test_np = np.array(X_test_aligned)

print(cf_array.shape)
print(X_test_np.shape)

y_test_aligned =y_test.loc[list(df_cf.index)]
y_test_np = np.array(y_test_aligned)

# Validity of CFs generated

y_pred_instances = model.predict(X_test_aligned)
y_pred_cf = model.predict(df_cf)

differing_predictions = sum(y_pred_instances != y_pred_cf)

# Calculate the total number of predictions
total_predictions = len(y_pred_instances)

# Calculate the percentage of differing predictions
percentage_differing = (differing_predictions / total_predictions) * 100

print(f"Percentage of differing predictions: {percentage_differing:.2f}%")
