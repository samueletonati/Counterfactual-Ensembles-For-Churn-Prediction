# %%
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import manhattan_distances
from scipy import spatial
import seaborn as sns
import math
import matplotlib.pyplot as plt

# %%
current_dir = os.getcwd()
current_dir

# %%

df = pd.read_csv(os.path.join(current_dir, 'df_clean.csv'), index_col=0)
X_test = pd.read_csv(os.path.join(current_dir, 'Test_set.csv'), index_col=0)

# LightGBM CFs
explanations_DiCE = pd.read_csv(current_dir + '/CF generation datasets/explanations_DiCE.csv', index_col=0)
explanations_GS = pd.read_csv(current_dir + '/CF generation datasets/explanations_GS.csv', index_col=0)
explanations_CP_ILS = pd.read_csv(current_dir + '/CF generation datasets/explanations_CP_ILS.csv', index_col=0).round(0)
explanations_CFRL = pd.read_csv(current_dir + '/CF generation datasets/explanations_CFRL.csv', index_col=0)


# %%
# Load the model
with open('lgb.pkl', 'rb') as file:
    model = pickle.load(file)

# %% [markdown]
# #### Checking Validity

# %%
exps = [explanations_DiCE, explanations_GS, explanations_CP_ILS, explanations_CFRL]
exp_names = ['DiCE', 'GS', 'CP_ILS', 'CFRL']

for exp, name in zip(exps, exp_names):
    # Get the subset of X_test that matches the index of the current explanation DataFrame
    X_test_e = X_test.loc[list(exp.index)]

    # Make predictions on the original instances and the counterfactual instances
    y_pred_instances = model.predict(X_test_e)
    y_pred_cf = model.predict(exp)

    # Calculate the number of differing predictions
    differing_predictions = sum(y_pred_instances != y_pred_cf)

    # Calculate the total number of predictions
    total_predictions = len(y_pred_instances)

    # Calculate the percentage of differing predictions
    percentage_differing = (differing_predictions / total_predictions) * 100

    # Print the result
    print(f"Percentage of differing predictions for {name}: {percentage_differing:.2f}%")


# %%
index_difference_count_DiCE = len(set(explanations_DiCE.index) ^ set(X_test.index))
index_difference_count_GS = len(set(explanations_GS.index) ^ set(X_test.index))
index_difference_count_CP_ILS = len(set(explanations_CP_ILS.index) ^ set(X_test.index))
index_difference_count_CFRL = len(set(explanations_CFRL.index) ^ set(X_test.index))
print("Number of differing indices between explanations_DiCE and X_test:", index_difference_count_DiCE)
print("Number of differing indices between explanations_GS and X_test:", index_difference_count_GS)
print("Number of differing indices between explanations_CP_ILS and X_test:", index_difference_count_CP_ILS)
print("Number of differing indices between explanations_DiCE and X_test:", index_difference_count_CFRL)
print('GS vs CP_ILS differing indices', len(set(explanations_GS.index) ^ set(explanations_CP_ILS.index)))


# %%
X_test.index.difference(explanations_CP_ILS.index)

# %%
counts = explanations_GS.index.value_counts()

xx=0

repeated_indices = counts[counts > xx].index.tolist()
count_idx = len(repeated_indices)
if repeated_indices:
    print("Repeated indices:", count_idx)
else:
    print("No index is repeated more than {xx} times.")

# %%
print(len(X_test))
print(len(explanations_DiCE))
print(len(explanations_GS))
print(len(explanations_CP_ILS))
print(len(explanations_CFRL))

# %%
# Create an empty DataFrame with the same columns as explanations_DiCE (since it has all the indices)
ensemble_df = pd.DataFrame(columns=explanations_DiCE.columns)

# Get the unique index values from explanations_DiCE
unique_indices_DiCE= explanations_DiCE.index.unique()

# Iterate over the unique index values from explanations_DiCE
for index_value in unique_indices_DiCE:
    # Extract rows with the current index value from both DataFrames
    rows_DiCE = explanations_DiCE.loc[explanations_DiCE.index == index_value].copy()
    
    # Append rows from the second DataFrame (all available rows)
    rows_DiCE['source'] = 'DiCE'
    ensemble_df = pd.concat([ensemble_df, rows_DiCE], axis=0)
    
    # Check if corresponding rows exist in explanations_DiCE
    if index_value in explanations_CFRL.index.unique():
        # Extract all rows from the second DataFrame for the same index value
        rows_CFRL = explanations_CFRL.loc[explanations_CFRL.index == index_value]
        
        # Append all rows from the second DataFrame
        rows_CFRL['source'] = 'CFRL'
        ensemble_df = pd.concat([ensemble_df, rows_CFRL], axis=0)

    if index_value in explanations_CP_ILS.index.unique():
        # Extract all rows from the third DataFrame for the same index value
        rows_ILS = explanations_CP_ILS.loc[explanations_CP_ILS.index == index_value]
        
        # Append all rows from the third DataFrame with a new 'source' column indicating LORE
        rows_ILS['source'] = 'CP-ILS'
        ensemble_df = pd.concat([ensemble_df, rows_ILS], axis=0)
    
    if index_value in explanations_GS.index.unique():
        # Extract all rows from the third DataFrame for the same index value
        rows_GS = explanations_GS.loc[explanations_GS.index == index_value]
        
        # Append all rows from the third DataFrame with a new 'source' column indicating LORE
        rows_GS['source'] = 'GS'
        ensemble_df = pd.concat([ensemble_df, rows_GS], axis=0)


ensemble_df.rename(columns={'Unnamed: 0': 'original index'}, inplace=True)

# %%
cols_to_convert = ['CreditScore', 'Age', 'Gender', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
ensemble_df[cols_to_convert] = ensemble_df[cols_to_convert].astype('int64')

# %%
ensemble_df.dtypes

# %%
# Duplicate rows according to ensemble_df index
X_test_repeated = X_test.loc[list(ensemble_df.index)]

# %% [markdown]
# #### Filtering only valid CFs

# %%
# Filter ensemble to include only valid CFs, i.e. with differing predictions
y_pred_instances = model.predict(X_test_repeated)
y_pred_ensemble = model.predict(ensemble_df.iloc[:,:-1])
indices_differing = np.where(y_pred_instances != y_pred_ensemble)[0]
ensemble_df = ensemble_df.iloc[indices_differing]

# %%
file_path = os.path.join(current_dir, 'ensemble_df.csv')
ensemble_df.to_csv(file_path, mode='w', index=True)

# %%
ensemble_df.shape

# %%
# Re-index again X_test
X_test_repeated = X_test.loc[list(ensemble_df.index)]

# %%
X_test_np = np.array(X_test_repeated)
cf_array = np.array(ensemble_df.iloc[:, :-1])
print(X_test_np.shape)
print(cf_array.shape)

# %%
ensemble_df_new = ensemble_df.copy()

# Measure 1.1: Proximity - Overall difference in feature values (the average Euclidean distance between x and the counterfactual x')
#ensemble_df_new['Proximity'] = np.diag(cdist(cf_array, X_test_np))
ensemble_df_new['Proximity'] = np.sqrt(np.sum((cf_array - X_test_np)**2, axis=1))


# Measure 1.2: L1 norm - Overall difference in feature values (the average Manhattan distance between x and the counterfactual x')
ensemble_df_new['Proximity_L1'] = np.sum(np.abs(cf_array - X_test_np), axis=1)
                                       
# Measure 1.3: L∞ norm - the maximum of element-wise absolute difference btw the two vectors
ensemble_df_new['Proximity_L_inf'] = np.max(np.abs(cf_array - X_test_np), axis=1)

# Measure 2: Sparsity - Proportion of differing features %
#ensemble_df_new['Sparsity'] = np.diag(cdist(cf_array, X_test_np, metric='hamming'))
ensemble_df_new['Sparsity'] = np.sum(cf_array != X_test_np, axis=1) / X_test_np.shape[1]


# %%
X_test_np.shape[1]


# %%
# Measure 3: Plausibility - The average distance of x' from the closest instance in the X_test population 

# Build a KDTree on X_test_np
tree = spatial.KDTree(X_test_np)

# Query the KDTree to find the nearest neighbor in X_test_np for each instance in cf_array (mindist contains the distances to the nearest neighbors)
mindist, _ = tree.query(cf_array)

ensemble_df_new['Plausibility'] = mindist

# %%
ensemble_df.iloc[:, :-1].shape

# %%
X_test_np.shape

# %%
ensemble_df_new.head()

# %%
l2_values = ensemble_df_new['Proximity'].values.reshape(-1, 1)
l1_values = ensemble_df_new['Proximity_L1'].values.reshape(-1, 1)
l_inf_values = ensemble_df_new['Proximity_L_inf'].values.reshape(-1, 1)
plausibility_values = ensemble_df_new['Plausibility'].values.reshape(-1, 1)

# Create a MinMaxScaler
min_max_scaler = MinMaxScaler()

# Fit and transform the proximity values
ensemble_df_new['Proximity_Normalized'] = min_max_scaler.fit_transform(l2_values)
ensemble_df_new['Proximity_L_1_Normalized'] = min_max_scaler.fit_transform(l1_values)
ensemble_df_new['Proximity_L_inf_Normalized'] = min_max_scaler.fit_transform(l_inf_values)
ensemble_df_new['Pausibility_Normalized'] = min_max_scaler.fit_transform(plausibility_values)
ensemble_df_new['AVG_Proximity'] = (
    ensemble_df_new['Proximity_Normalized'] *
    ensemble_df_new['Proximity_L_1_Normalized'] *
    ensemble_df_new['Proximity_L_inf_Normalized']
) ** (1 / 3)

ensemble_df_new = ensemble_df_new.drop(['Proximity', 'Plausibility','Proximity_L_inf', 'Proximity_L1','Proximity_Normalized', 'Proximity_L_1_Normalized', 'Proximity_L_inf_Normalized'], axis=1)

# %%
ensemble_df_new.head()

# %% [markdown]
# #### Diversity metric calculation

# %%
diversity_cols = ensemble_df_new.iloc[:,:-3].copy()

# %%
#V1
def calculate_diversity(df): # Mean of distances intra-group
    diversity_values = []
    for index, row in df.iterrows():
        source = row['source']
        group = df[(df['source'] == source) & (df.index == index)]
        if len(group) > 1:
            distances = pdist(group.iloc[:, :-1])  # Calculate distances within the same group
            diversity = np.mean(distances)
            diversity_values.append(diversity)
        else:
            diversity_values.append(np.nan) # if only 1 CF we can't caluclate diversity obv
    return diversity_values

# Add column 
diversity_cols['Diversity'] = calculate_diversity(diversity_cols)

# %%
# Normalize Diversity and append it to ensemble_df_new
mean_pairwise_distance_values = diversity_cols['Diversity'].values.reshape(-1, 1)

ensemble_df_new['Diversity_Normalized'] = min_max_scaler.fit_transform(mean_pairwise_distance_values)

# %%
# Fill NaN values for Diversity when only 1 CF was found (Diversity is equal 0 in that case)
ensemble_df_new.fillna(0, inplace=True)

# %%
ensemble_df_new.to_csv('ensemble_df_new.csv', index=True)
#ensemble_df_new.to_csv('ensemble_df_new_xgb.csv', index=True)

# %% [markdown]
# ### Import ensemble_df_new again from here

# %%
ensemble_df_new.iloc[:,:-3]

# %%
ensemble_df_new = pd.read_csv('ensemble_df_new.csv', index_col=0)

# %%
# Define weights for the linear combination
w_dis_dist, w_dis_count, w_impl, w_div = 0.25, 0.25, 0.25, 0.25 
# Compute the linear combination score
ensemble_df_new['Linear Combination'] = (
    w_dis_dist * ensemble_df_new['AVG_Proximity'] +   # TO BE MINIMIZED
    w_dis_count * ensemble_df_new['Sparsity'] +     # TO BE MINIMIZED
    w_impl * ensemble_df_new['Pausibility_Normalized'] +   # TO BE MINIMIZED
    w_div * (1-ensemble_df_new['Diversity_Normalized'])   # COMPLEMENT TO BE MINIMIZED
)

# %%
#ensemble_df_new.set_index(ensemble_df_new['index'], inplace=True)
#ensemble_df_new.drop('index', inplace=True, axis=1)

# %%
# Sort the DataFrame by 'Linear Combination' in asc order
ensemble_df_new_sorted = ensemble_df_new.sort_values(by='Linear Combination', ascending=True)

# Group by index and select the top 5 rows for each group  
# Result is 5 top CFs per index so for example for 1000 original instances --> 5000 rows

top_5_per_index = ensemble_df_new_sorted.groupby(level=0).head(5)
print(top_5_per_index.shape)

# Sort it
top_5_per_index_sorted = top_5_per_index.sort_index()


print(len(top_5_per_index_sorted[top_5_per_index_sorted.source=='DiCE'])) 
print(len(top_5_per_index_sorted[top_5_per_index_sorted.source=='GS']))
print(len(top_5_per_index_sorted[top_5_per_index_sorted.source=='CP-ILS']))
print(len(top_5_per_index_sorted[top_5_per_index_sorted.source=='CFRL']))

# %%
print(len(ensemble_df_new[ensemble_df_new.source=='DiCE'])) 
print(len(ensemble_df_new[ensemble_df_new.source=='GS']))
print(len(ensemble_df_new[ensemble_df_new.source=='CP-ILS']))
print(len(ensemble_df_new[ensemble_df_new.source=='CFRL']))

# %%
# Adding Predicted Churn to Raw Ensemble(Not ground truth from y_test)
y_pred = model.predict(ensemble_df_new.iloc[:,:-6].values)
ensemble_df_new['churn'] = y_pred


# Definition of optional Hierarchical Selection
def hierarchical_selection(df, cf_measures, weights, k_selected):
    """
    Perform hierarchical selection based on precomputed metrics,
    ordering the metrics by weights (higher values are better)
    """
    df_copy = df.copy()
    
    # Sort metrics in order according to weights importance (higher weights are better)
    sorted_metrics = [cf_measures[i] for i in np.argsort(weights)[::-1]]
    sorted_weights = np.sort(weights)[::-1]
 
    top_k = pd.DataFrame()
 
    for metric, weight in zip(sorted_metrics, sorted_weights):
        #proportion = int(k_selected * weight) 
        proportion = math.ceil(k_selected * weight)
        #print(f"Selecting {proportion} samples based on {metric}")
        # Save the selected proportion of examples for each level=0
        selected_proportion_df = df_copy.sort_values(by=metric, ascending=False).groupby(level=0).head(proportion)
        #print("Selected proportion df:\n", selected_proportion_df.shape, proportion * len(df_copy.groupby(level=0)))
 
        top_k = pd.concat([top_k, selected_proportion_df])
        #print(len(top_k),len(selected_proportion_df),df_copy.shape)
        #df_copy = df_copy[~df_copy.apply(tuple, 1).isin(selected_proportion_df.apply(tuple, 1))]
        #print(df_copy.shape, 'shape of df_copy after dropping selected')
    
        
    # If we have more than k_selected, we trim the excess
    #print('length of top k', len(top_k), 'with respect to k_selected', k_selected * len(df.groupby(level=0)))
    top_k = top_k.groupby(level=0).head(k_selected)
    
    
    return top_k