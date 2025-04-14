import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import json
from joblib import load
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import dice_ml
from dice_ml.diverse_counterfactuals import _DiverseCFV1SchemaConstants, _DiverseCFV2SchemaConstants, json_converter, CounterfactualExamples

synthetic_datasets = {
    'df_1_ctgan_model': "synth datasets/Dataset_1_CTGAN_synthetic.csv",
    'df_2_ctgan_model': "synth datasets/Dataset_2_CTGAN_synthetic.csv",
    'df_3_ctgan_model': "synth datasets/Dataset_3_CTGAN_synthetic.csv",
    'df_4_ctgan_model': "synth datasets/Dataset_4_CTGAN_synthetic.csv"
}

cat_cols_map = {
    'df_1_ctgan_model': 0,
    'df_2_ctgan_model': 1,
    'df_3_ctgan_model': 2,
    'df_4_ctgan_model': 3
}

# Import black box
model_dir = "saved_models"

models = {}
for f in os.listdir(model_dir):
    loaded_model = load(os.path.join(model_dir, f))
    model_name = f.replace('.joblib', '')
    models[model_name] = loaded_model


"""DICE requires to pass continuous variables and categorical variables as lists"""

# Features setup for DiCE processing

cat_cols_lists = [[],[],[],[]]#['Gender', 'HasCrCard', 'IsActiveMember'], ['Gender'], ['Age Group','Tariff Plan', 'Status']] ### for all 3 dfs
target_variable = 'churn'


# Process each synthetic dataset + model pair
results = []
for model_name, dataset_path in synthetic_datasets.items():
    print(f"Processing Model: {model_name}")
    
    # Assign dinamically list of categorical columns
    cat_cols = cat_cols_lists[cat_cols_map[model_name]]

    # Load model
    model = models[model_name]
    print(model_name)
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Split data into train-test split
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ensure DiCE's categorical vs continuous feature mapping is correctly done
    continuous_features = [col for col in X_train.columns if col not in cat_cols and col != target_variable]

    # Prepare DiCE Data
    d = dice_ml.Data(
        dataframe=df,
        categorical_features=cat_cols,
        continuous_features=continuous_features,
        outcome_name=target_variable
    )

    # Initialize DiCE model
    m = dice_ml.Model(model=model, backend="sklearn")
    exp_genetic = dice_ml.Dice(d, m, method="genetic")
    exp_random = dice_ml.Dice(d, m, method='random')

    # Generate counterfactual explanations
    print("Generating counterfactuals...")
    instances = X_test.copy()
    dice_exp_genetic_many = exp_genetic.generate_counterfactuals(
        instances,
        total_CFs=10,
        desired_class="opposite",
        sparsity_weight=1.0,
        proximity_weight=1.0,
        diversity_weight=1.0,
        categorical_penalty=0.1,
        posthoc_sparsity_algorithm="linear",
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

    # Save file
    save_path = f"synth CF datasets/{model_name}_cf.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_cf.to_csv(save_path, index=True)
    print(f"Saved CFs data to {save_path}")

    df_cf = pd.read_csv(save_path, index_col=0)

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
    
    # Log results
    results.append({
        "Model": model_name,
        "Found CFs": total_found,
        "Percentage of CFs found": percentage_found,
        "Diverging Predictions %": percentage_differing,
    })

# Output all logged results
for result in results:
    print(result)

