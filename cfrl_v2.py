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
from copy import deepcopy
from typing import List, Tuple, Dict, Callable

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from alibi.explainers import CounterfactualRLTabular, CounterfactualRL
from alibi.datasets import fetch_adult
from alibi.models.tensorflow import HeAE
from alibi.models.tensorflow import Actor, Critic
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder
from alibi.explainers.cfrl_base import Callback
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor, get_statistics, \
    get_conditional_vector, apply_category_mapping
import alibi.explainers

# Import black box

#file_path = 'rf_model.pkl'
file_path = r'C:\Users\samue\OneDrive\Bank Customer Churn XAI Project\3) Iranian Churn\lgb.pkl'

# Load the object from the pickle file
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Import original df

df = pd.read_csv(r'C:\Users\samue\OneDrive\Bank Customer Churn XAI Project\3) Iranian Churn\df4_clean.csv', index_col=0)

# Define feature and class names
feature_names = [c for c in df.columns if c != 'churn']
class_name = 'churn'

X = df.drop('churn', axis=1)
y = df['churn']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y, random_state=42)


# List of binary categorical column names
cat_features_list = []#['Tariff Plan', 'Status']

category_map = {}#9: [1, 2], 10: [1, 2]}

binary_categorical_columns = []#['Tariff Plan', 'Status']

# Create lists of categorical and numerical column names and their indices
categorical_names = []
categorical_ids = []
numerical_names = []
numerical_ids = []


for idx, col_name in enumerate(feature_names):
    if col_name in binary_categorical_columns:
        categorical_names.append(col_name)
        categorical_ids.append(idx)
    else:
        numerical_names.append(col_name)
        numerical_ids.append(idx)

predictor = lambda x: model.predict_proba(x)

class HeAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# Define attribute types, required for datatype conversion.
feature_types = {
    'Tariff Plan': int,
    'Status' :int
}

# Define data preprocessor and inverse preprocessor. The inverse preprocessor includes datatype conversions.
heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(X=X_train,
                                                               feature_names=feature_names,
                                                               category_map=category_map,
                                                               feature_types=feature_types)

# Define trainset
trainset_input = heae_preprocessor(X_train).astype(np.float32)
trainset_outputs = {
    "output_1": trainset_input[:, :len(numerical_ids)]
}

for i, cat_id in enumerate(categorical_ids):
    trainset_outputs.update({
        f"output_{i+2}": X_train.iloc[:, cat_id]
    })

trainset = tf.data.Dataset.from_tensor_slices((trainset_input, trainset_outputs))
trainset = trainset.shuffle(1024).batch(128, drop_remainder=True)

len(numerical_ids)

[len(category_map[cat_id]) for cat_id in categorical_ids]

# Custom Encoder
class CustomEncoder(keras.Model):
    def __init__(self, hidden_dim, latent_dim):
        super(CustomEncoder, self).__init__()
        self.hidden_layer = keras.layers.Dense(hidden_dim, activation='relu')
        self.latent_layer = keras.layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        z = self.latent_layer(x)
        return z

# Custom Decoder
class CustomDecoder(keras.Model):
    def __init__(self, hidden_dim, output_dims):
        super(CustomDecoder, self).__init__()
        self.hidden_layer = keras.layers.Dense(hidden_dim, activation='relu')
        self.output_layers = [keras.layers.Dense(dim) for dim in output_dims]

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        outputs = [output_layer(x) for output_layer in self.output_layers]
        return outputs

# Define Custom Autoencoder
class CustomAutoencoder(keras.Model):
    def __init__(self, encoder, decoder):
        super(CustomAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z = self.encoder(inputs)
        outputs = self.decoder(z)
        return outputs

# Instantiate Custom Encoder and Decoder
hidden_dim = 128
latent_dim = 5
output_dims = [len(numerical_ids)]
output_dims += [len(category_map[cat_id]) for cat_id in categorical_ids]
custom_encoder = CustomEncoder(hidden_dim, latent_dim)
custom_decoder = CustomDecoder(hidden_dim, output_dims)

# Define loss functions
mse_loss = keras.losses.MeanSquaredError()
categorical_losses = [keras.losses.SparseCategoricalCrossentropy(from_logits=True) for _ in range(len(output_dims))]

# Define loss weights
loss_weights = [1.0] + [1.0 / len(output_dims) for _ in range(len(output_dims))]

# Define metrics.
metrics = {}
for i, cat_name in enumerate(categorical_names):
    metrics.update({f"output_{i+2}": keras.metrics.SparseCategoricalAccuracy()})

# Instantiate Custom Autoencoder
custom_autoencoder = CustomAutoencoder(encoder=custom_encoder, decoder=custom_decoder)

# Compile Custom Autoencoder
custom_autoencoder.compile(optimizer='adam', loss=[mse_loss] + categorical_losses, loss_weights=loss_weights, metrics=metrics)

# Train Custom Autoencoder
custom_autoencoder.fit(trainset, epochs=50, batch_size=32)

# Save or Load the Model

# custom_autoencoder.save("custom_autoencoder.h5")
# loaded_model = keras.models.load_model("custom_autoencoder.h5")


# Define constants
COEFF_SPARSITY = 0.5               # sparisty coefficient
COEFF_CONSISTENCY = 0.5            # consisteny coefficient
TRAIN_STEPS = 10000                # number of training steps -> consider increasing the number of steps
BATCH_SIZE = 100                   # batch size (how many data points will be used in each training step during the fitting process)

explainer = CounterfactualRLTabular(predictor=predictor,
                                    encoder=custom_autoencoder.encoder,
                                    decoder=custom_autoencoder.decoder,
                                    latent_dim=latent_dim,
                                    encoder_preprocessor=heae_preprocessor,
                                    decoder_inv_preprocessor=heae_inv_preprocessor,
                                    coeff_sparsity=COEFF_SPARSITY,
                                    coeff_consistency=COEFF_CONSISTENCY,
                                    category_map=category_map,
                                    feature_names=feature_names,
                                    train_steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE,
                                    backend="tensorflow"
)

explainer = explainer.fit(X=X_train)


# Run on all Test Set

exps = []
C = []

for i, X in enumerate(X_test[:1000].values):  # Enumerate directly over the values
    pred = predictor(X.reshape(1, -1))
    predicted_class = np.argmax(pred)
    Y_t = np.array([0]) if predicted_class == 1 else np.array([1])
    explanation = explainer.explain(X=X.reshape(1, -1), Y_t=Y_t, C=C, diversity=True, num_samples=10, batch_size=100)#, patience=200, tolerance=0.1)
    exps.append((X_test.index[i], explanation))  # Store index along with explanation


cf_dfs = []

for original_index, explanation in exps:
    # Concat label column to the counterfactual instances.
    cf = explanation.data['cf']['X']

    df_cf = pd.DataFrame(
        apply_category_mapping(cf, category_map),
        columns=feature_names,
        index=[original_index] * len(cf)  # Set the index to the original instance index
    )

    cf_dfs.append(df_cf)

# Concatenate all DataFrames in cf_dfs while keeping the original index
df_cf = pd.concat(cf_dfs)




# Repeat indexes of X_test based on cf_df
X_test_aligned = X_test.loc[list(df_cf.index)]
X_test_aligned.shape

X_test_aligned = X_test_aligned.apply(pd.to_numeric, errors='ignore')
df_cf = df_cf.apply(pd.to_numeric, errors='ignore')

# Validity of CFs generated

y_pred_instances = model.predict(X_test_aligned)
y_pred_cf = model.predict(df_cf)

differing_predictions = sum(y_pred_instances != y_pred_cf)

# Calculate the total number of predictions
total_predictions = len(y_pred_instances)

# Calculate the percentage of differing predictions
percentage_differing = (differing_predictions / total_predictions) * 100

print(f"Percentage of differing predictions: {percentage_differing:.2f}%")

# Filter df_cf to include only valid CFs, i.e. with differing predictions
indices_differing = np.where(y_pred_instances != y_pred_cf)[0]

df_cf = df_cf.iloc[indices_differing]

# Save CFS Dataset
file_path = 'explanations_CFRL.csv'
#file_path = 'explanations_CFRL_xgb.csv'
df_cf.to_csv(file_path, mode='w', index=True)

cf_array = np.array(df_cf)
X_test_np = np.array(X_test_aligned)

# Measure 1: Proximity - Overall difference in feature values (the average Euclidean distance between x and the counterfactual x')
dis_dist = np.mean(np.diag(cdist(cf_array, X_test_np)))

# Measure 2: Sparsity - Proportion of differing features %
dis_count_h = np.mean(np.diag(cdist(cf_array, X_test_np, metric='hamming')))
dis_count = np.mean(np.mean(~np.isclose(cf_array, X_test_np), axis=1))

# Measure 3: Plausibility - The average distance of x' from the closest instance in the X_test population
impl = np.mean(np.min(cdist(cf_array, X_test_np), axis=1))

#Measure 4: Diversity 1 - The average distance between counterfactuals    ### div_1 and div_2 give error if, for each set of cf, less than 2 are found ###
div_dist = np.mean([np.mean(pdist(df_cf.loc[idx].values)) for idx in np.unique(df_cf.index)])

#Measure 5: Diversity 2 - The average differing features % between counterfactuals (first compute pairwise distance within each index, then calculate mean within the CFs the index and finally the mean of all indexed CFs)
div_count = np.mean([np.mean(pdist(df_cf.loc[idx].values, metric ='hamming')) for idx in np.unique(df_cf.index)])

print(dis_dist)
print(dis_count_h)
print(dis_count)
print(impl)
print(div_dist)
print(div_count)

