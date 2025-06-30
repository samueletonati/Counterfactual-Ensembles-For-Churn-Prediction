import os
import shutil
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
import json
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import MinMaxScaler
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.model_selection import train_test_split

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset_name = 'Dataset_4_CTGAN_synthetic.csv'
df = pd.read_csv('synth datasets/Dataset_4_CTGAN_synthetic.csv')

feature_names = [c for c in df.columns if c != 'churn']
class_name = 'churn'

X = df[feature_names]
y = df[class_name]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42, stratify=y)



clf = joblib.load('saved_models/df_4_ctgan_model.joblib')

#with open('xgb.pkl', 'rb') as file:
    #clf = pickle.load(file)

black_box = 'df_4_ctgan_model.joblib'
#black_box = 'xgb.pkl'

"""Train latent space"""

scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

def predict(x, scaler=None, return_proba=False):
    if scaler is not None:
        x = scaler.inverse_transform(x)
    if return_proba:
        return clf.predict_proba(x)[:,1].ravel()
    else: return clf.predict(x).ravel().ravel()

y_test_pred = predict(X_test, return_proba=True)
y_train_pred = predict(X_train, return_proba=True)

cat_features_list = []#'Gender','HasCrCard', 'IsActiveMember']

target_variable = 'churn'
continuous_features = [col for col in df.columns if col not in cat_features_list and col != target_variable]

X_train_latent = np.hstack((X_train_scaled, y_train_pred.reshape(-1,1)))
X_test_latent = np.hstack((X_test_scaled, y_test_pred.reshape(-1,1)))

idx_cat = [list(X.columns).index(cat_features_list[0])]#,
           #list(X.columns).index(cat_features_list[1])]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

random_seed = 42
torch.manual_seed(random_seed)

# Latent Space
latent_dim = 12
batch_size = 1024
sigma = 1
max_epochs = 10
early_stopping = 3
learning_rate = 1e-3

def compute_similarity_Z(Z, sigma):
    D = 1 - F.cosine_similarity(Z[:, None, :], Z[None, :, :], dim=-1)
    M = torch.exp((-D**2)/(2*sigma**2))
#     return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0)-1)).transpose(0,1)
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def compute_similarity_X(X, sigma, idx_cat=None):
    D_class = torch.cdist(X[:,-1].reshape(-1,1),X[:,-1].reshape(-1,1))
    X = X[:, :-1]
    if idx_cat:
        X_cat = X[:, idx_cat]
        X_cont = X[:, np.delete(range(X.shape[1]),idx_cat)]
        h = X_cat.shape[1]
        m = X.shape[1]
        D_cont = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
        D_cat = torch.cdist(X_cat, X_cat, p=0)/h
        D = h/m * D_cat + ((m-h)/m) * D_cont + D_class
    else:
        D_features = 1 - F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
        D = D_features + D_class
    M = torch.exp((-D**2)/(2*sigma**2))
#     return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0)-1)).transpose(0,1)
    return M / (torch.ones([M.shape[0],M.shape[1]]).to(device)*(torch.sum(M, axis = 0))).transpose(0,1)

def kld_loss_function(X, Z, idx_cat, sigma=1):
    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    # Compute similarity matrices
    Sx = compute_similarity_X(X, sigma, idx_cat)
    Sz = compute_similarity_Z(Z, sigma)
    # Compute Kullback-Leibler Divergence loss
    loss = similarity_KLD(torch.log(Sz), Sx)
    return loss

class LinearModel(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(LinearModel, self).__init__()
        # encoding components
        self.fc1 = nn.Linear(input_shape, latent_dim)
    def encode(self, x):
        x = self.fc1(x)
        return x
    def forward(self, x):
        z = self.encode(x)
        return z
    def get_weight(self):
        return self.fc1.weight

# Create Model
model = LinearModel(X_train_latent.shape[1], latent_dim=latent_dim).to(device)

train_dataset = TensorDataset(torch.tensor(X_train_latent).float().to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(torch.tensor(X_test_latent).float().to(device))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

ds_name = dataset_name.split('.')[0]
bb_name = black_box.split('.')[0]



#creating weights directory or clearing content if it exists
def check_and_clear(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)  # Remove the directory and its contents
        os.makedirs(dir_path)
check_and_clear('./weights')

model_params = list(model.parameters())
optimizer = torch.optim.Adam(model_params, lr=learning_rate)
# record training process
epoch_train_losses = []
epoch_test_losses = []
#validation parameters
epoch = 1
best = np.inf
# progress bar
pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
            postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])

# start training
while epoch <= max_epochs:
#     print(epoch)
    # ------- TRAIN ------- #
    # set model as training mode
    model.train()
    batch_loss = []
    for batch, (X_batch,) in enumerate(train_loader): # unpacking syntax to extract values from tuples
        optimizer.zero_grad()
        Z_batch = model(X_batch)
        W_model = model.get_weight()
        loss = kld_loss_function(X_batch, Z_batch, idx_cat, sigma)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    # save result
    epoch_train_losses.append(np.mean(batch_loss))
    pbar.postfix[3]["value"] = np.mean(batch_loss)

    # -------- VALIDATION --------
    # set model as testing mode
    model.eval()
    batch_loss = []
    with torch.no_grad():
        for batch, (X_batch,) in enumerate(test_loader):
            Z_batch = model(X_batch)
            W_model = model.get_weight()
            loss = kld_loss_function(X_batch, Z_batch, idx_cat, sigma)
            batch_loss.append(loss.item())

    # save information
    epoch_test_losses.append(np.mean(batch_loss))
    pbar.postfix[5]["value"] = np.mean(batch_loss)
    pbar.postfix[1]["value"] = epoch
    # Early Stopping
    if epoch_test_losses[-1] < best:
    # If the current test loss is better (lower) than the best test loss
        wait = 0  # Reset the counter for consecutive epochs without improvement
        best = epoch_test_losses[-1]  # Update the best test loss
        best_epoch = epoch  # Record the epoch where the best test loss occurred
        torch.save(model.state_dict(), f'./weights/LinearTransparent_df_clean.pt')
    else:
    # If the current test loss is not better than the best test loss
        wait += 1  # Increment the counter for consecutive epochs without improvement

    epoch += 1
    pbar.update()

model=model.cpu()
model.load_state_dict(torch.load(f'./weights/LinearTransparent_df_clean.pt'))
with torch.no_grad():
    model.eval()
    Z_train = model(torch.tensor(X_train_latent).float()).cpu().detach().numpy()
    Z_test = model(torch.tensor(X_test_latent).float()).cpu().detach().numpy()

torch.save(model.state_dict(),f'./weights/df_clean_latent.pt')

plt.plot(epoch_train_losses)

ds_name = dataset_name.split('.')[0]
bb_name = black_box.split('.')[0]
# Loading the trained model state
model.load_state_dict(torch.load(f'./weights/df_clean_latent.pt'))

# Moving the model to the CPU
model=model.cpu()

# Evaluation mode and transforming data
with torch.no_grad():
    model.eval()
    Z_train = model(torch.tensor(X_train_latent).float()).cpu().detach().numpy()
    Z_test = model(torch.tensor(X_test_latent).float()).cpu().detach().numpy()

w = model.fc1.weight.detach().numpy()
b = model.fc1.bias.detach().numpy()
y_contrib = model.fc1.weight.detach().numpy()[:,-1]
y_contrib_norm = y_contrib/np.sqrt(np.sum(y_contrib**2))

#scatter plot of the training data in the latent space where the color of the points is determined by the predicted values
plt.scatter(Z_train[:,0], Z_train[:,1], c=y_train_pred, cmap='coolwarm') #first and sec dims of latent space

#quiver plot is added to represent the contribution of the last weight vector in the latent space
plt.quiver(Z_train[:,0].mean(), Z_train[:,1].mean(),
           y_contrib[0], y_contrib[1], angles='xy', scale_units='xy', scale=2)
plt.grid() #adds grid

"""Counterfactual Search"""

def compute_cf(q, indexes, scaler=None, max_steps=100):
    # Predict the probability for the original input
    q_pred = predict(q, scaler, return_proba=True)
    q_cf = q.copy()  # Create a copy of the input
    q_cf_preds = []  # List to store predictions during iterations
    q_cf_preds.append(float(predict(q_cf, scaler, return_proba=True)))
    q_cf['prediction'] = q_pred

    # Determine the direction for perturbation based on the original prediction
    if q_pred > 0.5:
        m = -0.1
    else:
        m = +0.1

    # Iteratively modify the input to generate counterfactual instances
    for iteration in range(max_steps):
        if np.round(q_pred) == np.round(q_cf_preds[-1]):
            # Compute the vector to apply for perturbation
            adapt_coeff = 2 * float(abs(q_cf_preds[-1] - 0.5))
            v = (model(torch.tensor(q_cf.values).float()).detach().numpy() + m * y_contrib_norm * adapt_coeff).ravel()

            # Compute the changes (delta) in the input space
            c_l = [v[l] - np.sum(q_cf.values * w[l, :]) - b[l] for l in range(latent_dim)]
            M = []
            for l in range(latent_dim):
                M.append([np.sum(w[k, indexes] * w[l, indexes]) for k in range(latent_dim)])
            M = np.vstack(M)
            lambda_k = np.linalg.solve(M, c_l)
            delta_i = [np.sum(lambda_k * w[:, i]) for i in indexes]
            q_cf[q_cf.columns[indexes]] += delta_i
            q_cf = np.clip(q_cf, 0, 1)

            '''# Check for changes or null effects in the prediction
            if float(predict(q_cf.iloc[:, :-1], scaler, return_proba=True)) in q_cf_preds:
                return q_cf.iloc[:, :-1]'''
            
            # Check for changes or null effects in the prediction
            pred_proba = predict(q_cf.iloc[:, :-1], scaler, return_proba=True)
            if float(pred_proba) in q_cf_preds:
                return q_cf.iloc[:, :-1]

            q_cf_preds.append(float(predict(q_cf.iloc[:, :-1], scaler, return_proba=True)))
            q_cf[q_cf.columns[-1]] = q_cf_preds[-1]
        else:
            break

    return q_cf.iloc[:, :-1]

METRIC = 'euclidean'
WEIGHTS = None

def selected_cf_distance(x, selected, lambda_par=1.0, knn_dist=False, knn_list=None, lconst=None):

    if not knn_dist:
        dist_ab = 0.0
        dist_ax = 0.0
        for i in range(len(selected)):
            a = np.expand_dims(selected[i], 0)
            for j in range(i + 1, len(selected)):
                b = np.expand_dims(selected[j], 0)
                dist_ab += cdist(a, b, metric=METRIC, w=WEIGHTS)[0][0]
            dist_ax += cdist(a, x, metric=METRIC, w=WEIGHTS)[0][0]

        coef_ab = 1 / (len(selected) * len(selected)) if len(selected) else 0.0
        coef_ax = lambda_par / len(selected) if len(selected) else 0.0

    else:
        dist_ax = 0.0
        common_cfs = set()
        for i in range(len(selected)):
            a = np.expand_dims(selected[i], 0)
            knn_a = knn_list[a.tobytes()]
            common_cfs |= knn_a
            dist_ax += cdist(a, x, metric=METRIC, w=WEIGHTS)[0][0]
        dist_ab = len(common_cfs)

        coef_ab = 1.0
        coef_ax = 2.0 * lconst

    dist = coef_ax * dist_ax - coef_ab * dist_ab
    # dist = coef_ab * dist_ab - coef_ax * dist_ax
    return dist

def get_best_cf(x, selected, cf_list_all, lambda_par=1.0, submodular=True,
                    knn_dist=False, knn_list=None, lconst=None):
    min_d = np.inf
    best_i = None
    best_d = None
    d_w_a = selected_cf_distance(x, selected, lambda_par, knn_dist, knn_list, lconst)
    for i, cf in enumerate(cf_list_all):
        d_p_a = selected_cf_distance(x, selected + [cf], lambda_par)
        d = d_p_a - d_w_a if submodular else d_p_a  # submudular -> versione derivata
        if d < min_d:
            best_i = i
            best_d = d_p_a
            min_d = d

    return best_i, best_d

def greedy_kcover(x, cf_list_all, k=10, lambda_par=1.0, submodular=True, knn_dist=True):

#         x = np.expand_dims(x, 0)
#     nx = scaler.inverse_transform(x)
    nx = x.reshape(1, -1)

#     ncf_list_all = scaler.inverse_transform(cf_list_all)
    ncf_list_all = cf_list_all.copy()

    lconst = None
    knn_list = None
    if knn_dist:
        dist_x_cf = cdist(nx, ncf_list_all, metric=METRIC, w=WEIGHTS)
        d0 = np.argmin(dist_x_cf)
        lconst = 0.5 / (-d0) if d0 != 0.0 else 0.5

        # cf_dist_matrix = np.mean(self.cdist(ncf_list_all, ncf_list_all,
        #                                     metric=METRIC, w=WEIGHTS), axis=0)
        cf_dist_matrix = cdist(ncf_list_all, ncf_list_all, metric=METRIC, w=WEIGHTS)

        knn_list = dict()
        for idx, knn in enumerate(np.argsort(cf_dist_matrix, axis=1)[:, 1:k+1]):
            cf_core_key = np.expand_dims(cf_list_all[idx], 0).tobytes()
            knn_set = set([np.expand_dims(cf_list_all[nn], 0).tobytes() for nn in knn])
            knn_list[cf_core_key] = knn_set

    cf_list = list()
    cf_selected = list()
    ncf_selected = list()
    min_dist = np.inf
    while len(ncf_selected) < k:
        idx, dist = get_best_cf(nx, ncf_selected, ncf_list_all, lambda_par, submodular,
                                     knn_dist, knn_list, lconst)
#         cf_selected.append(self.scaler.inverse_transform(ncf_list_all[idx]))
        cf_selected.append(ncf_list_all[idx])
        ncf_selected.append(ncf_list_all[idx])
        ncf_list_all = np.delete(ncf_list_all, idx, axis=0)
        if dist < min_dist:
            min_dist = dist
            cf_list = cf_selected

    cf_list = np.array(cf_list)

    return cf_list

def generate_counterfactuals(df_instances, features_to_change, max_features_to_change, 
                             n_cfs=10, n_feats_sampled=1, topn_to_check=1):
    
    from itertools import combinations, chain
    
    all_cfs = []
    
    for _, row in tqdm(list(df_instances.iterrows())):
        
        q_cfs = []
        
        q = row.to_frame().T
        
        q_pred = predict(q, scaler, return_proba=False)
        s_i = [set()]
        s_f = set()
        l_i = []
        l_f = []
        
        ########

        for indexes in list(combinations(list(features_to_change),1)):    
            q_cf = compute_cf(q, list(indexes), scaler)
            q_cf_pred = predict(q_cf, scaler, return_proba=True)
            diff_probs = float(abs(q_cf_pred-0.5))
            if q_pred:
                if q_cf_pred<0.5:
#                     q_cf['n_changes'] = len(indexes)
                    q_cfs.append((q_cf, diff_probs))
                    s_i[-1].add(frozenset(list(indexes)))
                else:
                    l_i.append((list(indexes), diff_probs))
            else:
                if q_cf_pred>0.5:
#                     q_cf['n_changes'] = len(indexes)
                    q_cfs.append((q_cf, diff_probs))
                    s_i[-1].add(frozenset(list(indexes)))
                else:
                    l_i.append((list(indexes), diff_probs))
                    
        if len(l_i)>0:
            
            r = np.argsort(np.stack(np.array(l_i,dtype=object)[:,1]).ravel())[:topn_to_check]
            l_i = np.array(l_i,dtype=object)[r,0]
        
            while len(l_i[0])<max_features_to_change:
                for e in l_i:
                    features_to_check = list(np.delete(features_to_change, 
                             list(map(lambda f: (features_to_change).index(f), e))))

                    for i in np.random.choice(features_to_check, 
                                              size=min(len(features_to_check), n_feats_sampled), replace=False):

                        indexes = list(e)+[i]

                        skip_i = False
                        #check if the current indices already returned a cf
                        if frozenset(indexes) in s_f:
                            skip_i = True

                        if not skip_i:
                            #check if any subset of current indices already returned a cf
                            for comb_i in chain.from_iterable(combinations(indexes, r) 
                                                              for r in range(1, len(indexes))):
                                if frozenset(comb_i) in s_i[len(comb_i)-1]:
                                    skip_i = True
                                    break

                        if not skip_i:
                            q_cf = compute_cf(q, list(indexes), scaler)
                            q_cf_pred = predict(q_cf, scaler, return_proba=True)
                            diff_probs = float(abs(q_cf_pred-0.5))
                            if q_pred:
                                if q_cf_pred<0.5:
#                                     q_cf['n_changes'] = len(indexes)
                                    q_cfs.append((q_cf, diff_probs))
                                    s_f.add(frozenset(indexes))
                                else:
                                    l_f.append((list(indexes), diff_probs))
                            else:
                                if q_cf_pred>0.5:
#                                     q_cf['n_changes'] = len(indexes)
                                    q_cfs.append((q_cf, diff_probs))
                                    s_f.add(frozenset(indexes))
                                else:
                                    l_f.append((list(indexes), diff_probs))
                
                if len(l_f)==0:
                    break
                    
                s_i.append(s_f.copy())
                s_f = set()

                r = np.argsort(np.stack(np.array(l_f,dtype=object)[:,1]).ravel())[:topn_to_check]
                l_f = np.array(l_f,dtype=object)[r,0]
                l_i = l_f.copy()
                l_f = []
            
        if len(q_cfs)==0:
            all_cfs.append(None)
        else:
            q_cfs = [cf[0] for cf in q_cfs]
            #if len(q_cfs)>n_cfs:
                #cf_list = greedy_kcover(q.values, np.array(q_cfs).squeeze(), k=n_cfs)
                #q_cfs = [pd.Series(cf, index=q.columns, name=q.index[0]).to_frame().T for cf in cf_list]
            all_cfs.append(q_cfs)
    
    return all_cfs


# Batch-specific parameters
start_index = 0
end_index = len(X_test)
#batch = X_test
batch = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

def process_batch(batch, start_index, end_index):

    # Create a directory to save the pickle files
    output_directory = 'CP-ILS'

    # Generate counterfactuals for the current batch

    # indices of features to change
    feats_to_change = list(range(batch.shape[1]))

    # max number of features to change
    max_features_to_change = 12

    cf_batch = generate_counterfactuals(batch, feats_to_change, max_features_to_change)

    # Define the file name based on the starting and ending index of the batch
    output_file = os.path.join(output_directory, f'cpils_{latent_dim}dims_batch_{start_index}_{end_index}.pkl')

    # Save the counterfactual explanations as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(cf_batch, f)

process_batch(batch, start_index, end_index)

output_directory = 'CP-ILS'  # Define output directory

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

dis_dist = []
dis_count = []
impl = []

div_dist = []
div_count = []

cf_arrays = []

# Load the counterfactual explanations for the single batch
batch_pickle_file = os.path.join(output_directory, f'cpils_{latent_dim}dims_batch_{start_index}_{end_index}.pkl')

with open(batch_pickle_file, 'rb') as f:
    cf_array = pickle.load(f)

# Remove the first element if it's not the desired index range
if start_index > 0:
    cf_array = cf_array[1:]

# Concatenate counterfactuals according to the instance index
cf_array = pd.concat([pd.concat(cfs)
                      for cfs in cf_array if isinstance(cfs, list)]).clip(0, 1)

# Scaler inverse transform on continual features

cf_array.values[:] = np.concatenate((scaler.inverse_transform(cf_array.values)[:, :-2],
                                     cf_array.values[:, -2:]), axis=1)

# Discretize categorical features
cf_array.iloc[:, -1] = np.rint(cf_array.iloc[:, -1])
cf_array.iloc[:, -2] = np.rint(cf_array.iloc[:, -2])

cf_arrays.append(cf_array)


df_cf = pd.DataFrame(cf_arrays[0])

file_path = 'explanations_CP_ILS.csv'
#file_path = 'explanations_CP_ILS_xgb.csv'
df_cf.to_csv(file_path, mode='w', index=True)


'''
X_cf_df = cf_array.copy()
X_cf = cf_array.values.copy()

X_test_np_aligned = X_test.loc[list(X_cf_df.index)]

# Measure 1: Overall difference in feature values
dis_dist_batch = np.diag(cdist(X_cf, X_test_np_aligned))
dis_dist.append(dis_dist_batch)

# Measure 2: Proportion of differing features %
dis_count_batch = np.diag(cdist(X_cf, X_test_np_aligned, metric='hamming'))
dis_count.append(dis_count_batch)

# Measure 3: the average distance of x' from the closest instance in the X population
impl_batch = np.min(cdist(X_cf, X_train.values), axis=1)
impl.append(impl_batch)

# Measure 4: the average distance between counterfactuals
div_dist_batch = np.array([np.mean(pdist(cf_array.loc[idx].values)) for idx in np.unique(cf_array.index) if len(cf_array.loc[idx].values.shape) > 1])
div_dist.append(div_dist_batch)

# Measure 5: the average differing features % between counterfactuals
div_count_batch = np.array([np.mean(pdist(cf_array.loc[idx].values, metric='hamming')) for idx in np.unique(cf_array.index) if len(cf_array.loc[idx].values.shape) > 1])
div_count.append(div_count_batch)

print(f'Single Batch ({start_index}-{end_index}):')
print('dis_dist: ', np.mean(dis_dist_batch))
print('dis_count: ', np.mean(dis_count_batch))
print('impl: ', np.mean(impl_batch))
print('div_dist: ', np.mean(div_dist_batch))
print('div_count: ', np.mean(div_count_batch))
print('\n')
'''