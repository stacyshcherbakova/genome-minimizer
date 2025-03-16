import os
import sys
import itertools
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utilities.directories import *
from models.extras import *
from models.training import *
from models.VAE_models.VAE_model import VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, TEN_K_DATASET)
PHYLOGROUP_PATH = os.path.join(PROJECT_ROOT, TEN_K_DATASET_PHYLOGROUPS)
FIGURE_DIR = os.path.join(PROJECT_ROOT, "models/v0/figures/")

folder = "v0_run/"

print("** START OF THE SCRIPT **\n")

## Loading and preping the dataset
print("LOADING THE DATASET...")
large_data = pd.read_csv(DATA_PATH, index_col=[0], header=[0])
large_data.columns = large_data.columns.str.upper()
phylogroup_data = pd.read_csv(PHYLOGROUP_PATH, index_col=[0], header=[0])

data_without_lineage = large_data.drop(index=['Lineage'])
merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
# merged_df['Phylogroup'] = merged_df['Phylogroup'].fillna('Not determined')
print(merged_df['Phylogroup'].value_counts())

# numeric_cols = merged_df.select_dtypes(include='number')
# column_sums = numeric_cols.sum(axis=0)

# filtered_columns = column_sums[column_sums / 7512 >= 0.05].index
# filtered_data = merged_df[filtered_columns]

# filtered_data = merged_df[filtered_columns].copy()
# filtered_data['Phylogroup'] = merged_df['Phylogroup']

data_array_t = np.array(merged_df.iloc[:, :-1])
phylogroups_array = np.array(merged_df.iloc[:, -1])

print(f"Dataset shape: {data_array_t.shape}")
print(f"Phylogroups array shape: {phylogroups_array.shape}")

## Preping the dataset
print("PREPING THE DATASET...")
# Convert to PyTorch tensor
data_tensor = torch.tensor(data_array_t, dtype=torch.float32)

# Split into train and test sets
# train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=12345)
# train_data, test_data = train_test_split(data_tensor, test_size=0.25, random_state=12345)

# Spliting into train and test sets
train_data, temp_data, train_labels, temp_labels = train_test_split(data_tensor, phylogroups_array, test_size=0.3, random_state=12345)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.3333, random_state=12345)

test_phylogroups = test_labels

# TensorDataset
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
test_dataset = TensorDataset(test_data)

# DataLoaders for main training
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Preping model inputs
print("PREPPING MODEL INPUTS...")
# best hyperparams
# hidden_dim=1024, latent_dim=64, learning_rate=0.001
# Model inputs 
hidden_dim = 1024
latent_dim = 64
beta_start = 0.1
beta_end = 1.0
n_epochs = 1 # 100
max_norm = 1.0 
input_dim = data_array_t.shape[1]
print(f"Input dimention: {input_dim}")

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

## Trainign the model
print("TRAINING STARTED...")
train_loss_vals2, val_loss_vals, epochs = v0(model=model, folder=folder, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, train_loader=train_loader, val_loader=val_loader, beta_start=beta_start, beta_end=beta_end, max_norm=max_norm)

# Save trained model
torch.save(model.state_dict(), PROJECT_ROOT+"/models/trained_models/"+folder+"saved_VAE_v0.pt")
print("Model saved.")

## Generating a comparison graph 
print("GENERATING A COMPARISON GRAPH...")
# Generating points for graphs
epochs = np.linspace(1, epochs, num=epochs)
# Plot train vs val loss graph
plot_loss_vs_epochs_graph(epochs=epochs, train_loss_vals=train_loss_vals2, val_loss_vals=val_loss_vals, fig_name=FIGURE_DIR+"train_val_loss.pdf")

## Calculating F1 scores 
# Clacualting F1 score over all samples 
print("Calculating F1 scores...")
model.eval()
all_recon_x = []
all_test_data = []

# Ploting the data into batches
with torch.no_grad():
    for batch in test_loader:
        batch_data = batch[0].to(device)
        recon_x, mu, logvar = model(batch_data)
        all_recon_x.append(recon_x.to(device))
        all_test_data.append(batch_data.to(device))

all_recon_x = torch.cat(all_recon_x)
all_test_data = torch.cat(all_test_data)

# Setting thershold 
recon_x_binarized = (all_recon_x > 0.5).int()

all_test_data_np = all_test_data.cpu().numpy().flatten()
recon_x_binarized_np = recon_x_binarized.cpu().numpy().flatten()

f1 = sklearn.metrics.f1_score(all_test_data_np, recon_x_binarized_np)
print(f'F1 Score: {f1:.2f}')

accuracy = sklearn.metrics.accuracy_score(all_test_data_np, recon_x_binarized_np)
print(f'Accuracy Score: {accuracy:.2f}')

# Calcualting F1 score for each sample (comapring the target to reconstruction)
f1_scores = []
accuracy_scores = []
for genome_x, genome in zip(recon_x_binarized.cpu(), all_test_data.cpu().int()):
    f1_scores.append(sklearn.metrics.f1_score(genome_x.numpy(), genome.numpy()))
    accuracy_scores.append(sklearn.metrics.accuracy_score(genome_x.numpy(), genome.numpy()))

# Ploting a histogram of all calculated F1 scores 
plt.figure(figsize=(4,4), dpi=300)
plt.hist(f1_scores, color='dodgerblue')
plt.xlabel("F1 score")
plt.ylabel("Frequency")
plt.savefig(FIGURE_DIR+"f1_score_frequency_test_set.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Ploting a histogram of all calculated Accuracy scores scores 
plt.figure(figsize=(4,4), dpi=300)
plt.hist(accuracy_scores, color='dodgerblue')
plt.xlabel("Accuracy score")
plt.ylabel("Frequency")
plt.savefig(FIGURE_DIR+"accuracy_score_frequency_test_set.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ## Exploring latent space
print("EXPLORING THE LATENT SPACE...")
# Get latent variables
latents = get_latent_variables(model, test_loader, device)
# Apply t-SNE for dimensionality reduction
name = FIGURE_DIR+"tsne_latent_space_visualisation.pdf"
# do_tsne(n_components=2, latents=latents, fig_name=name)
tsne = TSNE(n_components=2)
tsne_latents = tsne.fit_transform(latents)
df_tsne = pd.DataFrame(tsne_latents, columns=['TSNE1', 'TSNE2'])
# print(f"len tsne latents: {len(tsne_latents)}")
# print(f"len test phylogroups: {len(test_phylogroups)}")
df_tsne['phylogroup'] = test_phylogroups

plt.figure(figsize=(4,4), dpi=300)
sns.scatterplot(x='TSNE1', y='TSNE2', hue = df_tsne['phylogroup'], data=df_tsne)
handles, labels = plt.gca().get_legend_handles_labels()  
plt.legend(handles, labels, fontsize=8) 
plt.savefig(name, format="pdf", bbox_inches="tight")
plt.show()

# Apply PCA
latents = get_latent_variables(model, test_loader, device)
pca = PCA(n_components=3)
data_pca = pca.fit_transform(latents)
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['phylogroup'] = test_phylogroups
print(df_pca.head())
fig, axes = plt.subplots(1, 2, figsize=(8,4), dpi=300)
sns.scatterplot(x='PC1', y='PC2', hue = df_pca['phylogroup'] , data=df_pca, ax=axes[0])
sns.scatterplot(x='PC2', y='PC3', hue = df_pca['phylogroup'] , data=df_pca, ax=axes[1])
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8)
plt.savefig(FIGURE_DIR+"pca_latent_space_test_set.pdf", format="pdf", bbox_inches="tight")
plt.show()

# print("\nHyperparameter tuning")
# # Gridsearch
# input_dim = data_array_t.shape[1]
# print(f"Input dimention: {input_dim}")
# hidden_dim_values = [256, 512, 1024]
# latent_dim_values = [32, 64, 128]
# learning_rate_values = [0.01, 1e-3] # Decrease of learning rate causes higher average train loss, better if 0.01, 0.001
# max_norm = 1.0 
# beta_start = 0.1
# beta_end = 1.0

# # beta_start, beta_end, max_norm
# for hidden_dim, latent_dim, learning_rate in itertools.product(
#     hidden_dim_values, latent_dim_values, learning_rate_values): #beta_start_values, beta_end_values, max_norm_values
#     print(f"Training with hidden_dim={hidden_dim}, latent_dim={latent_dim}, learning_rate={learning_rate}") # beta_start={beta_start}, beta_end={beta_end}, max_norm={max_norm}"
#     model = VAE(input_dim, hidden_dim, latent_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#     train_with_KL_annelaing(model=model, optimizer=optimizer, scheduler=scheduler, n_epochs=10, train_loader=train_loader, val_loader=val_loader, beta_start=beta_start, beta_end=beta_end, max_norm=max_norm)
#     print("--------------------------------------------------------------------------------------")