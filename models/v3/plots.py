import sys
import os
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from VAE_models.VAE_model import *
from VAE_models.VAE_model_enhanced import *
from training import *
from extras import *

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/F4_complete_presence_absence.csv")
PHYLOGROUP_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/accessionID_phylogroup_BD.csv")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "models/v3/figures/")
ESSENTIAL_GENE_POSITIONS_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/essential_gene_positions.pkl")

with open(ESSENTIAL_GENE_POSITIONS_PATH, "rb") as f:
    ESSENTIAL_GENE_POSITIONS = pickle.load(f)

# Load datasets
large_data = pd.read_csv(DATA_PATH, index_col=0).rename(columns=str.upper)
data_without_lineage = large_data.drop(index=['Lineage'])
large_data_t = data_without_lineage.T.values

phylogroup_data = pd.read_csv(PHYLOGROUP_PATH, index_col=0)
merged_df = data_without_lineage.T.merge(phylogroup_data, left_index=True, right_on='ID')
data_array_t, phylogroups_array = merged_df.iloc[:, :-1].values, merged_df.iloc[:, -1].values

data_tensor = torch.tensor(data_array_t, dtype=torch.float32)
train_data, temp_data, train_labels, temp_labels = train_test_split(data_tensor, phylogroups_array, test_size=0.3, random_state=12345)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.3333, random_state=12345)

test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)
train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=False)

weights = ['1_gammastart2', '2_gammastart2']
input_dim, hidden_dim, latent_dim = 55039, 512, 32

for weight in weights:
    model, binary_generated_samples = load_model(input_dim, hidden_dim, latent_dim, f"{PROJECT_ROOT}/models/trained_models/v3_run/saved_VAE_v3_{weight}.pt")
    plot_samples_distribution(binary_generated_samples, f"{FIGURE_DIR}/sampling_10000_genome_size_distribution_v3_{weight}.pdf", "dodgerblue", 2000, 5000)

    model.eval()

    latents = get_latent_variables(model, test_loader, DEVICE)
    data_pca = PCA(n_components=2).fit_transform(latents)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2']).assign(phylogroup=test_labels)

    plt.figure(figsize=(4,4))
    sns.scatterplot(x='PC1', y='PC2', hue='phylogroup', data=df_pca)
    plt.legend(fontsize=8)
    plt.savefig(f"{FIGURE_DIR}/pca_latent_space_visualisation_v3_{weight}.pdf", format="pdf", bbox_inches="tight")

    essential_genes_count_per_sample = count_essential_genes(binary_generated_samples, ESSENTIAL_GENE_POSITIONS)
    plot_essential_genes_distribution(essential_genes_count_per_sample, f"{FIGURE_DIR}/essential_genes_v3_{weight}.pdf", "violet", 250, 327)

    num_samples = 10000
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_samples = model.decode(z).cpu().numpy()
    binary_generated_samples = (generated_samples > 0.5).astype(float)

    min_ones_index = np.argmin(binary_generated_samples.sum(axis=1))
    latent_distances = np.linalg.norm(generated_samples - generated_samples[min_ones_index], axis=1)
    closest_latent_index = np.argmin(latent_distances)
    
    z_of_interest = z[closest_latent_index].unsqueeze(0)
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim) * 0.1
        additional_generated_samples = model.decode(z_of_interest + noise).cpu().numpy()
    additional_generated_samples = (additional_generated_samples > 0.5).astype(float)

    plot_samples_distribution(additional_generated_samples, f"{FIGURE_DIR}/additional_sampling_10000_genome_size_distribution_v3_{weight}.pdf", "dodgerblue", 2000, 2500)
    np.save(f"{FIGURE_DIR}/additional_generated_samples_v3_{weight}.npy", additional_generated_samples)

    essential_genes_count_per_sample = count_essential_genes(additional_generated_samples, ESSENTIAL_GENE_POSITIONS)
    plot_essential_genes_distribution(essential_genes_count_per_sample, f"{FIGURE_DIR}/additional_essential_genes_v3_{weight}.pdf", "violet", 250, 327)

    total_genes_count_per_sample = additional_generated_samples.sum(axis=1)
    plot_essential_vs_total(essential_genes_count_per_sample, total_genes_count_per_sample, f"{FIGURE_DIR}/additional_essential_vs_total_genes_v3_{weight}.pdf")
