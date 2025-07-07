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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.VAE_models.VAE_model import VAE
from models.VAE_models.VAE_model_enhanced import *
from models.training import *
from models.extras import *
from utilities.directories import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FIGURE_DIR = os.path.join(PROJECT_ROOT, "models/v3/figures_sample/")
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models/trained_models/v3/")

with open(ESSENTIAL_GENES_POSITIONS, "rb") as f:
    essential_gene_positions = pickle.load(f)

def write_samples_to_dataframe(binary_generated_samples, all_genes, output_file):
        """Convert binary generated samples to a DataFrame with genes as rows and samples as columns."""
        df = pd.DataFrame(binary_generated_samples, columns=all_genes)
        df.index = [f"Sample_{i+1}" for i in range(df.shape[0])]
        df = df.transpose()  # Transpose to get genes x samples
        df.columns = [f"Sample_{i+1}" for i in range(df.shape[1])]  # Rename columns
        df = df.reset_index()  # Move gene names from index to a column
        df = df.rename(columns={'index': 'Gene'})  # Rename that column to 'Gene'
        df.to_csv(output_file, index=False)

print("load datasets")
# Load datasets
large_data = pd.read_csv(TEN_K_DATASET, index_col=0).rename(columns=str.upper)
data_without_lineage = large_data.drop(index=['Lineage'])
large_data_t = data_without_lineage.T.values

phylogroup_data = pd.read_csv(TEN_K_DATASET_PHYLOGROUPS, index_col=0)
merged_df = data_without_lineage.T.merge(phylogroup_data, left_index=True, right_on='ID')
data_array_t, phylogroups_array = merged_df.iloc[:, :-1].values, merged_df.iloc[:, -1].values

merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
all_genes = merged_df.columns[:-1]
print("all genes size:", all_genes.shape)

data_tensor = torch.tensor(data_array_t, dtype=torch.float32)
train_data, temp_data, train_labels, temp_labels = train_test_split(data_tensor, phylogroups_array, test_size=0.3, random_state=12345)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.3333, random_state=12345)

test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)
train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=False)

print("setting params")
weights = ['1_gammastart2', '2_gammastart2']
input_dim, hidden_dim, latent_dim = 55039, 512, 32
nsamples = 100

print("looping through weights")
for weight in weights:
    print(f"\nProcessing weight: {weight}")

    ### DEFAULT SAMPLING
    model = load_model(input_dim, hidden_dim, latent_dim, f"{MODEL_DIR}/saved_VAE_v3_{weight}.pt")
    binary_generated_samples, generated_samples, z = sample_from_model(model, latent_dim, nsamples, device)
    
    print(f"\tGenerated samples: {binary_generated_samples.shape[0]}")
    genome_sizes = binary_generated_samples.sum(axis=1)
    print(f"\tMedian genome size: {np.median(genome_sizes)}")
    print(f"\tMin/Max of genome sizes: {np.min(genome_sizes)} - {np.max(genome_sizes)}")

    plot_samples_distribution(binary_generated_samples, f"{FIGURE_DIR}/plot_full_samples_v3_{weight}.pdf", "dodgerblue", 2000, 5000)
    np.save(f"{FIGURE_DIR}/data_full_samples_v3_{weight}.npy", binary_generated_samples)

    write_samples_to_dataframe(binary_generated_samples, all_genes, f"{FIGURE_DIR}/data_full_samples_df_{weight}.csv")

    latents = get_latent_variables(model, test_loader, device)
    data_pca = PCA(n_components=2).fit_transform(latents)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2']).assign(phylogroup=test_labels)

    plt.figure(figsize=(4,4))
    sns.scatterplot(x='PC1', y='PC2', hue='phylogroup', data=df_pca)
    plt.legend(fontsize=8)
    plt.savefig(f"{FIGURE_DIR}/plot_full_pca_latent_v3_{weight}.pdf", format="pdf", bbox_inches="tight")

    essential_genes_count_per_sample = count_essential_genes(binary_generated_samples, essential_gene_positions)
    plot_essential_genes_distribution(essential_genes_count_per_sample, f"{FIGURE_DIR}/plot_full_essential_genes_v3_{weight}.pdf", "violet", 250, 327)

    ### FOCUSSED SAMPLING
    min_ones_index = np.argmin(binary_generated_samples.sum(axis=1))
    latent_distances = np.linalg.norm(generated_samples - generated_samples[min_ones_index], axis=1)
    closest_latent_index = np.argmin(latent_distances)
    
    z_of_interest = z[closest_latent_index].unsqueeze(0)
    with torch.no_grad():
        noise = torch.randn(nsamples, latent_dim, device=device) * 0.1
        additional_generated_samples = model.decode(z_of_interest + noise).cpu().numpy()
    additional_generated_samples = (additional_generated_samples > 0.5).astype(float)

    print()
    print(f"\tGenerated additional samples: {additional_generated_samples.shape[0]}")
    genome_sizes = additional_generated_samples.sum(axis=1)
    print(f"\tMedian genome size: {np.median(genome_sizes)}")
    print(f"\tMin/Max of genome sizes: {np.min(genome_sizes)} - {np.max(genome_sizes)}")

    plot_samples_distribution(additional_generated_samples, f"{FIGURE_DIR}/plot_focus_samples_v3_{weight}.pdf", "dodgerblue", 2000, 5000)
    np.save(f"{FIGURE_DIR}/data_focus_samples_v3_{weight}.npy", additional_generated_samples)

    write_samples_to_dataframe(additional_generated_samples, all_genes, f"{FIGURE_DIR}/data_focus_samples_df_{weight}.csv")

    essential_genes_count_per_sample = count_essential_genes(additional_generated_samples, essential_gene_positions)
    plot_essential_genes_distribution(essential_genes_count_per_sample, f"{FIGURE_DIR}/plot_focus_essential_genes_v3_{weight}.pdf", "violet", 250, 327)

    total_genes_count_per_sample = additional_generated_samples.sum(axis=1)
    plot_essential_vs_total(essential_genes_count_per_sample, total_genes_count_per_sample, f"{FIGURE_DIR}/plot_focus_essential_vs_total_v3_{weight}.pdf")
