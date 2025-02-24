import os
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from VAE_models.VAE_model import VAE
from extras import *

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/F4_complete_presence_absence.csv")
PHYLOGROUP_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/accessionID_phylogroup_BD.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/trained_models/v0_run/saved_VAE_v0.pt")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "models/v0/figures/")
ESSENTIAL_GENE_POSITIONS_PATH = os.path.join(PROJECT_ROOT, "data_exploration/data/essential_gene_positions.pkl")

BATCH_SIZE = 32
INPUT_DIM = 55039
HIDDEN_DIM = 1024
LATENT_DIM = 64
NUM_SAMPLES = 10000

with open(ESSENTIAL_GENE_POSITIONS_PATH, "rb") as f:
    ESSENTIAL_GENE_POSITIONS = pickle.load(f)

# Helper Functions
def create_dataloaders(data_array, labels, batch_size):
    """Create DataLoaders for train, validation, and test splits."""
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    train_data, temp_data, train_labels, temp_labels = train_test_split(data_tensor, labels, test_size=0.3, random_state=12345)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.3333, random_state=12345)
    test_phylogroups = test_labels

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    test_dataset = TensorDataset(test_data)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, test_phylogroups

# Main Script
if __name__ == "__main__":
    print("start")
    # Load and preprocess data
    large_data = pd.read_csv(DATA_PATH,index_col=0, header=0)
    print(large_data.head())
    large_data.columns = large_data.columns.str.upper()
    data_without_lineage = large_data.drop(index=['Lineage'])
    phylogroup_data = pd.read_csv(PHYLOGROUP_PATH, index_col=[0], header=[0])

    merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
    data_array_t = np.array(merged_df.iloc[:, :-1])
    phylogroups_array = np.array(merged_df.iloc[:, -1])

    # Create DataLoaders
    train_loader, test_loader, test_phylogroups = create_dataloaders(data_array_t, phylogroups_array, BATCH_SIZE)

    # Load the trained VAE model
    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Generate new samples
    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(DEVICE)
        generated_samples = model.decode(z).cpu().numpy()

    # Binary thresholding for generated samples
    binary_generated_samples = (generated_samples > 0.5).astype(float)

    # Latent space plot (Figure 2a, b, c)
    latents = get_latent_variables(model, test_loader, DEVICE)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(latents)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    df_pca['phylogroup'] = test_phylogroups

    plt.figure(figsize=(4,4))
    sns.scatterplot(x='PC1', y='PC2', hue = df_pca['phylogroup'], data=df_pca)
    handles, labels = plt.gca().get_legend_handles_labels()  
    plt.legend(handles, labels, fontsize=8) 
    plt.savefig(os.path.join(FIGURE_DIR, "latent_space_v0.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    # F1 score plot (Figure 2d, e, f)
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

    recon_x_binarized = (all_recon_x > 0.5).int()
    all_test_data_np = all_test_data.cpu().numpy().flatten()
    recon_x_binarized_np = recon_x_binarized.cpu().numpy().flatten()
    f1 = sklearn.metrics.f1_score(all_test_data_np, recon_x_binarized_np)

    plot_color = "dodgerblue"

    f1_scores = []
    for genome_x, genome in zip(recon_x_binarized.cpu(), all_test_data.cpu().int()):
        f1_scores.append(sklearn.metrics.f1_score(genome_x.numpy(), genome.numpy()))

    median = np.median(f1_scores)
    min_value = np.min(f1_scores)
    max_value = np.max(f1_scores)

    plt.figure(figsize=(4,4))
    plt.hist(f1_scores, bins=10, color=plot_color)
    plt.xlim(0.9, 1)
    plt.xlabel('F1 score')
    plt.ylabel('Frequency')
    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')
    handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'), dummy_min, dummy_max] 
    plt.legend(handles=handles, fontsize=8)
    plt.savefig(FIGURE_DIR+"f1_scores_v0.pdf", format="pdf", bbox_inches="tight")

    # Samples size distribution plot (Figure 2g, h, i)
    plot_samples_distribution(binary_generated_samples, FIGURE_DIR+"normal_genomes_distribution_v0.pdf", plot_color, 2000, 5000)
    
    # Essential genes distributio ploy (Figure 2j, k, l)
    plot_color = "violet"
    essential_genes_count_per_sample = count_essential_genes(binary_generated_samples, ESSENTIAL_GENE_POSITIONS)
    plot_essential_genes_distribution(essential_genes_count_per_sample, FIGURE_DIR+"essential_genes_distribution_v0.pdf", plot_color, 250, 327)
