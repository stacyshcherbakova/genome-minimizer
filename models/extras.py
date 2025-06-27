import pandas as pd 
import torch 
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re
from models.VAE_models.VAE_model import *
from models.VAE_models.VAE_model_enhanced import *

def plot_essential_vs_total(essential_counts, total_counts, output_path):
    plt.figure(figsize=(4,4))
    plt.scatter(total_counts, essential_counts, color='violet')
    sns.regplot(x=total_counts, y=essential_counts, scatter=False, color='black')
    plt.xlabel("Total Genes")
    plt.ylabel("Essential Genes")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()

def extract_prefix(gene):
    match = re.match(r"([a-zA-Z0-9]+)", gene)
    if match:
        return match.group(1)
    return gene

def cleaned_100_samples_lists(essential_genes_count_per_sample, additional_generated_samples, all_genes, datatset_EG):
    # Step 1: Get the top 100 essential gene counts
    top_100_values = np.sort(essential_genes_count_per_sample)[-100:][::-1]

    # Step 2: Find the sequence indices in the array
    sequence_indices = []
    for value in top_100_values:
        indices = np.where(essential_genes_count_per_sample == value)[0]
        sequence_indices.extend(indices)

    # Ensure we only get the first 100 unique indices in case of duplicates
    sequence_indices = sequence_indices[:100]

    # Step 3: Get the samples from additional_generated_samples
    samples = additional_generated_samples[sequence_indices]

    # Step 4: Find what genes they have present
    present_genes_lists = []
    for sample in samples:
        present_genes = all_genes[:-1][sample == 1]
        present_genes_lists.append(present_genes)

    # Step 5: Clean up the gene names and add essential genes
    cleaned_genes_lists = []
    for genes in present_genes_lists:
        cleaned_gene_names = [extract_prefix(name) for name in genes]
        cleaned_gene_names.extend(datatset_EG) 
        cleaned_genes_lists.append(cleaned_gene_names)

    return cleaned_genes_lists

def count_essential_genes(binary_generated_samples, essential_gene_positions):
    '''
    count_essential_genes function counts essential genes in the generates samples

    Prameters:
    ----------
    binary_generated_samples - a 10000 x 55390 array with 10k samples ans 55390 boolean values 
    corresponding to genes

    essential_gene_positions - a boolean, pre-calcualted mask to spot essential genes

    Returns:
    -------
    essential_genes_count_per_sample - a 10000 element array where each elemnt shows the total number 
    of essential genes in that sample

    '''

    nsamples = binary_generated_samples.shape[0]

    binary_generated_samples = binary_generated_samples.astype(int)

    essential_genes_count_per_sample = np.zeros(nsamples, dtype=int)

    for sample_index in range(nsamples):
        present_essential_genes = 0
        
        for _, positions in essential_gene_positions.items():
            if len(positions) == 1:
                pos = positions[0]
                if pos < binary_generated_samples.shape[1]:
                    if binary_generated_samples[sample_index, pos] != 0:
                        present_essential_genes += 1
            else:
                for pos in positions:
                    if pos < binary_generated_samples.shape[1]:
                        if binary_generated_samples[sample_index, pos] != 0:
                            present_essential_genes += 1
                            break

        essential_genes_count_per_sample[sample_index] = present_essential_genes

    return essential_genes_count_per_sample

def plot_essential_genes_distribution(essential_genes_count_per_sample, figure_name, plot_color, x_min=0, x_max=0):
    '''
    plot_essential_genes_distribution fucntion plot the frequency of essential genes of the samples

    Prameters:
    ----------
    essential_genes_count_per_sample - a 10000 element array where each elemnt shows the total number 
    of essential genes in that sample (counted by count_essential_genes function)

    figure_name - name of the pdf figure

    plot_color - color of the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 

    '''

    # mean = np.mean(essential_genes_count_per_sample)
    median = np.median(essential_genes_count_per_sample)
    min_value = np.min(essential_genes_count_per_sample)
    max_value = np.max(essential_genes_count_per_sample)

    plt.figure(figsize=(4,4), dpi=300)
    plt.hist(essential_genes_count_per_sample, bins=10, color=plot_color)
    plt.xlim(x_min, x_max)
    plt.xlabel('Essential genes number')
    plt.ylabel('Frequency')

    # plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'), dummy_min, dummy_max] # plt.Line2D([], [], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}'),

    plt.legend(handles=handles, fontsize=8)

    plt.savefig(figure_name, format="pdf", bbox_inches="tight")

def plot_samples_distribution(binary_generated_samples, figure_name, plot_color, x_min=0, x_max=0):
    '''
    plot_samples_distribution finction plots the frequence distribution of genome sizes

    Prameters:
    ----------
    binary_generated_samples - a 10000 x 55390 array with 10k samples ans 55390 boolean values 
    corresponding to geness

    figure_name - name of the pdf figure

    plot_color - color of the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 
    
    '''

    samples_size_sum = binary_generated_samples.sum(axis=1)

    # mean = np.mean(samples_size_sum)
    median = np.median(samples_size_sum)
    min_value = np.min(samples_size_sum)
    max_value = np.max(samples_size_sum)

    plt.figure(figsize=(4,4), dpi=300)
    plt.hist(samples_size_sum, bins=10, color=plot_color)
    plt.xlim(x_min, x_max)
    plt.xlabel('Genome size')
    plt.ylabel('Frequency')

    # plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'), dummy_min, dummy_max] # plt.Line2D([], [], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}'),

    plt.legend(handles=handles, fontsize=8)

    plt.savefig(figure_name, format="pdf", bbox_inches="tight")

def load_model_enhanced(input_dim, hidden_dim, latent_dim, path_to_model):
    '''
    load_model_enhanced function loads a saved VAE model which 
    includes dropout layers

    Prameters:
    ----------
    input_dim - input dimention of the model 

    hidden_dim - hiden dimention of the model 

    latent_dim - latent dimention of the model 

    path_to_model - path to the model where its stored

    Returns:
    -------
    model - return the loaded model so it can then be subsequently use for pca plot

    binary_generated_samples - the samples array generated by the model 

    '''

    # Load trained model 
    input_dim = input_dim
    hidden_dim = hidden_dim
    latent_dim = latent_dim

    # changes layer norm layer to batch norm layer and 
    model = VAE_enhanced(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu'), weights_only=True))  
    model.eval()  

    # Generate 10 new samples
    num_samples = 1000
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # Sample from the standard normal distribution because the latent space follows normal distribution 
        generated_samples = model.decode(z).cpu().numpy() 

    threshold = 0.5
    binary_generated_samples = (generated_samples > threshold).astype(float)

    print("Generated samples (binary):\n", binary_generated_samples)
    print("\n")
    print("Generated samples (sigmoid function output):\n", generated_samples)

    return model, binary_generated_samples


def load_model(input_dim, hidden_dim, latent_dim, path_to_model):
    '''
    load_model function loads a saved VAE model 

    Prameters:
    ----------
    input_dim - input dimention of the model 

    hidden_dim - hiden dimention of the model 

    latent_dim - latent dimention of the model 

    path_to_model - path to the model where its stored

    Returns:
    -------
    model - return the loaded model so it can then be subsequently use for pca plot

    binary_generated_samples - the samples array generated by the model 
    
    '''

    # Load trained model 
    input_dim = input_dim
    hidden_dim = hidden_dim
    latent_dim = latent_dim

    # changes layer norm layer to batch norm layer and 
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu'), weights_only=True))  
    model.eval()  

    # Generate 10 new samples
    num_samples = 10000
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # Sample from the standard normal distribution because the latent space follows normal distribution 
        generated_samples = model.decode(z).cpu().numpy() 

    threshold = 0.5
    binary_generated_samples = (generated_samples > threshold).astype(float)

    print("Generated samples (binary):\n", binary_generated_samples)
    print("\n")
    print("Generated samples (sigmoid function output):\n", generated_samples)

    return model, binary_generated_samples

def l1_regularization(model, lambda_l1):
    '''
    l1_regularization function computes the L1 regularization term for a given model

    Parameters:
    ----------
    model - The neural network model whose parameters are to be regularized

    lambda_l1 - The regularization strength parameter for L1 regularization

    Returns:
    -------
    The computed L1 regularization term.
    
    '''

    l1_penalty = 0.0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty

def cosine_annealing_schedule(t, T, min_beta, max_beta):
    '''
    cosine_annealing_schedule function computes the value of beta using a cosine annealing schedule

    Parameters:
    ----------
    t - The current time step or epoch
    T - The total number of time steps or epochs
    min_beta - The minimum value of beta
    max_beta - The maximum value of beta

    Returns:
    -------
    The computed beta value for the given time step.
    
    '''

    return min_beta + (max_beta - min_beta) / 2 * (1 + np.cos(np.pi * (t % T) / T))

# def exponential_decay_schedule(t, initial_beta, decay_rate):
#     return initial_beta * np.exp(-decay_rate * t)

def get_latent_variables(model, data_loader, device):
    '''
    get_latent_variables function extracts latent variables from a given model using a data loader.

    Parameters:
    ----------
    model - The neural network model to extract latent variables from.
    data_loader - The data loader providing the input data.
    device - The device (CPU or GPU) to perform computations on.

    Returns:
    -------
    An array of latent variables extracted from the model.

    '''

    model.eval()
    latents = []
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(torch.float).to(device)
            mean, _ = model.encode(data)
            latents.append(mean.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents

def do_tsne(n_components, latents, fig_name):
    '''
    do_tsne function performs t-SNE dimensionality reduction on latent variables and plots the results

    Parameters:
    ----------
    n_components - The number of dimensions to reduce the data to
    latents - The latent variables to be reduced
    fig_name - The name of the file to save the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 

    '''

    tsne = TSNE(n_components=n_components)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(4,4), dpi=300)
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], color='dodgerblue')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()

def do_pca(n_components, latents, fig_name):
    '''
    do_pca function performs PCA dimensionality reduction on latent variables and plots the results

    Parameters:
    ----------
    n_components - The number of principal components to reduce the data to
    latents - The latent variables to be reduced
    fig_name - The name of the file to save the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 

    '''

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(latents)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    plt.figure(figsize=(4,4), dpi=300)
    sns.scatterplot(x='PC1', y='PC2', data=df_pca, color='dodgerblue')
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()

def plot_loss_vs_epochs_graph(epochs, train_loss_vals, val_loss_vals, fig_name):
    '''
    plot_loss_vs_epochs_graph function plots the training and validation loss versus epochs

    Parameters:
    ----------
    epochs - The list of epoch numbers
    train_loss_vals - The list of training loss values
    val_loss_vals - The list of validation loss values
    fig_name - The name of the file to save the plot

    Returns:
    -------
    None, saves a pdf image of the plot in the current working directory 

    '''

    plt.figure(figsize=(4,4), dpi=300)
    plt.scatter(epochs, train_loss_vals, color='dodgerblue')
    plt.plot(epochs, train_loss_vals, label='Train Loss', color='dodgerblue')
    plt.scatter(epochs, val_loss_vals, color='darkorange')
    plt.plot(epochs, val_loss_vals, label='Validation Loss', color='darkorange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()
