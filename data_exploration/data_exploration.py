import sys
import os

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from collections import defaultdict

import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from Bio import Phylo

# append a local path
sys.path.append('../utilities')
sys.path.append('../models')

from directories import *
from extras import *

large_data = pd.read_csv(TEN_K_DATASET, index_col=[0], header=[0])
large_data.columns = large_data.columns.str.upper()
large_data.sum(axis=1).sort_values()

# Filter out genomes without metadata
data_without_lineage = large_data.drop(index=['Lineage'])
large_data_t = np.array(data_without_lineage.transpose())
print(f"Dataset shape: {large_data_t.shape}")

phylogroup_data = pd.read_csv(PHYLOGROUPS_DATA, index_col=[0], header=[0])
merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
data_array_t = np.array(merged_df.iloc[:, :-1])
phylogroups_array = np.array(merged_df.iloc[:, -1])

print("Checking dataset shapes")
print(f"Values array: {data_array_t.shape}")
print(f"Phylogroups array: {phylogroups_array.shape}")

# Figure 1a
figure_name = "plot_genome_size_final.pdf"
plot_color = "darkorchid"
frequency2 = data_without_lineage.sum(axis=0).to_numpy()
frequency2 = frequency2.reshape(-1, 1)

plot_samples_distribution(frequency2, figure_name, plot_color)

# Figure 1b
frequency1 = data_without_lineage.sum(axis=1)
mean = np.mean(frequency1)
median = np.median(frequency1)
min_value = np.min(frequency1)
max_value = np.max(frequency1)

plt.figure(figsize=(4,4))
plt.hist(frequency1, color='darkorchid', bins=20)
plt.xlabel('Number of genomes')
plt.ylabel('Number of genes')
plt.savefig("plot_gene_count_final.pdf", format="pdf", bbox_inches="tight")
plt.close()

# Figure 1c
threshold_data = []
thresholds = np.linspace(0, 50, num=50)
data_without_lineage = large_data.drop(index=['Lineage'])

for i in thresholds:
    row_sums = data_without_lineage.sum(axis=1)
    threshold_data.append(len(data_without_lineage[row_sums >= i]))

plt.figure(figsize=(4,4))
plt.scatter(thresholds, threshold_data, color='darkorchid')
plt.plot(thresholds, threshold_data, color='darkorchid')
plt.xlabel("Number of genomes")
plt.ylabel("Number of genes")
plt.savefig("plot_gene_frequency_final.pdf", format="pdf", bbox_inches="tight")
plt.close()

# Essential genes preprocessing
essential_genes = pd.read_csv(PAPER_ESSENTIAL_GENES)
essential_genes_array = np.array(essential_genes).flatten()
print(f"Total number of essential genes present in the paper: {len(essential_genes_array)}")

all_genes = merged_df.columns
essential_genes_mask = np.isin(all_genes, essential_genes_array)
subset_not_in_essential_genes_mask = essential_genes[~np.isin(np.array(essential_genes), np.array(all_genes[essential_genes_mask]))]
subset_in_essential_genes_mask = essential_genes[np.isin(np.array(essential_genes), np.array(all_genes[essential_genes_mask]))]
absent_genes = np.array(subset_not_in_essential_genes_mask).flatten()
print(f"Number of genes not present in the dataset: {len(absent_genes)}")

present_genes = np.array(subset_in_essential_genes_mask).flatten()
print(f"Number of genes present in the dataset: {len(present_genes)}") 

matched_columns = []

for gene in absent_genes:
    pattern = re.compile(f"{gene}")
    matches = [col for col in merged_df.columns if pattern.match(col) and col not in present_genes]
    matched_columns.extend(matches)


divided_genes = np.array(matched_columns)
print(divided_genes)
print(len(divided_genes))

divided_genes_prefixes = ['msbA', 'fabG', 'lolD', 'topA', 'metG', 'fbaA', 'higA', 'lptB', 'ssb',  'lptG', 'dnaC'] # 'higA-1', 'higA1','higA-2', 'ssbA' dont count 
not_present = np.array(list(set(absent_genes) - set(divided_genes_prefixes)))

print(f"Genes which are still not present in the dataset after prefix extraction: {not_present}")
print(f"Total number: {len(not_present)}")

combined_array = np.concatenate((present_genes, divided_genes))
print(f"Total umber of genes that count as essential in the dataset: {len(combined_array)}")

essential_genes_mask = np.isin(all_genes, combined_array)
essential_genes_df = merged_df.loc[:, essential_genes_mask].copy()
gene_sums = essential_genes_df.sum()
zero_sum_genes = gene_sums[gene_sums == 0].index.tolist()
print(f"Genes that are not present (overall 0 in all samples): {zero_sum_genes}")

absent_essential_genes_df = pd.DataFrame()

for prefix in absent_genes:
    cols_to_merge = essential_genes_df.filter(regex=f'^{prefix}')
    absent_essential_genes_df[prefix] = (cols_to_merge.sum(axis=1) > 0).astype(int)

intermediate = essential_genes_df.drop(columns=divided_genes)

# Adding the absent essential genes that are present in the dataframe to the overall dataframe of the genes presemt in the datatframe
row_sums = absent_essential_genes_df.sum(axis=0)
columns_to_add = absent_essential_genes_df.columns[row_sums != 0]

for col in absent_essential_genes_df[columns_to_add].columns:
    intermediate[col] = absent_essential_genes_df[col]

np.save('essential_gene_in_ds.npy', intermediate.columns.to_list())

# Figure 1d
EG_distribution = intermediate.sum(axis=1)
median = np.median(EG_distribution)
min_value = np.min(EG_distribution)
max_value = np.max(EG_distribution)

plt.figure(figsize=(4,4))
plt.hist(EG_distribution, color='darkorchid', bins=50)
plt.xlim(300, 328)
plt.xlabel('Essential gene number')
plt.ylabel('Frequency')
plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

handles = [plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'),dummy_min, dummy_max]

plt.legend(handles=handles, fontsize=8)
plt.savefig("plot_EG_number.pdf", format="pdf", bbox_inches="tight")
plt.close()

# Figure 2a
pca = PCA(n_components=2)
data_pca = pca.fit_transform(merged_df.iloc[:, :-1])
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(4,4))
sns.scatterplot(x='PC1', y='PC2', hue = merged_df.Phylogroup.tolist(), data=df_pca)
handles, labels = plt.gca().get_legend_handles_labels()  
plt.legend(handles, labels, fontsize=8) 
plt.savefig("plot_PCA_by_phylogroup.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Figure 2b
# plt.figure(figsize=(4, 4))
# Phylo.draw(tree, do_show=False)
# plt.savefig("tree_plot.pdf", format='pdf', dpi=300)
# plt.show()