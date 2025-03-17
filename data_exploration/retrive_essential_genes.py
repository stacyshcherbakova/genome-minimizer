import pandas as pd
import numpy as np
import re
from collections import defaultdict
import os
import pickle
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.extras import *
from utilities.directories import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("start")

# Load and preprocess data
large_data = pd.read_csv(TEN_K_DATASET, index_col=0, header=0)
large_data.columns = large_data.columns.str.upper()
data_without_lineage = large_data.drop(index=['Lineage'])
phylogroup_data = pd.read_csv(TEN_K_DATASET_PHYLOGROUPS, index_col=[0], header=[0])

merged_df = pd
merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
essential_genes = pd.read_csv(PAPER_ESSENTIAL_GENES)
essential_genes_array = np.array(essential_genes).flatten()
all_genes = merged_df.columns
essential_genes_mask = np.isin(all_genes, essential_genes_array)
subset_not_in_essential_genes_mask = essential_genes[~np.isin(np.array(essential_genes), np.array(all_genes[essential_genes_mask]))]
subset_in_essential_genes_mask = essential_genes[np.isin(np.array(essential_genes), np.array(all_genes[essential_genes_mask]))]
absent_genes = np.array(subset_not_in_essential_genes_mask).flatten()
present_genes = np.array(subset_in_essential_genes_mask).flatten()

matched_columns = []

for gene in absent_genes:
    pattern = re.compile(f"{gene}")
    matches = [col for col in merged_df.columns if pattern.match(col) and col not in present_genes]
    matched_columns.extend(matches)


divided_genes = np.array(matched_columns)
# print(divided_genes)
# print(len(divided_genes))

divided_genes_prefixes = ['msbA', 'fabG', 'lolD', 'topA', 'metG', 'fbaA', 'higA', 'lptB', 'ssb',  'lptG', 'dnaC'] # 'higA-1', 'higA1','higA-2', 'ssbA' dont count 

not_present = np.array(list(set(absent_genes) - set(divided_genes_prefixes)))

combined_array = np.concatenate((present_genes, divided_genes))

essential_genes_mask = np.isin(all_genes, combined_array)

essential_genes_df = merged_df.loc[:, essential_genes_mask].copy()

gene_sums = essential_genes_df.sum()
zero_sum_genes = gene_sums[gene_sums == 0].index.tolist()
# print(f"Genes that are not present (overall 0 in all samples): {zero_sum_genes}")

absent_essential_genes_df = pd.DataFrame()

for prefix in absent_genes:
    cols_to_merge = essential_genes_df.filter(regex=f'^{prefix}')
    absent_essential_genes_df[prefix] = (cols_to_merge.sum(axis=1) > 0).astype(int)

intermediate = essential_genes_df.drop(columns=divided_genes)

row_sums = absent_essential_genes_df.sum(axis=0)
columns_to_add = absent_essential_genes_df.columns[row_sums != 0]

for col in absent_essential_genes_df[columns_to_add].columns:
    intermediate[col] = absent_essential_genes_df[col]

datatset_EG = list(intermediate.columns)

# Group gene positions by their prefix
groups_of_gene_positions = defaultdict(list)
for idx, gene in enumerate(all_genes):
    prefix = extract_prefix(gene)
    groups_of_gene_positions[prefix].append(idx)

# Convert defaultdict to a regular dict
groups_of_gene_positions = dict(groups_of_gene_positions)

# Print the dictionary to verify
# for prefix, positions in groups_of_gene_positions.items():
    # print(f"{prefix}: {positions}")
    


essential_gene_positions = {}
for gene in essential_genes_array:
    if gene in groups_of_gene_positions.keys():
        essential_gene_positions[gene] = groups_of_gene_positions[gene]

print(type(essential_gene_positions))

with open(PROJECT_ROOT+"/data/essential_gene_positions.pkl", "wb") as f:
    pickle.dump(essential_gene_positions, f)