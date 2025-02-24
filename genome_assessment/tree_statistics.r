## Calculates newick tree statictics

library(ape)
library(phytools)
library(treestats)
library(treebalance)

data_dir <- "/data/"
phylo_tree <- read.tree(data_dir+"upgma_tree_final_dataset.newick")

tree_height <- max(nodeHeights(phylo_tree))
cat("Tree height using nodeHeights:", tree_height, "\n")

pairwise_dist_matrix <- as.matrix(read.csv(data_dir+"pairwise_distances_final_dataset.csv", header=FALSE))

avg_pairwise_distance <- mean(pairwise_dist_matrix[lower.tri(pairwise_dist_matrix)])
cat("Average pairwise distance:", avg_pairwise_distance, "\n")

colless_val <- colless(phylo_tree)
cat("Colless imbalance index:", colless_val, "\n")

sackin_val <- sackin(phylo_tree)
cat("Sackin index:", sackin_val, "\n")

height_val <- tree_height(phylo_tree)
cat("Tree height:", height_val, "\n")

blum_val <- blum(phylo_tree)
cat("Blum index:", blum_val, "\n")

sym_val <- sym_nodes(phylo_tree)
cat("Total number of symmetric internal nodes:", sym_val, "\n")
