import os

# Paths to data and other files 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("PROJECT_ROOT:", PROJECT_ROOT)

TEN_K_DATASET = PROJECT_ROOT + "/data/F4_complete_presence_absence.csv"
TEN_K_DATASET_PHYLOGROUPS = PROJECT_ROOT + "/data/accessionID_phylogroup_BD.csv"
TEN_K_DATASET_METADATA = PROJECT_ROOT + "/data/metadata_BD.csv"
PAPER_ESSENTIAL_GENES = PROJECT_ROOT + "/data/essential_genes.csv"
ESSENTIAL_GENES_POSITIONS = PROJECT_ROOT + "/data/essential_gene_positions.pkl"
MINIMIZED_GENOME = PROJECT_ROOT + "/data/minimized_genome.fasta"
WILD_TYPE_SEQUENCE = PROJECT_ROOT + "/data/wild_type_sequence.gb"
DATA_DIR = PROJECT_ROOT+"/data/"