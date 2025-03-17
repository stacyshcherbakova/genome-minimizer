## Combines all the fasta sequences in your directory 

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.directories import *

fasta_dir = os.path.join(DATA_DIR, "generated_genomes") 
os.makedirs(fasta_dir, exist_ok=True)
output_file = "combined_sequences.fasta"

with open(output_file, 'w') as outfile:
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fasta"):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            with open(fasta_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")

print(f"All sequences have been combined into {output_file}.")
