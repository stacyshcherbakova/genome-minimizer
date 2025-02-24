## Combines all the fasta sequences in your directory 

import os

fasta_dir = "/data/generated_genomes"
output_file = "combined_sequences.fasta"

with open(output_file, 'w') as outfile:
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fasta"):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            with open(fasta_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write("\n")

print(f"All sequences have been combined into {output_file}.")
