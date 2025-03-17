## checks if you have any identical sequences in your fasta file 

import os
import sys
from Bio import SeqIO
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.directories import *

file_path = DATA_DIR+"generated_genomes.fasta"
sequences = list(SeqIO.parse(file_path, "fasta"))

sequence_dict = defaultdict(list)

for record in sequences:
    sequence_dict[str(record.seq)].append(record.id)

duplicates = {seq: ids for seq, ids in sequence_dict.items() if len(ids) > 1}

if duplicates:
    print("Identical sequences found:")
    for seq, ids in duplicates.items():
        print(f"Sequence: {seq[:30]}... (truncated) is shared by records: {', '.join(ids)}")
else:
    print("No identical sequences found.")
