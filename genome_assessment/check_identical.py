## checks if you have any identical sequences in your fasta file 

from Bio import SeqIO
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir=PROJECT_ROOT+"/data/"

file_path = data_dir+"generated_genomes.fasta"
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
