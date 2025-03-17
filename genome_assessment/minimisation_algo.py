import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.directories import *
import logging
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
# plt.style.use('ggplot')
log_file_path = "genome_minimiser.log"
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file_path,
                    filemode='w')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class GenomeMinimiser:
    '''
    A class used to minimize a genome sequence by removing non-essential genes
    '''
    def __init__(self, record: SeqRecord, needed_genes: list, idx: int):
        self.idx = idx
        self.record = record
        self.needed_genes = needed_genes
        self.features = self.__extract_non_essential_genes()
        self.positions_to_remove = self.__get_positions_to_remove()
        self.reduced_genome_str = self.__create_minimized_sequence()
    
    def __extract_non_essential_genes(self) -> list:
        '''
        Extracts non-essential genes from the genome sequence

        Parameters:
        record (SeqRecord): the genome sequence record
        needed_genes (list): the list of needed genes

        Returns:
        non_essential_features (list): A list of non-essential gene features

        '''
        non_essential_features = []

        for feature in self.record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers.get("gene", [""])[0]
                if gene_name not in self.needed_genes:
                    non_essential_features.append(feature)

        logging.debug(f"Non-non essential genes have been found in sequence no. {self.idx}.")

        return non_essential_features

    def __get_positions_to_remove(self) -> set:
        '''
        Gets the positions of non-essential genes to remove from the genome

        Parameters:
        features (list): list of non-essential gene features

        Returns:
        positions_to_remove (set): a set of unique positions to remove

        '''
        positions_to_remove = set()

        for feature in self.features:
            start_position = int(feature.location.start)
            end_position = int(feature.location.end)
            positions_to_remove.update(range(start_position, end_position))

        logging.debug(f"BP positions to remove have been found in sequence no. {self.idx}.")

        return positions_to_remove

    def __create_minimized_sequence(self) -> str:
        '''
        Creates a minimized genome sequence by removing non-essential genes

        Parameters:
        record (SeqRecord): the genome sequence record
        positions_to_remove (set): unique positions to remove

        Returns:
        reduced_genome_str (str): the minimized genome sequence

        '''
        reduced_genome = []

        for i, base in enumerate(self.record.seq):
            if i not in self.positions_to_remove:
                reduced_genome.append(base)
        reduced_genome_str = ''.join(reduced_genome)

        logging.debug(f"Minimised sequence for sequence no. {self.idx} has been created.")

        return reduced_genome_str

    def save_minimized_genome(self, file_path: str):
        '''
        Saves the minimized genome sequence to a file

        Parameters:
        sequence (str): the minimized genome sequence
        file_path (str): the file path whre to save the reduced sequence
        idx (int): index for file naming

        Returns:
        None

        '''
        try:
            with open(file_path, "w") as output_file:
                output_file.write(f">Minimized_E_coli_K12_MG1655_{self.idx+1}\n")
                output_file.write(str(self.reduced_genome_str))
                logging.info(f"Successfully saved reduced genome: {file_path}")

        except IOError as e:
            logging.error(f"Could not write to file: {file_path} - {e}")
            raise

def load_genome(file_path: str) -> SeqRecord:
    '''
    Loads the GeneBank genome file

    Parameters:
    file_path (str): path to the GeneBank file of the genome

    Returns:
    wildtype_sequence (SeqRecord): A SeqRecord object containing the genome sequence and annotations

    '''
    try:
        logging.info(f"Attempting to load wildtype genome from {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if not file_path.endswith(".gb") and not file_path.endswith(".genbank"):
            raise ValueError(f"The file {file_path} could not be read.\nEnsure the file holds a GeneBank format.")
        
        wildtype_sequence = SeqIO.read(file_path, "genbank")
        logging.info(f"Successfully loaded genome from {file_path}")

        # print(type(wildtype_sequence))

        return wildtype_sequence
    
    except FileNotFoundError as FNF_error:
        logging.error(f"Error: {FNF_error}")
        raise
    except ValueError as VAL_error:
        logging.error(f"Error: {VAL_error}")
        raise
    except Exception as e:
        logging.error(f"Error: {e}.\nAn unexpected error occured.")
        raise


def get_needed_genes(file_path: str) -> list:
    '''
    Loads the the numpy file which contains list of lists of the present in sequences genomes 

    Parameters:
    file_path (str): path to the numpy file containing the genes

    Returns:
    present_genes (list): a lost of lists containing the needed genes

    '''
    try:
        logging.info(f"Attempting to genes file from {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        if not file_path.endswith(".npy"):
            raise ValueError(f"The file {file_path} could not be read.\nEnsure the file holds numpy format.")
        
        present_genes = np.load(file_path, allow_pickle=True).tolist()
        logging.info(f"Successfully loaded gene file from {file_path}")

        return present_genes
    
    except FileNotFoundError as FNF_error:
        logging.error(f"Error: {FNF_error}")
        raise
    except ValueError as VAL_error:
        logging.error(f"Error: {VAL_error}")
        raise
    except Exception as e:
        logging.error(f"Error: {e}.\nAn unexpected error occured.")
        raise

def main():
    print("********************************************")
    print("* Welcome to genome minimiser application! *")
    print("********************************************")
    # DATA_DIR+"/wild_type_sequence.gb"
    WILD_TYPE_SEQUENCE = input("Please enter path for the GeneBank sequence file: ")
    # DATA_DIR+"/cleaned_genes_lists_1.npy"
    
    PRESENT_GENES = input("Please enter path to the list of lists of sample genes: ")
    WEIGHT = float(input("Please enter the weight of the model: "))
    
    print("Loading datasets...")
    wildtype_sequence = load_genome(WILD_TYPE_SEQUENCE)
    original_genome_length = len(wildtype_sequence.seq)
    present_genes = get_needed_genes(PRESENT_GENES)

    print("Starting to create reduced genomes...")
    minimised_genomes_sizes = []
    for idx, needed_genes in enumerate(present_genes):
        minimiser = GenomeMinimiser(wildtype_sequence, needed_genes, idx+1)
        minimized_genome_filename = PROJECT_ROOT+f"/genome_assessment/generated_genomes/minimized_genome_{minimiser.idx}.fasta"
        # minimiser.save_minimized_genome(minimized_genome_filename)

        print(f"Minimal set of genes no. {minimiser.idx}:")
        print(f"Original genome length: {original_genome_length}")
        print(f"Final reduced genome length: {len(minimiser.reduced_genome_str)}")
        minimised_genomes_sizes.append(len(minimiser.reduced_genome_str)/1e6)
        print(f"Saved as: {minimized_genome_filename}")
        print("----------------------------------------")

    if len(present_genes) >= 100:
        print("Plotting reduced genomes size distribution graph...")
        # mean = np.mean(minimised_genomes_sizes)
        median = np.median(minimised_genomes_sizes)
        min_value = np.min(minimised_genomes_sizes)
        max_value = np.max(minimised_genomes_sizes)
        plt.figure(figsize=(4,4))
        plt.hist(minimised_genomes_sizes, bins=10, color="dodgerblue")
        plt.xlabel("Genome size (Mbp)")
        plt.ylabel("Frequency")
        # plt.axvline(mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2f}")
        plt.axvline(median, color="b", linestyle="dashed", linewidth=2, label=f"Median: {median:.2f}")
        dummy_min = plt.Line2D([], [], color="black",  linewidth=2, label=f"Min: {min_value:.2f}")
        dummy_max = plt.Line2D([], [], color="black", linewidth=2, label=f"Max: {max_value:.2f}")
        handles = [plt.Line2D([], [], color="b", linestyle="dashed", linewidth=2, label=f"Median: {median:.2f}"),
                dummy_min, dummy_max] # plt.Line2D([], [], color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2f}"),
        plt.legend(handles=handles)
        plt.savefig(PROJECT_ROOT+f"/genome_assessment/figures/minimised_genomes_distribution_{WEIGHT}.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    
    print("Script has successfully finished executing.")

if __name__ == "__main__":
    main()
