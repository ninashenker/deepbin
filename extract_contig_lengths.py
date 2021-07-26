import argparse
from collections import defaultdict
import os
import gzip
import torch
import pickle

# read in list of per-sample assemblies
def create_contig_file_list(path_to_contig_file):
    print('Creating contig list from assemblies')
    contig_list = []
    with open(path_to_contig_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            contig_list.append(line)
    return contig_list

def file2seq(contig_list):
    print('Creating sequence')
    dict = defaultdict(str)

    for file in contig_list[0:1]:
        with open(file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if line[0] == '>':
                    key = line.strip('\n')
                else:
                    dict[key] += line.strip('\n')
    return dict

def main():
    contigs_file = '/mnt/data/CAMI/vamb/workflow/contigs.txt'
    out_path = 'contig_lengths.pickle'

    contig_list = create_contig_file_list(contigs_file)    
    sequence = file2seq(contig_list)

    # Get lengths from sequences
    lengths = {contig_name: len(seq) for contig_name, seq in sequence.items()}
    with open(out_path, 'wb') as fw:
        pickle.dump(lengths, fw)


if __name__ == "__main__":
    main()
