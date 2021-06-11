#!/usr/bin/env python

import argparse
from collections import defaultdict
import os
import gzip
import torch
from transformers import BertModel, BertConfig, DNATokenizer, BertTokenizer, BertForLongSequenceClassification
import pickle
import os.path

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

def seq2kmer(seq_dict,k):
    print("Converting sequence to kmers")
    """
    Convert original sequence to kmers

    Arguments:
    seq_dict -- dictionary (value), original sequence.                
    k -- int, kmer of length k specified.

    Returns:
    kmers -- dictionary (value), kmers separated by space
     """
    for key, value in seq_dict.items():
        kmer = [value[x:x+k] for x in range(len(value)+1-k)]
        kmers = " ".join(kmer)
        seq_dict[key] = kmers
    return seq_dict

def create_padding(kmers_dict):
    print('Padding the sequences')
    for key, kmers in kmers_dict.items():
        kmers_split = kmers.split() 
        token_inputs = [kmers_split[i:i+512] for i in range(0, len(kmers_split), 512)]
        num_to_pad = 512 - len(token_inputs[-1])
        token_inputs[-1].extend(['[PAD]'] * num_to_pad)
        kmers_dict[key] = token_inputs
    return kmers_dict

def convert_contigs_to_max_length(inputs_dict):
    inputs = inputs_dict.values()
    config.split = max([len(i) for i in inputs_dict.values()])
    max_length = config.split
    
    for contig in inputs:
        if len(contig) < 55:
            padding_length = max_length - len(inputs)
            print(padding_length)
    """
    pad_token = 0
    pad_token_segment_id=0
    mask_padding_with_zero=True

    if pad_on_left:
        inputs = ([pad_token] * padding_length) + inputs
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    """
def embedding(kmers_dict):
    print("Starting embedding")
    dir_to_pretrained_model = '/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0'

    config = BertConfig.from_pretrained('/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0/config.json')
    tokenizer = DNATokenizer.from_pretrained('dna4')
    #model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)
    #model = BertForLongSequenceClassification.from_pretrained(dir_to_pretrained_model, config=config)
    
    config.split = max([len(i) for i in kmers_dict.values()])
    inputs = kmers_dict.values()
    max_length = config.split

    model = BertForLongSequenceClassification.from_pretrained(dir_to_pretrained_model, config=config)

    for contig in inputs:
        if len(contig) < 55:
            padding_length = max_length - len(contig)

            pad_token = 0
            pad_token_segment_id=0
            mask_padding_with_zero=True
            pad_on_left = True
            attention_mask = [1 if mask_padding_with_zero else 0] * len(contig)

            if pad_on_left is True:
                contig = ([pad_token] * padding_length) + contig
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                contig = contig + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

    results = dict()
    for i, (key, kmers) in enumerate(kmers_dict.items()):
        print(f"{i} / {len(kmers_dict)}")
        model_input = tokenizer.encode_plus(kmers[0], add_special_tokens=True, max_length=512)["input_ids"]
        model_input = torch.tensor(model_input, dtype=torch.long)
        print(model_input.shape)
        model_input = model_input.view(1,512)
        output = model(model_input)
        results[key] = output[1]
        if i % 100 == 0:
            with open("output_long/"+ str(i) +".pickle", 'wb') as fw:
                pickle.dump(results, fw)
            del results
            results = dict()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--contigs",
            type=str,
            help="Contigs.txt contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    parser.add_argument(
            "--output_dir",
            default="deepbin_results",
            type=str,
            help="Path to your output directory"
            )
    args = parser.parse_args()
    
    cache_file = '/mnt/data/CAMI/DNABERT/long_bert.pickle'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            padded_kmers = pickle.load(fp)
    
    else:    
        contig_list = create_contig_file_list(args.contigs)    
        sequence = file2seq(contig_list)
        k=4
        kmers = seq2kmer(sequence,k)
        padded_kmers = create_padding(kmers)

        with open('long_bert.pickle', 'wb') as fp:
            pickle.dump(padded_kmers, fp)


    #full_length_seqs = convert_contigs_to_max_length(padded_kmers)     
    dnabert = embedding(padded_kmers)
    



if __name__ == "__main__":
    main()
