from typing import Dict
import random
import pickle
from collections import OrderedDict, defaultdict
from functools import partial

import lineflow as lf
import lineflow.datasets as lfds
import lineflow.cross_validation as lfcv

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel, BertConfig, DNATokenizer, BertForMaskedLM, BertForPreTraining
import argparse
import os
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, List, Tuple
from copy import deepcopy

import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
device = torch.device("cpu")

class KmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4):
        cache_file = '/mnt/data/CAMI/DNABERT/all_samples_validation.pickle'

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                self.tokens_dict = pickle.load(fp)
        else:
            self.tokenizer = DNATokenizer.from_pretrained('dna4')
            self.contigs = contigs
            contig_list = self.create_contig_file_list(contigs)    
            sequence = self.file2seq(contig_list)
            kmers = self.seq2kmer(sequence,k)
            padded_kmers = self.create_padding(kmers)
            self.tokens_dict = self.tokenize_all(padded_kmers)
            
            with open(cache_file, 'wb') as fp:
                pickle.dump(self.tokens_dict, fp)

        self.tokens = list(self.tokens_dict.values())
        self.tokenizer = DNATokenizer.from_pretrained('dna4')

    def __getitem__(self, idx):
        print("Getting item index", idx)
        kmer_item = self.tokens[idx]
        segment = random.choice(kmer_item)
        return segment

    def __len__(self):
        print("Getting length")
        return len(self.tokens)

    def create_contig_file_list(self, path_to_contig_file):
        print('Creating contig list from assemblies')
        contig_list = []
        with open(path_to_contig_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                contig_list.append(line)
        return contig_list

    def file2seq(self, contig_list):
        print('Creating sequence')
        seq_dict = defaultdict(str)

        for file in contig_list:
            with open(file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, seq_dict,k):
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
    
    def create_padding(self, kmers_dict):
        print('Padding the sequences')
        for key, kmers in kmers_dict.items():
            kmers_split = kmers.split() 
            token_inputs = [kmers_split[i:i+512] for i in range(0, len(kmers_split), 512)]
            num_to_pad = 512 - len(token_inputs[-1])
            token_inputs[-1].extend(['[PAD]'] * num_to_pad)
            kmers_dict[key] = token_inputs
        return kmers_dict
    
    def tokenize_all(self, kmers_dict):
        print('Tokenizing')
        #self.tokenizer = DNATokenizer.from_pretrained('dna4')

        for i, (key, kmer_512_segments) in enumerate(kmers_dict.items()):
            for idx, segment in enumerate(kmer_512_segments):
                tokenized_sequence = self.tokenizer.encode_plus(segment, add_special_tokens=True, max_length=512)["input_ids"]
                tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
                kmers_dict[key][idx] = tokenized_sequence
        return kmers_dict
    
 
NUM_LABELS = 1
MAX_LENGTH = 512

MASK_LIST = {
            "3": [-1, 1],
            "4": [-1, 1, 2],
            "5": [-2, -1, 1, 2],
            "6": [-2, -1, 1, 2, 3]
                        }
class BertBin(pl.LightningModule):

    def __init__(self, kmer_dataset, val_dataset):
        super(BertBin, self).__init__()
        print("Activating BERTBIN BBY")
        dir_to_pretrained_model = '/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0'

        config = BertConfig.from_pretrained('/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0/config.json', output_hidden_states=True, return_dict=True)
        self.model = BertForMaskedLM.from_pretrained(dir_to_pretrained_model, config=config)
        
        self._train_dataloader = DataLoader(kmer_dataset, batch_size=2, shuffle=True, num_workers=7, drop_last=True)
        self._val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=7, drop_last=False)
        print('train length', len(self._train_dataloader))
        print('val length', len(self._val_dataloader))
        
        self.train_tokenizer = kmer_dataset.tokenizer
        self.val_tokenizer = val_dataset.tokenizer

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                    "param": [p for n,p in self.model.named_parameters() if not any (nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {"params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

        optimizer = AdamW(
                self.model.parameters(),
                lr=2e-5,
                )
        return optimizer
    
    def mask_tokens(self, inputs, tokenizer, mlm_probability=0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        print("Masking tokens")
        mask_list = MASK_LIST[tokenizer.kmer]

        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        
        labels = inputs.clone().cuda()
            # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert)
        probability_matrix = torch.full(labels.shape, mlm_probability).cuda()
        special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device = probability_matrix.device), value=0.0)
        if tokenizer._pad_token is not None:
            padding_mask = labels.eq(tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask.cuda(), value=0.0)  

        masked_indices = torch.bernoulli(probability_matrix).bool()

        #change masked indices
        masks = deepcopy(masked_indices)
        for i, masked_index in enumerate(masks):
            end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
            mask_centers = set(torch.where(masked_index==1)[0].tolist())
            new_centers = deepcopy(mask_centers)
            for center in mask_centers:
                for mask_number in mask_list:
                    current_index = center + mask_number
                    if current_index <= end and current_index >= 1:
                        new_centers.add(current_index)
            new_centers = list(new_centers)
            masked_indices[i][new_centers] = True
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().cuda() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().cuda() & masked_indices & ~indices_replaced 
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].cuda()

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def training_step(self, batch, batch_idx):
        print("Start training")
        inputs, labels = self.mask_tokens(batch, self.train_tokenizer) #if True else (batch, batch)
        outputs = self.model(inputs, masked_lm_labels=labels) #if True else model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers
        
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("Start validation")
        inputs, labels = self.mask_tokens(batch[0], self.val_tokenizer) #if True else (batch[0], batch[0])
        outputs = self.model(inputs, masked_lm_labels=labels) #if True else model(inputs, labels=labels)
        val_loss = outputs[0]
        hidden_states = outputs[2]
        embedding_output = hidden_states[0]
        #attention_hidden_states = hidden_states[1:]
        last_hidden_state = hidden_states[1]
        cls = torch.mean(last_hidden_state, 1)
        self.log('val_loss', val_loss)
        taxonomy_labels = batch[1] 
        
        return {'loss': val_loss.cpu(), 'prediction': cls.cpu(), 'taxonomy': taxonomy_labels}
    
    def validation_epoch_end(self, validation_step_outputs):
        pred = [x['prediction'] for x in validation_step_outputs] 
        combined_feature_space = torch.cat(pred)
        print('feature shape', combined_feature_space.shape)
        labels = [x['taxonomy'] for x in validation_step_outputs]
        new_labels= [item for t in labels for item in t]

        pca = PCA(n_components=2)
        pca.fit(combined_feature_space)
        projection = pca.transform(combined_feature_space)

        genome_to_color_id = {k: i for k, i in zip((set(new_labels)), range(10))}
        print(genome_to_color_id)
        genome_keys = genome_to_color_id.keys()
        targets = list(genome_to_color_id[x] for x in new_labels)
        plt.figure(figsize=(7, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
        plt.text(3, 2.25, 'epoch:{nr}'.format(nr = self.current_epoch), color='black', fontsize=12)
        plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)


        plt.savefig('epoch{nr}.png'.format(nr = self.current_epoch))
        plt.clf()

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

class GenomeKmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4, genomes=10):

        cache_file = '/mnt/data/CAMI/DNABERT/all_samples_validation_2.pickle'

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                self.tokens_dict = pickle.load(fp)
        else:
            self.tokenizer = DNATokenizer.from_pretrained('dna4')
            self.contigs = contigs
            contig_list = self.create_contig_file_list(contigs)    
            sequence = self.file2seq(contig_list)
            kmers = self.seq2kmer(sequence,k)
            padded_kmers = self.create_padding(kmers)
            self.tokens_dict = self.tokenize_all(padded_kmers)
            
            with open(cache_file, 'wb') as fp:
                pickle.dump(self.tokens_dict, fp)

        self.tokens = list(self.tokens_dict.values())
        self.tokenizer = DNATokenizer.from_pretrained('dna4')
        taxonomy = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
        contig_to_genome = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'

        contig_to_genome_df = pd.read_csv(contig_to_genome, sep='\t', header=None)
        contig_to_genome_df = contig_to_genome_df.rename(columns={0: 'contig_name', 1: 'genome'})
        
        taxonomy_df = pd.read_csv(taxonomy, sep='\t', header = None)
        taxonomy_df = taxonomy_df.rename(columns={0: 'genome', 1: 'species', 2: 'genus'})

        merged_df = pd.merge(contig_to_genome_df, taxonomy_df, how="left", on=["genome"])
        
        genome_dict = dict()
        #NUM_TO_GROUPS_TO_PLOT = 10
        GROUP_KEY = "species"
        
        i = 0
        genome_dict = dict()
        list_of_tuples = []
        groups = list(merged_df.groupby(GROUP_KEY))
        random.shuffle(groups)
        for x_name, x in groups:
            if i >= 10:
                break
            genome_dict[x_name] = x['contig_name'].tolist()
            group_size = len(x)
            i += 1
        list_of_tuples = [(k, i) for k, l in genome_dict.items() for i in l if len(self.tokens_dict[i]) > 0]
        self.tuples = list_of_tuples
        random.shuffle(self.tuples)
        with open('tuples.txt', 'w') as fw:
            print(self.tuples, file=fw)
        

    def __getitem__(self, idx):
        species_contig_tuple = self.tuples[idx]
        genome_id = species_contig_tuple[0]
        segments = self.tokens_dict[species_contig_tuple[1]]
        segment = random.choice(segments)
        return segment, genome_id

    def __len__(self):
        print("Getting length")
        return len(self.tuples)

    def create_contig_file_list(self, path_to_contig_file):
        print('Creating contig list from assemblies')
        contig_list = []
        with open(path_to_contig_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                contig_list.append(line)
        return contig_list

    def file2seq(self, contig_list):
        print('Creating sequence')
        seq_dict = defaultdict(str)

        for file in contig_list:
            with open(file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, seq_dict,k):
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
    
    def create_padding(self, kmers_dict):
        print('Padding the sequences')
        for key, kmers in kmers_dict.items():
            kmers_split = kmers.split() 
            token_inputs = [kmers_split[i:i+512] for i in range(0, len(kmers_split), 512)]
            num_to_pad = 512 - len(token_inputs[-1])
            token_inputs[-1].extend(['[PAD]'] * num_to_pad)
            kmers_dict[key] = token_inputs
        return kmers_dict
    
    def tokenize_all(self, kmers_dict):
        print('Tokenizing')
        #self.tokenizer = DNATokenizer.from_pretrained('dna4')

        for i, (key, kmer_512_segments) in enumerate(kmers_dict.items()):
            for idx, segment in enumerate(kmer_512_segments):
                tokenized_sequence = self.tokenizer.encode_plus(segment, add_special_tokens=True, max_length=512)["input_ids"]
                tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
                kmers_dict[key][idx] = tokenized_sequence
        return kmers_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--contigs",
            type=str,
            help="Contigs.txt contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    parser.add_argument(
            "--val_contigs",
            type=str,
            help="Contigs.txt contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    args = parser.parse_args()
    kmers_dataset = KmerDataset(args.contigs)    
    dataset_length = len(kmers_dataset)
    print('Dataset length', dataset_length)
    print('contig name', kmers_dataset[1])
    val_dataset = GenomeKmerDataset(args.val_contigs)
    trainer = pl.Trainer(
            gpus=1,
            num_sanity_val_steps = 3,
            limit_val_batches = 100,
            limit_train_batches=100,
            max_epochs = 1
            )
    model = BertBin(kmers_dataset, val_dataset)
    trainer.fit(model)
if __name__ == "__main__":
    main()
