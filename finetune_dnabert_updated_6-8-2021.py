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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
device = torch.device("cpu")

class KmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4):
        cache_file = '/mnt/data/CAMI/DNABERT/validation.pickle'

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                self.tokens = pickle.load(fp)
        else:
            self.contigs = contigs
        
            contig_list = self.create_contig_file_list(contigs)    
            sequence = self.file2seq(contig_list)
            kmers = self.seq2kmer(sequence,k)
            padded_kmers = self.create_padding(kmers)
            tokens_dict = self.tokenize_all(padded_kmers)
            self.tokens = list(tokens_dict.values())
            
            with open('validation.pickle', 'wb') as fp:
                pickle.dump(self.tokens, fp)

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
        self.tokenizer = DNATokenizer.from_pretrained('dna4')

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

        config = BertConfig.from_pretrained('/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0/config.json')
        self.model = BertForMaskedLM.from_pretrained(dir_to_pretrained_model, config=config)
        
        self._train_dataloader = DataLoader(kmer_dataset, batch_size=2, shuffle=True, num_workers=7, drop_last=True)
        self._val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=7, drop_last=False)
        self.tokenizer = kmer_dataset.tokenizer
        #self.tokenizer = val_dataset.tokenizer

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
    
    def mask_tokens(self, inputs: KmerDataset, GenomeKmerDataset, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
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
        inputs, labels = self.mask_tokens(batch, self.tokenizer) if True else (batch, batch)
        outputs = self.model(inputs, masked_lm_labels=labels) if True else model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers
        
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("Start validation")
        print('Getting batch size:', batch)
        inputs, labels = self.mask_tokens(batch, self.tokenizer) if True else (batch, batch)
        outputs = self.model(inputs, masked_lm_labels=labels) if True else model(inputs, labels=labels)
        val_loss = outputs[0]  
        self.log('val_loss', val_loss)
        return val_loss
    
    def validation_epoch_end(self, validation_step_outputs):
        for out in validation_step_outputs:
            data = load_viz_data(out)

        return self._validation_epoch_end

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):

        return self._val_dataloader

class GenomeKmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4, genomes=10):
        self.contigs = contigs
        contig_list = self.create_contig_file_list(contigs)    
        sequence = self.file2seq(contig_list)
        kmers = self.seq2kmer(sequence,k)
        padded_kmers = self.create_padding(kmers)
        self.tokens_dict = self.tokenize_all(padded_kmers)
        self.tokens = list(self.tokens_dict.values())

        taxonomy = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
        contig_to_genome = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'

        contig_to_genome_df = pd.read_csv(contig_to_genome, sep='\t', header=None)
        contig_to_genome_df = contig_to_genome_df.rename(columns={0: 'contig_name', 1: 'genome'})
        
        taxonomy_df = pd.read_csv(taxonomy, sep='\t', header = None)
        taxonomy_df = taxonomy_df.rename(columns={0: 'genome', 1: 'species', 2: 'genus'})

        merged_df = pd.merge(contig_to_genome_df, taxonomy_df, how="left", on=["genome"])
        
        genome_dict = dict()
        NUM_TO_GROUPS_TO_PLOT = 10
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
            list_of_tuples = [(k, i) for k, l in genome_dict.items() for i in l]

        self.tuples = list_of_tuples

    def __getitem__(self, idx):
        #print(idx)
        species_contig_tuple = self.tuples[idx]
        print('Species contig tuple', species_contig_tuple)
        print(self.tokens_dict)
        segments = self.tokens_dict[species_contig_tuple[1]]
        #print('TUPLE', species_contig_tuple[1])
        print('Segments', segments)
        segment = random.choice(segments)
        #print('Segment', idx_contig_name)
        return segment

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
        dict = defaultdict(str)

        for file in contig_list[0:1]:
            with open(file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        # i added this in [1:] to get rid of >
                        key = line[1:].strip('\n')
                    else:
                        dict[key] += line.strip('\n')
        return dict

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
        self.tokenizer = DNATokenizer.from_pretrained('dna4')

        for i, (key, kmer_512_segments) in enumerate(kmers_dict.items()):
            for idx, segment in enumerate(kmer_512_segments):
                tokenized_sequence = self.tokenizer.encode_plus(segment, add_special_tokens=True, max_length=512)["input_ids"]
                tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
                kmers_dict[key][idx] = tokenized_sequence
        return kmers_dict
    
    def plot_pca():
        all_concat_np = concat_tensors.detach().numpy()
        genome_group_np = all_concat_np[genome_group_indices]
        
        pca = PCA(n_components=2)
        pca.fit(genome_group_np)
        projection = pca.transform(genome_group_np)
        
        genome_to_color_id = {k: i for k, i in zip(filtered_512_df.loc[genome_group_indices][GROUP_KEY].unique(), range(10))}
        targets = filtered_512_df.loc[genome_group_indices][GROUP_KEY].apply(lambda x: genome_to_color_id[x]).tolist()
        labels = filtered_512_df.loc[genome_group_indices][GROUP_KEY].unique().tolist()
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--contigs",
            type=str,
            help="Contigs.txt contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    args = parser.parse_args()
    kmers_dataset = KmerDataset(args.contigs)    
    dataset_length = len(kmers_dataset)
    print(dataset_length)
    print(kmers_dataset[1])
    val_dataset = GenomeKmerDataset(args.contigs)
    trainer = pl.Trainer(
            #gpus=1,
            num_sanity_val_steps = -1,
            #limit_train_batches = 1
            )
    model = BertBin(kmers_dataset, val_dataset)
    trainer.fit(model)
    #trainer.test()
if __name__ == "__main__":
    main()
