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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import psutil

random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class KmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4, genomes=10):
        self.tokenizer = DNATokenizer.from_pretrained('dna4')
        self.vocab = '/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0/vocab.txt'

        #if tuples already stored, read them in - note if any of the underlying val contig samples are deleted then make sure to remove the cache or if arguments change
        tuple_cache_file = '/mnt/data/CAMI/DNABERT/nsp_train_tuples.pickle'
        if os.path.exists(tuple_cache_file):
            with open(tuple_cache_file, 'rb') as fp:
                self.train_tuples = pickle.load(fp)
            return 

        #contigs are coming in as a list of paths to the samples. we need to open all of the samples and retrieve the sequences by their contig_name
        contig_list = self.create_contig_file_list(contigs)
        for x in range(len(contig_list)):
            sample_id = x
        sequence_by_contig_name = self.file2seq(contig_list)

        #genome information is stored in the taxonomy and gsa mapping files. We need to join these together and then sample 10 genomes and store their respective contigs.
        taxonomy = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
        contig_to_genome = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'

        contig_to_genome_df = pd.read_csv(contig_to_genome, sep='\t', header=None)
        contig_to_genome_df = contig_to_genome_df.rename(columns={0: 'contig_name', 1: 'genome'})
        
        taxonomy_df = pd.read_csv(taxonomy, sep='\t', header = None)
        taxonomy_df = taxonomy_df.rename(columns={0: 'genome', 1: 'species', 2: 'genus'})

        merged_df = pd.merge(contig_to_genome_df, taxonomy_df, how="left", on=["genome"])
    
        genome_dict = dict()
        GROUP_KEY = "species"
        
        genome_dict = dict()
        groups = list(merged_df.groupby(GROUP_KEY))
        random.shuffle(groups)
        for x_name, x in groups:
            genome_dict[x_name] = x['contig_name'].tolist()

        flatten_genome_dict =[(genome_name, contig_name) for genome_name,  contig_names in genome_dict.items() for contig_name in contig_names]

        #now that we have the validation contigs, we go through and find the sequence and tokenize it and then store it to disk ready to be read from get_item.
        self.train_tuples = []
        for genome_name, contig_name in flatten_genome_dict:
            if contig_name not in sequence_by_contig_name:
                continue

            sequence = sequence_by_contig_name[contig_name]
            kmers = self.seq2kmer(sequence, k)
            padded_kmers = self.create_padding(kmers)
            tokenized_kmers = self.tokenize_all(padded_kmers)
            cache_file = '/mnt/data/CAMI/DNABERT/nsp_contigs/contig_{idx}.pt'.format(idx = contig_name)
            with open(cache_file, 'w') as fp:
                torch.save(tokenized_kmers, cache_file)

            self.train_tuples.append((genome_name, contig_name, cache_file, sample_id))

        random.shuffle(self.train_tuples)

        with open(tuple_cache_file, 'wb') as fp:
            pickle.dump(self.train_tuples, fp, protocol=4)

        print('Length of train tuples', len(self.train_tuples))

    def __getitem__(self, idx):
        #print("Getting item index", idx)
        contig_cache_tuple = self.train_tuples[idx]
        #genome_id = contig_cache_tuple[0]
        #ATTN CHANGE BACK TO 2 WHEN YOU RECACHE
        contig_file_name = contig_cache_tuple[2]
        with open(contig_file_name, 'r') as fp:
            segments = torch.load(contig_file_name)
            if len(segments) >= 2:
                segment_idx = random.randint(0, len(segments) - 2)
            else:
                segment_idx = 0
            segment = segments[segment_idx]

        if random.random() > 0.5 and len(segments) > 1:
            next_segment_idx = segment_idx + 1
            next_segment = segments[next_segment_idx]
            nsp_label = 1
        else:
            next_sentence_tuple = random.choice(self.train_tuples) # Make sure this isn't the same as contig_cach_tuple above
            contig_file_name = next_sentence_tuple[2] 
            with open(contig_file_name, 'r') as fp:
                segments = torch.load(contig_file_name)
                next_segment = random.choice(segments)
                nsp_label = 0
        
        sep_token = torch.tensor([self.tokenizer.vocab['[SEP]']])
        cls_token = torch.tensor([self.tokenizer.vocab['[CLS]']])
        #combine cls + segment + sep + segment + sep
        combined_segment = torch.cat((cls_token, segment, sep_token, next_segment, sep_token))
        #print('combined_segment', combined_segment)
        token_type_ids = torch.tensor((2 + len(segment)) * [0] + (1 + len(next_segment)) * [1])
        return combined_segment, nsp_label, token_type_ids
        #return genome_id


    def __len__(self):
        #print("Getting length")
        return len(self.train_tuples)

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
        for sample_file in contig_list:
            with open(sample_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, value, k):
        print("Converting sequence to kmers")
        #for key, value in seq_dict.items():
        kmer = [value[x:x+k] for x in range(len(value)+1-k)]
        kmers = " ".join(kmer)
        return kmers
    
    def create_padding(self, kmers):
        print('Padding the sequences')
        kmers_split = kmers.split() 
        token_inputs = [kmers_split[i:i+254] for i in range(0, len(kmers_split), 254)]
        num_to_pad = 254 - len(token_inputs[-1])
        token_inputs[-1].extend(['[PAD]'] * num_to_pad)
        return token_inputs
    
    def tokenize_all(self, kmers_254_segments):
        print('Tokenizing')
        tokenized_254_segments = []
        for idx, segment in enumerate(kmers_254_segments):
            tokenized_sequence = self.tokenizer.encode(segment, add_special_tokens=False, max_length=254)
            tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
            tokenized_254_segments.append(tokenized_sequence)
        return tokenized_254_segments
    
NUM_LABELS = 1
MAX_LENGTH = 254

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

        config = BertConfig.from_pretrained('/mnt/data/CAMI/DNABERT/pretrained_models/4-new-12w-0/config.json', output_hidden_states=True)
        self.model = BertForPreTraining.from_pretrained(dir_to_pretrained_model, config=config) 

        self._train_dataloader = DataLoader(kmer_dataset, batch_size=16, shuffle=True, num_workers=7, drop_last=True)
        self._val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=7, drop_last=False)
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
        #print("Masking tokens")
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
        #print("Start training")
        inputs, labels = self.mask_tokens(batch[0], self.train_tokenizer) 
        outputs = self.model(inputs, masked_lm_labels=labels, next_sentence_label=batch[1], token_type_ids=batch[2]) 
        train_loss = outputs[0]  # model outputs are always tuple in transformers
        #taxonomy_labels = batch[1]
        #sample_labels = batch[2]
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss
    
        
    def validation_step(self, batch, batch_idx):
        #print("Start validation")
        inputs, labels = self.mask_tokens(batch[0], self.val_tokenizer) 
        outputs = self.model(inputs, masked_lm_labels=labels, next_sentence_label=batch[1], token_type_ids=batch[2])
        val_loss = outputs[0]
        hidden_states = outputs[3]
        # embedding_output = hidden_states[0]
        last_hidden_state = hidden_states[1]
        cls = last_hidden_state[:, 0]
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, on_step=True)
        taxonomy_labels = batch[3] 
        
        return {'loss': val_loss.cpu(), 'prediction': cls.cpu(), 'taxonomy': taxonomy_labels}
    
    def validation_epoch_end(self, validation_step_outputs):
        pred = [x['prediction'] for x in validation_step_outputs] 
        combined_feature_space = torch.cat(pred)
        labels = [x['taxonomy'] for x in validation_step_outputs]
        new_labels= [item for t in labels for item in t]

        pca = PCA(n_components=2)
        pca.fit(combined_feature_space)
        projection = pca.transform(combined_feature_space)

        genome_to_color_id = {k: i for k, i in zip(sorted(set(new_labels)), range(10))}
        genome_keys = genome_to_color_id.keys()
        targets = list(genome_to_color_id[x] for x in new_labels)
        plt.figure(figsize=(7, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
        #plt.text(3, 2.25, 'epoch:{nr}'.format(nr = self.current_epoch), color='black', fontsize=12)
        plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)


        plt.savefig('nsp_epoch{nr}.png'.format(nr = self.current_epoch))
        plt.clf()

        tsne = TSNE(n_components=2, perplexity=30)
        projection = tsne.fit_transform(combined_feature_space)

        genome_to_color_id = {k: i for k, i in zip((set(new_labels)), range(10))}
        genome_keys = genome_to_color_id.keys()
        targets = list(genome_to_color_id[x] for x in new_labels)
        plt.figure(figsize=(7, 7))
        scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
        #plt.text(3, 2.25, 'epoch:{nr}'.format(nr = self.current_epoch), color='black', fontsize=12)
        plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)
        
        plt.savefig('tsne_epoch{nr}.png'.format(nr = self.current_epoch))

        kmeans_pca = KMeans(n_clusters = len(genome_to_color_id), init = 'k-means++', random_state=None).fit(projection)
        file_name = 'kmeans_{nr}.txt'.format(nr = self.current_epoch)
        with open(file_name, 'w') as fw:
            print(kmeans_pca.cluster_centers_, file=fw)
    

    def test_step(self, batch):
        return self.validation_step(self, batch)

    def test_epoch_end(self, validation_step_outputs):
        return validation_epoch_end(self, validation_step_outputs)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
    

class GenomeKmerDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, k=4, genomes=10, cache_name="nsp_val_tuples"):
        self.tokenizer = DNATokenizer.from_pretrained('dna4')

        #if tuples already stored, read them in - note if any of the underlying val contig samples are deleted then make sure to remove the cache or if arguments change
        tuple_cache_file = f"/mnt/data/CAMI/DNABERT/{cache_name}.pickle"
        if os.path.exists(tuple_cache_file):
            with open(tuple_cache_file, 'rb') as fp:
                self.tuples = pickle.load(fp)
            return 

        #contigs are coming in as a list of paths to the samples. we need to open all of the samples and retrieve the sequences by their contig_name
        contig_list = self.create_contig_file_list(contigs)
        sequence_by_contig_name = self.file2seq(contig_list)

        #genome information is stored in the taxonomy and gsa mapping files. We need to join these together and then sample 10 genomes and store their respective contigs.
        taxonomy = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
        contig_to_genome = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'

        contig_to_genome_df = pd.read_csv(contig_to_genome, sep='\t', header=None)
        contig_to_genome_df = contig_to_genome_df.rename(columns={0: 'contig_name', 1: 'genome'})
        
        taxonomy_df = pd.read_csv(taxonomy, sep='\t', header = None)
        taxonomy_df = taxonomy_df.rename(columns={0: 'genome', 1: 'species', 2: 'genus'})

        merged_df = pd.merge(contig_to_genome_df, taxonomy_df, how="left", on=["genome"])
    
        genome_dict = dict()
        GROUP_KEY = "species"
        
        i = 0
        genome_dict = dict()
        groups = list(merged_df.groupby(GROUP_KEY))
        random.shuffle(groups)
        for x_name, x in groups:
            if i >= 10:
                break
            genome_dict[x_name] = x['contig_name'].tolist()
            i += 1

        flatten_genome_dict =[(genome_name, contig_name) for genome_name,  contig_names in genome_dict.items() for contig_name in contig_names]
        
        #now that we have the validation contigs, we go through and find the sequence and tokenize it and then store it to disk ready to be read from get_item.
        self.tuples = []
        for genome_name, contig_name in flatten_genome_dict:
            if contig_name not in sequence_by_contig_name:
                continue

            sequence = sequence_by_contig_name[contig_name]
            kmers = self.seq2kmer(sequence, k)
            padded_kmers = self.create_padding(kmers)
            tokenized_kmers = self.tokenize_all(padded_kmers)
            if len(tokenized_kmers) <= 1:
                continue

            segment_idx = random.randint(0, len(tokenized_kmers) - 2)
            cache_file = '/mnt/data/CAMI/DNABERT/nsp_contigs/val_sample_{idx}.pt'.format(idx = contig_name)
            with open(cache_file, 'w') as fp:
                segment = tokenized_kmers[segment_idx]
                next_segment = tokenized_kmers[segment_idx + 1]
                torch.save([segment, next_segment], cache_file)

            self.tuples.append((genome_name, contig_name, cache_file))

        random.shuffle(self.tuples)

        with open(tuple_cache_file, 'wb') as fp:
            pickle.dump(self.tuples, fp, protocol=4)
        
        print('Length of tuples', len(self.tuples))
        

    def __getitem__(self, idx):
        #print("Getting item index", idx)
        contig_cache_tuple = self.tuples[idx]
        genome_id = contig_cache_tuple[0]
        #ATTN CHANGE BACK TO 2 WHEN YOU RECACHE
        contig_file_name = contig_cache_tuple[2]
        with open(contig_file_name, 'r') as fp:
            segment, next_segment = torch.load(contig_file_name)
            nsp_label = 1

        if random.random() > 0.5:
            next_sentence_tuple = random.choice(self.tuples)
            contig_file_name = next_sentence_tuple[2]
            with open(contig_file_name, 'r') as fp:
                 _, next_segment = torch.load(contig_file_name)
                 nsp_label = 0
        
        sep_token = torch.tensor([self.tokenizer.vocab['[SEP]']])
        cls_token = torch.tensor([self.tokenizer.vocab['[CLS]']])
        #combine cls + segment + sep + segment + sep
        combined_segment = torch.cat((cls_token, segment, sep_token, next_segment, sep_token))
        token_type_ids = torch.tensor((2 + len(segment)) * [0] + (1 + len(next_segment)) * [1])
        
        
        return combined_segment, nsp_label, token_type_ids, genome_id

    def __len__(self):
        #print("Getting length")
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
        for val_file in contig_list:
            with open(val_file, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    if line[0] == '>':
                        key = line[1:].strip('\n')
                    else:
                        seq_dict[key] += line.strip('\n')
        return seq_dict

    def seq2kmer(self, value, k):
        print("Converting sequence to kmers")
        kmer = [value[x:x+k] for x in range(len(value)+1-k)]
        kmers = " ".join(kmer)
        return kmers
    
    def create_padding(self, kmers):
        print('Padding the sequences')
        kmers_split = kmers.split() 
        token_inputs = [kmers_split[i:i+254] for i in range(0, len(kmers_split), 254)]
        num_to_pad = 254 - len(token_inputs[-1])
        token_inputs[-1].extend(['[PAD]'] * num_to_pad)
        return token_inputs
    
    def tokenize_all(self, kmers_254_segments):
        print('Tokenizing')
        tokenized_254_segments=[]
        for idx, segment in enumerate(kmers_254_segments):
            tokenized_sequence = self.tokenizer.encode(segment, add_special_tokens=False, max_length=254)
            tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
            tokenized_254_segments.append(tokenized_sequence)
        return tokenized_254_segments        


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
            help="Val_contigs contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    """
    parser.add_argument(
            "-t",
            "--training",
            type=bool,
            help="Sets model to training mode if True else sets model to testing mode if False",
            default=True,
            required=True
            )
    """
    parser.add_argument(
            "-p",
            "--ckpt_path",
            type=str,
            help="ckpt_path=/path/to/my_checkpoint.ckpt",
            default=None,
            required=False
            )
    args = parser.parse_args()
    kmers_dataset = KmerDataset(args.contigs)    
    val_dataset = GenomeKmerDataset(args.val_contigs)
    checkpoint_callback = ModelCheckpoint(
            save_weights_only=False,
            verbose=True,
            monitor='val_loss_epoch',
            save_top_k=2,
            save_last=True,
            mode='min'
            )
    trainer = pl.Trainer(
            gpus=4,
            accelerator='ddp',
            callbacks=[checkpoint_callback],
            resume_from_checkpoint = args.ckpt_path,
            num_sanity_val_steps=3,
            limit_val_batches=1600,
            )
    model = BertBin(kmers_dataset, val_dataset)
    trainer.fit(model)
#    if args.training == True:
 #       trainer.fit(model)
  #  elif args.training == False and args.ckpt_path == None:
   #     trainer.test()
   # elif args.training == False and args.ckpt_path is not None:
    #    trainer.test(ckpt_path = args.ckpt_path)

if __name__ == "__main__":
    main()
