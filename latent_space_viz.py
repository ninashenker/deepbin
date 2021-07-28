import pytorch_lightning as pl
from collections import defaultdict
import sys
from sklearn.cluster import KMeans
from mlm_train import GenomeKmerDataset, BertBin
import argparse
import psutil
import pathlib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import vamb

def evaluate(model, data_loader, name, num_batches=None, visualize=True):
    validation_outputs=[]
    for i, batch in enumerate(data_loader):
        print(f"{i}/{num_batches}")
        if num_batches is not None and i >= num_batches:
            break

        outputs = model(batch[0]) 

        print(psutil.virtual_memory().percent)
        print('---')

        hidden_states = [x.detach().cpu().numpy() for x in outputs[1]]
        taxonomy_labels = batch[1] 
        contig_names = batch[2]
        validation_outputs.append({ 'hidden_states': hidden_states, 
                                    'taxonomy': taxonomy_labels,
                                    'contig_names': contig_names})

    num_hidden_states = len(hidden_states)
    for collapse_fn in ["mean_-1", "max_-1", "mean_-2", "max_-2", "pca1", "pca3", "flatten"]:
        print(f"collapse_fn: {collapse_fn}")
        for hidden_state_i in range(num_hidden_states):
            print(f"hidden_state_i: {hidden_state_i}")

            hidden_states= [x['hidden_states'][hidden_state_i] for x in validation_outputs] 
            combined_hidden_states = np.concatenate(hidden_states)
            if collapse_fn == "mean_-1":
                combined_feature_space = np.mean(combined_hidden_states, axis=-1)
            elif collapse_fn == "max_-1":
                combined_feature_space = np.max(combined_hidden_states, axis=-1)
            elif collapse_fn == "mean_-2":
                combined_feature_space = np.mean(combined_hidden_states, axis=-2)
            elif collapse_fn == "max_-2":
                combined_feature_space = np.max(combined_hidden_states, axis=-2)
            elif collapse_fn == "flatten":
                b, s, f = combined_hidden_states.shape
                combined_feature_space = combined_hidden_states.reshape(b, s * f)
            elif collapse_fn == "pca3":
                b, s, f = combined_hidden_states.shape
                components = 3
                combined_feature_space = np.zeros((b, s * components))
                pca = PCA(n_components=components)
                for b_i in range(b):
                    pca.fit(combined_hidden_states[b_i])
                    projection = pca.transform(combined_hidden_states[b_i])
                    combined_feature_space[b_i] = projection.reshape(s * components)
            elif collapse_fn == "pca1":
                b, s, f = combined_hidden_states.shape
                components = 1
                combined_feature_space = np.zeros((b, s * components))
                pca = PCA(n_components=components)
                for b_i in range(b):
                    pca.fit(combined_hidden_states[b_i])
                    projection = pca.transform(combined_hidden_states[b_i])
                    combined_feature_space[b_i] = projection.reshape(s * components)
            else:
                raise ValueError(f"{collapse_fn} not supported")

            labels = [x['taxonomy'] for x in validation_outputs]
            new_labels= [item for t in labels for item in t]

            pca = PCA(n_components=2)
            pca.fit(combined_feature_space)
            projection = pca.transform(combined_feature_space)
            genome_to_color_id = {k: i for k, i in zip(sorted(set(new_labels)), range(10))}

            if visualize: 
                genome_keys = genome_to_color_id.keys()
                targets = list(genome_to_color_id[x] for x in new_labels)
                plt.figure(figsize=(7, 7))
                scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
                plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)

                os.makedirs(name, exist_ok=True)
                plt.savefig(f"{name}/viz_{hidden_state_i}_{collapse_fn}_pca.png")
                plt.clf()

                tsne = TSNE(n_components=2, perplexity=30)
                projection = tsne.fit_transform(combined_feature_space)

                genome_to_color_id = {k: i for k, i in zip((set(new_labels)), range(10))}
                genome_keys = genome_to_color_id.keys()
                targets = list(genome_to_color_id[x] for x in new_labels)
                plt.figure(figsize=(7, 7))
                scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
                plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)
                
                plt.savefig(f"{name}/viz_{hidden_state_i}_{collapse_fn}_tsne.png")

            kmeans_clusters = KMeans(n_clusters = len(genome_to_color_id), init = 'k-means++', random_state=None).fit_predict(projection)
            
            #ATTN should have this file loaded in previously???
            reference_path = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'
            with open(reference_path) as reference_file:
                reference = vamb.benchmark.Reference.from_file(reference_file)

            taxonomy_path = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
            with open(taxonomy_path) as taxonomy_file:
                    reference.load_tax_file(taxonomy_file)
            
            # Target format:
            # clusters = {
            #     "bin1": [ "contig1", "contig2"],
            #     "bin2": [ "contig3""],
            # }

            # Current format:
            # Index is index to contig
            # Value is cluster name
            # kmeans = [ 0, 0, 1 ]

            #Format of contig_names
            #list_of_tuples = [(x,y) for x in kmeans_clusters for y in contig_name]
            #clusters = {k: [v for s, v in list_of_tuples if s == k] for k, _ in list_of_tuples}
            #print(clusters)

            clusters = defaultdict(list)
            contig_names= [x['contig_names'] for x in validation_outputs] 
            combined_contig_names = np.concatenate(contig_names)
            
            for i, x in enumerate(kmeans_clusters):
                clusters[x].append(combined_contig_names[i])
            
            deepbin_bins = vamb.benchmark.Binning(clusters, reference, minsize=1)
            for rank in deepbin_bins.summary():
                print('\t'.join(map(str, rank)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--ckpt_path",
            type=str,
            help="Loads in model from checkpoint path provided",
            default=None,
            required=True
            )
    parser.add_argument(
            "--val_contigs",
            type=str,
            help="Val_contigs contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    parser.add_argument(
            "--train_contigs",
            type=str,
            help="Train_contigs contains path to each of the per-sample assemblies (.fna.gz)",
            required=True
            )
    parser.add_argument(
            "--visualize",
            type=bool,
            help="Option to choose to do the visualizations",
            default=True,
            required=False
            )
    args = parser.parse_args()
    
    val_dataset = GenomeKmerDataset(args.val_contigs, cache_name="viz_val", genomes=None)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    train_dataset = GenomeKmerDataset(args.train_contigs, cache_name="viz_train", genomes=None)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)

    print('train length', len(train_dataloader))
    print('val length', len(val_dataloader))

    model = BertBin.load_from_checkpoint(args.ckpt_path, kmer_dataset=val_dataset, val_dataset=val_dataset)
    model.eval()
    
    evaluate(model, val_dataloader, name=f"viz_out/viz_val_{pathlib.Path(args.ckpt_path).stem}", num_batches=None, visualize = args.visualize) #1000
    evaluate(model, train_dataloader, name=f"viz_out/viz_train_{pathlib.Path(args.ckpt_path).stem}", num_batches=None, visualize = args.visualize) #31 


    mincontiglength=254
    
    #Created in mlm train
    contig_list = self.create_contig_file(args.train_contigs)

    #TNF calculation
   # tnfs_per_fasta = []
   # contignames_per_fasta = []
   # for fasta in contig_list:
    #    with vamb.vambtools.Reader(fasta, 'rb') as tnffile:
     #       tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile, minlength=mincontiglength)
      #      tnfs_per_fasta.append(tnfs)
       #     contignames_per_fasta.extend(contignames)

    #tnfs_per_fasta = np.concatenate(tnfs_per_fasta)

if __name__ == "__main__":
    main()
