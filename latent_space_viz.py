import pytorch_lightning as pl
from collections import defaultdict
import vamb.vambtools as _vambtools
import sys
from sklearn.cluster import KMeans
from vamb._vambtools import _kmercounts
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
        genome_id = batch[3]
        validation_outputs.append({ 'hidden_states': hidden_states, 
                                    'taxonomy': taxonomy_labels,
                                    'contig_names': contig_names,
                                    'genome_id': genome_id })

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

            #scree plot

            for j in range(2, 11):
                pca = PCA(n_components=j)
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

                list_of_genome_id= [x['genome_id'] for x in validation_outputs] 
                genome_id = [genome_id for sublist in list_of_genome_id for genome_id in sublist]

                list_of_contig_names= [x['contig_names'] for x in validation_outputs] 
                contig_names = [contig_name for sublist in list_of_contig_names for contig_name in sublist]

                contigs_for_genome = defaultdict(list)
                for contig_name, genome in zip(contig_names, genome_id):
                    contig = vamb.benchmark.Contig.subjectless(contig_name, 512)
                    contigs_for_genome[genome].append(contig)

                genomes = [] 
                for genome_instance in set(genome_id):
                # Example of creating 1 genome:
                    genome = vamb.benchmark.Genome(genome_instance)
                    for contig_name in contigs_for_genome[genome_instance]:
                        genome.add(contig_name)

                    genomes.append(genome)

                for genome in genomes:
                    genome.update_breadth()
                                
                reference = vamb.benchmark.Reference(genomes)
                taxonomy_path = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
                with open(taxonomy_path) as taxonomy_file:
                        reference.load_tax_file(taxonomy_file)

                clusters = defaultdict(list)
                for i, x in enumerate(kmeans_clusters):
                    clusters[x].append(contignames[i])
                
                deepbin_bins = vamb.benchmark.Binning(clusters, reference, minsize=1)
                i = 2
                with open('pca{i}.txt'.format(i=i), 'w') as f:
                    for rank in deepbin_bins.summary():
                        print('\t'.join(map(str, rank)), file = f)
                    i+=1

_KERNEL = _vambtools.read_npz("./kernel.npz")

def _project(fourmers, kernel=_KERNEL):
    "Project fourmers down in dimensionality"
    s = fourmers.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    fourmers *= 1/s
    fourmers += -(1/256)
    return np.dot(fourmers, kernel)

def _convert(raw, projected):
    "Move data from raw PushArray to projected PushArray, converting it."
    raw_mat = raw.take().reshape(-1, 256)
    projected_mat = _project(raw_mat)
    projected.extend(projected_mat.ravel())
    raw.clear()

def evaluate_tnf(dataloader):
    num_batches = len(dataloader)
    for i, batch in enumerate(dataloader):
        print(f"{i}/{num_batches}")

def create_contig_file_list(path_to_contig_file):
    contig_list = []
    with open(path_to_contig_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            contig_list.append(line)
    return contig_list


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
    parser.add_argument('--viz', dest='visualize', action='store_true')
    parser.add_argument('--no-viz', dest='visualize', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    
    val_dataset = GenomeKmerDataset(args.val_contigs, cache_name="viz_val", genomes=None)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    train_dataset = GenomeKmerDataset(args.train_contigs, cache_name="viz_train", genomes=None)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)

    contig_name_to_genome = {}

    genomes = set()
    for batch in train_dataloader:
        taxonomy_labels = batch[1] 
        contig_names = batch[2]
        genome_id = batch[3]

        for name, taxonomy, genome in zip(contig_names, taxonomy_labels, genome_id):
          contig_name_to_genome[name] = (taxonomy, genome)
          genomes.add(genome)

    mincontiglength = 10
    file_list = create_contig_file_list(args.train_contigs)

    assert len(file_list) == 1
    for fasta in file_list:
        with vamb.vambtools.Reader(fasta, 'rb') as tnffile:
            tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(tnffile, minlength=mincontiglength)

    kmeans_clusters = KMeans(n_clusters = len(genomes), init = 'k-means++', random_state=None).fit_predict(tnfs)

    genome_id = [contig_name_to_genome[x][1] for x in contignames]

    contigs_for_genome = defaultdict(list)
    for contig_name, genome, contiglength in zip(contignames, genome_id, contiglengths):
        contig = vamb.benchmark.Contig.subjectless(contig_name, contiglength)
        contigs_for_genome[genome].append(contig)

    genomes = [] 
    for genome_instance in set(genome_id):
        genome = vamb.benchmark.Genome(genome_instance)
        for contig_name in contigs_for_genome[genome_instance]:
            genome.add(contig_name)

        genomes.append(genome)

    for genome in genomes:
        genome.update_breadth()
                    
    reference = vamb.benchmark.Reference(genomes)
    taxonomy_path = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
    with open(taxonomy_path) as taxonomy_file:
            reference.load_tax_file(taxonomy_file)

    clusters = defaultdict(list)
    for i, x in enumerate(kmeans_clusters):
        clusters[x].append(contig_names[i])
    
    deepbin_bins = vamb.benchmark.Binning(clusters, reference, minsize=1)
    i = 2
    with open('pca{i}.txt'.format(i=i), 'w') as f:
        for rank in deepbin_bins.summary():
            print('\t'.join(map(str, rank)), file = f)
        i+=1

    # detokenized_val_dataset = GenomeKmerDataset(args.val_contigs, cache_name="viz_val_no_token", genomes=None, tokenize=False)
    # detokenized_val_dataloader = DataLoader(detokenized_val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    # evaluate_tnf(detokenized_val_dataloader)

    print('train length', len(train_dataloader))
    print('val length', len(val_dataloader))

    model = BertBin.load_from_checkpoint(args.ckpt_path, kmer_dataset=val_dataset, val_dataset=val_dataset)
    model.eval()

    
    # evaluate(model, val_dataloader, name=f"viz_out/viz_val_{pathlib.Path(args.ckpt_path).stem}", num_batches=2, visualize = args.visualize) #1000
    # evaluate(model, train_dataloader, name=f"viz_out/viz_train_{pathlib.Path(args.ckpt_path).stem}", num_batches=2, visualize = args.visualize) #31 


if __name__ == "__main__":
    main()
