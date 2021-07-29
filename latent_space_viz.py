import pytorch_lightning as pl
import random
from collections import defaultdict
import vamb.vambtools as _vambtools
import sys
from sklearn.cluster import KMeans
import pickle
import csv
from vamb._vambtools import _kmercounts
from mlm_evaluate import GenomeKmerDataset, BertBin
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

#random.seed(42)

SPECIES_TO_PLOT = ['Neisseria meningitidis', 'Clostridioides difficile', 'Porphyromonas gingivalis', 'Pasteurella multocida', 'Streptococcus suis', 'Moraxella bovoculi', 'Streptococcus mutans', '[Haemophilus] ducreyi', 'Carnobacterium sp. CP1', 'Streptococcus pneumoniae']

CSV_HEADER = ['name', 'species_0.3_recall', 'species_0.4_recall', 'species_0.5_recall', 'species_0.6_recall', 'species_ 0.7_recall', 'species_0.8_recall', 'species_0.9_recall', 'species_0.95_recall', 'species_0.99_recall', 'genome_0.3_recall', 'genome_0.4_recall', 'genome_0.5_recall', 'genome_0.6_recall', 'genome_0.7_recall', 'genome_0.8_recall', 'genome_0.9_recall', 'genome_0.95_recall', 'genome_0.99_recall', 'genus_0.3_recall',  'genus_0.4_recall',  'genus_0.5_recall',  'genus_0.6_recall',  'genus_0.7_recall',  'genus_0.8_recall',  'genus_0.9_recall',  'genus_0.95_recall',  'genus_0.99_recall']

def evaluate(model, data_loader, name, file_name, num_batches=None, visualize=True):
    validation_outputs=[]
    for i, batch in enumerate(data_loader):
        print(f"{i}/{len(data_loader)}")
        if num_batches is not None and i >= num_batches:
            break

        outputs = model(batch[0]) 

        print(psutil.virtual_memory().percent)
        print('---')
        import nvidia_smi

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Total memory:", info.total)
        print("Free memory:", info.free)
        print("Used memory:", info.used)

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
        for hidden_state_i in range(num_hidden_states):

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
                method_name = f"fn{collapse_fn}_layer{hidden_state_i}_pca{j}"
                print(method_name)

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

                kmeans_clusters = KMeans(n_clusters = len(genome_to_color_id), init = 'k-means++', random_state=1).fit_predict(projection)

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
                    clusters[x].append(contig_names[i])
                
                deepbin_bins = vamb.benchmark.Binning(clusters, reference, minsize=1)

                with open('results_{x}.csv'.format(x=file_name), 'a') as f:
                    writer = csv.writer(f)
                    flatten_bins = [str(rank) for sublist in deepbin_bins.summary() for rank in sublist]
                    print(flatten_bins)
                    writer.writerow([method_name] + flatten_bins)

def plot(features, targets, legend_labels, name):
    pca = PCA(n_components=2)
    pca.fit(features)
    projection = pca.transform(features)

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        alpha=0.9,
        s=5.0,
        c=targets,
        cmap='tab10'
      )
    plt.legend(
        loc="upper left",
        prop={'size': 6},
        handles=scatter.legend_elements()[0],
        labels=legend_labels
    )

    plt.savefig(f"{name}.png")
    plt.clf()


def evaluate_tnf(dataloader, contig_file, file_name):

    # Retrive map from contig name to genome.
    contig_name_to_genome = {}
    species_to_contig_name = defaultdict(list)
    genomes_set = set()
    for batch in dataloader:
        taxonomy_labels = batch[1] 
        contig_names = batch[2]
        genome_id = batch[3]

        for name, taxonomy, genome in zip(contig_names, taxonomy_labels, genome_id):
          contig_name_to_genome[name] = (taxonomy, genome)
          species_to_contig_name[taxonomy].append(name)
          genomes_set.add(genome)

    print("evaluate_tnf: dataloader length", len(dataloader))
    print("evaluate_tnf: dataset length", len(dataloader.dataset))

    mincontiglength = 10
    file_list = create_contig_file_list(contig_file)
    assert len(file_list) == 1

    # Retrive tnf frequencies for all contigs.
    for fasta in file_list:
        with vamb.vambtools.Reader(fasta, 'rb') as tnffile:
            tnfs, contignames, contiglengths = vamb.parsecontigs.read_contigs(
                tnffile,
                minlength=mincontiglength
            )

    print("evaluate_tnf: tnfs length:", len(contignames))

    # Plot 10 genomes
    index_for_contig = {}
    for i in range(len(contignames)):
      index_for_contig[contignames[i]] = i

    tnfs_to_plot = []
    species_label = []
    plot_contigs = []
    for species in SPECIES_TO_PLOT:
        contig_names = species_to_contig_name[species]
        for contig_name in contig_names[:150]:
            contig_tnfs = tnfs[index_for_contig[contig_name]]
            tnfs_to_plot.append(contig_tnfs)
            species_label.append(species)
            plot_contigs.append(contig_name)

    tnfs_to_plot = np.stack(tnfs_to_plot)
    print(tnfs_to_plot.shape)

    genome_to_color_id = {k: i for k, i in zip(sorted(set(species_label)), range(10))}
    genome_keys = genome_to_color_id.keys()
    targets = list(genome_to_color_id[x] for x in species_label)
    plot(tnfs_to_plot, targets, genome_keys, name="tnf_gt_{dataset}".format(dataset=file_name))

    cache_file = "tnf_clusters_{dataset}.pkl".format(dataset=file_name)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            kmeans_clusters = pickle.load(fp)

    else:

        # Cluster tnfs.
        kmeans = KMeans(n_clusters = len(genomes_set), init = 'k-means++', random_state=1)
        kmeans_clusters = kmeans.fit_predict(tnfs)

        with open(cache_file, 'wb') as fp:
            pickle.dump(kmeans_clusters, fp, protocol=4)

    print("Finished clustering")

    bin_labels = [str(i) for i in range(len(SPECIES_TO_PLOT))]
    targets = []
    for contig in plot_contigs:
        target = kmeans_clusters[index_for_contig[contig]]
        targets.append(target)

    plot(tnfs_to_plot, targets, bin_labels, name="tnf_clusters_{dataset}".format(dataset=file_name))

    # Create reference file.
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
        clusters[x].append(contignames[i])

    deepbin_bins = vamb.benchmark.Binning(clusters, reference, minsize=1)

    # Save results.
    with open('results_{x}.csv'.format(x=file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        flatten_bins = [str(rank) for sublist in deepbin_bins.summary() for rank in sublist]
        print(flatten_bins)
        writer.writerow(["tnf"] + flatten_bins)


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
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=False)
    train_dataset = GenomeKmerDataset(args.train_contigs, cache_name="viz_train", genomes=None)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=False)

    train_file_name = "train"
    val_file_name = "val"
    evaluate_tnf(train_dataloader, args.train_contigs, file_name=train_file_name)
    evaluate_tnf(val_dataloader, args.val_contigs, file_name=val_file_name)

    print('train length', len(train_dataloader))
    print('val length', len(val_dataloader))

    model = BertBin.load_from_checkpoint(args.ckpt_path, kmer_dataset=val_dataset, val_dataset=val_dataset)
    model.eval()
    
    evaluate(model, val_dataloader, name=f"viz_out/viz_val_{pathlib.Path(args.ckpt_path).stem}", file_name=val_file_name, num_batches=900, visualize = args.visualize)
    evaluate(model, train_dataloader, name=f"viz_out/viz_train_{pathlib.Path(args.ckpt_path).stem}", file_name=train_file_name, num_batches=900, visualize = args.visualize)


if __name__ == "__main__":
    main()
