import random
import pytorch_lightning as pl
from collections import defaultdict
import vamb.vambtools as _vambtools
from tqdm import tqdm
from umap import UMAP
import sys
import hdbscan
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import pickle
import csv
import itertools
from vamb._vambtools import _kmercounts
from mlm_evaluate import GenomeKmerDataset, BertBin
from nsp_mlm_evaluate import GenomeKmerDatasetNSP, BertBinNSP 
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
import pandas as pd

#random.seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')

SPECIES_TO_PLOT = ['Neisseria meningitidis', 'Clostridioides difficile', 'Porphyromonas gingivalis', 'Pasteurella multocida', 'Streptococcus suis', 'Moraxella bovoculi', 'Streptococcus mutans', '[Haemophilus] ducreyi', 'Carnobacterium sp. CP1', 'Streptococcus pneumoniae']

CSV_HEADER = ['name', 'species_0.3_recall', 'species_0.4_recall', 'species_0.5_recall', 'species_0.6_recall', 'species_ 0.7_recall', 'species_0.8_recall', 'species_0.9_recall', 'species_0.95_recall', 'species_0.99_recall', 'genome_0.3_recall', 'genome_0.4_recall', 'genome_0.5_recall', 'genome_0.6_recall', 'genome_0.7_recall', 'genome_0.8_recall', 'genome_0.9_recall', 'genome_0.95_recall', 'genome_0.99_recall', 'genus_0.3_recall',  'genus_0.4_recall',  'genus_0.5_recall',  'genus_0.6_recall',  'genus_0.7_recall',  'genus_0.8_recall',  'genus_0.9_recall',  'genus_0.95_recall',  'genus_0.99_recall']

def evaluate(model, data_loader, name, file_name, layer,  num_batches=None, visualize=True, nsp=False):
    if os.path.exists("batches.pickle"):
        os.remove("batches.pickle")

    t = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, batch in t:
        # print(f"{i}/{len(data_loader)}")
        num_results = num_batches if num_batches is not None and i >= num_batches else len(data_loader)
        if num_batches is not None and i >= num_batches:
            break   

        outputs = model(batch[0].cuda()) 

        memory=psutil.virtual_memory().percent
        t.set_description(f"cpu_mem: {memory}")

        # hidden_states = [x.detach().cpu().numpy() for x in outputs[1]]
        selected_hidden_state = outputs[1][layer].detach().cpu().numpy()
        taxonomy_labels = batch[1] 
        contig_names = batch[2]
        genome_id = batch[3]
        contig_length = batch[4]
        with open('batches.pickle', 'ab') as fp:
            pickle.dump({ 'selected_hidden_state': selected_hidden_state,
                          'taxonomy': taxonomy_labels,
                          'contig_names': contig_names,
                          'genome_id': genome_id,
                          'contig_length': contig_length }, 
                           fp)
    new_labels = []
    genome_id = []
    contig_names = []

    selected_hidden_state = np.empty((len(data_loader.dataset), 512, 768), dtype='uint8')
    with open('batches.pickle', 'rb') as fr:
        try:
            i = 0
            while True:
                step_result = pickle.load(fr)
                new_labels.extend(step_result["taxonomy"])
                selected_hidden_state[i:i+data_loader.batch_size] = step_result["selected_hidden_state"]
                genome_id.extend(step_result["genome_id"])
                contig_names.extend(step_result["contig_names"])
                i += data_loader.batch_size
        except EOFError:
            pass

    print('len of selected_hidden_state', len(selected_hidden_state))
    print('len of genome ids', len(genome_id))
    print('len of labels', len(new_labels))
    print('len of contig names', len(contig_names))
    collapse_options = ["mean_-2", "max_-1", "mean_-1", "max_-2", "flatten"] #mean-2
    for collapse_fn in collapse_options:
        if collapse_fn == "mean_-1":
            combined_feature_space = np.mean(selected_hidden_state, axis=-1).astype(np.float32)
        elif collapse_fn == "max_-1":
            combined_feature_space = np.max(selected_hidden_state, axis=-1).astype(np.float32)
        elif collapse_fn == "mean_-2":
            combined_feature_space = np.mean(selected_hidden_state, axis=-2).astype(np.float32)
        elif collapse_fn == "max_-2":
            combined_feature_space = np.max(selected_hidden_state, axis=-2).astype(np.float32)
        elif collapse_fn == "flatten":
            b, s, f = selected_hidden_state.shape
            combined_feature_space = selected_hidden_state.reshape(b, s * f)
        elif collapse_fn == "pca3":
            b, s, f = selected_hidden_state.shape
            components = 3
            combined_feature_space = np.zeros((b, s * components))
            pca = PCA(n_components=components)
            for b_i in range(b):
                pca.fit(selected_hidden_state[b_i])
                projection = pca.transform(selected_hidden_state[b_i])
                combined_feature_space[b_i] = projection.reshape(s * components)
        elif collapse_fn == "pca1":
            b, s, f = selected_hidden_state.shape
            components = 1
            combined_feature_space = np.zeros((b, s * components))
            pca = PCA(n_components=components)
            for b_i in range(b):
                pca.fit(selected_hidden_state[b_i])
                projection = pca.transform(selected_hidden_state[b_i])
                combined_feature_space[b_i] = projection.reshape(s * components)
        else:
            raise ValueError(f"{collapse_fn} not supported")
        hidden_state_i = layer
        projection_dims = [2] if visualize else [500] 
        for j in projection_dims:
            for projection_method in ["pca"]: # "umap"
                if projection_method == "pca":
                    pca = PCA(n_components=j)
                    pca.fit(combined_feature_space)
                    projection = pca.transform(combined_feature_space)
                elif projection_method == "umap":
                    umap = UMAP(
                        n_neighbors=40,
                        min_dist=0.0,
                        n_components=j,
                        random_state=42,
                    )
                    projection = umap.fit_transform(combined_feature_space)

                if visualize: 
                    sorted_labels = sorted(set(new_labels))
                    genome_to_color_id = {k: i for i, k in enumerate(sorted_labels)}
                    genome_keys = genome_to_color_id.keys()
                    targets = list(genome_to_color_id[x] for x in new_labels)
                    plt.figure(figsize=(7, 7))
                    scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
                    plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)

                    os.makedirs(name, exist_ok=True)
                    plt.savefig(f"{name}/viz_{hidden_state_i}_{collapse_fn}_{projection_method}.png")
                    plt.clf()

                    # tsne = TSNE(n_components=2, perplexity=30)
                    # projection = tsne.fit_transform(combined_feature_space)

                    # genome_to_color_id = {k: i for k, i in zip((set(new_labels)), range(10))}
                    # genome_keys = genome_to_color_id.keys()
                    # targets = list(genome_to_color_id[x] for x in new_labels)
                    # plt.figure(figsize=(7, 7))
                    # scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=targets, cmap='tab10')
                    # plt.legend(loc="upper left", prop={'size': 6}, handles=scatter.legend_elements()[0], labels=genome_keys)
                    # 
                    # plt.savefig(f"{name}/viz_{hidden_state_i}_{collapse_fn}_tsne.png")

                # Build reference
                #list_of_genome_id= [x['genome_id'] for x in validation_outputs] 
                #genome_id = [genome_id for sublist in list_of_genome_id for genome_id in sublist]
                #genome_id = [genome_id for sublist in new_genomes for genome_id in sublist]
                

                #list_of_contig_names= [x['contig_names'] for x in validation_outputs] 
                #contig_names = [contig_name for sublist in list_of_contig_names for contig_name in sublist]
                #contig_names = [contig_name for sublist in new_contig_names for contig_name in sublist]
                
                #list_of_lengths = [x['contig_length'] for x in validation_outputs]
                #lengths = [length for sublist in list_of_lengths for length in sublist]

                gsa_mapping = '/mnt/data/CAMI/data/short_read_oral/reformatted_manually_combined_gsa_mapping.tsv'
                df = pd.read_csv(gsa_mapping, sep='\t', header=None)
                df = df.rename(columns={0: 'contig_name', 1: 'genome', 2: 'subject', 3: 'start', 4: 'end'})

                contigs_for_genome = defaultdict(list)
                for contig_name, genome in zip(contig_names, genome_id):
                    length_positions = df.loc[df['contig_name'] == contig_name, ['start', 'end']]
                    start = length_positions['start'].item()
                    end = length_positions['end'].item()
                    contig = vamb.benchmark.Contig(contig_name, genome, start, end)
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

                #print("Reference:")
                #print(reference.ngenomes)
                #print(reference.ncontigs)

                k_centers = [int(len(genomes) * 0.2), int(len(genomes) * 0.5), len(genomes)]
                k_methods = ["kmeans", "kmedoids"]
                k_combos = itertools.product(k_methods, k_centers)

                cluster_sizes = [5, 10, 15, 20]
                methods = ["hdbscan", "vamb_clustering"]
                cluster_combos = itertools.product(methods, cluster_sizes)

                methods = list(k_combos) + list(cluster_combos) 
                #methods = list(cluster_combos)
                for cluster_method, k in methods:
                        method_name = f"{cluster_method}_{k}_fn{collapse_fn}_layer{hidden_state_i}_{projection_method}{j}"
                        print(method_name)
                        if cluster_method == "kmeans":
                            cluster_results = KMeans(n_clusters = k, init = 'k-means++', random_state=1).fit_predict(projection)
                            clusters = defaultdict(list)
                            for i, x in enumerate(cluster_results):
                                clusters[x].append(contig_names[i])
                        elif cluster_method == "kmedoids":
                            cluster_results = KMedoids(n_clusters = k, random_state=1).fit_predict(projection)
                            clusters = defaultdict(list)
                            for i, x in enumerate(cluster_results):
                                clusters[x].append(contig_names[i])
                        elif cluster_method == "hdbscan":
                            cluster_results = hdbscan.HDBSCAN(min_cluster_size=k).fit_predict(projection)
                            clusters = defaultdict(list)
                            for i, x in enumerate(cluster_results):
                                clusters[x].append(contig_names[i])
                            if -1 in clusters:
                                del clusters[-1]  # Remove "no bin" from dbscan
                        elif cluster_method == "vamb_clustering":
                            filtered_labels = [n for n in contig_names]
                            vamb_clusters = vamb.cluster.cluster(projection, labels=filtered_labels)
                            cluster_results = list(vamb_clusters)
                            sorted_labels = sorted(set(filtered_labels))
                            genome_to_color_id = {k: i for i, k in enumerate(sorted_labels)}
                            genome_keys = genome_to_color_id.keys()
                            cluster_results = list(genome_to_color_id[x] for x in filtered_labels)

                            with open('bins.tsv', 'w') as binfile:
                                    vamb.vambtools.write_clusters(binfile, vamb_clusters)
                            with open('bins.tsv') as clusters_file:
                                    clusters = vamb.vambtools.read_clusters(clusters_file)
                            #clusters = defaultdict(list)
                            #for key, value in cluster_results.items():
                             #   clusters[key].append(contig_names)

                        deepbin_bins = vamb.benchmark.Binning(clusters, 
                                                              reference, 
                                                              recalls=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], 
                                                              precisions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
                        
                        #print('Vamb bins:')
                        #for rank in deepbin_bins.summary():
                         #       print('\t'.join(map(str, rank)))

                        print("Binning:")
                        print(deepbin_bins.nbins)
                        print(deepbin_bins.ncontigs)

                        print(deepbin_bins.print_matrix(rank=0))
                        for i, x in enumerate(clusters.keys()):
                            print(i)
                            print(deepbin_bins.confusion_matrix(genome, x))
                           # print(deepbin_bins.mcc(genome, x))
                            #print(deepbin_bins.f1(genome, x))
                            #print()

                        #print(clusters)

                        with open('results_{x}.csv'.format(x=file_name), 'a') as f:
                            writer = csv.writer(f)
                            flatten_bins = [str(rank) for sublist in deepbin_bins.summary() for rank in sublist]
                            print(flatten_bins)
                            writer.writerow([method_name] + flatten_bins)

                        if visualize: 
                            plt.figure(figsize=(7, 7))
                            scatter = plt.scatter(projection[:, 0], projection[:, 1], alpha=0.9, s=5.0, c=cluster_results, cmap='tab10')
                            os.makedirs(name, exist_ok=True)
                            plt.savefig(f"{name}/viz_{hidden_state_i}_{collapse_fn}_{projection_method}_{cluster_method}_{k}_cluster.png")
                            plt.clf()


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
    #assert len(file_list) == 1

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
            if contig_name in index_for_contig:
                contig_tnfs = tnfs[index_for_contig[contig_name]]
                tnfs_to_plot.append(contig_tnfs)
                species_label.append(species)
                plot_contigs.append(contig_name)

    tnfs_to_plot = np.stack(tnfs_to_plot)

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
        if contig in index_for_contig:
            target = kmeans_clusters[index_for_contig[contig]]
            targets.append(target)

    plot(tnfs_to_plot, targets, bin_labels, name="tnf_clusters_{dataset}".format(dataset=file_name))

    # Create reference file.
    to_evaluate_idxes = [
        i for i in range(len(contignames)) 
        if contignames[i] in contig_name_to_genome
    ]
    contigs_for_genome = defaultdict(list)

    genome_id = [contig_name_to_genome[contignames[i]][1] for i in to_evaluate_idxes]
    contignames = [contignames[i] for i in to_evaluate_idxes]
    contiglengths = [contiglengths[i] for i in to_evaluate_idxes]
    for contig_name, genome, contiglength in zip(
                contignames,
                genome_id,
                contiglengths
            ):
        contig = vamb.benchmark.Contig.subjectless(contig_name, contiglength)
        contigs_for_genome[genome].append(contig)

    genomes = [] 
    for genome_instance in set(genome_id):
        genome = vamb.benchmark.Genome(genome_instance)
        for contig_name in contigs_for_genome[genome_instance]:
            genome.add(contig_name)

        genomes.append(genome)

    print("Number of genomes:", len(genomes))

    for genome in genomes:
        genome.update_breadth()
                    
    reference = vamb.benchmark.Reference(genomes)

    taxonomy_path = '/mnt/data/CAMI/data/short_read_oral/taxonomy.tsv'
    with open(taxonomy_path) as taxonomy_file:
          reference.load_tax_file(taxonomy_file)

    clusters = defaultdict(list)
    kmeans_clusters = [kmeans_clusters[i] for i in to_evaluate_idxes]
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
    
    parser.add_argument('--mlm', dest='nsp_training_task', action='store_false')
    parser.add_argument('--nsp', dest='nsp_training_task', action='store_true')
    parser.set_defaults(feature=False)
    args = parser.parse_args()
   
    train_file_name = "train"
    val_file_name = "val"

    if args.nsp_training_task == False:
        val_dataset = GenomeKmerDataset(args.val_contigs, cache_name="viz_val", genomes=None, random_segment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=7, drop_last=False)

        train_dataset = GenomeKmerDataset(args.train_contigs, cache_name="viz_train_1_sample_over_512", genomes=SPECIES_TO_PLOT, random_segment=False)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=7, drop_last=False)

        print('train length', len(train_dataloader))
        print('val length', len(val_dataloader))

        model = BertBin.load_from_checkpoint(args.ckpt_path, kmer_dataset=val_dataset, val_dataset=val_dataset).cuda()
        model.eval()
    
    elif args.nsp_training_task == True:
        val_dataset = GenomeKmerDatasetNSP(args.val_contigs, cache_name="nsp_viz_val", genomes=None, random_segment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=7, drop_last=False)

        train_dataset = GenomeKmerDatasetNSP(args.train_contigs, cache_name="nsp_viz_train_1_sample_over_512", genomes=SPECIES_TO_PLOT, random_segment=False)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=7, drop_last=False)

        print('train length', len(train_dataloader))
        print('val length', len(val_dataloader))

        model = BertBinNSP.load_from_checkpoint(args.ckpt_path, kmer_dataset=val_dataset, val_dataset=val_dataset).cuda()
        model.eval()

    evaluate_tnf(train_dataloader, args.train_contigs, file_name=train_file_name)
    #evaluate_tnf(val_dataloader, args.val_contigs, file_name=val_file_name)

    #for i in reversed(range(13)):
    #evaluate(model, val_dataloader, name=f"viz_out/viz_val_{pathlib.Path(args.ckpt_path).stem}", file_name=val_file_name, layer=i, collapse_fn=x, num_batches=None, visualize = args.visualize)
    evaluate(model, train_dataloader, name=f"viz_out/viz_train_{pathlib.Path(args.ckpt_path).stem}", file_name=train_file_name, layer=12, num_batches=None, visualize = args.visualize, nsp = args.nsp_training_task)


if __name__ == "__main__":
    main()
