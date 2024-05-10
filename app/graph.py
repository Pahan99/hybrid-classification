import pandas as pd
import numpy as np

from tqdm import tqdm
import os

def get_idx_maps(read_ids_file_path):
    read_id_idx = {}
    # global read_id_idx
    with open(read_ids_file_path) as read_ids_file:
        for rid in tqdm(read_ids_file):
            rid = rid.strip().split()[0][1:]
            read_id_idx[rid] = len(read_id_idx)

    return read_id_idx


def load_read_degrees(degrees_file_path,read_id_idx):
    degree_array = np.zeros((len(read_id_idx),), dtype=int)
    for line in tqdm(open(degrees_file_path, 'r')):
        i, d = line.strip().split()
        d = int(d)
        # print(degree_array)
        degree_array[read_id_idx[i]] = d

    return degree_array


def load_edges_as_numpy(edges_txt_path, edges_npy_path):
    if not os.path.isfile(edges_npy_path):
        edges_txt = [x.strip() for x in tqdm(open(edges_txt_path))]
        edges = np.zeros((len(edges_txt), 2), dtype=np.int32)

        for i in tqdm(range(len(edges_txt))):
            e1, e2 = edges_txt[i].strip().split()
            edges[i]  = [int(e1), int(e2)]

        np.save(edges_npy_path, edges)

    return np.load(edges_npy_path)

def alignments_to_edges(alignments_file_path, edges_txt_path, read_id_idx):
    TP = 0
    FP = 0

    if not os.path.isfile(edges_txt_path):
        with open(edges_txt_path, "w+") as ef:
            for line in tqdm(open(alignments_file_path, "r")):
                u, v = line.strip().split('\t')

                if u == v:
                    continue
                try:
                    ef.write(f"{read_id_idx[u]}\t{read_id_idx[v]}\n")
                except:
                    print(f'Missing {u,v}')


exp = "./output/"
alignments_file_path = exp + "reads.alns"
degrees_file_path = exp + "degree"

comp = pd.read_csv(exp + "4-mers.tsv", header=None).to_numpy()

read_id_idx = get_idx_maps(exp + 'read_ids')
degree_array = load_read_degrees(degrees_file_path,read_id_idx)
alignments_to_edges(exp +"reads.alns", exp + "edges.txt", read_id_idx)

edges = load_edges_as_numpy(exp + "edges.txt", exp + "edges.npy")
sample_weights = np.zeros_like(degree_array, dtype=np.float32)
sample_scale = np.ones_like(degree_array, dtype=np.float32)

for n, d in enumerate(degree_array):
    sample_weights[n] = 1.0/d if d>0 else 0
    sample_scale[n] = max(1, np.log10(max(1, d)))

scaled = comp * sample_scale.reshape(-1, 1)

np.savez(exp + 'data.npz', edges=edges, scaled=scaled)