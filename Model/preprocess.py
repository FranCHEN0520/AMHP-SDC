import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import pickle

def normalize_sym(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_KG(dataset):
    prefix = "../Data/Luo"
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating']).reset_index(drop=True)
    dd = pd.read_csv(os.path.join(prefix, "drug_drug.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)
    simdd = pd.read_csv(os.path.join(prefix, "sim_drugs.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)
    pp = pd.read_csv(os.path.join(prefix, "pro_pro.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    simpp = pd.read_csv(os.path.join(prefix, "sim_proteins.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    de = pd.read_csv(os.path.join(prefix, "drug_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'dis', 'weight']).reset_index(drop=True)
    pe = pd.read_csv(os.path.join(prefix, "protein_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'dis', 'weight']).reset_index(drop=True)
    ds = pd.read_csv(os.path.join(prefix, "drug_se.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'se', 'weight']).reset_index(drop=True)
    cp = pd.read_csv(os.path.join(dataset, "cellLines_protein.csv"), encoding='utf-8', delimiter=',',
                     names=['cid', 'pid', 'weight']).reset_index(drop=True)
    np.random.seed(1)
    
    # num_cline = 48 #ALMANAC_Loewe
    num_cline = 44 #DrugCombDB
    offsets = {'drug': 708, 'protein': 708 + 1512}
    offsets['disease'] = offsets['protein'] + 5603
    offsets['sideeffect'] = offsets['disease'] + 4192
    offsets['cline'] = offsets['sideeffect'] + num_cline
    print(offsets['cline'])
    # * node types
    node_types = np.zeros((offsets['cline'],), dtype=np.int32)
    node_types[offsets['drug']:offsets['protein']] = 1
    node_types[offsets['protein']:offsets['disease']] = 2
    node_types[offsets['disease']:offsets['sideeffect']] = 3
    node_types[offsets['sideeffect']:] = 4

    np.save("./preprocessed/node_types", node_types)

    #* positive pairs
    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    #* adjs with offset
    adjs_offset = {}

    # drug-protein
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dp_pos[:, 0], dp_pos[:, 1] + offsets['drug']] = 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)
    print(len(dp_pos))

    # drug-disease
    de_npy = de.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[de_npy[:, 0], de_npy[:, 1] + offsets['protein']] = 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)
    print(len(de_npy))
    ed_matrix = np.zeros((5603, 708), dtype=int)
    ed_matrix[de_npy[:, 1], de_npy[:, 0]] = 1

    # protein-disease
    pe_npy = pe.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pe_npy[:, 0] + offsets['drug'], pe_npy[:, 1] + offsets['protein']] = 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)
    print(len(pe_npy))

    # drug-sideeffect
    ds_npy = ds.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[ds_npy[:, 0], ds_npy[:, 1] + offsets['disease']] = 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)
    print(len(ds_npy))
    sd_matrix = np.zeros((4192, 708), dtype=int)
    sd_matrix[ds_npy[:, 1], ds_npy[:, 0]] = 1

    # celline-protein
    cp_npy = cp.to_numpy(int)[:, :2]
    cp_score = cp['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(cp_npy[:, 0] + offsets['sideeffect'], cp_npy[:, 1] + offsets['drug'], cp_score):
        adj_offset[i, j] = k
    adjs_offset['4'] = sp.coo_matrix(adj_offset)
    print(len(cp_npy)) 
    cp_matrix = np.zeros((num_cline, 1512), dtype=float)
    for i, j, k in zip(cp_npy[:, 0], cp_npy[:, 1], cp_score):
        cp_matrix[i, j] = k

    # drug-drug
    dd_npy = dd.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[dd_npy[:, 0], dd_npy[:, 1]] = 1
    adjs_offset['5'] = sp.coo_matrix(adj_offset)
    print(len(dd_npy))
    dd_matrix = np.zeros((708, 708), dtype=int)
    dd_matrix[dd_npy[:, 0], dd_npy[:, 1]] = 1

    # protein-protein
    pp_npy = pp.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pp_npy[:, 0] + offsets['drug'], pp_npy[:, 1] + offsets['drug']] = 1
    adjs_offset['6'] = sp.coo_matrix(adj_offset)
    print(len(pp_npy))
    pp_matrix = np.zeros((1512, 1512), dtype=int)
    pp_matrix[pp_npy[:, 0], pp_npy[:, 1]] = 1

    f2 = open("./preprocessed/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()
    np.savez('./preprocessed/combined_matrices.npz', ed_matrix=ed_matrix, sd_matrix=sd_matrix, cp_matrix=cp_matrix)
    
    
if __name__ == '__main__':
    # preprocess_KG("../Data/ALMANAC_Loewe")
    preprocess_KG("../Data/DrugCombDB")
    