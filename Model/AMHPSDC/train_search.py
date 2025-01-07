import os
import numpy as np
import pickle
import scipy.sparse as sp
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold
import warnings


from node2vec import node2Vec_main
from model_search import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--gpu', type=int, default=)
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=, help='probability of random sampling')
parser.add_argument('--decay', type=float, default=, help='decay factor for eps')
parser.add_argument('--DCbeta', type=float, default=0.0, help = "The weight percentage of DrugCombination in the total loss")
parser.add_argument('--seed', type=int, default=)
args = parser.parse_args()

cstr_source = [0]
cstr_target = [8]

def search_main(synergy_train, synergy_validation, drug_sim_mat, cline_sim_mat): 
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "preprocessed"
    prefix = os.path.join("..", datadir)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    # Luo
    for i in range(0, 5):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    for i in range(5, 7):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_sym(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())

    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse_coo_tensor(size=adjs_offset['1'].shape).cuda())

    # embedding
    in_dims = []
    num_drug = 0
    num_cline = 0
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        if (k == 0):
            num_drug = in_dims[-1]
        elif (k == 4):
            num_cline = in_dims[-1]
    offset = sum(in_dims[:-1])
    train_f0 = synergy_train.copy()
    train_f0[:, 2] = train_f0[:, 2] - num_drug + offset
    val_f0 = synergy_validation.copy()
    val_f0[:, 2] = val_f0[:, 2] - num_drug + offset

    node_feats = []
    data = np.load('../preprocessed/combined_matrices.npz')
    dp_matrix = np.zeros((708, 1512), dtype=int)
    prefix = "../../Data/Luo"
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating']).reset_index(drop=True)
    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    dp_matrix[dp_pos[:, 0], dp_pos[:, 1]] = 1
    features = node2Vec_main(dp_matrix)
    node_feats.append(torch.FloatTensor(features[:dp_matrix.shape[0]]).cuda())
    node_feats.append(torch.FloatTensor(features[dp_matrix.shape[0]:]).cuda())

    for k, d in enumerate(data):
        matrix = data[d]
        features = node2Vec_main(matrix)
        node_feats.append(torch.FloatTensor(features[:matrix.shape[0]]).cuda())


    t = [3]
    model_s = Model(in_dims, args.n_hid, len(adjs_pt), t, cstr_source).cuda()
    model_t = Model(in_dims, args.n_hid, len(adjs_pt), t, cstr_target).cuda()
    loss_func = torch.nn.BCELoss()

    optimizer_w = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model_s.alphas() + model_t.alphas(),
        lr=args.alr
    )

    eps = args.eps
    minval_error = None
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_pt, drug_sim_mat, cline_sim_mat, train_f0, val_f0,
                                       model_s, model_t, loss_func, optimizer_w, optimizer_a, args.DCbeta, eps)
        if (minval_error == None or minval_error > val_error):
            minval_error = val_error
            s = model_s.parse()
            t = model_t.parse()
        eps = eps * args.decay
    print("AMG End!")
    print(s)
    print(t)
    return s, t


def train(node_feats, node_types, adjs, drug_sim_mat, cline_sim_mat, train_f0, val_f0,
          model_s, model_t, loss_func, optimizer_w, optimizer_a, beta, eps):
    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    optimizer_w.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    drug1_emb_train = out_s[train_f0[:, 0]]
    drug2_emb_train = out_s[train_f0[:, 1]]
    cline_emb_train = out_t[train_f0[:, 2]]
    label = torch.from_numpy(train_f0[:, 3]).float().to(device)
    combine_drug = torch.max(drug1_emb_train, drug2_emb_train)
    logits = torch.sigmoid((combine_drug * cline_emb_train).sum(dim=1))
    drug_emb = out_s[node_types == 0]
    cline_emb = out_t[node_types == 4]
    rec_drug = torch.sigmoid(torch.mm(drug_emb, drug_emb.t()))
    rec_cline = torch.sigmoid(torch.mm(cline_emb, cline_emb.t()))
    
    loss = loss_func(logits, label)
    loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
    loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
    loss_w = beta*loss + (1-beta)*(loss_rec_1 + loss_rec_2)

    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t)
    drug1_emb_val = out_s[val_f0[:, 0]]
    drug2_emb_val = out_s[val_f0[:, 1]]
    cline_emb_val = out_t[val_f0[:, 2]]
    label = torch.from_numpy(val_f0[:, 3]).float().to(device)
    combine_drug = torch.max(drug1_emb_val, drug2_emb_val)
    logits = torch.sigmoid((combine_drug * cline_emb_val).sum(dim=1))
    drug_emb = out_s[node_types == 0]
    cline_emb = out_t[node_types == 4]
    rec_drug = torch.sigmoid(torch.mm(drug_emb, drug_emb.t()))
    rec_cline = torch.sigmoid(torch.mm(cline_emb, cline_emb.t()))

    loss = loss_func(logits, label)
    loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
    loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
    loss_a = beta*loss + (1-beta)*(loss_rec_1 + loss_rec_2)

    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()
