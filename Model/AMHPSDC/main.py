import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as Data
from model import BioEncoder, AMHPSDC, HgnnEncoder, Decoder, ASMGEncoder
from sklearn.model_selection import KFold
from node2vec import node2Vec_main
import train_search
import os
import glob
import warnings
import sys

sys.path.append('..')
from drug_util import GraphDataset, collate
from utils import metrics_graph, set_seed_all
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from process_data import getData
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(dataset):
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)

    # ALMANAC_Loewe: -17.5, -2.9
    # DrugCombDB_ZIP:-3.9, 1.78 
    threshold1 = -3.9
    threshold3 = 1.78
    synergy_final = [row for row in synergy if row[3] >= threshold3 or row[3] <= threshold1]
    for row in synergy_final:
        if row[3] >= threshold3:
            row[3] = 1
        if row[3] <= threshold1:
            row[3] = 0

    drug_sim_matrix, cline_sim_matrix = get_sim_mat(drug_smiles_fea, np.array(gene_data, dtype='float32'))

    return drug_fea, cline_fea, synergy_final, drug_sim_matrix, cline_sim_matrix



def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    # -----split synergy into 5CV,test set
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    # --CV set
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # --test set
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(synergy_test).to(device)
    return synergy_cv_data, test_ind, test_label


def get_sim_mat(drug_fea, cline_fea):
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(cline_fea))
    return torch.from_numpy(drug_sim_matrix).type(torch.FloatTensor).to(device), torch.from_numpy(
        cline_sim_matrix).type(torch.FloatTensor).to(device)


# --train+test
def train(node_feats, adjs, drug_num, cline_num, drug_fea_set, cline_fea_set, synergy_adj, index, label, alpha):
    loss_train = 0
    true_ls, pre_ls = [], []
    optimizer.zero_grad()
    for batch, (drug, cline) in enumerate(zip(drug_fea_set, cline_fea_set)):
        pred, rec_drug, rec_cline = model(node_feats, node_types, adjs, archs, drug.x, drug.edge_index, drug_num, cline_num, drug.batch, cline[0], synergy_adj,
                                          index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        loss_rec_1 = rec_func(rec_drug, drug_sim_mat)
        loss_rec_2 = rec_func(rec_cline, cline_sim_mat)
        loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        true_ls += label_train.cpu().detach().numpy().tolist()
        pre_ls += pred.cpu().detach().numpy().tolist()
    auc_train, aupr_train, f1_train, acc_train = metrics_graph(true_ls, pre_ls)
    return [auc_train, aupr_train, f1_train, acc_train], loss_train


def test(node_feats, adjs, drug_num, cline_num, drug_fea_set, cline_fea_set, synergy_adj, index, label, alpha):
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline) in enumerate(zip(drug_fea_set, cline_fea_set)):
            pred, rec_drug, rec_cline = model(node_feats, node_types, adjs, archs, drug.x, drug.edge_index, drug_num, cline_num, drug.batch, cline[0], synergy_adj,
                                              index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        loss_rec_1 = rec_func(rec_drug, drug_sim_mat)
        loss_rec_2 = rec_func(rec_cline, cline_sim_mat)
        loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        auc_test, aupr_test, f1_test, acc_test = metrics_graph(label.cpu().detach().numpy(),
                                                               pred.cpu().detach().numpy())
        return [auc_test, aupr_test, f1_test, acc_test], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    dataset_name = 'DrugCombDB_ZIP'  # ALMANAC_Loewe, DrugCombDB_ZIP
    seed = 0
    epochs = 1000
    learning_rate = 0.0001
    L2 = 1e-4
    alpha = 0.4

    n_hid = 64 #AMGEncoder
    dropout = 0.2

    archs = {
    "source":([[3, 12, 0]], [[13, 13, 13]]),
    "target": ([[10, 8, 8]], [[13, 13, 13]])}

    datadir = "../preprocessed"
    prefix = os.path.join(datadir)

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
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    # embedding
    in_dims = []
    drug_num = 0
    cline_num = 0
    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        if(k == 0):
            drug_num = in_dims[-1]
        elif (k == 4):
            cline_num = in_dims[-1]
    offset = sum(in_dims[:-1])

    node_feats_load = []
    data = np.load('../preprocessed/combined_matrices.npz')
    dp_matrix = np.zeros((708, 1512), dtype=int)
    prefix = "../../Data/Luo"
    dp = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['did', 'pid', 'rating']).reset_index(drop=True)
    dp_pos = dp[dp['rating'] == 1].to_numpy()[:, :2]
    dp_matrix[dp_pos[:, 0], dp_pos[:, 1]] = 1
    features = node2Vec_main(dp_matrix)
    node_feats_load.append(torch.FloatTensor(features[:dp_matrix.shape[0]]).cuda())
    node_feats_load.append(torch.FloatTensor(features[dp_matrix.shape[0]:]).cuda())

    for k, d in enumerate(data):
        matrix = data[d]
        features = node2Vec_main(matrix)
        node_feats_load.append(torch.FloatTensor(features[:matrix.shape[0]]).cuda())
    
    path = 'result/' + dataset_name  + '_'
    file = open(path + 'result.txt', 'w')
    set_seed_all(seed)
    drug_feature, cline_feature, synergy_data, drug_sim_mat, cline_sim_mat = load_data(dataset_name)
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                                   collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
    cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                    batch_size=len(cline_feature), shuffle=False)
    # -----split synergy into 5CV,test set
    synergy_cv, index_test, label_test = data_split(synergy_data)
    cv_data = synergy_cv
    # ---5CV
    final_metric = np.zeros(4)
    fold_num = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in kf.split(cv_data):
        node_feats = []
        for fea in node_feats_load:
            node_feats.append(fea)

        # ---construct train_set+validation_set
        synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
        
        # ---ASMG searching
        archs["source"], archs["target"] = train_search.search_main(synergy_train, synergy_validation, drug_sim_mat, cline_sim_mat)
        steps_s = [len(meta) for meta in archs["source"][0]]
        steps_t = [len(meta) for meta in archs["target"][0]]

        np.savetxt(path + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
        label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
        label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
        index_train = torch.from_numpy(synergy_train).to(device)
        index_validation = torch.from_numpy(synergy_validation).to(device)
        # -----construct hyper_synergy_graph_set
        edge_data = synergy_train[synergy_train[:, 3] == 1, 0:3]
        synergy_edge = edge_data.reshape(1, -1)
        index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
        synergy_num = np.concatenate((index_num, index_num, index_num), axis=1)
        synergy_num = np.array(synergy_num).reshape(1, -1)
        synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
        synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

        # ---model_build
        model = AMHPSDC(BioEncoder(dim_drug=75, dim_cellline=cline_feature.shape[-1], output=100),
                        HgnnEncoder(in_channels=100, out_channels=256), Decoder(in_channels=960), 
                        ASMGEncoder(in_dims, n_hid, steps_s, dropout = dropout),
                        ASMGEncoder(in_dims, n_hid, steps_t, dropout = dropout)).to(device)
        loss_func = torch.nn.BCELoss()
        rec_func = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

        # ---run
        best_metric = [0, 0, 0, 0]
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            train_metric, train_loss = train(node_feats, adjs_pt, drug_num, cline_num, drug_set, cline_set, synergy_graph,
                                            index_train, label_train, alpha)
            val_metric, val_loss, _ = test(node_feats, adjs_pt, drug_num, cline_num, drug_set, cline_set, synergy_graph,
                                            index_validation, label_validation, alpha)
            if epoch % 20 == 0:
                print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                        'AUC: {:.6f},'.format(train_metric[0]), 'AUPR: {:.6f},'.format(train_metric[1]),
                        'F1: {:.6f},'.format(train_metric[2]), 'ACC: {:.6f},'.format(train_metric[3]),
                        )
                print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                        'AUC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                        'F1: {:.6f},'.format(val_metric[2]), 'ACC: {:.6f},'.format(val_metric[3]))
            torch.save(model.state_dict(), '{}.pth'.format(epoch))
            if val_metric[0] > best_metric[0]:
                best_metric = val_metric
                best_epoch = epoch
            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(f)
        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
              'AUC: {:.6f},'.format(best_metric[0]),
              'AUPR: {:.6f},'.format(best_metric[1]), 'F1: {:.6f},'.format(best_metric[2]),
              'ACC: {:.6f},'.format(best_metric[3]))
        model.load_state_dict(torch.load('{}.pth'.format(best_epoch)))
        val_metric, _, y_val_pred = test(node_feats, adjs_pt, drug_num, cline_num, drug_set, cline_set, synergy_graph, index_validation, label_validation,
                                            alpha)
        test_metric, _, y_test_pred = test(node_feats, adjs_pt, drug_num, cline_num, drug_set, cline_set, synergy_graph, index_test, label_test, alpha)
        np.savetxt(path + 'val_' + str(fold_num) + '_pred.txt', y_val_pred)
        np.savetxt(path + 'test_' + str(fold_num) + '_pred.txt', y_test_pred)
        file.write('val_metric:')
        for item in val_metric:
            file.write(str(item) + '\t')
        file.write('\ntest_metric:')
        for item in test_metric:
            file.write(str(item) + '\t')
        file.write('\n')
        final_metric += test_metric
        fold_num = fold_num + 1
    final_metric /= 5
    print('Final 5-cv average results, AUC: {:.6f},'.format(final_metric[0]),
          'AUPR: {:.6f},'.format(final_metric[1]),
          'F1: {:.6f},'.format(final_metric[2]), 'ACC: {:.6f},'.format(final_metric[3]))
