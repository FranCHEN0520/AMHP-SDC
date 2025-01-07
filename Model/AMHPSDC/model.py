import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, GINConv, JumpingKnowledge, global_max_pool, global_mean_pool
from geniepath import GeniePath, GeniePathLazy
import sys

sys.path.append('..')
from utils import reset


class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)


class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(Cell, self).__init__()

        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid) if use_norm is True else lambda x: x
        self.use_nl = use_nl
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(Op())

    def forward(self, x, adjs, idxes_seq, idxes_res):

        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])  # ! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)

        output = self.norm(states[-1])
        if self.use_nl:
            output = F.gelu(output)
        return output


class ASMGEncoder(nn.Module):

    def __init__(self, in_dims, n_hid, n_steps, dropout=None, attn_dim=64, use_norm=True, out_nl=True):
        super(ASMGEncoder, self).__init__()
        self.n_hid = n_hid
        self.ws = nn.ModuleList()
        assert (isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(64, n_hid))
        assert (isinstance(n_steps, list))
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, use_norm=use_norm, use_nl=out_nl))

        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x


    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)
        temps = [];
        attns = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)

        return out

class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = HypergraphConv(256, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.batch1(self.act(self.conv1(x, edge)))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        # GeniePathLazy
        self.GeniePathLazy = GeniePath(dim_drug, output)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.GeniePathLazy(drug_feature, drug_adj, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.4)
        self.act = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h1 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=1))


class AMHPSDC(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, model_s, model_t):
        super(AMHPSDC, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.model_s = model_s
        self.model_t = model_t
        self.drug_rec_weight = nn.Parameter(torch.rand(320, 320))
        self.cline_rec_weight = nn.Parameter(torch.rand(320, 320))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)
        reset(self.model_s)
        reset(self.model_t)

    def forward(self, node_feats, node_types, adjs, archs, drug_feature, drug_adj, drug_num, cline_num, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        out_s = self.model_s(node_feats, node_types, adjs, archs["source"][0], archs["source"][1])
        out_t = self.model_t(node_feats, node_types, adjs, archs["target"][0], archs["target"][1])
        drug_emb1 = out_s[:drug_num]
        cline_emb1 = out_t[-cline_num:]
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        drug_emb2, cline_emb2 = graph_embed[:drug_num], graph_embed[drug_num:]
        drug_emb = torch.cat((drug_emb1, drug_emb2), 1)
        cline_emb = torch.cat((cline_emb1, cline_emb2), 1)
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        graph_embed_full = torch.cat((drug_emb, cline_emb), 0)
        res = self.decoder(graph_embed_full, druga_id, drugb_id, cellline_id)
        return res, rec_drug, rec_cline
