import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.function as fn
import random
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import pandas as pd
from scipy.stats import ttest_rel
import uuid


parser = argparse.ArgumentParser()
parser.add_argument('--in_features', type=int, default=50, help='Input feature dimension & the original feature dimension')
parser.add_argument('--hid_feats1', type=int, default=100, help='First hidden layer dimension')
parser.add_argument('--hid_feats2', type=int, default=200, help='Second hidden layer dimension')
parser.add_argument('--hid_feats3', type=int, default=100, help='Third hidden layer dimension')
parser.add_argument('--out_features', type=int, default=50, help='Output feature dimension')
parser.add_argument('--num_negatives', type=int, default=20, help='Number of negative samples')
parser.add_argument('--epoch', type=int, default=200, help='Number of epoch')
parser.add_argument('--times', type=int, default=3, help='Number of cross validations')
args = parser.parse_args()
parser.print_help()


run_id = str(uuid.uuid4())[:4]  # Use only the first 4 characters for brevity
print(f'Unique Run ID: {run_id}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats1, hid_feats2, hid_feats3, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats1)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats1, hid_feats2)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats2, hid_feats3)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats3, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hid_feats1, hid_feats2, hid_feats3, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}).to(device)


def compute_mrr(pos_score, neg_score):
    num_edges = pos_score.shape[0]
    neg_score = neg_score.view(num_edges, -1).detach().cpu().numpy()
    pos_score = pos_score.detach().cpu().numpy()
    mrr = []
    for i in range(len(pos_score)):
        rank = np.sum(neg_score[i] > pos_score[i]) + 1
        mrr.append(1/rank)
    return mrr


def get_inverse_edge(g, etype):
    edges = g.canonical_etypes
    s, d = etype[0], etype[2]
    for edge in edges:
        if edge[0] == d and edge[2] == s:
            break
    return edge


def get_data(g, etype, n):
    inv_etype = get_inverse_edge(g, etype)
    src, dst = g.edges(etype=etype)
    num_to_choose = max(1, len(src) // n)
    print('number of mask: ', num_to_choose)
    edge_to_re = random.sample(range(len(src)), num_to_choose)
    src_test, dst_test = src[torch.tensor([edge_to_re])].view(-1), dst[torch.tensor([edge_to_re])].view(-1)
    edge_to_re = g.edge_ids(src_test, dst_test, etype=etype)
    train_g = dgl.remove_edges(g, edge_to_re, etype=etype)
    if inv_etype != etype:
        edge_to_re_inv = g.edge_ids(dst_test, src_test, etype=inv_etype)
        train_g = dgl.remove_edges(train_g, edge_to_re_inv, etype=inv_etype)
    train_g = train_g.to(device)
    src_test = src_test.to(device)
    dst_test = dst_test.to(device)

    return train_g, src_test, dst_test


ds1 = dgl.data.CSVDataset('data/pkg')
ds2 = dgl.data.CSVDataset('data/merge')
pkg = ds1[0].to(device)
merge = ds2[0].to(device)

etypes = [('anatomy', 'anatomy_protein', 'gene/protein'),
          ('disease', 'disease_drug', 'drug'),
          ('disease', 'disease_effect', 'effect/phenotype'),
          ('disease', 'disease_protein', 'gene/protein'),
          ('drug', 'drug_effect', 'effect/phenotype'),
          ('biological_process', 'biological_process_protein', 'gene/protein'),
          ('drug', 'drug_protein', 'gene/protein')]

k = args.num_negatives
in_features = args.in_features
hid_feats1 = args.hid_feats1
hid_feats2 = args.hid_feats2
hid_feats3 = args.hid_feats3
out_features = args.out_features
epochs = args.epoch
k_test = 20

num_nodes_dict_pkg = {ntype: pkg.number_of_nodes(ntype) for ntype in pkg.ntypes}
num_nodes_dict_merge = {ntype: merge.number_of_nodes(ntype) for ntype in merge.ntypes}

### get the mapping ###
pkg_map = pd.read_csv('data/pkgid.csv')
merge_map = pd.read_csv('data/mergeid.csv')
pkg_map = pkg_map[['x_index', 'x_name', 'x_type']]
merge_map = merge_map[['x_index', 'x_name', 'x_type']]
pkg_map = pkg_map.rename(columns={'x_index': 'pkg_index'})
merge_map = merge_map.rename(columns={'x_index': 'merge_index'})
mappings = pd.merge(pkg_map, merge_map, how='left', on=['x_name', 'x_type'])
# ID of PrimeKG should be consistent with MERGE
assert len(mappings[mappings['pkg_index'] != mappings['merge_index']]) == 0

mrr_score = {}
for etype in tqdm(etypes):
    print('current edge type is:', etype)
    mrr_mean = []
    for i in range(args.times):
        train_pkg, src_test, dst_test = get_data(pkg, etype, 5)
        inv_etype = get_inverse_edge(merge, etype)
        edge_to_re = merge.edge_ids(src_test, dst_test, etype=etype)
        train_merge = dgl.remove_edges(merge, edge_to_re, etype=etype)
        if inv_etype != etype:
            edge_to_re_inv = merge.edge_ids(dst_test, src_test, etype=inv_etype)
            train_merge = dgl.remove_edges(train_merge, edge_to_re_inv, etype=inv_etype)

        n_hetero_features = in_features
        for ntype, num_nodes in num_nodes_dict_pkg.items():
            train_pkg.nodes[ntype].data['feature'] = torch.randn(num_nodes, n_hetero_features).to(device)
        for ntype, num_nodes in num_nodes_dict_merge.items():
            train_merge.nodes[ntype].data['feature'] = torch.randn(num_nodes, n_hetero_features).to(device)
        node_features_pkg = {}
        for ntype in train_pkg.ntypes:
            node_features_pkg[ntype] = train_pkg.nodes[ntype].data['feature']
        node_features_merge = {}
        for ntype in train_merge.ntypes:
            node_features_merge[ntype] = train_merge.nodes[ntype].data['feature']
        ### training process ###
        model_pkg = Model(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, train_pkg.etypes).to(device)
        opt_pkg = torch.optim.Adam(model_pkg.parameters())

        for epoch in range(epochs):
            negative_graph = construct_negative_graph(train_pkg, k, etype)
            pos_score, neg_score = model_pkg(train_pkg, negative_graph, node_features_pkg, etype)
            loss = compute_loss(pos_score, neg_score)
            opt_pkg.zero_grad()
            loss.backward()
            opt_pkg.step()

        model_merge = Model(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, train_merge.etypes).to(device)
        opt_merge = torch.optim.Adam(model_merge.parameters())

        for epoch in range(epochs):
            negative_graph = construct_negative_graph(train_merge, k, etype)
            pos_score, neg_score = model_merge(train_merge, negative_graph, node_features_merge, etype)
            loss = compute_loss(pos_score, neg_score)
            opt_merge.zero_grad()
            loss.backward()
            opt_merge.step()

        ### testing process ###
        src_node, dst_node = etype[0], etype[2]
        test_nodes_dict_pkg = {src_node: num_nodes_dict_pkg[src_node], dst_node: num_nodes_dict_pkg[dst_node]}
        test_nodes_dict_merge = {src_node: num_nodes_dict_merge[src_node], dst_node: num_nodes_dict_merge[dst_node]}
        test_graph_pkg = dgl.heterograph({etype: (src_test, dst_test)}, num_nodes_dict=test_nodes_dict_pkg,
                                         device=device)
        test_graph_merge = dgl.heterograph({etype: (src_test, dst_test)}, num_nodes_dict=test_nodes_dict_merge,
                                           device=device)

        model_pkg.eval()
        model_merge.eval()
        with torch.no_grad():
            trained_features_pkg = model_pkg.sage(train_pkg, node_features_pkg)
            trained_features_merge = model_merge.sage(train_merge, node_features_merge)

        test_features_pkg = {src_node: trained_features_pkg[src_node], dst_node: trained_features_pkg[dst_node]}
        test_features_merge = {src_node: trained_features_merge[src_node], dst_node: trained_features_merge[dst_node]}

        neg_graph_test_pkg = construct_negative_graph(test_graph_pkg, k_test, etype)
        neg_graph_test_merge = construct_negative_graph(test_graph_merge, k_test, etype)
        with torch.no_grad():
            pos_score_pkg = model_pkg.pred(test_graph_pkg, test_features_pkg, etype)
            neg_score_pkg = model_pkg.pred(neg_graph_test_pkg, test_features_pkg, etype)
            pos_score_merge = model_merge.pred(test_graph_merge, test_features_merge, etype)
            neg_score_merge = model_merge.pred(neg_graph_test_merge, test_features_merge, etype)

        mrr_pkg = compute_mrr(pos_score_pkg, neg_score_pkg)
        mrr_merge = compute_mrr(pos_score_merge, neg_score_merge)
        mrr_mean.append((np.mean(mrr_pkg), np.mean(mrr_merge)))

    pkg_score = [x[0] for x in mrr_mean]
    merge_score = [x[1] for x in mrr_mean]
    t_statistic, p_value = ttest_rel(pkg_score, merge_score)
    print('t-statistic: ', t_statistic)
    print('p-value: ', p_value)

    mrr_score[etype] = mrr_mean

with open(f'mrr_score/mrr_{run_id}.pkl', 'wb') as f:
    pickle.dump(mrr_score, f)

