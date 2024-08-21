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
import uuid
import os
from scipy.stats import ttest_rel
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='target_gene', help='Dataset name')
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


# src should be a list of len 1
def construct_negative_edge(g, k, etype, src):
    utype, _, vtype = etype
    _, dst = g.out_edges(src, etype=etype)
    dst_list = dst.tolist()
    neg_dst = random.sample(list(set(range(g.num_nodes(vtype))) - set(dst_list)), k)
    src = torch.tensor(src)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.tensor(neg_dst)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})


# g is on cpu and src is a list of len 1, dst is also a list.
def build_data(g, src, dst, edge_type, k):
    # find out the dst which are not in the graph
    _, dst_og = g.out_edges(src, etype=edge_type)
    dst_not_og = list(set(dst) - set(dst_og.tolist()))
    new_g = g.clone()
    new_g.add_edges(src * len(dst_not_og), dst_not_og, etype=edge_type)  # new_g on cpu
    edge_re_id = new_g.edge_ids(src * len(dst), dst, etype=edge_type)
    train_g = dgl.remove_edges(new_g, edge_re_id, etype=edge_type)
    test_neg_g = construct_negative_edge(new_g, k * len(dst), edge_type, src)
    test_g = dgl.heterograph(
        {edge_type: (src * len(dst), dst)},
        num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    new_g = new_g.to(device)
    train_g = train_g.to(device)
    test_g = test_g.to(device)
    test_neg_g = test_neg_g.to(device)
    return new_g, train_g, test_g, test_neg_g


ds1 = dgl.data.CSVDataset('data/pkg')
ds2 = dgl.data.CSVDataset('data/merge')
pkg = ds1[0]
merge = ds2[0]

k = args.num_negatives
in_features = args.in_features
hid_feats1 = args.hid_feats1
hid_feats2 = args.hid_feats2
hid_feats3 = args.hid_feats3
out_features = args.out_features
epochs = args.epoch
k_test = 10
edge_type = ('gene/protein', 'protein_protein', 'gene/protein')
num_nodes_dict_pkg = {ntype: pkg.number_of_nodes(ntype) for ntype in pkg.ntypes}
num_nodes_dict_merge = {ntype: merge.number_of_nodes(ntype) for ntype in merge.ntypes}
files = os.listdir(args.data)
mrr_score = {}

for file in tqdm(files):
    print('current gene is:', file[:-4])
    mrr_mean = []
    for i in range(args.times):
        df = pd.read_csv(f'{args.data}/{file}')
        src = [df['src'][0].tolist()]
        dst = df['dst'].tolist()

        _, train_pkg, test_pkg, test_neg_pkg = build_data(pkg, src, dst, edge_type, k_test)
        _, train_merge, test_merge, test_neg_merge = build_data(merge, src, dst, edge_type, k_test)

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
            negative_graph = construct_negative_graph(train_pkg, k, edge_type)
            pos_score, neg_score = model_pkg(train_pkg, negative_graph, node_features_pkg, edge_type)
            loss = compute_loss(pos_score, neg_score)
            opt_pkg.zero_grad()
            loss.backward()
            opt_pkg.step()

        model_merge = Model(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, train_merge.etypes).to(device)
        opt_merge = torch.optim.Adam(model_merge.parameters())

        for epoch in range(epochs):
            negative_graph = construct_negative_graph(train_merge, k, edge_type)
            pos_score, neg_score = model_merge(train_merge, negative_graph, node_features_merge, edge_type)
            loss = compute_loss(pos_score, neg_score)
            opt_merge.zero_grad()
            loss.backward()
            opt_merge.step()

        ### testing process ###

        model_pkg.eval()
        with torch.no_grad():
            trained_features_pkg = model_pkg.sage(train_pkg, node_features_pkg)
            pos_score_pkg = model_pkg.pred(test_pkg, trained_features_pkg, edge_type)
            neg_score_pkg = model_pkg.pred(test_neg_pkg, trained_features_pkg, edge_type)
            mrr_pkg = compute_mrr(pos_score_pkg, neg_score_pkg)

        model_merge.eval()
        with torch.no_grad():
            trained_features_merge = model_merge.sage(train_merge, node_features_merge)
            pos_score_merge = model_merge.pred(test_merge, trained_features_merge, edge_type)
            neg_score_merge = model_merge.pred(test_neg_merge, trained_features_merge, edge_type)
            mrr_merge = compute_mrr(pos_score_merge, neg_score_merge)

        mrr_mean.append((np.mean(mrr_pkg), np.mean(mrr_merge)))

    pkg_score = [x[0] for x in mrr_mean]
    merge_score = [x[1] for x in mrr_mean]
    t_statistic, p_value = ttest_rel(pkg_score, merge_score)
    print('t-statistic: ', t_statistic)
    print('p-value: ', p_value)

    mrr_score[file[:-4]] = mrr_mean

with open(f'mrr_score/case_{run_id}_{args.times}.pkl', 'wb') as f:
    pickle.dump(mrr_score, f)

