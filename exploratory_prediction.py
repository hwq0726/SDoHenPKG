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
import scipy.sparse as sp
# import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--edge_type', type=tuple, default=('gene/protein', 'protein_SDoH factor', 'SDoH factor'), help='Target edges')
parser.add_argument('--in_features', type=int, default=100, help='Input feature dimension & the original feature dimension')
parser.add_argument('--hid_feats1', type=int, default=200, help='First hidden layer dimension')
parser.add_argument('--hid_feats2', type=int, default=400, help='Second hidden layer dimension')
parser.add_argument('--hid_feats3', type=int, default=200, help='Third hidden layer dimension')
parser.add_argument('--out_features', type=int, default=100, help='Output feature dimension')
parser.add_argument('--num_negatives', type=int, default=50, help='Number of negative samples')
parser.add_argument('--epoch', type=int, default=200, help='Number of epoch')
args = parser.parse_args()
parser.print_help()


run_id = str(uuid.uuid4())[:4]  # Use only the first 4 characters for brevity
print(f'Unique Run ID: {run_id}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
ds = dgl.data.CSVDataset('data/merge')
g = ds[0]
g = g.to(device)


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            # Apply sigmoid to get the likelihood (probability) of the edge
            score = graph.edges[etype].data['score']
            return torch.sigmoid(score)


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
    # Binary cross-entropy loss
    pos_label = torch.ones_like(pos_score)
    neg_label = torch.zeros_like(neg_score)

    loss_pos = F.binary_cross_entropy(pos_score, pos_label)
    loss_neg = F.binary_cross_entropy(neg_score, neg_label)

    return loss_pos + loss_neg


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


k = args.num_negatives
in_features = args.in_features
hid_feats1 = args.hid_feats1
hid_feats2 = args.hid_feats2
hid_feats3 = args.hid_feats3
out_features = args.out_features
relation = args.edge_type
epochs = args.epoch

num_nodes_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
for ntype, num_nodes in num_nodes_dict.items():
    g.nodes[ntype].data['feature'] = torch.randn(num_nodes, in_features).to(device)


model = Model(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, g.etypes).to(device)
node_features = {}
for ntype in g.ntypes:
    node_features[ntype] = g.nodes[ntype].data['feature']
opt = torch.optim.Adam(model.parameters())
for epoch in tqdm(range(epochs)):
    negative_graph = construct_negative_graph(g, k, relation)
    pos_score, neg_score = model(g, negative_graph, node_features, relation)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

u, v = g.edges(etype=relation)
u = u.cpu()
v = v.cpu()
num_of_nodes_0 = num_nodes_dict[relation[0]]
num_of_nodes_1 = num_nodes_dict[relation[2]]
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(num_of_nodes_0, num_of_nodes_1))
if relation[0] == relation[2]:
    adj_neg = 1 - adj.todense() - np.eye(num_of_nodes_0)
else:
    adj_neg = 1 - adj.todense()
neg_u, neg_v = np.where(adj_neg != 0)
predict_target = dgl.heterograph(
        {relation: (neg_u, neg_v)},
        num_nodes_dict=num_nodes_dict, device=device)


model.eval()
with torch.no_grad():
    features = model.sage(g, node_features)
    likelihoods = model.pred(predict_target, features, relation)

# save the results
results = {'edges': (neg_u, neg_v), 'score': likelihoods}
with open(f'exp_pred/exp_pred_{run_id}_{relation[1]}.pkl', 'wb') as f:
    pickle.dump(results, f)

# # plot histogram
# plt.hist(likelihoods.cpu().numpy(), bins=100)
# plt.yscale('log')
# plt.xlabel('Likelihood')
# plt.ylabel('Frequency')
# plt.title('Likelihood Distribution')
# plt.savefig(f'histogram_{run_id}.png')
# plt.show()
#
# # plot heatmap
# heatmap_data = np.full((num_of_nodes, num_of_nodes), np.nan)
# for i in tqdm(range(len(neg_u))):
#     heatmap_data[neg_u[i], neg_v[i]] = likelihoods.view(-1)[i].item()
#
# plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.savefig(f'heatmap_{run_id}.png')
# plt.show()

