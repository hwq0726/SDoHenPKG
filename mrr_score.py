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


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/pkg', help='Dataset name')
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


ds = dgl.data.CSVDataset(args.data)
g = ds[0]
num_nodes_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
k = args.num_negatives
in_features = args.in_features
hid_feats1 = args.hid_feats1
hid_feats2 = args.hid_feats2
hid_feats3 = args.hid_feats3
out_features = args.out_features
epochs = args.epoch
k_test = 10
etypes = [('anatomy', 'anatomy_protein', 'gene/protein'),
          ('disease', 'disease_drug', 'drug'),
          ('disease', 'disease_effect', 'effect/phenotype'),
          ('disease', 'disease_protein', 'gene/protein'),
          ('drug', 'drug_effect', 'effect/phenotype'),
          ('biological_process', 'biological_process_protein', 'gene/protein'),
          ('drug', 'drug_protein', 'gene/protein')]
mrr_score = {}

for etype in tqdm(etypes):
    print('current edge type is:', etype)
    mrr_mean = []
    for i in range(args.times):
        train_g, src_test, dst_test = get_data(g, etype, 5)
        n_hetero_features = args.in_features
        for ntype, num_nodes in num_nodes_dict.items():
            train_g.nodes[ntype].data['feature'] = torch.randn(num_nodes, n_hetero_features).to(device)

        model = Model(in_features, hid_feats1, hid_feats2, hid_feats3, out_features, train_g.etypes).to(device)

        node_features = {}
        for ntype in train_g.ntypes:
            node_features[ntype] = train_g.nodes[ntype].data['feature']

        ### training process ###
        opt = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            negative_graph = construct_negative_graph(train_g, k, etype)
            pos_score, neg_score = model(train_g, negative_graph, node_features, etype)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()

        ### testing process ###
        src_node, dst_node = etype[0], etype[2]
        test_nodes_dict = {src_node: num_nodes_dict[src_node], dst_node: num_nodes_dict[dst_node]}
        test_graph = dgl.heterograph({etype: (src_test, dst_test)}, num_nodes_dict=test_nodes_dict, device=device)

        model.eval()

        with torch.no_grad():
            trained_features = model.sage(train_g, node_features)

        test_features = {src_node: trained_features[src_node], dst_node: trained_features[dst_node]}

        neg_graph_test = construct_negative_graph(test_graph, k_test, etype)

        with torch.no_grad():
            pos_score = model.pred(test_graph, test_features, etype)
            neg_score = model.pred(neg_graph_test, test_features, etype)

        mrr = compute_mrr(pos_score, neg_score)
        mrr_mean.append(np.mean(mrr))

    mrr_score[etype] = mrr_mean

with open(f'mrr_score/mrr_{args.data[5:]}_{run_id}.pkl', 'wb') as f:
    pickle.dump(mrr_score, f)
