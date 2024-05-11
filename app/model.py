import pandas as pd
import numpy as np
import subprocess

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch

tv = torch.__version__
command = f'pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-"{tv}".html'
subprocess.run(command, shell=True)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm
from torch_geometric.data import Data, Dataset
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from torch_cluster import random_walk
import torch.optim as optim

import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, device):
        super(GNNModel,self).__init__()

        self.num_layers = num_layers
        # hidden_channels = (in_channels + out_channels)//2
        hidden_channels = 128

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.fc2 = torch.nn.Linear(hidden_channels//2, out_channels)

        self.device = device

        self.to(device)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # print(x.shape, edge_index.shape)

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_idx_maps(read_ids_file_path, truth):
    reads_truth = {}
    read_id_idx = {}

    with open(read_ids_file_path) as read_ids_file:
        for t, rid in tqdm(zip(truth, read_ids_file)):
            rid = rid.strip().split()[0][1:]
            reads_truth[rid] = t
            read_id_idx[rid] = len(read_id_idx)

    return reads_truth, read_id_idx

def get_train_data(truth, mask):
    lb = LabelEncoder()
    lb.fit(truth[mask])

    y = np.full(len(truth), -1)

    y[train_idx] = lb.transform(truth[train_idx])
    y = torch.tensor(y, dtype=torch.long)

    no_classes = len(set(truth[train_idx]))

    return y, no_classes, lb



def get_graph_data(features, edges,y,train_idx,test_idx,val_idx):
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    train_indices = torch.tensor(train_idx, dtype=torch.long)
    test_indices = torch.tensor(test_idx, dtype=torch.long)
    val_indices = torch.tensor(val_idx, dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=y)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True
    val_mask[val_indices] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    return data

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        weighted_train_loss = torch.mean(loss * sample_weights_train)
        weighted_train_loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        val_loss = criterion(out[graph.val_mask], graph.y[graph.val_mask])
        weighted_val_loss = torch.mean(val_loss * sample_weights_val)

        train_losses.append(weighted_train_loss.item())
        val_losses.append(weighted_val_loss.item())


        weighted_train_loss_np = weighted_train_loss.detach().numpy()
        weighted_val_loss_np = weighted_val_loss.detach().numpy()

        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {weighted_train_loss_np:.4f}, Val Acc: {acc:.4f}, Val Loss: {weighted_val_loss_np:.4f}')


        if acc > 0.995:
            break

    return model, train_losses, val_losses

def eval_node_classifier(model, graph, mask):
    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc


result_path = f"output"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = np.load(f'{result_path}/labels.npy', allow_pickle=True)
data = np.load(f'{result_path}/data.npz')
sample_weights = np.load(f"{result_path}/sample_weights.npy")

sample_weights = torch.tensor(sample_weights).to(device)
reads_truth, read_id_idx = get_idx_maps(f"{result_path}/read_ids", labels)

edges = data['edges']
comp = data['scaled']
comp = torch.from_numpy(comp).float()

id_list = np.array(list(read_id_idx.items()))
edge_index = torch.tensor(edges, dtype=torch.long)

train_idx = np.load(f"{result_path}/train_idx.npy")
test_idx = np.load(f"{result_path}/test_idx.npy")


y, no_classes, encoder = get_train_data(labels, train_idx)

train_idx, val_idx, weight_idx_train, weight_idx_val = train_test_split(train_idx, np.arange(len(train_idx)), test_size=0.1, random_state=42)

sample_weights_train = sample_weights[weight_idx_train]
sample_weights_val = sample_weights[weight_idx_val]
sample_weights_train.shape, sample_weights_val.shape


data = get_graph_data(comp, edges,y,train_idx,test_idx,val_idx)

num_layers = 2

model = GNNModel(data.x.shape[1], no_classes, num_layers, device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=10e-6)

history = train_node_classifier(model, data, optimizer, criterion, n_epochs=200)
# torch.save(model, f'{result_path}/model.pkl')
model.eval()
pred = model(data).argmax(dim=1)
y_pred = encoder.inverse_transform(pred[data.test_mask].cpu())

labels = np.load(f'{result_path}/labels.npy', allow_pickle=True)
# print(id_list)

labels[test_idx] = y_pred
pred_df = pd.DataFrame({'Read ID': id_list[:,0], 'Prediction': labels})

pred_df.to_csv(f'{result_path}/predictions.csv', index=False)