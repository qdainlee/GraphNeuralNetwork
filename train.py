import os
import time
import itertools
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import models
from utils import read_data, dict_collate_fn, dict_to_device


class MolDataSet(Dataset):
    '''Molecule dataset, list of dicts'''
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def run(dataloader, train):
    model.train() if train else model.eval()
    losses, preds, trues = [], [], []

    for data in dataloader:
        model.zero_grad()
        data = dict_to_device(data, device)

        pred = model(data)
        loss = criterion(pred, data['y'].view(-1, 1))

        losses.append(loss.data.cpu().numpy())
        preds.extend(pred.view(-1, 1).tolist())
        trues.extend(data['y'].view(-1, 1).tolist())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    loss = np.mean(losses)
    return loss, preds, trues


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay, L2')
parser.add_argument('--batch_size', type=int, default=32, help='number of batches')
parser.add_argument('--dropout', type=float, default=.2, help='dropout rate')
parser.add_argument('--readout', type=str, default='sum', help='readout method')
parser.add_argument('--num_gnn', type=int, default=3, help='number of gnn layers')
parser.add_argument('--dim_gnn', type=int, default=64, help='number of hidden dimension for gnn')
parser.add_argument('--num_fc', type=int, default=2, help='number of fc layers')
parser.add_argument('--dim_fc', type=int, default=512, help='number of hidden dimension for fc')
args = parser.parse_args()

# create folder to save local files
try:
    os.makedirs('models')
except FileExistsError:
    pass

# read data and split to train, validation
data = read_data(min_atom=5, max_atom=50)
train_data, val_data = train_test_split(data, test_size=.2, shuffle=True)

args.dim_af = train_data[0]['x'].shape[1]

# model, optimizer, criterion
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.GCN(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.BCELoss()
print(model)

train_dataset = MolDataSet(train_data)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=dict_collate_fn,
                              shuffle=True)
val_dataset = MolDataSet(val_data)
val_dataloader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            collate_fn=dict_collate_fn,
                            shuffle=True)

# train, validate
for epoch in range(1, args.num_epoch+1):
    s = time.time()
    train_epoch_loss, _, _ = run(train_dataloader, True)
    val_epoch_loss, preds, trues = run(val_dataloader, False)

    preds = np.array(list(itertools.chain(*preds)))
    trues = np.array(list(itertools.chain(*trues)))

    acc = accuracy_score(trues, preds>.5)
    auc = roc_auc_score(trues, preds)
    pr = precision_score(trues, preds>.5, zero_division=1)
    rc = recall_score(trues, preds>.5, zero_division=1)

    if epoch % 1 == 0:
        torch.save(model.state_dict(), f'models/E{epoch}.pt')
        print(f'Epoch: {epoch:04d} '
              f'Train Loss: {train_epoch_loss:.5f} '
              f'Val Loss: {val_epoch_loss:.5f} '
              f'AUC: {auc:.3f} '
              f'PR: {pr:.3f} '
              f'RC {rc:.3f}')
