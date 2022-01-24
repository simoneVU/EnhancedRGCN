import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.train_test_split_edges import train_test_split_edges
from utils.RGCN_conv import negative_sampling
from dataset.dataset import EntitiesIOSPress
from models import R_GCN


dataset = EntitiesIOSPress()
dataset.train_mask = dataset.val_mask = dataset.test_mask = dataset.y = None
dataset.process()
print("Data process done!")
data = train_test_split_edges(dataset.data)
with open('dataset/data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = R_GCN(data,device).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
data.potential_energy_for_train_index.to(device)

def train(data):
    model.train()
    neg_edge_index = negative_sampling(data.train_pos_edge_index,data.num_nodes, device=device)
    optimizer.zero_grad()
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index, data.train_pos_edge_index_edge_type)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    cross_entropy_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    reg_loss = z.pow(2).mean() + model.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)

    results = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        edge_type = data[f'{prefix}_pos_edge_index_edge_type']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index, edge_type)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

best_val_auc = test_auc = 0

for epoch in range(100):
    train_loss = train(data) 
    val_auc, tmp_test_auc = test(data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc= tmp_test_auc
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_auc, test_auc))

torch.save(model, 'frozen_models/trained_vanilla_RGCN.pth')
