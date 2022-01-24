import torch
from utils.RGCN_conv import negative_sampling
from utils.train_test_split_edges import calculate_potential_energy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('frozen_models/trained_PE_estimator.pth')
with open('dataset/data.pkl', 'rb') as handle:
    data = pickle.load(handle)

def decode_MLP(z, edge_index, MLP):
        src, dst = z[edge_index[0]], z[edge_index[1]]
        node_embeddings = torch.cat([src,dst], dim=-1)
        return MLP(node_embeddings)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
    
#Freeze RGCN parameters
for param in model.parameters():
    param.requires_grad = False 

model.scaling_factor = nn.parameter.Parameter(torch.tensor([0.0]))
optimizer_MLP_updated_decoder = torch.optim.Adam([model.scaling_factor], lr=0.01, weight_decay=0.0005)

def train_optimized_RGCN(data, model):
    model.train()
    neg_edge_index = negative_sampling(data.train_pos_edge_index,data.num_nodes, device=device)
    neg_edge_index_tuples = []
    for source, dest in zip(neg_edge_index[0], neg_edge_index[1]):
        neg_edge_index_tuples.append((source,dest))
    neg_edge_index_PE = calculate_potential_energy(neg_edge_index_tuples)
    optimizer_MLP_updated_decoder.zero_grad()
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)
    PE_pos = data.potential_energy_for_train_index[data.train_pos_edge_index.unbind()]
    PE_neg = neg_edge_index_PE[neg_edge_index.unbind()]
    PE = torch.cat([PE_pos, PE_neg])
    link_logits = model.decode_optimized(PE, z, data.train_pos_edge_index, neg_edge_index, data.train_pos_edge_index_edge_type, model.scaling_factor)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    cross_entropy_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    reg_loss = z.pow(2).mean() + model.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss
    loss.backward()
    optimizer_MLP_updated_decoder.step()
    return loss

@torch.no_grad()
def test_optimized_RGCN(data):
    model.eval()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)
    neg_edge_index_tuples = []
    results = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        edge_type = data[f'{prefix}_pos_edge_index_edge_type']
        potential_energy_pos = data[f'{prefix}_potential_energy'][pos_edge_index.unbind()].to(device)
        for source, dest in zip(neg_edge_index[0], neg_edge_index[1]):
            neg_edge_index_tuples.append((source,dest))
        neg_edge_index_PE = calculate_potential_energy(neg_edge_index_tuples)
        potential_energy_neg = neg_edge_index_PE[neg_edge_index.unbind()]
        PE = torch.cat([potential_energy_pos, potential_energy_neg])
        link_logits = model.decode_optimized(PE, z, pos_edge_index, neg_edge_index, edge_type, model.scaling_factor)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

#Train RGCN with DistMult again
best_val_auc = test_auc = 0
for epoch in range(1, 300):
    train_loss = train_optimized_RGCN(data,model)  
    val_auc, tmp_test_auc = test_optimized_RGCN(data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc= tmp_test_auc
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Alpha: {:.15f}'
    print(log.format(epoch, train_loss, best_val_auc, test_auc, model.scaling_factor.item()))
