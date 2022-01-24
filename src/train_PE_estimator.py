import torch
import torch.nn as nn
import pickle
from utils.train_test_split_edges import calculate_potential_energy

model = torch.load('frozen_models/trained_vanilla_RGCN.pth')
with open('dataset/data.pkl', 'rb') as handle:
    data = pickle.load(handle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for param in model.parameters():
    param.requires_grad = False

model.MLP = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

optimizer_MLP = torch.optim.Adam(model.MLP.parameters(), lr=0.01, weight_decay=0.005)

def decode_MLP(z, edge_index, MLP):
        src, dst = z[edge_index[0]], z[edge_index[1]]
        node_embeddings = torch.cat([src,dst], dim=-1)
        return MLP(node_embeddings)

def train_MLP(data, model):
    model.train()
    optimizer_MLP.zero_grad()
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)
    source_values = (decode_MLP(z,data.train_pos_edge_index, model.MLP).reshape(-1))
    target_values = data.potential_energy_for_train_index[data.train_pos_edge_index.unbind()]
    loss_L1= nn.L1Loss()
    loss =  loss_L1(source_values, target_values)
    loss.backward()
    optimizer_MLP.step()
    return loss

@torch.no_grad()
def test_MLP(data):
    model.eval()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type, data.embeddings)

    results = {}
    neg_edge_index_tuples = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        for source, dest in zip(neg_edge_index[0], neg_edge_index[1]):
            neg_edge_index_tuples.append((source,dest))
        neg_edge_index_PE = calculate_potential_energy(neg_edge_index_tuples)
        potential_energy = data[f'{prefix}_potential_energy']
        print(type(potential_energy[pos_edge_index.unbind()]))
        PE_pos = potential_energy[pos_edge_index.unbind()]
        PE_neg = neg_edge_index_PE[neg_edge_index.unbind()]
        PE = torch.cat([PE_pos, PE_neg])
        edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1)
        source_values = (decode_MLP(z,edge_index, model.MLP).reshape(-1))
        target_values = PE
        assert not any(target_values.isnan())
        loss_L1= nn.L1Loss()
        loss = loss_L1(source_values, target_values)
        results[prefix] = loss.detach().cpu()
    return results

#Train MLP
best_val_L1 = float('inf')
for epoch in range(0, 4000):
    train_loss = train_MLP(data, model)
    if epoch % 100 == 0:
        val_test_L1 = test_MLP(data)
        tmp_val = val_test_L1['val']
        tmp_test = val_test_L1['test']
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}' 
        print(log.format(epoch, train_loss, tmp_val, tmp_test))

torch.save(model, 'frozen_models/trained_PE_estimator.pth')