import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from train_test_split_edges import train_test_split_edges
from RGCN_conv import negative_sampling
from dataset import EntitiesIOSPress
from RGCN_conv import RGCNConv
import logging
print("starting network.py", flush=True)
dataset = EntitiesIOSPress()
dataset.train_mask = dataset.val_mask = dataset.test_mask = dataset.y = None
dataset.process()
print('done processing', flush=True)
data = train_test_split_edges(dataset.data)
print('done edges', flush=True)
print(data)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(data.num_nodes, 16, data.num_relations)
        self.conv2 = RGCNConv(16, data.num_classes, data.num_relations)

    def encode(self, edge_index, edge_type):
        print("Doing self.conv1()...")
        x = self.conv1(data.embeddings, edge_index, edge_type) 
        print("Doing x.relu...")
        x.relu() 
        print("Doing retunr self.conv2()")
        return self.conv2(x, edge_index, edge_type)


    def decode(self, z, pos_edge_index, neg_edge_index):
         print("Doing decode() edge_index...")
         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
         print("Return decode...")
         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

print("Getting device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Doing model, data = Net().to(device), data.to(device)...")
model, data = Net().to(device), data.to(device) 
print("Setting optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data):
    print("Start training...")
    model.train()
    print("Start negative sampling...")
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()

    print("Start encode...")
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type)
    print("Start link logits...")
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    print("Start link labels...")
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    print("Start loss...")
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    print("Start link probs")
    link_probs = link_logits.sigmoid()
    print("Start backward...")
    loss.backward()#retain_graph = True)
    print("Start optimizer...")
    optimizer.step()
    print("Return loss")
    return loss


@torch.no_grad()
def test(data):
    print("Doing model evaluation...")
    model.eval()
    print("Doing encode...")
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type)

    results = []
    for prefix in ["val", "test"]:
        print("Doing pos_edge_index ...")
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        print("Doing neg_egde_index...")
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        print("Doing link_logits...")
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        print("Doing linkg_probs...")
        link_probs = link_logits.sigmoid()
        print("Doing link labels...")
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        print("Doing roc_auc_score")
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        print("Return results") 
   return results

best_val_auc = test_auc = 0
for epoch in range(1, 101):
    train_loss = train(data)
    val_auc, tmp_test_auc = test(data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc= tmp_test_auc
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_auc, test_auc))

z = model.encode(data.train_pos_edge_index,data.train_pos_edge_index_edge_type)
final_edge_index = model.decode_all(z)
