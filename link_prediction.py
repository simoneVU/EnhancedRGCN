from numpy.core.arrayprint import printoptions
from  EntitiesIOSPress import EntitiesIOSPress

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from negative_sampling import negative_sampling
from rgcn_conv import RGCNConv
from train_test_split_edges import train_test_split_edges
import logging

logging.basicConfig(filename='RGCNConv.log', filemode='w', format='%(asctime)s %(message)s')
dataset = EntitiesIOSPress()
dataset.train_mask = dataset.val_mask = dataset.test_mask = dataset.y = None
dataset.process()
print('done processing')
data = train_test_split_edges(dataset.data)
print('done edges')

'''              
                PROBLEM with def forward() in rgcn_conv.py in pytorch_geometric when the train is run.

                 The masked_edge_index_function does not return the masked_edge_index correctly.
                 If I return tmp (as masked edge_index) it gives me an error when it checks the input with __check_input__ in
                 def propagate(...) in torch_geometric/nn/conv/message_passing.py at line 220. I tried to change tmp to edge_index
                 but I still get another problem with the R-GCN because in its setting the node.dim = 0
                 meanwhile, for the GCN used in their pytorch_geometric/examples/link_pred.py example, the node.dim = -2. Then, I got an error
                 still in the torch_geometric/nn/conv/message_passing.py but line 160 for the function __lift__
                 which uses self. and expects it to have node.dim = -2. 
'''

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(data.num_nodes, 16, data.num_relations,
                                num_bases=30)
        self.conv2 = RGCNConv(16, data.num_classes, data.num_relations,
                              num_bases=30)

    def encode(self, edge_index, edge_type):
        #print("Edge_index dim: " + str(data.edge_index)) returns None???
        #logging.warning(f'self.conv1(data.x = {data.x}, data.train_pos_edge_index_shape = {data.train_pos_edge_index.shape}, edge_type_shape = {(edge_type.shape)})')
        x = self.conv1(None, edge_index, edge_type)
        x.relu() 
        return self.conv2(x, edge_index, edge_type)


    def decode(self, z, pos_edge_index, neg_edge_index):
         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()

    #logging.warning(f'edge_type_shape_in_training = {(data.edge_type.shape)})')
    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    #print("Loss is:" + str(loss))
    return loss


@torch.no_grad()
def test(data):
    model.eval()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type)

    results = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        print(str(link_logits) + "this is link_logits")
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

best_val_auc = test_auc = 0
for epoch in range(1, 101):
    print('Start Training')
    train_loss = train(data)
    print('Finish Training')
    print("Start Test Phase")
    val_auc, tmp_test_auc = test(data)
    print("Finish Test Phase")
    print("val_auc: " + str(val_auc) + "> " + str(best_val_auc) + "best_value_auc")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc= tmp_test_auc
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_auc, test_auc))

z = model.encode(data.x, data.train_pos_edge_index)
final_edge_index = model.decode_all(z)
