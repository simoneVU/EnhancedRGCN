import torch
import torch.nn as nn
from utils.RGCN_conv import RGCNConv

class R_GCN(torch.nn.Module):
        def __init__(self, data, device):
            super(R_GCN, self).__init__()
            self.node_emb = data.embeddings.to(device)
            self.fc = nn.Linear(768, 64)
            self.rand_embed = nn.parameter.Parameter(torch.Tensor(data.embeddings.shape[0], 64))
            self.conv1 = RGCNConv(64, 14, data.num_relations, num_bases=2)
            self.conv2 = RGCNConv(14, 32, data.num_relations, num_bases=2)
            self.conv3 = RGCNConv(32, 32, data.num_relations, num_bases=2)
            self.rel_emb = nn.parameter.Parameter(torch.Tensor(data.num_relations, 32))
            self.reset_parameters()
        
        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.node_emb)
            torch.nn.init.xavier_uniform_(self.rand_embed)
            self.fc.reset_parameters()
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()
            self.conv3.reset_parameters()
            torch.nn.init.xavier_uniform_(self.rel_emb)

        def encode(self, edge_index, edge_type, embeddings):
            embeddings = self.fc(self.node_emb).add(self.rand_embed)
            x = self.conv1(embeddings, edge_index, edge_type)
            x = x.relu_()
            x = self.conv2(x, edge_index, edge_type)
            x = self.conv3(x, edge_index, edge_type)
            return x
            
        def decode(self, z, pos_edge_index, neg_edge_index, edge_type):
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            rel = self.rel_emb[torch.cat([edge_type, edge_type])]
            i = z[edge_index[0]] * rel
            j = i * z[edge_index[1]]
            return torch.sum(j, dim=1) 

        def decode_optimized(self, MLP_output, z, pos_edge_index, neg_edge_index, edge_type, alpha):
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            rel = self.rel_emb[torch.cat([edge_type, edge_type])]
            i = z[edge_index[0]] * rel
            j = i * z[edge_index[1]]
            k = torch.sum(j, dim=1) 
            l = (1-alpha)*k + (MLP_output * alpha)
            return l

        