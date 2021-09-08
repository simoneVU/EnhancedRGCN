import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings("error")

class Net(torch.nn.Module):
    def __init__(self, params, data):
        super(Net, self).__init__()
        self.fc = nn.Linear(768, params["in_features"])
        self.conv1 = RGCNConv(params["in_features"], 16, data.num_relations, num_bases = params['num_bases'])
        self.conv2 = RGCNConv(16, params["num_classes"], data.num_relations, num_bases = params['num_bases'])
        
    def encode(self, edge_index, edge_type,data):
        embeddings = self.fc(data.embeddings)
        x = self.conv1(embeddings, edge_index, edge_type) 
        x = x.relu() 
        return self.conv2(x, edge_index, edge_type)

    def decode(self, z, pos_edge_index, neg_edge_index):
         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

def get_link_labels(pos_edge_index, neg_edge_index, device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(data, optimizer, model, device):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type,data)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    link_probs = link_logits.sigmoid()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data, optimizer, model, device):
    model.eval()

    z = model.encode(data.train_pos_edge_index, data.train_pos_edge_index_edge_type,data)

    results = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index,device)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

def running_train(data, optimizer, model,device, trial):
  best_val_auc = test_auc = 0
  for epoch in range(1, 10001):
      train_loss = train(data, optimizer, model, device)
      val_auc, tmp_test_auc = test(data,optimizer, model, device)
      if val_auc > best_val_auc:
          best_val_auc = val_auc
          test_auc= tmp_test_auc      
      trial.report(test_auc, epoch)
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
  return test_auc
  
def objective(trial):
    params = { 
          "in_features" : trial.suggest_int("in_features", low=32, high=256, step=32),
          "num_classes" : trial.suggest_int("n_classes", low=4, high=32, step=4),
          "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
          "num_bases" : trial.suggest_int("num_bases", 1, 6)
          }
    dataset = EntitiesIOSPress()
    dataset.train_mask = dataset.val_mask = dataset.test_mask = dataset.y = None
    dataset.process()
    data = train_test_split_edges(dataset.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(params,data).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0.0005)
    test_acc = running_train(data, optimizer, model, device, trial)
    return test_acc


#trials
study = optuna.create_study(study_name='RGCN_test_3',direction="maximize",storage='sqlite:///db.sqlite3')
study.optimize(objective, n_trials = 20)
print("best trial:")
trial_ = study.best_trial
print(trial_.values)
print(trial_.params)
