import torch
import pandas as pd
import rdflib as rdf
from torch_geometric.data import (InMemoryDataset, Data)
from collections import Counter
import logging

logging.basicConfig(filename='RGCNConv.log', filemode='a', format='%(asctime)s %(message)s')
class EntitiesIOSPress(InMemoryDataset):    


    def __init__(self):
        super(EntitiesIOSPress, self)
        self.data = None

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self):
        return self.data.train_y.max().item() + 1

    def process(self):

        g  = rdf.Graph()

        g.parse(location='TriplydbJS/results.ttl', format='ttl')
        freq_ = Counter(g.predicates())

        def freq(rel):
            return freq_[rel] if rel in freq_ else 0

        relations_set = sorted(set(g.predicates()), key=lambda rel: -freq(rel))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        relations_dict = {rel: i for i, rel in enumerate(list(relations_set))}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edge_list = []
        relations = []
        nodes_subjects = []
        nodes_objects = []
        train_labels = {}
        train_labels_set = set()

        for e1,r,e2 in g.triples((None, None, None)):
            if (e1 and r and e2) is not None:
                nodes_subjects.append(e1)
                nodes_objects.append(e2)
                relations.append(r)
            src, dst, rel = nodes_dict[e1], nodes_dict[e2], relations_dict[r]
            edge_list.append([src, dst, 2 * rel])
            edge_list.append([dst, src, 2 * rel + 1])
            train_labels[src] = relations_dict[r]
            train_labels_set.add(train_labels[src])

        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2]
        logging.warning(f'edge = {torch.tensor(edge_list, dtype=torch.long).t().contiguous()}')
        print("Edge_index dim: " + str(len(edge_index[1])) + "Edge_type dim: " + str(len(edge_type)))
        train_y = torch.tensor(list(train_labels_set), dtype=torch.long)
        ######################################### ONE-HOT Enconding for node-feature matrix###################################################
        nodes_dict_final = {"Nodes" : nodes_subjects, "Features" : [i for i in range(len(nodes_objects))]} 
        df = pd.DataFrame(nodes_dict_final)     
        ###############################################################################################################

        data = Data(edge_index=edge_index)
        f = df[['Nodes']].join(pd.get_dummies(df['Features']).add_prefix('FEATURE_')).groupby('Nodes').max()
        f_array = f.values
        f_tensor = torch.from_numpy(f_array)
        data.x = f_tensor 
        data.edge_list = edge     
        data.edge_index = edge_index
        data.edge_type = edge_type 
        data.train_y = train_y
        data.num_nodes = edge_index.max().item() + 1
        data.num_relations = edge_type.max().item() + 1
        data.num_classes = data.train_y.max().item() + 1
        self.data = data