import torch.nn
import torch
import rdflib as rdf
from torch_geometric.data import (InMemoryDataset, Data)

def create_dictionaries(set_unique_entities, set_unique_relations):
  nodes_dict = {value: nodeID for nodeID, value in enumerate(set_unique_entities)}
  relations_dict = {value: relID for relID, value in enumerate(set_unique_relations)}
  return nodes_dict, relations_dict
 
class EntitiesIOSPress(InMemoryDataset):    


    def __init__(self):
        super(EntitiesIOSPress, self)
        self.data = None

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_features(self):
        return self.data.tmp_num_features


    #def num_classes(self):
        #return self.data.train_y.max().item() + 1

    #Produces an embedding of size torch.Size([1,768]), 
    #it does not matter how long the list is.


    def process(self):
        g = rdf.Graph()
        graph = g.parse(location='dataset/10K_dataset_without_final.ttl', format='ttl')        
        entities = []
        relations = []
        for e1,r,e2 in graph.triples((None, None, None)):
         if type(e2) != rdf.term.Literal:
          entities.append(e1)
          entities.append(e2)
          relations.append(r)
        
        #Get unique set of relations
        relations = set(relations)
        #Get unique set of entities
        entities = set(entities)
        #Create dictionaries for entities' IDs and relations' IDs
        entities_to_entities_IDs, relations_to_relations_IDs = create_dictionaries(entities,relations) 
        edge_list = []
        #for each row in nodes add the correct embedding
        for e1,r,e2 in graph.triples((None, None, None)):
           assert (e1 and r and e2) is not None
           if type(e2) != rdf.term.Literal: 
            src, dst, rel = entities_to_entities_IDs[e1], entities_to_entities_IDs[e2], relations_to_relations_IDs[r]
            edge_list.append([src, dst, 2 * rel])
            edge_list.append([dst, src, 2 * rel + 1])
          
        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype = torch.long).t().contiguous()
        edge_list_dict = {}
        for item in edge_list:
            edge_list_dict[(item[0], item[1])] = item[2]
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index, edge_type = edge[:2], edge[2]

        data = Data(edge_index=edge_index)
        data.entities_to_entities_IDs = entities_to_entities_IDs
        data.graph = graph
        data.edge_list_dict = edge_list_dict
        data.edge_list = edge     
        data.edge_index = edge_index
        data.edge_type = edge_type 
        data.num_relations = edge_type.max().item() + 1
        data.num_classes = 32
        data.num_nodes = data.edge_index.max().item() + 1
        self.data = data
