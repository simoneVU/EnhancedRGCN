
import torch.nn
import torch
import rdflib as rdf
from torch_geometric.data import (InMemoryDataset, Data)

def create_dictionaries(set_unique_entities, set_unique_relations):
  nodes_dict = {value: nodeID for nodeID, value in enumerate(set_unique_entities)}
  relations_dict = {value: relID for relID, value in enumerate(set_unique_relations)}
  return nodes_dict, relations_dict
'''
def create_entities_embeddings(rdf_graph, entities_dict):
  entities_IDs_to_embeddings = torch.zeros(len(entities_dict), 54)
  #entity_embeddings_dict = {}
  #ID_to_remove = []
  #for loop through all unique entities and their IDs
  for entity, entity_ID in entities_dict.items():
    entity_textual_information = []
    #for loop through all the triples in the graph which have as entity the 
    #entity for entity_ID and where the entity e2 is a textual attribute for e1
    for e1,r,e2 in rdf_graph.triples((entity, None, None)):
      #check for literal attribute for entity e1
      if type(e2) == rdf.term.Literal:
      #create embedding for entity embeddings
        entity_textual_information.append(e2)
    if len(entity_textual_information) > 0:
      #entity_embeddings_dict[entity_ID] = get_bert_embedding([" ".join(entity_textual_information)])
      entities_IDs_to_embeddings[entity_ID]  = embeddingDimensionalityReduction(get_bert_embedding(entity_textual_information))
      #print(entity_embeddings_dict[entity_ID].size())
    #remove the entity from the set of unique entities if it does not have any textual attribute
   # else: ID_to_remove.append(entity)
    #print(list(entity_embeddings_dict.items())[:3])
  #print(len(ID_to_remove))  
  #numpy.sum(bc.encode("keyword1", "keyword2", "keyword3"), axis=0)
  #for entity in ID_to_remove:
    #del entities_dict[entity] 
  #entities_IDs_to_embeddings = torch.stack([embedding for embedding in entity_embeddings_dict.values()])     
  print(entities_IDs_to_embeddings)
  return entities_IDs_to_embeddings

def embeddingDimensionalityReduction(embedding):
    #reduces the dimension of the embedding through a dense layer
    return fc(embedding)
'''   
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
        graph = g.parse(location='prepro_without_5000_triples.ttl', format='ttl')        
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
        #print(len(entities_IDs_to_entities))
        #print(list(entities_IDs_to_entities.items())[:2])
        #print(list(relations_IDs_to_relations.items())[:2])  
        #Created the entity_to_embedding tensor   
        #entity_embeddings_dict = create_entities_embeddings(graph, entities_IDs_to_entities)
        #embeddings = torch.zeros(len(entity_embeddings_dict), 768)
        edge_list = []
        #print(list(entities_IDs_to_entities.items())[:5])
        #for each row in nodes add the correct embedding
        for e1,r,e2 in graph.triples((None, None, None)):
           assert (e1 and r and e2) is not None
           if type(e2) != rdf.term.Literal: 
            src, dst, rel = entities_to_entities_IDs[e1], entities_to_entities_IDs[e2], relations_to_relations_IDs[r]
            edge_list.append([src, dst, 2 * rel])
            edge_list.append([dst, src, 2 * rel + 1])
               # print(nodes[e1].size())
                #input_e1 = nodes[e1].view(batch_size, -1)
                #input_e2 = nodes[e2].view(batch_size, -1)
                #print(input_e1)
          
        #
        # Simulate a 28 x 28 pixel, grayscale "image"
       # input = torch.randn(1, 28, 28)
        # Use view() to get [batch_size, num_features].
        # -1 calculates the missing value given the other dim.
       # input = input.view(batch_size, -1) # torch.Size([1, 784])
        # Intialize the linear layer.
       #
        # Pass in the simulated image to the layer.
        #output = fc(embeddings)
       # print(output.shape)
     #   labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
     #   train_labels, tpytorch geometric
       #  est_labels = list(labels_dict.values())[:80,:], list(labels_dict.values())[80:,:]
       # train_y = torch.tensor(train_labels, dtype=torch.long)
        # IF for loop for edge_list edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2])) # deterministic 
        #TO ADD: for loop for edge_list
        #Edge list must be sorted
        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
        edge = torch.tensor(edge_list, dtype = torch.long).t().contiguous()
        edge_list_dict = {}
        for item in edge_list:
            edge_list_dict[(item[0], item[1])] = item[2]
        edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        #embeddings = entity_embeddings_dict
        #print(embeddings)
        edge_index, edge_type = edge[:2], edge[2]
        #print("edge_index is " + str(edge_index))
        #embeddings = output
        #print(embeddings[:3])
        #print(edge_index.shape)
        #print(edge_index[:3])

        data = Data(edge_index=edge_index)
        data.entities_to_entities_IDs = entities_to_entities_IDs
        data.graph = graph
        data.edge_list_dict = edge_list_dict
        print("Edge list lenght is " + str(len(edge_list)))
        #data.embeddings = embeddings
        data.edge_list = edge     
        data.edge_index = edge_index
        data.edge_type = edge_type 
        #data.train_y = train_y
        data.num_relations = edge_type.max().item() + 1
        data.num_classes = 32
        data.num_nodes = data.edge_index.max().item() + 1
        self.data = data
