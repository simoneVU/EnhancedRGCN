from os import path
import math
import torch
from torch_geometric.utils import to_undirected
import rdflib as rdf
import networkx as nx
from itertools import combinations
from transformers import BertTokenizer, BertModel

print("I am here before BERT")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print("Im here after BERT")
def get_bert_embedding(list_of_strings):
  tensor_length = len(list_of_strings)
  result_embedding = torch.zeros(tensor_length, 768)
  for sentence in list_of_strings:
    input_ids = torch.tensor(tokenizer.encode(sentence[:512])).unsqueeze(0)  #batch size 1
    outputs = model(input_ids)
    embeddings_of_last_layer = outputs[0]
    cls_embeddings = embeddings_of_last_layer[0]
    result_embedding[list_of_strings.index(sentence)] = torch.sum(cls_embeddings, dim=0)
  return torch.sum(result_embedding, dim=0)

def create_entities_embeddings(rdf_graph, entities_dict, embedding_dimension):
  entities_IDs_to_embeddings = torch.zeros(len(entities_dict), embedding_dimension)
  #for loop through all unique entities and their IDs
  for rdf_entity, entity_ID in entities_dict.items():
    entity_textual_information = []
    #for loop through all the triples in the graph which have as entity the 
    #entity for entity_ID and where the entity e2 is a textual attribute for e1
    for e1,r,e2 in rdf_graph.triples((rdf_entity, None, None)):
      #check for literal attribute for entity e1
      if type(e2) == rdf.term.Literal:
      #create embedding for entity embeddings
        entity_textual_information.append(e2)
    if len(entity_textual_information) > 0:
      entities_IDs_to_embeddings[entity_ID]  =  get_bert_embedding(entity_textual_information)
    
  return entities_IDs_to_embeddings.detach()

def calculate_potential_energy(edge_index_tuples):
    edge_index_graph = nx.Graph()
    edge_index_graph.add_edges_from(edge_index_tuples)
    #Calculate connect_components
    connected_component_graphs = [edge_index_graph.subgraph(c).copy() for c in nx.connected_components(edge_index_graph)]
    #Calculate the common_neighbours for all the pair of nodes in train_pos_edge_index
    potential_energy_edge_index = torch.zeros((max(edge_index_graph.nodes())+1, max(edge_index_graph.nodes())+1), dtype=torch.float)
    for connected_component in connected_component_graphs:
        sp = dict(nx.all_pairs_shortest_path_length(connected_component))
        for node_1, node_2 in combinations(list(connected_component.nodes()), r=2):
          product_of_degree_of_nodes = connected_component.degree(node_1) * connected_component.degree(node_2)
          common_neighbours = list(nx.common_neighbors(edge_index_graph, node_1, node_2))
          clustering_coefficient_sum = 0 
          if common_neighbours: 
            for common_neighbour in common_neighbours:
               clustering_coefficient_sum += nx.clustering(edge_index_graph, common_neighbour)
          else:
            clustering_coefficient_sum = 0.1
          shortest_path_length = sp[node_1][node_2]
          potential_energy = product_of_degree_of_nodes * clustering_coefficient_sum * (1.0/shortest_path_length)
          potential_energy_edge_index[node_1][node_2] = potential_energy
          potential_energy_edge_index[node_2][node_1] = potential_energy
    return potential_energy_edge_index

def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):

    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    
    print("Start train_test_split_edges")
    assert 'batch' not in data  # No batch-mode.
    edge_list = data.edge_list
    num_nodes = data.num_nodes
    entities_to_IDs_dict = data.entities_to_entities_IDs
    row, col= data.edge_index
    graph = data.graph
    edge_list_dict = data.edge_list_dict
    mask = row < col 
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
  

    data.train_pos_edge_index_edge_type = []
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    train_pos_edge_index_entites_to_entites_IDs = []
    train_pos_edge_index_tuples = []
    for sub, obj in zip(data.train_pos_edge_index[0], data.train_pos_edge_index[1]):
      if (int(sub),int(obj)) in edge_list_dict.keys():
          train_pos_edge_index_tuples.append((int(sub),int(obj)))
          data.train_pos_edge_index_edge_type.append(edge_list_dict[(int(sub), int(obj))])
    data.train_pos_edge_index_edge_type =  torch.tensor(data.train_pos_edge_index_edge_type, dtype=torch.long).t().contiguous()   

    data.val_pos_edge_index_edge_type = []
    data.test_pos_edge_index_edge_type = []
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    val_pos_edge_index_tuples = []
    for sub, obj in zip(data.val_pos_edge_index[0], data.val_pos_edge_index[1]):
      if (int(sub),int(obj)) in edge_list_dict.keys():
          val_pos_edge_index_tuples.append((int(sub),int(obj)))
          data.val_pos_edge_index_edge_type.append(edge_list_dict[(int(sub), int(obj))])
    data.val_pos_edge_index_edge_type =  torch.tensor(data.val_pos_edge_index_edge_type, dtype=torch.long).t().contiguous()
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    test_pos_edge_index_tuples = []
    for sub, obj in zip(data.test_pos_edge_index[0], data.test_pos_edge_index[1]):
      if (int(sub),int(obj)) in edge_list_dict.keys():
          test_pos_edge_index_tuples.append((int(sub),int(obj)))
          data.test_pos_edge_index_edge_type.append(edge_list_dict[(int(sub), int(obj))])
    data.test_pos_edge_index_edge_type =  torch.tensor(data.test_pos_edge_index_edge_type, dtype=torch.long).t().contiguous()
    
    
    current_graph = list(train_pos_edge_index_tuples)
    potential_energy_for_train_index = calculate_potential_energy(current_graph)
    val_pos_edge_index_tuples =  set(val_pos_edge_index_tuples).difference(train_pos_edge_index_tuples)
    
    print("I am here 0")
    current_graph.extend(val_pos_edge_index_tuples)
    potential_energy_for_val_index = calculate_potential_energy(current_graph)
    print("I am here 1")
    test_pos_edge_index_tuples =  set(test_pos_edge_index_tuples).difference(val_pos_edge_index_tuples)
    test_pos_edge_index_tuples =  set(test_pos_edge_index_tuples).difference(train_pos_edge_index_tuples)
    current_graph.extend(test_pos_edge_index_tuples)
    print("I am here 2")
    potential_energy_for_test_index = calculate_potential_energy(current_graph)
  
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    print("Im here 3")
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    if path.exists("/Users/simonecolombo/Desktop/eRGCN/embeddings/embeddings10K.pt"):
      print("loeading embeddings")
      train_pos_edge_index_entity_embeddings = torch.load("/Users/simonecolombo/Desktop/eRGCN/embeddings/embeddings10K.pt")
      print("embeddings detected")
    else:
      print("creating new embeddings")
      train_pos_edge_index_entity_embeddings = create_entities_embeddings(graph, entities_to_IDs_dict, 768)
      torch.save(train_pos_edge_index_entity_embeddings, "/home/colombo/eRGCN/utils/embeddings10K.pt")
      print("embeddings not detected")
     

    data.embeddings = train_pos_edge_index_entity_embeddings
    data.potential_energy_for_train_index = potential_energy_for_train_index
    data.val_potential_energy = potential_energy_for_val_index
    data.test_potential_energy = potential_energy_for_test_index
    return data
