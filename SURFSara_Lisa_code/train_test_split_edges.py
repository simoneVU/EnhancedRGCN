import math
import torch
from torch_geometric.utils import to_undirected
import numpy
import logging
import rdflib as rdf
import faulthandler
faulthandler.enable()
logging.basicConfig(filename='link_pred_simone.log', filemode='w', level=logging.DEBUG)
logging.debug('Bert server is loading the server...')
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(list_of_strings):
  tensor_length = len(list_of_strings)
  result_embedding = torch.zeros(tensor_length, 768)
  for sentence__ in list_of_strings:
    sentence = str(sentence__)
    input_ids = torch.tensor(tokenizer.encode(sentence[:512])).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    embeddings_of_last_layer = outputs[0]
    cls_embeddings = embeddings_of_last_layer[0]
    result_embedding[list_of_strings.index(sentence__)] = torch.sum(cls_embeddings, dim=0)
  return torch.sum(result_embedding, dim=0)
def create_entities_embeddings(rdf_graph, entities_dict, dimension_for_reduction):
  entities_IDs_to_embeddings = torch.zeros(len(entities_dict), dimension_for_reduction)
  #print("Entities_dict is " + str(entities_dict))
  #entity_embeddings_dict = {}
  #ID_to_remove = []
  #for loop through all unique entities and their IDs
  for index, tuple_ in enumerate(entities_dict):
    #print((index,tuple_))
    entity_textual_information = []
    #print(tuple_)
    #for loop through all the triples in the graph which have as entity the 
    #entity for entity_ID and where the entity e2 is a textual attribute for e1
    for e1,r,e2 in rdf_graph.triples((tuple_[0], None, None)):
      #check for literal attribute for entity e1
      if type(e2) == rdf.term.Literal:
      #create embedding for entity embeddings
        entity_textual_information.append(e2.encode('utf-8'))
    if len(entity_textual_information) > 0:
      #print("Making the embeddings...")
      #print(entity_textual_information)
      #entity_embeddings_dict[entity_ID] = get_bert_embedding([" ".join(entity_textual_information)])
      entities_IDs_to_embeddings[index]  = embeddingDimensionalityReduction(get_bert_embedding(entity_textual_information),dimension_for_reduction)
      logging.debug(f"Making the embeddings ... {entities_IDs_to_embeddings[index]} and entity textual info is {entity_textual_information}")
      #print(entities_IDs_to_embeddings[:5])
      #print(entity_embeddings_dict[entity_ID].size())
    #remove the entity from the set of unique entities if it does not have any textual attribute
   # else: ID_to_remove.append(entity)
    #print(list(entity_embeddings_dict.items())[:3])
  #print(len(ID_to_remove))  
  #numpy.sum(bc.encode("keyword1", "keyword2", "keyword3"), axis=0)
  #for entity in ID_to_remove:
    #del entities_dict[entity] 
  #entities_IDs_to_embeddings = torch.stack([embedding for embedding in entity_embeddings_dict.values()])     
  print("return embeddings") 
  return entities_IDs_to_embeddings

def embeddingDimensionalityReduction(embedding, dimension_for_reduction):
    #reduces the dimension of the embedding through a dense layer
    fc  = torch.nn.Linear(768, dimension_for_reduction)
    return fc(embedding)


def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.
    edge_list = data.edge_list
    #num_nodes = data.num_nodes
    entities_to_IDs_dict = data.entities_to_entities_IDs
    row, col = data.edge_index
    graph = data.graph
    edge_list_dict = data.edge_list_dict
    data.edge_index = None
    #print(edge_list)
    #print("Row before mask" + str(row) + "Col before mask" + str(col))
    # Return upper triangular portion.
    mask = row < col 
    #print(row[mask])
    #print(mask) 
    row, col = row[mask], col[mask]
    #print("Row " + str(row) + "Col " + str(col))

    n_v = int(math.floor(val_ratio * row.size(0)))
    #print("n_v" + str(n_v))
    #print("row.size(0) = " + str(row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    #print("n_t" + str(n_t))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    #print("perm" + str(perm))
    #print("Lenght of row is " + str(len(row)))
    #print("row[perm[0]] is " + str(row[perm[0]]))
    row, col = row[perm], col[perm]
    #print("row, col" + str(row[perm]) + str(col[perm]))

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    
    data.train_pos_edge_index_edge_type = []
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    #print()
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    #print("r is " + str(r) + "and c is " + str(c))
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    #print("train_pos_edge_index is " + str(data.train_pos_edge_index))
    #print(edge_list[0])
    #print(edge_list[1])
   # print(edge_list[2])
    #for sub, obj in zip(data.train_pos_edge_index[0], data.train_pos_edge_index[1]):
    #  if sub in edge_list[0]:
     #   sub_index = edge_list[0].tolist().index(sub)
     #   obj_index = edge_list[1].tolist().index(sub)
    edge_list_entites_to_entites_IDs = []
    print("Start for loop")
    for sub, obj in zip(data.train_pos_edge_index[0], data.train_pos_edge_index[1]):
      if (int(sub),int(obj)) in edge_list_dict.keys():
          data.train_pos_edge_index_edge_type.append(edge_list_dict[(int(sub), int(obj))])
    data.train_pos_edge_index_edge_type =  torch.tensor(data.train_pos_edge_index_edge_type, dtype=torch.long).t().contiguous()
    print("Finish for loop")   
    print("starts negative edges 0")
    # Negative edges.
    num_nodes = data.num_nodes #list(data.train_pos_edge_index.size())[1]
    print("Number of nodes is :" + str(num_nodes))
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    print("Negative adj. mask is " + str(neg_adj_mask))
    print("starts negative edges 1")
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    print("starts negative edges 2")
    neg_adj_mask[row, col] = 0
    print("Final neg adj mask is " + str(neg_adj_mask))
    print("Size of neg adj mask is " +  str(neg_adj_mask.size()))

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    print("Neg_row size is " + str(neg_row.size()))
    print("Neg_col size is " + str(neg_col.size()))
    print("starts negative edges 3")
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    print("Neg_row perm is " + str(neg_row))
    print("Neg_col perm is " + str(neg_col))
    print("Size of neg_row is" + str(neg_row.size()))
    print("Stops here 0")
    neg_adj_mask[neg_row, neg_col] = 0
    print("Size of neg_adj_mask is " + str(neg_adj_mask.size()))
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    print("Stops here 1")
    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
       
    for entity_ID_e1 in edge_list[0]:
     #print(int(entity_ID_e1))
     #print(int(entity_ID_e2))  
     #print(entities_to_IDs_dict.values())
     assert int(entity_ID_e1) in entities_to_IDs_dict.values()
       #print(list(entities_to_IDs_dict.keys())[list(entities_to_IDs_dict.values()).index(int(entity_ID))])   
     #rdf_entity_e1 is the entity in rdf format corresponding to the entity_ID_e1 in train_post_edge_index[0]]. I know
     #that I am usaing the dict in a wrong way since I am looking through the keys and I am retrieving the entity rdf value
     #corresponding to entitity_ID_e1 in train_pos_edge_index 
     rdf_entity_e1 = list(entities_to_IDs_dict.keys())[list(entities_to_IDs_dict.values()).index(int(entity_ID_e1))]
     edge_list_entites_to_entites_IDs.append((rdf_entity_e1,int(entity_ID_e1)))
    #I do not need to check entity_ID_e2 because all the unique_nodes/entities are listed in 
    #assert int(entity_ID_e2) in entities_to_IDs_dict.values()
     #rdf_entity_e2 = list(entities_to_IDs_dict.keys())[list(entities_to_IDs_dict.values()).index(int(entity_ID_e2))]
     #train_pos_edge_index_entites_to_entites_IDs.append((rdf_entity_e2,int(entity_ID_e2)))   
    #print("Length of train_pos_edge_index_entites_to_entites_IDs"+ str(len(train_pos_edge_index_entites_to_entites_IDs)))

    entity_embeddings = create_entities_embeddings(graph, edge_list_entites_to_entites_IDs, 16)
    #print("data.embeddings is. : " + str(train_pos_edge_index_entity_embeddings))
    data.embeddings = entity_embeddings
    
    #print("data.embeddings is. : " + str(data.embeddings))
    #print("data.train_pos_edge_index" + str(data.train_pos_edge_index.size()))
    #print("data.embeddings size is. : " + str(data.embeddings.size()))
    print(data)
    logging.debug(f"This is the data : {data}") 
    return data
