import torch

def get_GraphSAGE_embedding_feature(model, graph_collection):
  train_data = None
  label = None 
  for graph in graph_collection:
    x = graph.x
    y = graph.y
    edge_index = graph.edge_index
    graph_level_feature = model(x, edge_index)
    if train_data==None:
      label = y.unsqueeze(0)
      train_data = graph_level_feature.unsqueeze(0)
    else:
      train_data = torch.cat((train_data, graph_level_feature.unsqueeze(0)), dim=0)
      label = torch.cat((label, y.unsqueeze(0)), dim=0)
  return train_data, label