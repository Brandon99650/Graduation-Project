import torch
import torch.utils.data as data_utils
from torch_geometric.utils import negative_sampling 


class EdgeSampler(data_utils.Dataset):

    def __init__(self, g, batchsize, epochs):

        self.batchsize = batchsize
        self.epochs = epochs
        self.edges = g.edge_index
        self.uptri_adj =  self.upperTri_adj_idx(self.edges)

    def __getitem__(self, idx):

        edge_sample = self.random_choose_edge(self.uptri_adj)
        neg_sample = negative_sampling(edge_index = edge_sample)
        return edge_sample, neg_sample

    def upperTri_adj_idx(self, edge_index):
        tri_mask = edge_index[0] < edge_index[1]
        uptri_edges = edge_index[:,tri_mask]
        return uptri_edges

    def random_choose_edge(self, uptri_e:torch.tensor):
        selection = uptri_e[:,torch.randperm(uptri_e.size()[1])][:,:self.batchsize]
        return selection
    
    def __len__(self):
        return self.epochs 

def collate_fn(batch):
    edges, nonedges = batch[0]
    return (edges, nonedges)

def get_edge_sampler(g, batchsize, epochs):
    data_source = EdgeSampler(g, batchsize, epochs)
    return data_utils.DataLoader(data_source,collate_fn=collate_fn)
