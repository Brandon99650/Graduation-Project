from torch_geometric.data import InMemoryDataset
import os.path as osp
import torch
from sklearn.preprocessing import normalize


class LoadDataset(InMemoryDataset):
    def __init__(self, root, transform=None):    
        super().__init__(root, transform)

    @property
    def processed_file_names(self):
        p = osp.join('processed','data.pt')
        return p

    def get(self):
        
        data = torch.load(osp.join(self.root, self.processed_file_names))
        try:
            data[0].num_classes = data[0].y.size()[1]
            data[0].affiliation = self.affiliation_distribution(data[0].y)
        except:
            pass
    
        return data[0]

    def affiliation_distribution(self, y):

        labels_num = {}
        for nodeid_labels in y:
            n = 0
            for l , label in enumerate(nodeid_labels):
                if label > 0:
                    n += 1
            if n in labels_num:
                labels_num[n] += 1
            else:
                labels_num[n] = 1
        return labels_num



def getdata(datafolder, nor=False):
    loader = None

    loader= LoadDataset(root = datafolder)
    gdata = loader.get()

    if nor:
        gdata.x = torch.tensor(normalize(gdata.x, axis = 1), dtype = torch.float)
    
    return gdata
