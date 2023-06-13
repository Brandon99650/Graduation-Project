from genericpath import isfile
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
import numpy as np
import torch 
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as GData
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer,util
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#NLPmodel=SentenceTransformer("all-mpnet-base-v2", cache_folder="./utils/")

class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])

def textsim(tx):
    print("textsim")
    numnodes = tx.size()[0]
    ordered = torch.zeros((numnodes, numnodes), dtype=torch.long)
    coss_value = torch.zeros((numnodes, numnodes),dtype=torch.float)
    unsorted_coss_value = torch.zeros((numnodes, numnodes),dtype=torch.float)
    for i in tqdm(range(numnodes)):
        xi = tx[i,:].expand(tx.size())
        coss= util.pairwise_cos_sim(xi, tx)
        unsorted_coss_value[i] = coss
        coss_sorted = torch.sort(coss, dim=0, descending=True)
        coss_value[i]= coss_sorted.values
        ordered[i]=coss_sorted.indices
    return ordered, coss_value, unsorted_coss_value

def get_edge_index_from_tuple_list(nblist):
    s = []
    t = []
    for p in nblist:
        s.append(p[0])
        t.append(p[1])
    e1 = torch.tensor(s,dtype=torch.long)
    e2 = torch.tensor(t, dtype=torch.long)
    edge_index = torch.stack((e1,e2), dim=0)
    return edge_index

def navieKNNG(ordered, k):
    nblist = set()
    for i in tqdm(range(ordered.size()[0])):
        for j in range(k+1):
            if ordered[i][j].item() == i:
                continue
            nblist.add((i, ordered[i][j].item())) 
            nblist.add((ordered[i][j].item(), i))
        
    nblist = list(nblist)
    return get_edge_index_from_tuple_list(nblist)

def thresholdNNG(ordered, coss_value,threshold=0.5):
    nblist = list()
    w = list()
    for i in tqdm(range(ordered.size()[0])):
        nb_num = 0
        
        for j in range(ordered.size()[1]):
            if coss_value[i][j].item() < threshold:
                break
            nb_num += 1
        
        for j in range(nb_num):
            if ordered[i][j].item() == i:
                continue
            
            if ((i, ordered[i][j].item())) not in nblist:
                nblist.append((i, ordered[i][j].item()))
                nblist.append((ordered[i][j].item(), i))
                w.append(coss_value[i][j].item())
                w.append(coss_value[i][j].item())
            
    e = get_edge_index_from_tuple_list(nblist=nblist)
    w = torch.tensor(w, dtype=torch.float)
    return e, w


def cknn(orederd, sortedcoss_value,unsortedcoss_value, k=5, cntr=0.5):
    nblist = set()
    for i in tqdm(range(orederd.size()[0])):
        k_th_n = orederd[i][k].item()
        cosik = sortedcoss_value[i][k]
        for j in range(orederd.size()[1]):
            thisnode =  orederd[i][j].item()
            cosij = sortedcoss_value[i][j].item()
            cosjk = unsortedcoss_value[k_th_n][thisnode].item()
            if cosij > (cntr*((cosik*cosjk)**0.5)):
                nblist.add((i, orederd[i][j]))
                nblist.add((orederd[i][j], i))
    return get_edge_index_from_tuple_list(nblist=list(nblist))


def main():

    datafolder = os.path.join("data", "attraction")
    featurepath = os.path.join(datafolder,"ToldescribeEBD.pt")
    #labelpath =  os.path.join(datafolder,"processed", "labels.pt")
    #labels = torch.load(labelpath)
    features = torch.load(featurepath)
    ordered = None
    cossvalue = None
    unsorted = None
    if not os.path.isfile(os.path.join(datafolder,"processed","order.pt")):
        ordered, cossvalue, unsorted = textsim(features)
        torch.save(ordered, f = os.path.join(datafolder,"processed","order.pt"))
        torch.save(cossvalue, f= os.path.join(datafolder,"processed","cosine.pt"))
        torch.save(unsorted, f= os.path.join(datafolder,"processed","unsorted_cosine.pt"))
    else:
        ordered = torch.load( os.path.join(datafolder,"processed","order.pt"))
        cossvalue = torch.load( os.path.join(datafolder,"processed","cosine.pt"))
        unsorted = torch.load(f= os.path.join(datafolder,"processed","unsorted_cosine.pt"))
    
    for ki in range(5,7,2):
        print(ki)
        #edge_index, w = thresholdNNG(ordered=ordered, coss_value=cossvalue)
        edge_index = navieKNNG(ordered=ordered,k=ki)
        
        """
        edge_index = cknn(
            orederd=ordered, sortedcoss_value=cossvalue, 
            unsortedcoss_value=unsorted, k=ki,cntr=1.0
        )
        """
        
        print(edge_index.size())
        #print(w.size())
        data = GData(x=features,edge_index=edge_index)
        savedir = os.path.join(datafolder,f"K_{ki}")
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
        s = MyDataset(root=savedir ,data_list=[data])
        s.process()
        print("Write as .th data OK")
        

if __name__ =="__main__":
    main()