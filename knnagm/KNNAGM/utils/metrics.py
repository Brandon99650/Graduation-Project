import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import normalize

class Neglikelihood(torch.nn.Module):

    def __init__(self,  num_nodes, num_edges):
        super(Neglikelihood, self).__init__()
        #print(f"nodes: {num_nodes} edges : {num_edges}")
        self.num_edges = num_edges
        self.all_possible_edge_num =  num_nodes**2 -  num_nodes
        self.num_neg_edges = self.all_possible_edge_num -  num_edges

        """back ground community"""
        self.eps = -np.log( 1- self.num_edges/self.all_possible_edge_num) 


    def batch_loss(self, emd, edge_index, neg_edge_index):
    
        te1, te2 = edge_index
        ne1, ne2 = neg_edge_index
      
        te_prob = torch.sum(emd[te1]*emd[te2], dim =1)
        te_prob = -torch.mean(torch.log(-torch.expm1(-self.eps-te_prob)))

        ne_prob = torch.mean(torch.sum(emd[ne1]*emd[ne2], dim = 1))
    
        return (te_prob+ne_prob)/2.0
    
    def global_loss(self, emd, edge_index):
        te1, te2 = edge_index      
        tdot = torch.sum(emd[te1]*emd[te2], dim =1)+self.eps
        te_prob = -torch.mean(torch.log(- torch.expm1(-tdot)))
        total_dot = torch.sum(emd @ torch.sum(emd, dim=0, keepdim=True).t())
        ne_prob = (total_dot - torch.sum(emd * emd) - torch.sum(tdot))/self.num_neg_edges

        return (te_prob+ne_prob)/2.0


class PRF1_Metrics(torch.nn.Module):

    def __init__(self) -> None:
        super(PRF1_Metrics, self).__init__()
    

    def forward(self, pre, gt):
       
        c = self.confusionmatrix(pre, gt)
        #self.print_confusion(c)

        acc = (c['TP']+c['TN'])/(c['TP']+c['TN']+c['FN']+c['FP']+1e-8)
        
        precision = c['TP']/(c['FP']+c['TP']+1e-8)
        
        recall = c['TP']/(c['FN']+c['TP']+1e-8)
      
        F1 = 2.0*precision*recall/(precision+recall+1e-8)
        
        return {
            'accuracy':acc,
            'precision':precision, 
            'recall':recall, 
            'F1':F1
        }

    def confusionmatrix(self, pre, gt):
        TN = ((1-pre).t()) @ (1-gt) # 0 0
        FN = ((1-pre).t()) @ (gt) # 0 1
        FP = (pre.t()) @ (1-gt) # 1 0
        TP = (pre.t()) @ (gt) # 1 1
        return {'TN':TN, 'FN':FN, 'FP':FP, 'TP':TP}

    def print_confusion(self, c):
        print("TP:")
        print(c['TP'])

        print("TN:")
        print(c['TN'])
        
        print("FP:")
        print(c['FP'])
        print("FN:")
        print(c['FN'])


def avg_F1_score(Pre, Gt):

    metric = PRF1_Metrics()
    Pre_Gt = metric(Pre, Gt)
    Gt_Pre = metric(Gt, Pre)

    Pre_Gt_bestF1, idx = torch.max(Pre_Gt['F1'], dim = 1)
    Gt_Pre_bestF1, idx2 = torch.max(Gt_Pre['F1'], dim = 1)

    avg_F1 = torch.sum(Pre_Gt_bestF1)/Pre.size()[1]+torch.sum(Gt_Pre_bestF1)/Gt.size()[1]
    avg_F1 = 0.5*avg_F1

    return avg_F1.item()


class Overlapping_NMI(torch.nn.Module):
    def __init__(self) -> None:
        super(Overlapping_NMI, self).__init__()
        self.confusion = PRF1_Metrics()
    def forward(self, Y_hat, Y):

        HY = self.__UncondEntropy(Y)
        HYhat = self.__UncondEntropy(Y_hat)
        
        HYYhat = self.__CondEntropy(Y, Y_hat)
        HYhatY = self.__CondEntropy(Y_hat, Y)
        #print(f"HYhat:{HYhat} HY:{HY} \n HYhY:{HYhatY} HYYh:{HYYhat}")
        nmi = 0.5*(HY+HYhat-HYYhat- HYhatY)/torch.max(HY, HYhat)
        return nmi.item()

    def __CondEntropy(self, X, Y):

        n = X.size()[0]
        confusion = self.confusion.confusionmatrix(X,Y)
        #a = ((1-X).t()) @ (1-Y) 
        a=confusion['TN']
        ha = self.__h(a,n)

        #b =  ((1-X).t()) @ (Y) 
        b = confusion['FN']
        hb = self.__h(b,n)

        #c =  (X.t()) @ (1-Y) 
        c = confusion['FP']
        hc = self.__h(c,n)

        #d =  (X.t()) @ (Y)
        d = confusion['TP']
        hd = self.__h(d,n)

        hab = self.__h(a+b,n)
        hcd = self.__h(c+d,n)
        hbd = self.__h(b+d,n)
        hac = self.__h(a+c,n)

        cond = ha+hb+hc+hd - hbd-hac
        mask = ( (ha+hd) >= (hb+hc) ).type(torch.float)

        cond = cond*mask
        restric = (hab+hcd)*(1.0-mask)
        cond += restric
        mindst,i = torch.min(cond, dim=1)
        return torch.sum(mindst)
        
    def __UncondEntropy(self, X):
        onenum = torch.sum(X,dim=0)
        zerosnum = X.size()[0] - onenum
        Info = torch.sum(
            self.__h(onenum, X.size()[0])
            +self.__h(zerosnum, X.size()[0])
        )
        return Info
        
    def __h(self, w, n):
        entropy = -(w)*torch.log2((w+1e-8)/n)
        return entropy

def modularity(adjlist:torch.Tensor, labels:torch.Tensor, d:torch.device)->float:
    
    adj = to_dense_adj(adjlist)[0]
    edge_num_double = adjlist.size()[1] ##2*edge_num
    """
    build ki*kj/2*edge_num as deg_matrix
    """
    deg_list = torch.sum(adj,dim=1)
    deg_list = deg_list.unsqueeze(0).to(device=d)
    deg_matrix = deg_list.t()@deg_list #ki*kj
    deg_matrix = deg_matrix/edge_num_double

    """
    build the matrix wether node i and node j are 
    share the same comunity
    (0 or 1)
    """
    l = torch.tensor(normalize(labels,axis=1,norm='l1')).to(device=d)
    belong_matrix = (l@l.t())

    """
    sum( 
        (adj-deg_matrix/(2*edge_index))*samecomm(i,j) 
    )/(2*edge_index)
    """
    md = torch.sum(
        (adj - deg_matrix)*belong_matrix
    )/edge_num_double

    return md.item()
