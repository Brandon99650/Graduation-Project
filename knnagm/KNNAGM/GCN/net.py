import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,GraphConv,GATConv,SGConv

def getgat(ind, outd, setting={}):
    gatlayer = GATConv(in_channels=ind, out_channels=outd,bias=setting['bias'])
    return gatlayer

def getgcn(ind, outd, setting={}):
    gcnlayer = GCNConv(
        in_channels=ind, out_channels=outd,
        normalize=setting['nor'],bias=setting['bias'],
        aggr=setting['aggr']
    )
    if 'init' in setting:
        if setting['init']=="xavier":
            torch.nn.init.xavier_uniform_(gcnlayer.lin.weight)
    
        elif setting['init']=="kaiming":
            torch.nn.init.kaiming_uniform_(gcnlayer.lin.weight)


    return gcnlayer

class NOCDnet(torch.nn.Module):

    def __init__(self, gcntrans):

        super(NOCDnet, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.gcnnorlayer= torch.nn.ModuleList()
        fininsh = self.__construct_gcn(gcntrans)
        """activation function"""
        self.act = torch.nn.LeakyReLU()
      
    def __construct_gcn(self, gcntrans):

        gcnlayernum = len(gcntrans)

        transsetting={'nor':True,'bias':True,'aggr':'mean', 'init':'xavier'}

        for i in range(gcnlayernum-2):
            self.convs.append(
                getgcn(gcntrans[i],gcntrans[i+1], transsetting)
                #getgat(gcntrans[i],gcntrans[i+1], transsetting)
            )
            self.gcnnorlayer.append(
                torch.nn.BatchNorm1d(gcntrans[i+1],affine=False)
            )
        
        outsetting={'nor':False,'bias':False,'aggr':'mean','init':'xavier'}

        self.convs.append(
            getgcn(gcntrans[gcnlayernum-2], gcntrans[gcnlayernum-1],outsetting)
            #getgat(gcntrans[gcnlayernum-2], gcntrans[gcnlayernum-1],outsetting)
        )

        return gcntrans[gcnlayernum-1]

    def forward(self, x, edge_index, edge_weight=None):
        xi = x
        for i in range(0,len(self.convs)-1):
            
            if edge_weight is None:
                xi = self.convs[i](xi, edge_index) 
            else:
                xi = self.convs[i](xi, edge_index,edge_weight) 
            
            xi = self.act(xi)
            xi = self.gcnnorlayer[i](xi)
            xi = F.dropout(xi, p=0.5, training=self.training)
        
        if edge_weight is None:
            xi = self.convs[len(self.convs)-1](xi, edge_index)
        else:
            xi = self.convs[len(self.convs)-1](xi, edge_index, edge_weight)
        
        return xi.sigmoid_()
        
    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

def l2_reg_loss(model, scale=1e-2):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale       