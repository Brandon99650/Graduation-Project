import os
from tqdm import tqdm
import torch
from GCN.net import NOCDnet
from utils.metrics import Neglikelihood, Overlapping_NMI, modularity
from utils.pygdata_utils import get_edge_sampler

def train(g, hyp, modelsavepath , Device):

    edgesample = get_edge_sampler(g=g, batchsize=hyp['batchsize'],epochs=hyp['epochs'])
    layers =hyp['gcntrans']
    gnnmodel= NOCDnet(gcntrans=layers).to(Device)
    
    edge_index = g.edge_index.to(Device)
    Loss = Neglikelihood(g.num_nodes, g.num_classes)
    #NMIcriteria = Overlapping_NMI()
    optr = torch.optim.Adam(gnnmodel.parameters(), lr = hyp['lr'])
    print(modelsavepath)
    pbar = tqdm(edgesample, total=hyp['epochs'])
    x = g.x.to(Device)
    #y = (g.y.type(torch.float)).to(Device)
    nmilst = []; nmi = 0.0
    Lhis = []; lv = torch.inf
    mdlist = []; md_max = -2.0
    for _, batch in enumerate(pbar):
        gnnmodel.train()
        emd = gnnmodel(x, edge_index)
        l =Loss.batch_loss(emd, batch[0].to(Device), batch[1].to(Device))
        optr.zero_grad()
        l.backward()
        optr.step()

        with torch.no_grad():
            gloss = Loss.global_loss(emd.cpu(), g.edge_index)
            labels = (emd>=0.5).type(torch.float).cpu()
            md = modularity(adjlist=g.edge_index, labels=labels)

            #thisnmi = NMIcriteria((emd>=0.5).type(torch.cuda.FloatTensor), y)
            Lhis.append(l.item())
            mdlist.append(md)
            if gloss < lv:
                lv = gloss
            if md > md_max:
                md_max = md
                torch.save(gnnmodel.state_dict(),modelsavepath)

        pbar.set_postfix(
            ordered_dict={
                "l":f"{gloss.item():.2f}", 
                "md":f"{md:.2f}", 
                "md_max":f"{md_max:.2f}"
            }
        )
    
    pbar.close()
    return Lhis,mdlist
