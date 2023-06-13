from dataclasses import dataclass
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']="6"
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import torch
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from GCN.net import NOCDnet
from utils.getdata import getdata
from utils.train import train
from utils.plot import plot_change
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def emb(modelpath,layers ,g, resultstore):
    print("embedding")
    r=None
    model = NOCDnet(gcntrans=layers)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    with torch.no_grad():
        r = (model(g.x, g.edge_index)>=0.5).type(torch.LongTensor)
    torch.save(r, os.path.join(resultstore,"cluster.pt"))


def plot_clustering_result(g_num,featurefilepath,df,outputdir):
    
    told = torch.load(featurefilepath)
    told = told.numpy()
    tsneTold = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(told)
    tsneTold = normalize(tsneTold, axis=0)
    each_g = []
    for l in tqdm(range(g_num)):
        gi = []
        for i in range(df.shape[0]):
            if df.iloc[i][str(l)] == 1:
                gi.append(tsneTold[i])
        gj = np.array(gi)
        each_g.append(gj)

    glist = list(i for i in range(g_num))
    colors = cm.rainbow(np.linspace(0, 1, len(glist)))
    plt.figure(figsize=(12,12))
    for y, c in tqdm(zip(glist, colors), total=len(glist)):
        this_g = each_g[y]
        plt.scatter(this_g[:, 0], this_g[:, 1], color=c)
    plt.savefig(os.path.join(outputdir,"vis.jpg"))
    plt.close()

def write_result(setting, filedir, resultpath):
    
    g_num = setting['num_classes']
    graph = graph = setting['graph']

    attraction  = pd.read_csv(filedir)
    name = attraction['Name'].tolist()
    toldscribe = attraction['Toldescribe'].tolist()
    clusteringresult = torch.load(os.path.join(resultpath, "cluster.pt"))
    clusteringresult = clusteringresult.numpy()
    z = 0
    for idx, i in enumerate(clusteringresult):
        s = i.sum()
        if s == 0:
            #print(idx)
            z += 1
    print(f"orphan:{z}")
    clusters = {}
    for i in range(g_num):
        clusters[i]= []
    for idx, i in enumerate(clusteringresult):
        belong = np.nonzero(i)[0].tolist()
        for groupid in belong:
            clusters[groupid].append(idx)
    outputdir = os.path.join(resultpath,"clustering_result")
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
        os.mkdir(os.path.join(outputdir,"eachC"))

    columns=['name','description']+list(str(i) for i in range(g_num))
    attrcl = []
    for idx, attrgroup in enumerate(clusteringresult):
        thisattr = [name[idx], toldscribe[idx]]
        thisattr = thisattr+attrgroup.tolist()
        attrcl.append(thisattr)
    df = pd.DataFrame(attrcl,columns=columns)
    df.to_csv(os.path.join(outputdir,"cluster.csv"), index=False, encoding='utf-8')

    for i in range(g_num):
        ci = df[df[str(i)]==1]
        ci.to_csv(os.path.join(outputdir,"eachC",f"{g_num}-c{i}.csv"),index=False, encoding='utf-8')

    plot_clustering_result(
        g_num=g_num,featurefilepath=os.path.join("data",setting['dataname'],"ToldescribeEBD.pt"),
        df = df, outputdir=outputdir
    )


def clusering(setting):
    
    dataroot = os.path.join("data",setting['dataname'])
    hyp = setting['hyp']

    gdata = getdata(datafolder=os.path.join(dataroot,f"{setting['graph']}"),nor=True)
    try:
        n = gdata.num_classes
    except:
        gdata.num_classes = setting['num_classes']
    hyp['gcntrans']= [gdata.num_features, hyp['hidden'], gdata.num_classes]
    outpath=os.path.join(
        "result",setting['dataname'],
        f"{setting['graph']}_{setting['num_classes']}"
    )
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    modelsavepath=os.path.join(
        outpath, 
        f"{setting['graph']}_{setting['num_classes']}.pt"
    )
    loss, md = train(
        gdata, hyp=hyp, modelsavepath=modelsavepath,
        Device=Device
    )            
    plot_change(
        loss,"BPloss",['min'],
        os.path.join(outpath,"loss.jpg")
    )
    plot_change(
        md, "modularity",['max'], 
        os.path.join(outpath,"md.jpg") 
    )
    emb(modelsavepath,hyp['gcntrans'], gdata, outpath)
    filedir = dataroot
    for i in setting['filedir']:
        filedir = os.path.join(filedir,i)
    write_result(
        setting=setting,
        filedir=filedir,
        resultpath=outpath
    )


if __name__ == "__main__":
    setting = None
    with open("setting.json","r") as js:
        setting = json.load(js)

    #print(setting)
    clusering(setting)
