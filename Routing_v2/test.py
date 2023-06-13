import os
from utils.attractiondata import AttractionDataset as ad
dataroot = "Attractiondata"
if __name__ == "__main__":
    csvfilepath=os.path.join(dataroot,"cluster_with_coo.csv")
    a = ad(csvfilepath=csvfilepath)
    path = [960,1017,1073,991]
    p = []
    for x in path:
        p.append(a.get_picurl(placeid=x))
    for x, pidx in zip(p, path):
        itname = a.get_attraction(placeid=pidx, targetcol=['name'])['name']
        itname = itname.values.tolist()[0]
        print(pidx, itname)
        for i in x:
            print(i)
        print("="*20)
