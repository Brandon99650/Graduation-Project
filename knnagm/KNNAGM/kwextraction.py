from opcode import opname
import os
import jieba.analyse as ja
import pandas as pd
from tqdm import tqdm

def extractkw(g_num, filesdir, topk=5, nonsenselst=[]):
    outd = os.path.join(filesdir[:filesdir.rfind("/")],"kw")
    if not os.path.isdir(outd):
        os.mkdir(outd)
    print(outd)
    for i in range(g_num):
        print(i)
        df_i = pd.read_csv(os.path.join(filesdir,f"{g_num}-c{i}.csv"))
        des = df_i['description'].tolist()
        with open(os.path.join(outd,f"{g_num}-c{i}.kw"),"w+") as f:
            for d in tqdm(des):
                keyword = ja.extract_tags(d, allowPOS=['n','ns','nr','nz'],topK=topk)
                for w in keyword:
                    if w not in nonsenselst:
                        f.write(f"{w} ")
                f.write("\n")


def extraction_all(csvfile, storeat, topk=5, nonsenselst=[]):
    df = pd.read_csv(csvfile)
    des = df['description'].tolist()
    with open(os.path.join(storeat,f"all.kw"),"w+") as f:
        for d in tqdm(des):
            keyword = ja.extract_tags(
                d, allowPOS=['n','ns','nr','nz'],topK=topk
            )
            for w in keyword:
                if w not in nonsenselst:
                    f.write(f"{w} ")
            f.write("\n")


if __name__ == "__main__":
    clusteringresult_dir = os.path.join(
        os.path.join("result", "attraction")
    )
    g_num = 7
    graph = f"K_5"

    nonsense=[]
    nonsensefile = os.path.join(
        'data','attraction','nonsense.txt'
    )
    with open(nonsensefile, "r") as n:
        for line in n.readlines():
            li = line.strip()
            if len(li):
                nonsense.append(li)
 
    extractkw(
        g_num=g_num , 
        filesdir=os.path.join(
            clusteringresult_dir,
            f"{graph}_{g_num}","clustering_result","eachC"
        ), 
        nonsenselst=nonsense, topk=7
    )
    """
    extraction_all(
        csvfile=os.path.join(
            clusteringresult_dir,
            f"{graph}_{g_num}","clustering_result", "cluster.csv"
        ),
        storeat=os.path.join(
            clusteringresult_dir,
            f"{graph}_{g_num}","clustering_result","kw"
        ), nonsenselst=nonsense, topk=30
    )
    """