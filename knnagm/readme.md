# KNN-AGM Fuzzy clustering
For short text clustering by semantic similarity

reference : https://github.com/shchur/overlapping-community-detection.git

## Algorithm:
1. Using python Sentence-Transformer pretrain model (BERT) to embedding the short text.
   - The pretrain model is fine-tuned for __semantic similarity__ task. (Also being pretrained) 
 
2. Construct KNN graph 
    - K : a hyperparameter, in this project we set it to 6. 
3. Using 2 layers GCN and optimize by min negative bernoulli–poisson model.
   - Hidden layer : 256
   - learning rate : 0.001
   - optmizer: Adam  

### Note:   
Number of clusters are also a hyperparameter, can try several values and get the best cluster number.    

In this project, the way we determine the best number of clusters is try several value and checking wether each cluster has their unique semantic meaning.

If group them into $n$ is the best way to distingush semantic meaning, then the result we using to add labels will be $n$ clusters result.


## Analysis:
For each cluster, we using __TF-IDF__ to extract the keywords for each element, and choosing top 15 frequency words to determine the semantic labels for that cluster.

- TF-IDF: using python package __jieba__:  __jieba.analyse.extract()__ to get the keyword for a short text.

## Python Package:
```
numpy
pandas
torch
torch_geometric
sentence_transformers
jieba
sklearn
matplotlib
```

## Run code:
data path :  
put data in ```data/dataname/``` directory

### Construct KNN graph:
Execute ```condstructKNNG.py```

The $K$ of KNN and the __directory__ of torch_geometric.data.Data of the KNN graph that being generated are needed to be set in the ```condstructKNNG.py```

### clustering:
Add the directory ```dataname``` under the ```result``` directory to save the clustering result.

The data path and hyperparameter are needed to be set at ```setting.json``` to cluster the specific data 

- Example of setting.json
```
{   
    "dataname":"attraction",
    "filedir":["raw","legaldata.csv"],
    "graph":"K_5",
    "num_classes":7,
    "hyp":{
        "batchsize":20000, 
        "epochs":1000, 
        "lr":0.001, 
        "hidden":128
    }
}
```
After that, 
executing ```multilabeling.py``` to cluster.

The result directory will be at:
```result/dataname/K_#K_#cluster/``` (result_root)


The labels of each element are saved at ```result_root/cluster.pt```
And the model will be saved at
```result_root/K_#K_#cluster.pt```
It will plot the  
- $\text{-BPLoss}$ (__loss.jpg__)
- $\text{modularity}$ (__md.jpg__)

for each training epochs under ```result_root/```

It also will plot the scatter plot for clustering result under ```result_root/``` 
(__vis.jpg__)


And, will write the clustering result to ```rresult_root/clustering_result/```
- eachC: each cluster members

### Keyword extraction
execute ```kwextracion.py``` to extract the keyword from each cluster. The result will be write at
```result_root/clustering_result/kw/```

After that, running ```TextClusteringAnalyze.ipynb``` to extract top K frequency keyword that appear in each cluster (will write to ```result_root/clustering_result/dupkw/```)
