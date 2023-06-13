import sys
sys.path.append("..")
from utils.dicttool import *
import os
from routingpy import OSRM
from tqdm import tqdm

class Router():
    
    def __init__(self, base_url="https://routing.openstreetmap.de/routed-car"):
        
        """
        Please go to check the document of routingpy 
        https://routingpy.readthedocs.io/en/latest/#osrm
        OSRM part 
        and then setting the ```base_url```
        """
        
        self.client = OSRM(base_url=base_url)
        self.__pbar = None

    def _gen_batch(self, nodedict:dict, batchsize)->tuple:
        coolist = list(nodedict.values())
        idlist = list(nodedict.keys())
        idslices = []
        cooslices = []
        for i in range(0, len(coolist), batchsize):
            idslices.append(idlist[i:i+batchsize])
            cooslices.append(coolist[i:i+batchsize])
        return idslices, cooslices

    def _single_source(self, srcinfo:list, idslices:list, cooslices:list)->dict:
        nodelinks = {}
        batch_count =0
        for Dstidbatch, Coobatch in zip(idslices,cooslices):
            
            dstidbatch = [srcinfo[0]]+Dstidbatch
            coobatch = [srcinfo[1]]+Coobatch
            
            batch_count += 1
            try:
                r = self.client.matrix(locations=coobatch,sources=[0])
                self.__pbar.set_postfix_str(f"{srcinfo[0]}_{batch_count} ok")
            except:
                self.__pbar.set_postfix_str(f"{srcinfo[0]}_{batch_count} err")
                r = self.client.matrix(locations=coobatch,sources=[0])
        
            d = r.distances
            t = r.durations
            for j , dstid in enumerate(dstidbatch):
                nodelinks[dstid] = {
                    'time':t[0][j],
                    'dist':d[0][j]
                }
        return nodelinks

    def routing(self, nodedict:dict, saveingdir:os.PathLike,batchsize = 250)->None:
        """
        ### Parameters:

        - nodedict:
            The dict about query nodes.

            { nodeid:[node_lng, node_lat], .... }
        
        - saveingdir:
            The path of result saving.   

            Note that the result of Each node will be
            store to saveingdir/nodeid.json

        ### result :
        Seeing saveingdir/nodeid.json:

        Each nodeid.json stores the driving time & distances
        to the destinations.

        ```
        source0id.json:

        {
            dst0nodeid:{
                'time':secs,
                'dist':meters
            }, 
            dat1nodeid: ...
        }
        ```
        

        ### Note:
        I am using batch to get single source all pair distance, but it
        is not the fastest method.

        Each query of ```OSRM().matrix``` will return the all
        pair of querying list, but it has a max length limit.

        For the purpose of easily managing, I choose 
        single source all pair method
        although it will be slower.
        """
        idslices , cooslices= self._gen_batch(nodedict, batchsize)
        self.__pbar = tqdm(list(nodedict.keys()))
        for i, srcid in enumerate(self.__pbar):
            resultsave = os.path.join(saveingdir, f"{srcid}.json")
            if os.path.exists(resultsave):
                self.__pbar.set_postfix_str(f"gotten {srcid}")
                continue
        
            r = self._single_source([srcid, nodedict[srcid]],idslices, cooslices)
            writejson(r,resultsave)
        
