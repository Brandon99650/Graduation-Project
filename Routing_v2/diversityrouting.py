import os
import pandas as pd
import numpy as np
from utils.attractiondata import AttractionDataset 
from utils.dicttool import *

class DiversityRouter():
    
    def __init__(self, dbpath:os.PathLike) -> None:
        
        self._attrdata = AttractionDataset(csvfilepath=dbpath)
        self._dist_info_root= os.path.join("OSM", "osmdist")
        self._label_index = list(str(i) for i in range(6))

    def _acceptable_travel_time_candidate(self, current, timelimit):
        placeinfo = os.path.join(
            self._dist_info_root,
            current['region2_en'].values[0],
            f"{current['placeid'].values[0]}.json"
        )
        #print(placeinfo)    
        current_dist_info = loadjson(placeinfo)
        candidate = {}
        for k, v in current_dist_info.items():
            if v['time'] > 0 and v['time'] <= timelimit:
                candidate[k] = v
        cid = list(map(lambda x:int(x), list(candidate.keys())))
        return candidate, cid

    def _build_labels_counting_df(self,pidlist:list, table:np.ndarray)->pd.DataFrame:
        
        df = pd.DataFrame(
            columns=self._label_index,
            data=table
        )
        df['placeid'] = pidlist
        return df

    def _max_n_improvement(self, candidate:pd.DataFrame, current_n:int)->pd.DataFrame:
        
        labeltable = candidate[self._label_index].values
        improve = np.count_nonzero(labeltable, axis=1)
        n_imporve_max = np.max(improve)
        if current_n == n_imporve_max:
            return candidate
        cid = np.where(improve == n_imporve_max)[0]
        return candidate.iloc[cid]

    def _min_sum_of_var_time(self, candidate:pd.DataFrame, travel_time:dict)->pd.DataFrame:
        
        labeltable = candidate[self._label_index].values
        varlist = np.var(labeltable, axis=1)
        travel_time_list = np.array(
            list(travel_time[str(i)]['time']/3600 for i in candidate.placeid.tolist())
        )
        balance_and_time = varlist+travel_time_list
        return candidate.iloc[np.argmin(balance_and_time)]
    
    def eva(self,labels, dtime, wt = 5, scaleout=10):
        var = np.mean((labels-np.mean(labels))**2)
        n = np.where(labels>0)[0].shape[0]
        score = n+np.exp(-(var+dtime*wt)/scaleout)
        return var, score


    def routing(self,sourceID, timelimit=30, pathlen = 4, debug=True)->dict:
        
        src = self._attrdata.get_attraction(
            placeid=sourceID, targetcol=[
                'name','description','addr','region2_en', 'placeid'
            ]+self._label_index
        )
        route = {
            'path':[int(src['placeid'].values[0])],
            'attrname':[src['name'].values[0]],
            'attr_desc':[src['description'].values[0]],
            'address':[src['addr'].values[0]],
            'travel_time':[],
            'time':0,
            'label':[src[self._label_index].values[0].tolist()],
            'table':src[self._label_index].values[0],
            'eva':0
        }
        for _ in range(pathlen-1):
            current = src
            
            if debug:
                print(current.values[0], end= "->")

            travel_time_limit, cid = self._acceptable_travel_time_candidate(current=current,timelimit=timelimit*60)
            if not len(cid):
                return route
            
            cid = list(x for x in cid if x not in route['path'])
            candidates = self._attrdata.df[self._attrdata.df['placeid'].isin(cid)]
            
            if debug:
                candidates.to_csv(os.path.join("buf", f"time_{_}.csv"),index=False,encoding='utf-8')
            
            
            appending_table = \
                np.tile(route['table'], (candidates.shape[0], 1)) \
                + candidates[self._label_index].values
            candidates = self._build_labels_counting_df(pidlist=candidates.placeid.tolist(), table=appending_table)
            
            #Coverage
            candidates = self._max_n_improvement(
                candidate=candidates, 
                current_n=np.count_nonzero(route['table'])
            )
            if debug:
                candidates.to_csv(os.path.join("buf", f"coverage_{_}.csv"),encoding='utf-8', index = False)
            
            #Balance & Travel time
            candidates = self._min_sum_of_var_time(candidate=candidates,travel_time=travel_time_limit)
            
            nextsteppid = candidates['placeid']
            nx = self._attrdata.get_attraction(
                placeid=nextsteppid, targetcol=[
                    'name','addr','description',
                    'region2_en','placeid'
                ]+self._label_index
            )

            if debug:
                print(nx.values[0])

            to_next_traveltime = travel_time_limit[f"{nextsteppid}"]['time']/60
            
            if debug:
                print(f"time = {to_next_traveltime}")
            
            route['path'].append(nextsteppid)
            route['attrname'].append(nx['name'].values[0])
            route['address'].append(nx['addr'].values[0])
            route['attr_desc'].append(nx['description'].values[0])
            route['table'] = candidates[self._label_index].values
            route['time'] += to_next_traveltime
            route['travel_time'].append(to_next_traveltime)
            route['label'].append(nx[self._label_index].values[0].tolist())
            src = nx
        
        route['time'] /= 60
        route['eva'] = self.eva(labels=route['table'], dtime=route['time'])
        return route
        
def getpath(sourceID=915, timelimit=30)->dict:
    router = DiversityRouter(dbpath=os.path.join("Attractiondata", "cluster_with_coo.csv"))
    p = router.routing(sourceID=sourceID,debug=False, timelimit=timelimit)
    return p

if __name__ == "__main__":
    p=getpath(sourceID=915)
    print(p)
