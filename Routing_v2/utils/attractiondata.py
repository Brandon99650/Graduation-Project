import os
import pandas as pd

class AttractionDataset():

    def __init__(self, csvfilepath:os.PathLike) -> None:
        self.df = pd.read_csv(csvfilepath)
        self.region = None
        self.regiondf = None
    
    def divide_according_regions(self, region:dict)->None:
        self.region = region
        diff_region = {}
        for region, name in self.region.items():
            diff_region[region] = self.df[self.df['region2'].isin(name['chi2'])]
        self.regiondf = diff_region


    def getpath(self, path, targetcol = ['name','description','0','1','2','3','4','5'])->pd.DataFrame:
        path_ = []
        for pi in path:
            path_.append(self.df[self.df['placeid']==pi][targetcol])
        
        return pd.concat(path_)

    def get_attraction(self, placeid, targetcol)->pd.DataFrame:
        return self.df[self.df['placeid']==placeid][targetcol]
    
    def get_picurl(self, placeid)->list:
        picurl = self.df.loc[
            self.df.placeid == placeid
        ][['pic0','pic1','pic2']].dropna(axis=1).values.tolist()
        return picurl

    def extract_places_coordinate(self, target_region:str|list="all")->dict:
        
        def extraction_places_coo(df:pd.DataFrame)->dict:
            placeidlist = df['placeid'].tolist()
            lnglist = df['lng'].tolist()
            latlist = df['lat'].tolist()
            lng_lat =[[lnglist[i], latlist[i]] for i in range(len(lnglist))]
            return dict(zip(placeidlist,lng_lat))

        if target_region == "all":
            return extraction_places_coo(self.df)
        if isinstance(target_region, list):
            #print(target_region)
            targetdf = pd.concat( list((self.regiondf[t] for t in target_region)))
            return extraction_places_coo(targetdf)
        
        return extraction_places_coo(self.regiondf[target_region])
    