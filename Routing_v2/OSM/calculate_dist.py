import os
import sys
from utils.dicttool import *
from utils.attractiondata import AttractionDataset as attrd
from osrmapi import Router


def each_region():
    attractions = attrd(
        csvfilepath=os.path.join("..", "Attractiondata", "cluster_with_coo.csv"),
    )
    attractions.divide_according_regions(
        region=loadjson(jsonfilepath=os.path.join(".","tw_region.json"))
    )

    resultroot=os.path.join(".","osmdist")
    if os.path.exists(resultroot):
        os.mkdir(resultroot)
    
    router = Router()
    for k in attractions.twregion.keys():
        print(k)
        savedir = os.path.join(resultroot, f"{k}")
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        
        print(savedir)
        pid_coo = attractions.extract_places_coordinate(target_region=k)
        router.routing(pid_coo, savedir, batchsize=250)

def taiwan_main_island():
    attractions = attrd(
        csvfilepath=os.path.join("..", "Attractiondata", "cluster_with_coo.csv"),
    )
    attractions.divide_according_regions(region=loadjson(jsonfilepath=os.path.join(".","tw_region.json")))


    outlying = ["Penghu", "Kinmen","Lienchiang","Greenisland", "Lanyu"]
    main_island = list(t for t in attractions.twregion.keys() if t not in outlying )
    pid_coo = attractions.extract_places_coordinate(target_region=main_island)
    writejson(pid_coo, os.path.join(".", "mainisland.json"))
    c=input("calculate all pair ? Total may need 19 hours for all pair.\nType \"R\" to discard.")
    if c == "R":
        return 
    
    savedir = os.path.join("osmdist", "Allpair")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    
    router= Router()
    router.routing(pid_coo, saveingdir=savedir,batchsize=250)

if __name__ == "__main__":
    #taiwan_main_island()
    each_region()