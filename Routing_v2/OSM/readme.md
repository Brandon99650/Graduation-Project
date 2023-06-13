# OpenStreetMap OSRM API To get driving time

This is a project about using python ```routingpy.osrm``` API 
to get driving distance & driving time from __OpenStreetMap__ data by using osrm API.

In this project, I use osrm to get the driving time and distance about the pairs of Taiwanese toursit attractions. 

#### ```osrmapi.py``` 
is for calling the api of osrm.

**Note** The class ```Router()``` is customized according to the structure of our dataset.

#### ```attractiondata.py```
is for ineracting with our dataset. 

#### ```calculate_dist.py``` 
is the main file of this project.

#### ```script.ipynb```
The mainly useage of this notebook is for unit execution.

#### ```dicttool.py```
The kit file  is about using 
```json``` as dict I/O. It is just for convenience
to deal with .json file rather than writing json method 
each time whenever needing I/O from .json file.

### Environment
python package:
- routingpy
  
  **Note: need ```GDAL``` & ```geopandas```** package and these 2 may be a little bit hard to setup.
- pandas 
- numpy 
