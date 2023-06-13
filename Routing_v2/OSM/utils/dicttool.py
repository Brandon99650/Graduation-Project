import json

__all__ = ["loadjson","writejson","print_dict"]


def loadjson(jsonfilepath)->dict:
    with open(jsonfilepath,'r', encoding="utf-8") as jf:
        return json.load(jf)

def writejson(d:dict, jsonfilepath)->None:
    with open(jsonfilepath,'w+', encoding="utf-8") as jf:
        json.dump(d, jf, indent=4, ensure_ascii=False)

def print_dict(d:dict,iter_range:int = None)->None:
    i = 0
    s = iter_range
    if s is None:
        s = len(d.items())
    for k, v in d.items():
        if i >= s:
            break
        print(f"{k} : {v}")
        i += 1