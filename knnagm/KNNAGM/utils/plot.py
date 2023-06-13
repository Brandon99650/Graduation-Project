from cmath import inf
import matplotlib.pyplot as plt

def plot_change(l:list, name:str, desc=[], savepath = None):
    
    epochs_list = list(e for e in range(len(l)))
    maxlist = [max(l)]*len(l)
    
    plt.figure(figsize=(5, 4))
    plt.plot(epochs_list, l, label=name)
    if "max" in desc:
        plt.plot(epochs_list, maxlist, label=f"max: {maxlist[0]:.3f}")
    if "min" in desc:
        minlist = [min(l)]*len(l)
        plt.plot(epochs_list, minlist, label=f"min: {minlist[0]:.3f}")
    
    plt.xlabel("epoch", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(savepath)


def plot_interaction(l:list, name:list,desc=[], savepath=None):
    epoch_list  = list(q for q in range(len(l[0])) )
    plt.figure(figsize=(5, 4))

    if "max" in desc:
        maxv = -inf
        for li in l:
            c = max(li)
            if maxv < c:
                maxv = c
        maxlist = [maxv]*len(li)
        plt.plot(epoch_list, maxlist)

    if "min" in desc:
        minv = inf
        for li in l:
            c = min(li)
            if minv > c:
                minv = c
        minlist = [minv]*len(li)
        plt.plot(epoch_list, minlist)
            

    for idx, li in enumerate(l):
        lname= f"{name[idx]}"
        if "max" in desc:
            lname += f" max:{max(li):.3f}"
        if "min" in desc:
            lname += f" min:{min(li):.3f}"
        plt.plot(epoch_list, li, label= lname)

    plt.xlabel("epoch", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(savepath)
        
    