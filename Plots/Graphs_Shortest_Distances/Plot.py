import networkx as nx
from torch_geometric.datasets import TUDataset,MoleculeNet
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

_name_dt=['PROTEINS','NCI109','NCI1','DD','REDDIT-BINARY','REDDIT-MULTI-5K']

hist={}
for name_dt in _name_dt:

    Mutag = TUDataset(root='./N_'+name_dt, name=name_dt)
    for i in range(0,len(Mutag)):
        graph=to_networkx(Mutag[i])
        n = nx.all_pairs_shortest_path_length(graph)
        n=dict(n)
        #print(n)
        for key,val in n.items():
            #print(val)
            for k2,v2 in val.items():
                #print(k2)
                t=hist.get((v2),0)
                hist[(v2)]=t+1
    df=pd.DataFrame({'x':list(hist.keys()),'y':list(hist.values())})

    fig, ax = plt.subplots(figsize =(5, 5))
    Colors={10:"#1F4690",30:"#D61C4E",50:"#E6B325",80:"#66BFBF",150:"#2C3639"}
    try:
        plt.plot(df[:30]['x'],df[:30]['y'],marker='o',color=Colors[10])
    except:
        plt.plot(df[:20]['x'],df[:20]['y'],marker='o',color=Colors[10])
    plt.axvline(x=5, c=Colors[50])
    plt.axvline(x=4, c=Colors[30])
    plt.axvline(x=3, c=Colors[80])
    #plt.axvline(x=2, c='#61481C')
    ax.set_xticks([1,3,4,5,10,20,30])
    # Show plot
    print(name_dt)
    print("3",df[0:3]['y'].sum()/df['y'].sum())
    print("4",df[0:4]['y'].sum()/df['y'].sum())
    print("5",df[0:5]['y'].sum()/df['y'].sum())
    print("6",df[0:6]['y'].sum()/df['y'].sum())
    print("max",df[0:df['y'].idxmax()]['y'].sum()/df['y'].sum())
     

    plt.title(name_dt)
    plt.xlabel('Lenght of Shortest Distance (Hops)')
    plt.ylabel('No. of Pairs')
    plt.savefig(name_dt+'.jpg',bbox_inches='tight', dpi=150)
    
