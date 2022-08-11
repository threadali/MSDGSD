import multiprocessing as mp
import concurrent.futures
import threading
import socket 
from igraph import *
import numpy as np
from itertools import repeat
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import pickle
import copy
import codecs
import sys
import psutil
import os
import math
import time


NODE_PER_REQUEST=128  #Data of how many nodes will the worker get from server in a single request

NEIGHBOUR_LVL=3  #how many level deep should neighbours be considered  for distance calculation
IMPORTANCE=60  #how much % of top important nodes should be considered for distance calculation

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DGSD:
    def __init__(self):
        self.graph = None
        self.bins=None

    def request_neighbors(self, node):
        N_i = list(self.graph.neighbors(node))
        return N_i

    def get_nodes(self):
        return self.graph.number_of_nodes()
        
    
    def spawn_controllers(self, graph, bins = 50, workers=1):
        
        self.bins=bins
        self.graph = nx.convert_node_labels_to_integers(graph)
        nodes = list(self.graph.nodes())
        
        if workers>len(nodes):
            workers = int(len(nodes))
        p = mp.Pool(processes = workers, maxtasksperchild =1)
        
        batches = np.array_split(nodes, workers)  
        emb = p.starmap_async(self.Generate_Embeddings, zip(batches, repeat(self.bins)))
        
        emb=emb.get()
        p.terminate()
        p.close()
        p.join()

        
        return np.sum(np.array(emb),axis = 0)


    def request_neighbors_list(self, nodes,v,N_v):
        l1=[]
        common=0
        d_u=0
        for node in nodes:
            N_u=list(self.graph.neighbors(node))
            delta=0
            if (node==v):
                common=-1  
            else:
                if (node in N_v):
                    delta=1
                common = len(list(set(N_u) & set(N_v)))
                common=common+delta
                d_u = len(N_u)
            l1.append( [common,d_u] )
        return l1    

    def request_1_level_nodes(self, seta):
        t_set=copy.deepcopy(seta)
        for v in seta:
            t_set.update(set(self.graph.neighbors(v)))
            #extra check
            if(len(t_set)==self.graph.number_of_nodes()):
                break
        return t_set
        
    def request_hub_node(self, node,number=1,importance=100):
        
        seta=set(self.request_neighbors(node))
        for i in range(1,number):
            if(len(seta)==self.graph.number_of_nodes()):
                break
            seta=self.request_1_level_nodes(seta)

        if(importance<100):
            to_pick=math.ceil((len(seta)/100)*importance)
            res = {element:(len(list(self.graph.neighbors(element)))) for element in seta}
            mydict={k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
            
            important_nodes=[]
            for i, (k, v) in enumerate(mydict.items()):
                if (i>to_pick):
                    break
                important_nodes.append(k)
            return important_nodes
            
        return list(seta) 
                   

        

    def Generate_Embeddings( self,batch, nbins):
        

        hist=np.zeros(nbins)
        dist = 0
        

        for v in batch:
            N_v = self.request_neighbors(v)
            d_v = len(N_v)
            
            hub_nodes=self.request_hub_node(v,NEIGHBOUR_LVL,IMPORTANCE)
            if(len(hub_nodes)<1):
                hist[0]=hist[0]+1
                continue
            
            spl=math.floor(len(hub_nodes)/NODE_PER_REQUEST)
            try:
                splits = np.array_split(hub_nodes, spl)
            except:
                try:
                    splits = np.array_split(hub_nodes, len(hub_nodes))
                except:
                    splits = np.array_split(hub_nodes, 1)
            
            for split in splits:
                split=split.tolist()
                node_list=self.request_neighbors_list(split,v,N_v)
                for res in node_list:
                    common=res[0]
                    d_u=res[1]
                    if(common==-1):
                        dist = 0
                    elif ((d_u + d_v) + common )>0:
                        dist = (d_u + d_v) / ((d_u + d_v) + common)
                    ind=(math.ceil(dist*nbins))
                    if (ind<=0.0):
                        #print("............................")
                        hist[ind]=hist[ind]+1
                    else:
                        hist[ind-1]=hist[ind-1]+1   

        return hist


_bins=[20, 100, 200, 500]
_name_dt=['NCI109','AIDS','DD','TOX21','COLLAB','IMDB-BINARY','IMDB-MULTI','REDDIT-BINARY','REDDIT-MULTI-5K','REDDIT-MULTI-12K']

def main():
    mp.set_start_method('fork')

    workers = 8
    dgsd=DGSD()
    for name_dt in _name_dt:
        for bins in _bins:
            Mutag = TUDataset(root='../N_'+name_dt, name=name_dt)
            descriptor=[]
            y=[]
            start = time.perf_counter()
            for i in range(0,len(Mutag)):
                if(i%50==0):
                    print("Have processed graph number",i)
                descriptor.append(dgsd.spawn_controllers(nx.Graph(to_networkx(Mutag[i])),bins, workers))
                y.append(Mutag[i].y.item())  
            finish=time.perf_counter()
            #print("Descriptor generated !",descriptor)
          
            
            with open('descriptor-s-'+name_dt+"-bin-"+str(bins), 'wb') as f:
                pickle.dump(descriptor, f)
            with open('labels-s-'+name_dt+"-bin-"+str(bins), 'wb') as f:
                pickle.dump(y, f) 
            with open('time-s-'+name_dt+"-bin-"+str(bins), 'wb') as f:
                pickle.dump(str(round(finish-start, 2)), f)
            print(f'Finished {name_dt} in {round(finish-start, 2)} second(s)',mp.current_process())

if __name__ == "__main__":
    main()