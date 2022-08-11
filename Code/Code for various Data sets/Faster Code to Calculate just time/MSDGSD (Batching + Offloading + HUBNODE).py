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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

HEADER = 64
PORT = [12084,19344,12092,12954,16642,21198,22313,27818,21991]
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
GET_NODE='get_nodes'
GET_NEIGHBOUR='get_neighbour'
GET_NEIGHBOUR_LIST='get_neighbour_list'
GET_HUB_NODE='get_hub_node'

NODE_PER_REQUEST=128  #Data of how many nodes will the worker get from server in a single request

NEIGHBOUR_LVL=3  #how many level deep should neighbours be considered  for distance calculation
IMPORTANCE=60  #how much % of top important nodes should be considered for distance calculation


class DGSD:
    def __init__(self):
        self.graph = None
        
    def limit_cpu(self):
        p = psutil.Process(os.getpid())
        p.cpu_affinity({2,3})
        return
    def request_neighbors(self, node):
        N_i = list(self.graph.neighbors(node))
        return N_i

    def get_nodes(self):
        return self.graph.number_of_nodes()
    
    def spawn_server(self,batches,workers,bins,server_index):
    
        ADDR=None
        server=None

        while True:
        
            try:
                ADDR = (SERVER, PORT[server_index])
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(ADDR)
                break;
            except:
                PORT[server_index]=PORT[server_index]+1
        
        thread = threading.Thread(target=self.start, args=(server,workers,server_index))
        thread.daemon=True
        thread.start()
        
        p = mp.Pool(processes = workers,initializer=self.limit_cpu, maxtasksperchild =1)
        batches=np.array_split(batches, workers)
        
        
        emb = p.starmap_async(Generate_Embeddings, zip(batches, repeat(bins),repeat(ADDR)))

        
        emb=emb.get()
        p.terminate()
        p.close()
        p.join()

        server.close()
        return
        
    
    def spawn_controllers(self, graph, bins = 50, workers=1,servers=1):
        
        self.graph = nx.convert_node_labels_to_integers(graph)
        nodes = list(self.graph.nodes())
        


        if workers>len(nodes):
            workers = int(len(nodes))
        if(workers<servers):
            servers=workers
        
            
        worker_per_server=int(workers/servers)
        
        batches = np.array_split(nodes, servers)  
        
     
        processes=[]
        for s in range(0, servers):
            p=mp.Process(target=self.spawn_server, args=(batches[s],worker_per_server,bins,s) )
            p.start()
            processes.append(p)
        

        for process in processes:
            process.join()

            
        return 0


        
    def start(self,server,workers,controller_no):
        
        server.listen(workers)

        for i in range(0,workers):
            conn, addr=(None,None)
            while(True):
                try:
                    conn, addr = server.accept()
                    break
                except:
                    print("Server Can't accept request from worker")
            thread = threading.Thread(target=self.handle_client, args=(conn, addr,controller_no))
            thread.daemon=True
            thread.start()
        return
    
    def handle_client(self,conn, addr,controller_no):
        p = psutil.Process(os.getpid())
        p.cpu_affinity({controller_no})
        connected = True
        while connected:
            msg_length=0
            msg=None
            while(msg_length==0):
                (msg,msg_length)=make_msg_to_receive(msg,conn)                
            if(msg==None):
                print(msg,msg_length)
                print("opps")
          
            res=None
            if(msg[0]==GET_NODE):
                res=self.get_nodes()
            elif (msg[0]==GET_NEIGHBOUR):
                res=self.request_neighbors(msg[1])
            elif (msg[0]==GET_NEIGHBOUR_LIST):
                res=self.request_neighbors_list(msg[1],msg[2],msg[3])
            elif (msg[0]==GET_HUB_NODE):
                res=self.request_hub_node(msg[1],NEIGHBOUR_LVL,IMPORTANCE)
            elif (msg[0] == DISCONNECT_MESSAGE):
                connected = False
            else:
                print("Faulty input from worker")
            if(res!=None):
                (send_length,message)=make_msg_to_send(res)
                
                conn.send(send_length)
                conn.send(message)

        conn.shutdown(socket.SHUT_RDWR)
        conn.close()
        return

    def request_neighbors_list(self, nodes,v,N_v):
        l1=[]
        common=0
        d_u=0
        for node in nodes:
            N_u=list(self.graph.neighbors(node))
            d_u = len(N_u)
            delta=0
            if (node==v):
                common=-1  
                d_u=0
            else:
                if (node in N_v):
                    delta=1
                common = len((set(N_u) & set(N_v)))
                common=common+delta
                
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

                   
def make_msg_to_send(msg):
    message = pickle.dumps(msg)
    msg_length = len(message)
    send_length = pickle.dumps(msg_length)
    send_length = send_length+ b' ' * (HEADER - len(send_length))
    return(send_length,message)

def make_msg_to_receive(msg,client):
    
    msg_length = client.recv(HEADER)
    try:
        msg_length=pickle.loads(msg_length)
    except EOFError:
        print("EOFError")
        return None,0
    if msg_length:
        msg_length = int(msg_length)
        msg = client.recv(msg_length)
        msg=pickle.loads(msg)
        return (msg,msg_length)
    return None,0
    
def receive_from_server(msg,client):

    (send_length,message)=make_msg_to_send(msg)
    client.send(send_length)
    client.send(message)
    msg_length=0
    msg=None
    while(msg_length==0):
        msg,msg_length=make_msg_to_receive(msg,client)



    return msg
    

def Generate_Embeddings( batch, nbins,ADDR):
    

    hist=np.zeros(nbins)
    dist = 0
    splits=[]
    while(True):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(ADDR)
            break
        except Exception as e:
            continue
            print("issue while connecting to Server ",e,ADDR)
         

            
    for v in batch:
        N_v = receive_from_server([GET_NEIGHBOUR,v],client)
        d_v = len(N_v) 
        
        hub_nodes=receive_from_server([GET_HUB_NODE,v],client)
        if(len(hub_nodes)<1):
            hist[0]=hist[0]+1
            continue
        
        spl=math.floor(len(hub_nodes)/NODE_PER_REQUEST)
        try:
            splits = np.array_split(hub_nodes, spl)
        except:
            splits = np.array_split(hub_nodes, 1)
        for split in splits:
            split=split.tolist()
            node_list=receive_from_server([GET_NEIGHBOUR_LIST,split,v,N_v],client)
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

    (send_length,message)=make_msg_to_send([DISCONNECT_MESSAGE,None])
    client.send(send_length)
    client.send(message)
    
    client.close()

    return hist

_name_dt=['MUTAG','PTC_MR','PROTEINS','NCI1','NCI109','AIDS','IMDB-BINARY','IMDB-MULTI']
#_name_dt=['REDDIT-BINARY','REDDIT-MULTI-5K','REDDIT-MULTI-12K']

def main():
    mp.set_start_method('fork')
    bins = 20
    workers =8
    dgsd=DGSD()
    servers=2
    for name_dt in _name_dt:
        Mutag = TUDataset(root='./N_'+name_dt, name=name_dt)
        descriptor=[]
        y=[]
        start = time.perf_counter()
        for i in range(0,len(Mutag)):
            if (i%100==0):
                print("Have processed graph number",i)
            descriptor.append(dgsd.spawn_controllers(nx.Graph(to_networkx(Mutag[i])),bins, workers,servers))
            y.append(Mutag[i].y.item())  
        finish=time.perf_counter()
        #print("Descriptor generated !",descriptor)
      
        
        with open('descriptor-3-'+name_dt, 'wb') as f:
            pickle.dump(descriptor, f)
        with open('labels-3-'+name_dt, 'wb') as f:
            pickle.dump(y, f) 
        with open('time-3-'+name_dt, 'wb') as f:
            pickle.dump(str(round(finish-start, 2)), f)
        print(f'Finished {name_dt} in {round(finish-start, 2)} second(s)',mp.current_process())

if __name__ == "__main__":
    main()
