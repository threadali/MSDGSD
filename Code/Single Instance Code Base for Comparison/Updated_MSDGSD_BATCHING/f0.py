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

start = time.perf_counter()

HEADER = 64
PORT = [12084,19344,12092,12954,16642,21198,22313,27818,21991]
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
GET_NODE='get_nodes'
GET_NEIGHBOUR='get_neighbour'
GET_NEIGHBOUR_LIST='get_neighbour_list'

SERVER_POWER=128
NODE_PER_REQUEST=128  #Data of how many nodes will the worker get from server in a single request


class DGSD:
    def __init__(self):
        self.graph = None
        self.bins=None
        
    def limit_cpu(self):
        p = psutil.Process(os.getpid())
        p.cpu_affinity({0,1,2,3})

        return
    def request_neighbors(self, node):
        N_i = list(self.graph.neighbors(node))
        return N_i

    def get_nodes(self):
        return self.graph.number_of_nodes()
    
    def spawn_server(self,workers,batches,server_index,q):
    
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
        
        batches=np.array_split(batches, workers)
        
        p = mp.Pool(processes = workers,initializer=self.limit_cpu, maxtasksperchild =1)
        emb = p.starmap_async(Generate_Embeddings, zip(batches, repeat(self.bins),repeat(ADDR)))
        
        emb=emb.get()
        embeddings=([item[0] for item in emb])
        execution_time_list=([item[1] for item in emb])
        communication_time_list=([item[2] for item in emb])
        embeddings = np.sum(np.array(embeddings),axis = 0)
        p.terminate()
        p.close()
        p.join()
        
        ne=len(execution_time_list)


        execution_time=sum(execution_time_list)/ne
        communicaiton_time=sum(communication_time_list)/ne
        
        server.close()
        q.put( [embeddings,execution_time,communicaiton_time] )
        return
        
    
    def spawn_controllers(self, graph, bins = 50, workers=1,servers=1):
        
        self.bins=bins
        self.graph = nx.convert_node_labels_to_integers(graph)
        nodes = list(self.graph.nodes())
        
        if workers>len(nodes):
            workers = int(len(nodes))
        if(workers<servers):
            servers=workers
            
        worker_per_server=int(workers/servers)
        
        batches = np.array_split(nodes, servers)  
        
        queue = mp.Queue()
        processes=[]
        for s in range(0, servers):
            p=mp.Process(target=self.spawn_server, args=(worker_per_server,batches[s],s,queue) )
            p.start()
            processes.append(p)
        
        embeddings=[]
        execution_time=0
        communication_time=0
        for process in processes:
            process.join()
            r=queue.get()
            embeddings.append(r[0])
            execution_time=execution_time+r[1]
            communication_time=communication_time+r[2]
        
        

        tt=execution_time/servers
        ct=communication_time/servers
        print("Avg Execution time",tt-ct)
        print("Avg Communicaiton time",ct)
        
        
        return np.sum(np.array(embeddings),axis = 0)


        
    def start(self,server,workers,controller_no):

        server.listen()

        sem_obj = threading.Semaphore(SERVER_POWER)
        threads=list()
        for i in range(0,workers):
            conn, addr=(None,None)
            while(True):
                try:
                    conn, addr = server.accept()
                    break
                except:
                    print("Server Can't accept request from worker")
            thread = threading.Thread(target=self.handle_client, args=(conn, addr,sem_obj,controller_no))
            thread.daemon=True
            threads.append(thread)
            thread.start()
 
        return
    
    def handle_client(self,conn, addr,sem_obj,controller_no):
        p = psutil.Process(os.getpid())
        p.cpu_affinity({controller_no})
        connected = True
        while connected:
            msg_length=0
            msg=None
            sem_obj.acquire()
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
            elif (msg[0] == DISCONNECT_MESSAGE):
                connected = False
            else:
                print("Faulty input from worker")
            if(res!=None):
                (send_length,message)=make_msg_to_send(res)
                
                conn.send(send_length)
                conn.send(message)
            sem_obj.release()

        conn.shutdown(socket.SHUT_RDWR)
        conn.close()
        return

    def request_neighbors_list(self, nodes,v,N_v):
        l1={}
        common=0
        d_u=0
        for node in nodes:
            l1[node]=list(self.graph.neighbors(node))
            
        return l1    
        
                   
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
    
def receive_from_server(msg,client,communication_time_list):
    t_s1=time.time()
    (send_length,message)=make_msg_to_send(msg)
    client.send(send_length)
    client.send(message)
    msg_length=0
    msg=None
    while(msg_length==0):
        msg,msg_length=make_msg_to_receive(msg,client)

    communication_time_list[0]=communication_time_list[0]+(time.time()-t_s1)

    return msg
    

def Generate_Embeddings( batch, nbins,ADDR):
    
    communication_time_list={0:0}
    start2=time.time()
    hist=np.zeros(nbins)
    dist = 0
    

    while(True):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(ADDR)
            break
        except Exception as e:
            continue
            print("issue while connecting to Server ",e,ADDR)
    
    total_nodes = receive_from_server([GET_NODE,None],client,communication_time_list)
    
    
    nodes_range=range(total_nodes)
    all_nodes_index=[*nodes_range]
    t_split = math.floor(total_nodes/NODE_PER_REQUEST)
    
    try:
        splits = np.array_split(all_nodes_index, t_split)
    except:
        splits = np.array_split(all_nodes_index, 1)
            
            
    for v in batch:
        N_v = receive_from_server([GET_NEIGHBOUR,v],client,communication_time_list)
        d_v = len(N_v)
        
        for split in splits:
            split=split.tolist()

            node_list=receive_from_server([GET_NEIGHBOUR_LIST,split,v,N_v],client,communication_time_list)
            for u,N_u in node_list.items():
                dist = 0
                delta = 0

                if (u!=v):
                    if u in N_v:
                        delta = 1

                    d_u = len(N_u)
                    common = len((set(N_u) & set(N_v)))
                    
                    if (((d_u + d_v) + common + delta))>0:
                        dist = (d_u + d_v) / ((d_u + d_v) + common + delta)
                    
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
    execution_time_list=(time.time()-start2)

    return (hist,execution_time_list,communication_time_list[0])



dgsd=DGSD()
bins = 20
name_dt=["ER5k.gxl"]
_servers=[2]
_workers=[50]
def main():
    mp.set_start_method('fork')
    
    for name in name_dt:
        print (f"Dataset={name}")
        g=Graph.Read_GraphML(name)
        A = g.get_adjacency()
        A = np.array(A.data)
        g=nx.from_numpy_array(A)
        for servers in _servers:
            print(f"S={servers}")
            for workers in _workers:                                         
                _start = time.time()
                descriptor = dgsd.spawn_controllers(g,bins,workers,servers)
                print(f'W={workers}Finished in {round(time.time()-_start, 2)} second(s)',mp.current_process())

if __name__ == "__main__":
    main()
