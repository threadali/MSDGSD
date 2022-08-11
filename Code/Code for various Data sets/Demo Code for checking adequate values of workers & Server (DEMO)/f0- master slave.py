import multiprocessing as mp
import concurrent.futures
import threading
import socket 
from igraph import *
import numpy as np
from itertools import repeat
from torch_geometric.datasets import TUDataset,MoleculeNet
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
PORT = [21991]
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
GET_NODE='get_nodes'
GET_NEIGHBOUR='get_neighbour'
SERVER_POWER=32


class DGSD:
    def __init__(self):
        self.graph = None
        
    def limit_cpu(self):
        p = psutil.Process(os.getpid())
        p.cpu_affinity({2,3})

        return
    def request_neighbors(self, node):
        #print("i am suppose to be parent", mp.current_process())
        N_i = list(self.graph.neighbors(node))
        return N_i

    def get_nodes(self):
        return self.graph.number_of_nodes()
    
    def spawn_controllers(self, graph, bins = 50, workers=1):
        controller_no=0
        
        self.graph = nx.convert_node_labels_to_integers(graph)
        nodes = list(self.graph.nodes())
        
        

        p = mp.Pool(processes = workers,initializer=self.limit_cpu, maxtasksperchild =1)
         
        if workers>len(nodes):
            workers = int(len(nodes))   
        batches = np.array_split(nodes, workers)
        
        
        ADDR=None
        server=None

        while True:
        
            try:
                #print("trying to connect to server")
                ADDR = (SERVER, PORT[controller_no])
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(ADDR)
                break;
            except:
                PORT[controller_no]=PORT[controller_no]+1
                
        thread = threading.Thread(target=self.start, args=(server,workers,controller_no))
        thread.daemon=True
        thread.start()

        emb = p.starmap_async(Generate_Embeddings, zip(batches, repeat(bins),repeat(ADDR)))

        
        
        emb=emb.get()
        embeddings = np.sum(np.array(emb),axis = 0)
        p.terminate()
        p.close()
        p.join()
        server.close()

        
        return embeddings
    
    def start(self,server,workers,controller_no):
        #p.nice(psutil.HIGH_PRIORITY_CLASS)
        sem_obj = threading.Semaphore(SERVER_POWER)

        server.listen()
        #print(f"[LISTENING] Server is listening on {SERVER}")
        
        #print("workers nodes",int(workers))
        threads=list()
        for i in range(0,workers):
            #print(f'controller {controller_no} spawed thread {i}')
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
            #print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
        

        
        return
    
    def handle_client(self,conn, addr,sem_obj,controller_no):
        #print(f"[NEW CONNECTION] {addr} connected.")
        p = psutil.Process(os.getpid())
        p.cpu_affinity({controller_no})
        connected = True
        while connected:
            #print("server is waiting...")
            
            msg_length=0
            msg=None
            sem_obj.acquire()
            while(msg_length==0):
                (msg,msg_length)=make_msg_to_receive(msg,conn)
                #print("msg lenght",msg_length)                
            if(msg==None):
                print(msg,msg_length)
                print("opps")
            #print("server receive msg",msg)
            
            res=None
            if(msg[0]==GET_NODE):
                res=self.get_nodes()
            elif (msg[0]==GET_NEIGHBOUR):
                res=self.request_neighbors(msg[1])
            elif (msg[0] == DISCONNECT_MESSAGE):
                connected = False
            else:
                print("Faulty input from worker")
            if(res!=None):
                (send_length,message)=make_msg_to_send(res)
                
                conn.send(send_length)
                conn.send(message)
            sem_obj.release()
            
        #print("shutting down thread......")
        conn.shutdown(socket.SHUT_RDWR)
        conn.close()
        return

        
        
                   
def make_msg_to_send(msg):
    #print("in make msg",FORMAT,HEADER)
    message = pickle.dumps(msg)
    #print("message encoded",message)
    msg_length = len(message)
    send_length = pickle.dumps(msg_length)
    #print("message lenght",send_length)
    send_length = send_length+ b' ' * (HEADER - len(send_length))
    #print("msg lenght computed")
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
    #print("sending to server successfully ")
    
    msg_length=0
    msg=None
    while(msg_length==0):
        #print("msg lenght is 0")
        msg,msg_length=make_msg_to_receive(msg,client)



    return msg
    

def Generate_Embeddings( batch, nbins,ADDR):
    

    
    hist=np.zeros(nbins)
    while(True):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(ADDR)
            break
        except Exception as e:
            continue
            print("issue while connecting client",e,ADDR)
    
    #print("i am running child first call")
    #print("calling all nodes")
    total_nodes = receive_from_server([GET_NODE,None],client)

    #print("i completed child first call")


    for v in batch:
        #print("calling neighbours of v")
        N_v = receive_from_server([GET_NEIGHBOUR,v],client)
        d_v = len(N_v)
        for u in range(total_nodes):
            dist = 0
            delta = 0
            if u != v:

                #print("calling neighbours of u")
                N_u = receive_from_server([GET_NEIGHBOUR,u],client)
                #print("neighbours",N_u)
                
                if u in N_v:
                    delta = 1

                d_u = len(N_u)
                common = len(list(set(N_u) & set(N_v)))
                
                if (((d_u + d_v) + common + delta))>0:
                    dist = (d_u + d_v) / ((d_u + d_v) + common + delta)

            #print("distance",dist)
            ind=(math.ceil(dist*nbins))
            #print(ind,dist)
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


dgsd=DGSD()
bins = 20
g2 = nx.erdos_renyi_graph(700, 0.5) 

_workers=[4,8,16,32,64,80,100]
def main():
    mp.set_start_method('fork')
    print("for 40 avg nodes")

    for workers in _workers:                                         
        _start = time.time()
        descriptor = dgsd.spawn_controllers(g2,bins,workers)
        print(f'W={workers}Finished in {round(time.time()-_start, 2)} second(s)',mp.current_process())


if __name__ == "__main__":
    main()