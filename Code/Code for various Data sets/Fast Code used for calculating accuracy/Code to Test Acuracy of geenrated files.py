from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import networkx as nx
import multiprocessing
import pickle
import pandas as pd
import numpy as np


def apply_RF_Grid(data_X,data_y):
    num_cores = 3
    max_arr = []
    estimator  = RandomForestClassifier(criterion='gini', max_depth=None, min_weight_fraction_leaf=0.0,
                                            max_leaf_nodes=None, bootstrap=True, 
                                            oob_score=False, n_jobs=num_cores,verbose=0, warm_start=False,
                                            class_weight=None)
    for seed in [567,890,5678,78, 6,1122,101,11111,42,345]:

        param_grid = {'n_estimators':[50,100,500], 'max_features':['sqrt'], 
                      'min_samples_split':[2,3,4,5,10], 'min_samples_leaf':[1,2,5]}
        kf = StratifiedKFold(n_splits=10, random_state = seed, shuffle = True)
        grid_rf    = GridSearchCV(estimator, param_grid, scoring='accuracy', n_jobs=num_cores, 
                     refit=True, cv=kf, verbose=1, pre_dispatch='n_jobs', 
                     error_score='raise')

        grid_rf.fit(data_X, data_y)
        max_arr.append(grid_rf.best_score_)
    print("max arr",max_arr)
    print('mean score:',np.mean(max_arr))

_dt=["DD","COLLAB",'IMDB-BINARY','IMDB-MULTI','REDDIT-BINARY','REDDIT-MULTI-5K','REDDIT-MULTI-12K']
_c=["s"]
_b=[20,100,200,500]
for dt in _dt:
    print(dt)
    for c in _c:
        for b in _b:
            print(b)
            y = []
            with (open("labels-"+c+"-"+dt+"-bin-"+str(b), "rb")) as openfile:
                while True:
                    try:
                        y=(pickle.load(openfile))
                    except EOFError:
                        break
            X = []
            with (open("descriptor-"+c+"-"+dt+"-bin-"+str(b), "rb")) as openfile:
                while True:
                    try:
                        X=(pickle.load(openfile))
                    except EOFError:
                        break



            try:

                X_df=pd.DataFrame(X)
                Y_df=pd.DataFrame(y)

                apply_RF_Grid(X,y)
            except:
                try:
                    X1=[]
                    for x in X:

                        X1.append(x[1])
                    X=X1
                    X_df=pd.DataFrame(X)
                    Y_df=pd.DataFrame(y)

                    apply_RF_Grid(X,y)
                except:
                    print("huge error")
