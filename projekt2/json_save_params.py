import json 

import numpy as np
 
import argparse 

 
# Przyklad: tak zapisane zostaly ponizsze parametry:


# position weight matrix:
#tmp = np.array([[3/8,1/8,2/8,2/8],[1/10,2/10,3/10,4/10],[1/7,2/7,1/7,3/7]])
#tmp = tmp.T
tmp = np.random.randint(1000, size = (4,20)) + 1
tmp = tmp/tmp.sum(axis = 0)
Theta = tmp

# background distribution
# ThetaB=np.array([1/4,1/4,1/4,1/4])
ThetaB = np.random.randint(100, size=4) + 1
ThetaB = ThetaB/np.sum(ThetaB)
#ThetaB=np.array([1/4,1/4,1/4,1/4])

params = {
    "w" : Theta.shape[1],
    "alpha" : 0.3,
    "k" : 50,
    "Theta" : Theta.tolist(),
    "ThetaB" : ThetaB.tolist()
}

#Uwaga: powzyej nie mozna podac macierzy -- tutaj sa zamieniane na listy
#(potem po wczytaniu, wystarczy zamienic na macierz np.asarray(.))


with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)

    
