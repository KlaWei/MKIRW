import json 
import numpy as np
import argparse
from random import randint
from random import choices


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 
 
w = params['w']
k = params['k']
alpha = params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB = np.asarray(params['ThetaB'])


# TO DO: wywymulowac x_1, ..., x_k, gdzie x_i=(x_{i1},..., x_{iw}) 
# zgodnie z opisem jak w skrypcie i zapisac w pliku .csv
# i-ta linijka = x_i

# Dla przykladu, zalozmy, ze x_1,... x_k sa zebrane w macierz X
# (dla przypomnienia: kazdy x_{ij} to A,C,G lub T, co utozsamiamy z 1,2,3,4


X=np.random.randint(4, size=(k,w))+1
#print(X)
#
#print(Theta[:,0])


#for i in range(20):
#    print(choices([1,2,3,4], Theta[:,0])[0])

for i in range(k):
    for j in range(w):
        ct = choices([0,1], [1 - alpha, alpha])[0]
        if ct==0:
            X[i, j] = choices([1,2,3,4], Theta[:,j])[0]
        else:
            X[i, j] = choices([1,2,3,4], ThetaB)[0]
         
np.savetxt("gen_data.csv", X, delimiter=",")

# Musimy zapisac powyzszy X oraz alpha (k i w mozna potem odczytac z X)

gen_data = {    
    "alpha" : alpha,
    "X" : X.tolist()
}



with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
 
