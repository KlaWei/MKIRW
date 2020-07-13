import json 
import numpy as np
import argparse 

epsilon = 10**(-6)
 
# Musimy wczytać parametry

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


# k - number of rows, w - numer of columns
def initial_guess(X, k, w, alpha):
    # Random probabilities > 0
    ThetaB_rand = np.random.randint(100, size=4) + 1
    ThetaB_rand = ThetaB_rand/np.sum(ThetaB_rand)
    
    Theta_rand = np.random.randint(1000, size = (4, w)) + 1
    Theta_rand = Theta_rand/Theta_rand.sum(axis = 0)
    
    X = X - 1 
    ind = np.random.choice(k, size = int(alpha*k), replace = False)
    indB = np.setdiff1d(np.arange(k), ind)
    
    # number of occurrences of the letter i/number of all elements in the submatrix
    ThetaB = np.array([ np.sum(X[indB, :] == i)/np.size(X[indB, :]) for i in range(4)])
#    print(ThetaB, np.sum(ThetaB))
    
    # number of occurrences of the letter i in a column / number of all elements in that column
    Theta = np.reshape(np.array([ np.sum(X[ind, j] == i)/np.size(X[ind, j]) 
                                  for j in range(w)  for i in range(4) ]), (-1,4)).T
    
    ThetaB = (ThetaB + ThetaB_rand)/2
    Theta = (Theta + Theta_rand)/2
    
    return Theta, ThetaB
    

# Returns vector of probabilities P(x_i; Theta) or P(x_i; ThetaB) for every i
def prob(X, k, w, Theta, ThetaB, B = False) :
    if B == False:
        return Theta[X, np.tile(np.arange(w), (k, 1))].prod(axis = 1)
    else:
        return ThetaB[X].prod(axis = 1)
    
def EM(k, w, X, alpha):
    # Generate random theta and ThetaB
    ThetaB = np.random.randint(1000, size=4) + 1
    ThetaB = ThetaB/np.sum(ThetaB)
#    ThetaB = np.random.dirichlet(np.ones(4))
    
#    print("initial theta b = ")
#    print(ThetaB)
    
    Theta = np.random.randint(1000, size = (4, w)) + 1
    Theta = Theta/Theta.sum(axis = 0)
    
    #Theta, ThetaB = initial_guess(X, k, w, alpha)
    
    Orig_X = X
    X = X - 1
    Loglik_old = 0
    Loglik_new = 100
    while(abs(Loglik_new - Loglik_old) > epsilon):
#    for t in range(20):
        # E Step
        
        a = np.full(k, alpha) * Theta[X, np.tile(np.arange(w), (k, 1))].prod(axis = 1)
        b = np.full(k, 1 - alpha) * ThetaB[X].prod(axis = 1)
        
        Q_zero = b/(a + b)
        Q_one = a/(a + b)
        
        # M Step
        old_ThetaB = ThetaB
        old_Theta = Theta
        
        # New ThetaB
        ThetaB = np.array([ np.sum(Q_zero * (X == i).sum(axis = 1)) for i in range(4)]) / (w * np.sum(Q_zero))
        
        # New Theta
        for ti in range(4):
            for tj in range(w):
                a = np.sum(Q_one * (X[:, tj] == ti).astype(int))
                b = np.sum(Q_one)
                Theta[ti, tj] = a/b
        
        # Log lik, numpy style
        Loglik_old = Loglik_new              
        Loglik_new = np.sum(Q_zero * np.full(k, 1 - alpha) * np.ma.log(ThetaB[X]).filled(0).sum(axis = 1) + 
                            Q_one * np.full(k, alpha) * np.ma.log(Theta[X, np.tile(np.arange(w), (k, 1))]).filled(0).sum(axis = 1))
        
#        print(Loglik_new)
#        print(ThetaB, ThetaB.sum())
#        print(Theta, Theta.sum(axis = 0))
            
    return Theta, ThetaB   
    

np.set_printoptions(suppress=True)
    
input_file, output_file, estimate_alpha = ParseArguments()
 

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
 
 
 
alpha=data['alpha']
X = np.asarray(data['X'])
k,w = X.shape

# initial_guess(X, k, w)
Theta, ThetaB = EM(k, w, X, alpha)


estimated_params = {
    "alpha" : alpha,            # "przepisujemy" to alpha, one nie bylo estymowane 
    "Theta" : Theta.tolist(),   # westymowane
    "ThetaB" : ThetaB.tolist()  # westymowane
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    
    
    
