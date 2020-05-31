#!/usr/bin/python3

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import sys

# np.set_printoptions(threshold=sys.maxsize)

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--alg', default = "NMF", required = True, help = 'Choose algorithm from NMF, SVD1, SVD2, SGD')
    parser.add_argument('--train', default = "training_ratings.csv", required = True, help = 'Provide the train file.')
    parser.add_argument('--test', default = "test_ratings.csv", required = True, help = 'Provide the test file.')
    parser.add_argument('--result', default = "results_274598", required = True, help = 'Provide the results file.')
    args = parser.parse_args()
    
    return (args.alg, args.train, args.test, args.result)

def alg_NMF(Z, r = 21):
    model = NMF(n_components = r , init = 'random' , random_state = 0)
    W = model.fit_transform(Z)
    H = model.components_
    X_approximated = np.dot(W, H)
    
    return X_approximated

def alg_SVD1(Z, r = 11):
    svd = TruncatedSVD(n_components =  r, random_state = 42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    
    return np.dot(W, H)


def alg_SVD2(Z, r, N = 500):
    Z_n = alg_SVD1(Z, r)
    U, Sigma, VT = randomized_svd(Z, 
                        n_components=15,
                        n_iter=5,
                        random_state=None)
    
    n = 0
    while n < N:
        Z_np = np.where(Z > 0, Z, Z_n)
        Z_np = alg_SVD1(Z_np, r)
        Z_n = Z_np
        n = n+1
    
    return Z_n


def get_rmse2(Z_p, V, index_set):
    s = 0
    for (i, j) in index_set:
        s = s + (Z_p[i, j] - V[i, j])**2
    
    return math.sqrt(s/len(index_set))


def get_matrix_number(df, u, movieId_dict, num = 0):
    
    df = df.set_index(['userId', 'movieId'])
    
    A = np.empty((u, len(movieId_dict)))
    A[:] = num
    index_set = list()
    
    for (i, j) in df.index.values:
        k = movieId_dict[j]
        A[i - 1, k] = df.loc[i, j]
        index_set.append((i-1, k))
        
    return A, index_set
    

def get_matrix_mean(df, u, movieId_dict, R = True):
    
    df = df.set_index(['userId', 'movieId'])
    
    A = np.empty((u, len(movieId_dict)))
    A[:] = np.nan
    index_set = list()
    
    for (i, j) in df.index.values:
        k = movieId_dict[j]
        A[i - 1, k] = df.loc[i, j]
        index_set.append((i-1, k))
    
    if R is True:
        row_mean = np.nanmean(A, axis = 1)
        ind = np.where(np.isnan(A))
        A[ind] = np.take(row_mean, ind[0])
        A[np.where(np.isnan(A))] = 0
    else:
        col_mean = np.nanmean(A, axis = 0)
        ind = np.where(np.isnan(A))
        A[ind] = np.take(col_mean, ind[1])
        A[np.where(np.isnan(A))] = 0
        
    return A, index_set


# Filmom przypisujemy nowe id, od 0 do liczby filmów
def assign_new_ids(train_df, test_df):
    movieId_dict = dict()
    
    keys = np.unique(np.concatenate([train_df['movieId'].values, test_df['movieId'].values]))
    
    val = 0
    for k in keys:
        movieId_dict[k] = val
        val = val + 1
    
    return movieId_dict

if __name__ == "__main__":

    (alg, train_filename, test_filename, result_file) = ParseArguments()
    
    alg = alg.lower()
    
    train_df = pd.read_csv(train_filename, usecols = ['userId', 'movieId', 'rating'])
    test_df = pd.read_csv(test_filename, usecols = ['userId', 'movieId', 'rating'])
    
    
    movieId_dict = assign_new_ids(train_df, test_df)

    user_num = max(train_df['userId'].max(), test_df['userId'].max())
    
    u, m = max(train_df['userId'].max(), test_df['userId'].max()), max(train_df['movieId'].max(), test_df['movieId'].max())

    
    if alg == 'svd2':
        Z, index_set_train = get_matrix_number(train_df, user_num, movieId_dict)
    
    else:
        Z, index_set_train = get_matrix_mean(train_df, user_num, movieId_dict)
    
    V, index_set = get_matrix_number(test_df, user_num, movieId_dict)

    np.set_printoptions(suppress=True)
    if alg == 'nmf':
        Z_approx = alg_NMF(Z)
        rmse = get_rmse2(Z_approx, V, index_set)
    elif alg == 'svd1':
        Z_approx = alg_SVD1(Z)
        rmse = get_rmse2(Z_approx, V, index_set)

    elif alg == 'svd2':
        r = 7
        Z_approx = alg_SVD2(Z, r)
        rmse = get_rmse2(Z_approx, V, index_set)
    
#    f = open(result_file,"w+")
#    f.write(str(rmse))
#    f.close()
    f = open(result_file,"a")
    f.write('{}\n'.format(rmse))
    f.close()

