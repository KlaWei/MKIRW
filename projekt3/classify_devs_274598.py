import numpy as np
import csv
import matplotlib.pyplot as plt
from hmmlearn import hmm
import glob
import pandas as pd
import argparse
import os


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--train', default="house3_5devices_train.csv", required=False, help='Train file')	
    parser.add_argument('--test', default="test_folder", required=False, help='Test folder')	
    parser.add_argument('--output', default="results.txt", required=False, help='Results')
    
    args = parser.parse_args()	
    return  args.train, args.test, args.output


class HMM:
    def __init__(self,num_components = 4, num_iter = 1000):
        self.n_components = num_components
        self.n_iter = num_iter
        
        self.hmm_model = hmm.GaussianHMM(n_components = num_components, covariance_type="full", n_iter = num_iter)
        
    def train(self, train_set):
        self.hmm_model.fit(train_set)
    
    def get_score(self, data):
        return self.hmm_model.score(data)
        
    def get_means(self):
        return self.hmm_model.means_.round().astype(int).flatten().tolist()
        
    def get_predictions(self, dataset):
        means = self.get_means()
        hidden_states = self.hmm_model.predict(dataset)
        return np.array([means[state] for state in hidden_states])
    
    def n_features(self):
        return self.hmm_model.n_features
    
    def n_components(self):
        return self.hmm_model.n_components
    
    def get_covars(self):
        return self.hmm_model.covars_
        

def best_fit(train_data):
    model = HMM(num_components = 2)
    model.train(train_data)
    max_score = model.get_score(train_data)
    n_comp = 2
    
    for i in range(2, 9):
        model = HMM(num_components = i)
        model.train(train_data)
        s = model.get_score(train_data)
        if s > max_score:
            max_score = s
            n_comp = i
    
    return n_comp


def best_fit_bic(train_set):
    # print("BIC")
    lowest_bic = np.infty
    n_comp_best = 0
    
    for n_components in range(2, 13):
        hmm = HMM(num_components = n_components)
        hmm.train(train_set)
        
        n_features = hmm.n_features()
        parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)
        
        bic_curr = np.log(train_set.shape[0])*parameters - 2*hmm.get_score(train_set)

        # print(n_components, bic_curr)
        
        if bic_curr < lowest_bic:
            lowest_bic = bic_curr
            n_comp_best = n_components

    return n_comp_best




if __name__ == "__main__":
    train_file, test_folder, result_file   =  ParseArguments()
    #print(os.path.abspath(os.getcwd()))
    
    # get names of all the csv files in the test folder
    csvfiles = glob.glob(os.path.join(test_folder, '*.csv'))
    # print(os.path.join(os.path.dirname(os.path.abspath(train_file)), train_file)) 
    csvfiles.sort()
    # read the csv files form the test folder
    dataframes = [] 
    for csvfile in csvfiles:
        df = pd.read_csv(csvfile, usecols = ['dev'])
        dataframes.append((os.path.basename(csvfile), df))
        
    train_data = pd.read_csv(train_file, usecols = ['lighting2','lighting5','lighting4','refrigerator','microwave'])
    
    lighting2_train = np.column_stack([train_data['lighting2']])
    lighting5_train = np.column_stack([train_data['lighting5']])
    lighting4_train = np.column_stack([train_data['lighting4']])
    refrigerator_train = np.column_stack([train_data['refrigerator']])
    microwave_train = np.column_stack([train_data['microwave']])
    
    
    lighting2_hmm = HMM(num_components = best_fit_bic(lighting2_train))
    lighting2_hmm.train(lighting2_train)
    
    lighting5_hmm = HMM(num_components = best_fit_bic(lighting5_train))
    lighting5_hmm.train(lighting5_train)
    
    lighting4_hmm = HMM(num_components = best_fit_bic(lighting4_train))
    lighting4_hmm.train(lighting4_train)
    
    refrigerator_hmm = HMM(num_components = best_fit_bic(refrigerator_train))
    refrigerator_hmm.train(refrigerator_train)
    
    microwave_hmm = HMM(num_components = best_fit_bic(microwave_train))
    microwave_hmm.train(microwave_train)
    
    
    res_hmm = []

    for csv_name, dataframe in dataframes:
        data = np.column_stack([dataframe])
        scores = []
        scores.append(("lighting2", lighting2_hmm.get_score(data)))
        scores.append(("lighting4", lighting4_hmm.get_score(data)))
        scores.append(("lighting5", lighting5_hmm.get_score(data)))
        scores.append(("microwave", microwave_hmm.get_score(data)))
        scores.append(("refrigerator", refrigerator_hmm.get_score(data)))
        
        scores.sort(key=lambda x: x[1], reverse = True)
        res_hmm.append((csv_name, scores[0][0]))
        
    with open(result_file, "w+") as rf:
        rf.write("file, dev_classified\n")
        for name, dev in res_hmm:
            rf.write("%s, %s\n" % (name, dev))


