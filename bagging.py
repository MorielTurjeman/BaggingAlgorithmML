from numpy.lib.function_base import append
import pandas as pd
import numpy  as np
import random
import csv
from pandas.core import api
from sklearn import tree

# random.randrange(0, 101)
class Bagging:
    def __init__(self, csvFile) -> None:
        super().__init__()
        # random.seed()

        self.dataset = []
        self.test = []
        
        codes = { '1st' : 1, '2nd': 2, '3rd' : 3, 'crew' : 0 }

        with open(csvFile, "r") as input_file:
            reader = csv.DictReader(input_file, delimiter=",")
            for i in reader:
                self.dataset.append([codes[i['pclass']], 1 if i['age'] == 'adult' else 0, 1 if i['gender'] == 'male' else 0, 1 if i['survived'] == 'yes' else -1])
        
        with open('titanikTest.csv', "r") as input_file:
            reader = csv.DictReader(input_file, delimiter=",", fieldnames=['pclass','age','gender','survived'])
            for i in reader:
                self.test.append([codes[i['pclass']], 1 if i['age'] == 'adult' else 0, 1 if i['gender'] == 'male' else 0, 1 if i['survived'] == 'yes' else -1])
                
        
        
    def create_all_ensembles(self):

        self.trees = []

        for i in range(100):
            Xall = self.create_ensemble(self.dataset) #Xall is an ensemble from the dataSet
            # x= data, y= classification
            X = np.delete(Xall, 3, 1)
            Y = np.delete(Xall, [0,1,2], 1)

            #build the tree, train it and add it to the other trees.
            cls = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
            cls.fit(X,Y) 
            self.trees.append(cls)

        
    #the classification:
    def run(self):
        self.test_x = np.delete(self.test, 3, 1)
        self.test_y = np.delete(self.test, [0,1,2], 1)

        predictions = [] # this will hold all the predictions
        results = []
        for i, tree in enumerate(self.trees):
            predictions.append(tree.predict(self.test_x)) # append each prediction array to the predicitons list



        for i in range(0,len(predictions[0])):
            sum = 0
            for j in range(0,len(self.trees)):
                sum += predictions[j][i]

            if sum > 0:
                results.append(1)
            else:
                results.append(0)
        
        self.results = results
        correct_results = 0
        for test_res, real_res in zip(self.results, self.test_y):
            if test_res == real_res:
                correct_results+=1
        
        self.score = (correct_results / len(self.results))

    def create_ensemble(self,dataSet: list):
        ensemble=[]
        indexes =[]

        for idx,i in enumerate(self.dataset):
            num = random.randint(0,100)
            if num<63:
                ensemble.append(i)
            else:
                indexes.append(idx)
                
        
        while len(ensemble)<len(dataSet):
            index= random.randrange(0,len(ensemble))
            ensemble.append(ensemble[index])

        return ensemble

    def arrToLabel(self, arr):
        codes = { 1: '1st', 2: '2nd' , 3: '3rd' , 0: 'crew' }
        adult = [ 'child', 'adult' ]
        gender = ['female', 'male']

        return f"{codes[arr[0]]}, {adult[arr[1]]}, {gender[arr[2]]}"


    def __str__(self):
        rows = []
        for idx, row in enumerate(self.test_x):
            rows.append(f"{self.arrToLabel(row)} {'yes' if self.results[idx] == 1 else 'no'}")
        
        return '\n'.join(rows)


if __name__ == "__main__":
    bag=Bagging('titanikData.csv')
    bag.create_all_ensembles()
    bag.run()
    print(bag)
    print(bag.score)

