import pandas as pd
import numpy as np
import sklearn as skl
from apyori import apriori

def main():
    data = pd.read_csv('drug_consumption.data')
    #data.drop(data.columns[[0]], axis=1, inplace=True)
    #data.drop(data.columns[[6]], axis=1, inplace=True)
    #data.drop(data.columns[[7]], axis=1, inplace=True)
    #data.drop(data.columns[[8]], axis=1, inplace=True)
    #data.drop(data.columns[[9]], axis=1, inplace=True)
    #data.drop(data.columns[[10]], axis=1, inplace=True)
    #data.drop(data.columns[[11]], axis=1, inplace=True)
    #data.drop(data.columns[[12]], axis=1, inplace=True)
    #drugs = []
    #for i in range(0, data.shape[0]):
        #drugs.append([str(data.values[i, j]) for j in range(0, 24)])
    print(drugs[0])

    rules = apriori(drugs, min_support = 0.05, min_confidence = 0.2, min_lift = 3, min_length = 2)

    results = list(rules)

    results = pd.DataFrame(results)
    print(results)

if __name__ == "__main__":
    main()