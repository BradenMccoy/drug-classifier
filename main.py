import pandas as pd
import numpy as np
import sklearn as skl
from apyori import apriori

def main():
    data = pd.read_csv('drug_consumption.data')
    drugs = []
    for i in range(0, data.shape[0]):
        drugs.append([str(data.values[i, j]) for j in range(0, 32)])
    print(drugs[0])

    rules = apriori(drugs, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

    results = list(rules)

    results = pd.DataFrame(results)
    print(results.head())

if __name__ == "__main__":
    main()