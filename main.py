import pandas as pd
import numpy as np
import sklearn as skl
from apyori import apriori

def main():
    data = pd.read_csv('drug_consumption.data', header=None)
    data.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    data.drop('Id', axis=1, inplace=True)
    data.drop('Nscore', axis=1, inplace=True)
    data.drop('Escore', axis=1, inplace=True)
    data.drop('Oscore', axis=1, inplace=True)
    data.drop('Ascore', axis=1, inplace=True)
    data.drop('Cscore', axis=1, inplace=True)
    data.drop('Impulsive', axis=1, inplace=True)
    data.drop('SS', axis=1, inplace=True)
    drugs = []
    for i in range(0, data.shape[0]):
        drugs.append([str(data.values[i, j]) for j in range(0, 24)])
    print(drugs[0])

    rules = apriori(drugs, min_support = 0.05, min_confidence = 0.2, min_lift = 3, min_length = 2)

    results = list(rules)

    results = pd.DataFrame(results)
    print(results)

if __name__ == "__main__":
    main()