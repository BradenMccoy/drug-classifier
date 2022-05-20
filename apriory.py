import pandas as pd
import numpy as np
import sklearn as skl
from apyori import apriori

def main():
    data = pd.read_csv('drug_consumption.data', header=None)
    data.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    data.drop('Id', axis=1, inplace=True)

    # Legal Drugs
    data.drop('Caff', axis=1, inplace=True)
    data.drop('Alcohol', axis=1, inplace=True)
    data.drop('Cannabis', axis=1, inplace=True)
    data.drop('Legalh', axis=1, inplace=True)
    data.drop('Nicotine', axis=1, inplace=True)
    
    # Illegal Drugs
    # data.drop('Coke', axis=1, inplace=True)
    # data.drop('Amphet', axis=1, inplace=True)
    # data.drop('Amyl', axis=1, inplace=True)
    # data.drop('Benzos', axis=1, inplace=True)
    # data.drop('Crack', axis=1, inplace=True)
    # data.drop('Ecstasy', axis=1, inplace=True)
    # data.drop('Heroin', axis=1, inplace=True)
    # data.drop('Ketamine', axis=1, inplace=True)
    # data.drop('LSD', axis=1, inplace=True)
    # data.drop('Meth', axis=1, inplace=True)
    # data.drop('Mushrooms', axis=1, inplace=True)
    # data.drop('Semer', axis=1, inplace=True)
    # data.drop('VSA', axis=1, inplace=True)

    # Personality Traits
    # data.drop('Nscore', axis=1, inplace=True)
    # data.drop('Escore', axis=1, inplace=True)
    # data.drop('Oscore', axis=1, inplace=True)
    # data.drop('Ascore', axis=1, inplace=True)
    # data.drop('Cscore', axis=1, inplace=True)
    # data.drop('Impulsive', axis=1, inplace=True)
    # data.drop('SS', axis=1, inplace=True)

    #print(data.head)
    drugs = []
    for i in range(0, data.shape[0]):
        drugs.append([str(data.values[i, j]) for j in range(0, 6)])
    #print(drugs[0])

    rules = apriori(drugs, min_support = 0.01, min_confidence = 0.7, min_lift = 1.5, min_length = 2, use_colnames = True)

    results = list(rules)

    #results = pd.DataFrame(results)
    for rule in results:
        print(rule)
        print()

if __name__ == "__main__":
    main()