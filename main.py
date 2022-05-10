import pandas as pd
import numpy as np
import sklearn as skl

def main():
    drugsComDB = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv', sep='\t')
    print(drugsComDB)

if __name__ == "__main__":
    main()