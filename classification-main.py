import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.metrics import f1_score, accuracy_score # metric for evaluation of accuracy and f1-score
from sklearn.tree import DecisionTreeClassifier # decision tree 
from sklearn.model_selection import train_test_split # split into train and test set
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def main():
    train = pd.read_csv('drug_consumption.data', header=None)
    train.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    feature_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive']

    # y is the output, the class we want to predict, and x contains the other columns of the dataset
    y = train.Cannabis
    X = train[feature_cols] # only include the columns that we want

    # Now we split the dataset in train and test sets
    # initial values are train set is 75% and test set is 25%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)   

    # Use the function imported above and apply fit() on it
    DT = DecisionTreeClassifier()
    DT.fit(X_train,y_train)

    # We use the predict() on the model to predict the output
    pred=DT.predict(X_test)
    
    # for classification we use accuracy and F1 score
    print(accuracy_score(y_test,pred))
    print(f1_score(y_test,pred,average='micro'))

    dot_data = StringIO()
    export_graphviz(DT, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('decisiontree.png')
    Image(graph.create_png())


if __name__ == "__main__":
    main()