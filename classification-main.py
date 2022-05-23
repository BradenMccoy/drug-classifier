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
from sklearn.tree import _tree

def main():
    train = pd.read_csv('drug_consumption.data', header=None)
    train.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    # empty for now
    feature_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive'] 
    
    # y is the output, the class we want to predict, and x contains the other columns of the dataset
    legal_drugs = ['Alcohol', 'Caff', 'Cannabis', 'Choc', 'Legalh', 'Nicotine', 'Mushrooms']
    drugname = 'Alcohol' # iterate over legal drugs

    y = train[drugname]
    X = train[feature_cols] # only include the columns that we want

    # Now we split the dataset in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)   

    # Use the function imported above and apply fit() on it
    DT = DecisionTreeClassifier(criterion="entropy", max_depth=4) # depth is 
    DT.fit(X_train,y_train)

    # We use the predict() on the model to predict the output
    pred=DT.predict(X_test)
    
    # for classification we use accuracy and F1 score
    print(accuracy_score(y_test,pred))
    print(f1_score(y_test,pred,average='micro'))

    # png(DT, feature_cols, "depth5-alc.png")
    # tree_to_code(DT, feature_cols)

def png(DT, feature_cols, file_name):
    dot_data = StringIO()
    export_graphviz(DT, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['Never Used', 'Used a Decade Ago', 'Used in Last Decade', 
                    'Used in Last Year', 'Used in Last Month', 'Used in Last Week', 'Used in Last Day'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(file_name)
    Image(graph.create_png())

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)


if __name__ == "__main__":
    main()