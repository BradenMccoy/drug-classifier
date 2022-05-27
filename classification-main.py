import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.metrics import f1_score, accuracy_score # metric for evaluation of accuracy and f1-score
from sklearn.tree import DecisionTreeClassifier # decision tree 
from sklearn.model_selection import train_test_split # split into train and test set
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
#simport pydotplus
from sklearn.tree import _tree
from itertools import combinations

def main():
    train = pd.read_csv('drug_consumption.data', header=None)
    train.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    # columns we are interested in
    feature_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive']

    combo_list = [] # contains all possible combinations
    combo_list.append(feature_cols[:1])
    combo_list.append(feature_cols[:2])
    combo_list.append(feature_cols[:3])
    combo_list.append(feature_cols[:4])
    combo_list.append(feature_cols[:5])
    combo_list.append(feature_cols[:6])
    combo_list.append(feature_cols[:7])
    combo_list.append(feature_cols[:8])
    combo_list.append(feature_cols[:9])
    combo_list.append(feature_cols[:10]) 
    combo_list.append(feature_cols)
    
    # print(combo_list)
    
    # y is the output, the class we want to predict, and x contains the other columns of the dataset
    legal_drugs = ['Alcohol', 'Caff', 'Cannabis', 'Choc', 'Legalh', 'Nicotine', 'Mushrooms']
    illegal_drugs = ['SS', 'Amphet', 'Amyl', 'Benzos', 'Coke', 'Crack', 'Ecstasy',
                     'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms', 'Semer', 'VSA']
    
    # change to a binary classification, used or never used
    for drug in legal_drugs:
        train[drug] = train[drug].apply(mod_classification)
    
    print("------ Beginning Training sequence ------")
    outfile = open("output.txt", "w")
    for drug in legal_drugs:
        y = train[drug]
        outfile.write("Predicting for " + drug + ":\n")
        
        # track the maximum accuracy rule for each drug
        max_accuracy = 0
        for combo in combo_list:
            X = train[list(combo)]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
            DT = DecisionTreeClassifier(criterion="entropy", max_depth=4) # depth is 
            DT.fit(X_train,y_train)
            pred=DT.predict(X_test)
            outfile.write("\tUsing feature columns: ")
            
            for item in list(combo):
                outfile.write(item + " ")
            outfile.write("Accuracy: " + str(accuracy_score(y_test, pred)) + "\n")

            # check if this is a maximum accuracy
            if(accuracy_score(y_test, pred) > max_accuracy):
                max_accuracy = accuracy_score(y_test, pred)

        outfile.write("Maximum accuracy: " + str(max_accuracy) + "\n\n")
    
    outfile.close()

    # y = train[drugname]
    # X = train[feature_cols] # only include the columns that we want

    # Now we split the dataset in train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)   

    # Use the function imported above and apply fit() on it
    # DT = DecisionTreeClassifier(criterion="entropy", max_depth=4) # depth is 
    # DT.fit(X_train,y_train)

    # We use the predict() on the model to predict the output
    # pred=DT.predict(X_test)
    
    # for classification we use accuracy and F1 score
    # print(accuracy_score(y_test,pred))
    # print(f1_score(y_test,pred,average='micro'))

    # png(DT, feature_cols, "depth5-alc.png")
    # tree_to_code(DT, feature_cols)

def mod_classification(x):
    if (x == 'CL0'):
        return 'Never Used'
    else:
        return 'Used' 

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
    #print "def tree({}):".format(", ".join(feature_names))

    # def recurse(node, depth):
    #     indent = "  " * depth
    #     if tree_.feature[node] != _tree.TREE_UNDEFINED:
    #         name = feature_name[node]
    #         threshold = tree_.threshold[node]
    #         print "{}if {} <= {}:".format(indent, name, threshold)
    #         recurse(tree_.children_left[node], depth + 1)
    #         print "{}else:  # if {} > {}".format(indent, name, threshold)
    #         recurse(tree_.children_right[node], depth + 1)
    #     else:
    #         print "{}return {}".format(indent, tree_.value[node])

    #recurse(0, 1)


if __name__ == "__main__":
    main()