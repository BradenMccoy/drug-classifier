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
from itertools import combinations
import os

def main():
    train = pd.read_csv('drug_consumption.data', header=None)
    train.columns = ['Id', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy',
                    'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    # columns we are interested in
    feature_cols = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                    'Impulsive']
    
    # y is the output, the class we want to predict, and x contains the other columns of the dataset
    legal_drugs = ['Alcohol', 'Caff', 'Cannabis', 'Choc', 'Legalh', 'Nicotine']
    illegal_drugs = ['Amphet', 'Amyl', 'Benzos', 'Coke', 'Crack', 'Ecstasy',
                     'Heroin', 'Ketamine', 'LSD', 'Meth', 'Mushrooms', 'Semer', 'SS', 'VSA']
    
    # change to a binary classification, used or never used
    for drug in legal_drugs:
        train[drug] = train[drug].apply(mod_classification)
    for drug in illegal_drugs:
        train[drug] = train[drug].apply(mod_classification)

    train['Ethnicity'] = train['Ethnicity'].apply(mod_ethnicity)
    
    combo_list = generate_combo_list(feature_cols) #all possible combinations of feature columns

    # get file for legal drugs accuracy stats and rules
    generate_tree(legal_drugs, feature_cols, combo_list, "legal", train)
    print("\n")
    # ditto for illegal drugs
    generate_tree(illegal_drugs, feature_cols, combo_list, "illegal", train)

# given a list of drugs and feature_columns, generate the trees for the possible
# permutations of rules and find the highest accuracy score
def generate_tree(drug_list, feature_columns, combo_list, filename, train):
    print("\033[4m" + filename + " drugs:\033[0m")
    
    outfile = open(filename + ".txt", "w")

    # make directories for the pngs
    path = "./" + filename + "-pngs"
    if False == os.path.isdir(path):
        os.mkdir(path)

    for drug in drug_list:
        y = train[drug]

        outfile.write("Predicting for " + drug + ":\n")
        
        # track the maximum accuracy rule for each drug
        max_accuracy = 0
        max_feature_list = []
        max_DT = DecisionTreeClassifier(criterion="entropy") # depth is 4
        
        # go through all possible attribute lists
        for combo in combo_list:
            # generate the tree
            X = train[list(combo)]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
            DT = DecisionTreeClassifier(criterion="entropy", max_depth=4) # depth is 4
            DT.fit(X_train,y_train)
            pred=DT.predict(X_test)

            outfile.write("\tUsing feature columns: ")
            
            for item in list(combo):
                outfile.write(item + " ")
            outfile.write("Accuracy: " + str(accuracy_score(y_test, pred)) + "\n")

            # check if this is a maximum accuracy
            if(accuracy_score(y_test, pred) > max_accuracy):
                max_accuracy = accuracy_score(y_test, pred)
                max_feature_list = combo
                max_DT = DT

        outfile.write("Maximum accuracy: " + str(max_accuracy) + "\n")
        outfile.write("Maximum accuracy attributes: " + str(max_feature_list) + "\n\n")
        print(drug + " maximum accuracy: " + "{:.2f}".format(max_accuracy))
        
        # create a png for the maximum accuracy rule
        pngname = path + "/" + drug + "-" + "{:.2f}".format(max_accuracy) + ".png"
        png(max_DT, max_feature_list, pngname)
    
    outfile.close()

def generate_combo_list(feature_cols):
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
    return combo_list

def mod_classification(x):
    if (x == 'CL0'):
        return 'Never Used'
    else:
        return 'Used'

def mod_ethnicity(x):
    if x == -.50212:
        # asian
        return 1
    if x == 1.90725:
        # mixed-black/asian
        return 2
    if x == 0.12600:
        # mixed-white/asian
        return 3
    if x == -0.22166:
        # mixed-white/black
        return 4
    if x == -1.10702:
        # black
        return 5
    if x == -0.31685:
        # white
        return 6
    if x == 0.11440:
        # other
        return 7
    else:
        return x


def png(DT, feature_cols, file_name):
    dot_data = StringIO()
    export_graphviz(DT, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['Never Used', 'Used'])
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