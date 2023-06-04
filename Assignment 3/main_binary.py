import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import glob
import sklearn
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import scratch

parser = argparse.ArgumentParser(prog="assignment")
parser.add_argument("--train_path", help="path to train file")
parser.add_argument("--test_path", type=str, help="path to test file")
parser.add_argument("--out_path", type=str, help="path to generated output scores")

args = parser.parse_args()

training = args.train_path
testing = args.test_path
output = args.out_path

cv_img = []

count = 0

main_path = training + "/person"
path = os.listdir(main_path)
for img in path:
    if img.endswith(".png"):
        n = np.array(cv.imread(main_path+'/'+img).flatten())
        cv_img.append(n)
        count += 1
y_train = np.ones(count)

count = 0
main_path = training + "/car"
path = os.listdir(main_path)
for img in path:
    if img.endswith(".png"):
        n = np.array(cv.imread(main_path+'/'+img).flatten())
        cv_img.append(n)
        count += 1

main_path = training + "/dog"
path = os.listdir(main_path)
for img in path:
    if img.endswith(".png"):
        n = np.array(cv.imread(main_path+'/'+img).flatten())
        cv_img.append(n)
        count += 1
    
main_path = training + "/airplane"
path = os.listdir(main_path)
for img in path:
    if img.endswith(".png"):
        n = np.array(cv.imread(main_path+'/'+img).flatten())
        cv_img.append(n)
        count += 1

x_train = np.array(cv_img)
y_train = np.hstack((y_train, np.zeros(count)))

cv_img = []
img_names = []
path = os.listdir(testing)
for img in path:
    if img.endswith(".png"):
        n = np.array(cv.imread(testing+'/'+img).flatten())
        cv_img.append(n)
        img_names.append(img.split(".")[0])

img_names = np.array(img_names)
x_test = np.array(cv_img)



def three_a(x_train, y_train, x_test, img_names):
    selection = SelectKBest(k=10).fit(x_train, y_train)
    x_train = selection.transform(x_train)
    x_test = selection.transform(x_test)
    
    clf = scratch.DecisionTree_Gain()
    clf.fit(x_train, y_train)
    
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31a.csv"), index=False, header=None
    )

def three_b(x_train, y_train, x_test, img_names):
    clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 7)
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31b.csv"), index=False, header=None
    )
    
def three_c(x_train, y_train, x_test, img_names):
    selection = SelectKBest(k=10).fit(x_train, y_train)
    x_train = selection.transform(x_train)
    x_test = selection.transform(x_test)
    
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_split = 4)
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31c.csv"), index=False, header=None
    )

def three_d(x_train, y_train, x_test, img_names):
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.0027352941176470593)
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31d.csv"), index=False, header=None
    )
    
def three_e(x_train, y_train, x_test, img_names):
    clf = RandomForestClassifier(criterion = 'entropy', max_depth = None, min_samples_split = 10, n_estimators = 150)    
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31e.csv"), index=False, header=None
    )
    
def three_f(x_train, y_train, x_test, img_names):
    clf = xgb.XGBClassifier(n_estimators = 40, max_depth = 6, subsample = 0.6)   
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31f.csv"), index=False, header=None
    )
    
def three_h(x_train, y_train, x_test, img_names):
    clf = RandomForestClassifier(criterion = 'entropy', max_depth = None, min_samples_split = 10, n_estimators = 150)     
    clf = clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.array([int(i) for i in y_test_predicted])
    y_test_predicted = pd.Series(y_test_predicted)
    img_names = pd.Series(img_names)
    
    output_df = pd.concat({0: img_names, 1: y_test_predicted}, axis=1)
    output_df.to_csv(
        os.path.join(output, "test_31h.csv"), index=False, header=None
    )

three_a(x_train, y_train, x_test, img_names)    
three_b(x_train, y_train, x_test, img_names)
three_c(x_train, y_train, x_test, img_names)
three_d(x_train, y_train, x_test, img_names)
three_e(x_train, y_train, x_test, img_names)
three_f(x_train, y_train, x_test, img_names)
three_h(x_train, y_train, x_test, img_names)

