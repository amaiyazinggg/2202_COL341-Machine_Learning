import os
import argparse
import pandas as pd
from linear_regression import run_linear_regression
from ridge import run_ridge
from logistic_regression import run_logistic_regression

parser = argparse.ArgumentParser(prog="assignment")
parser.add_argument("--train_path", help="path to train file")
parser.add_argument("--val_path", type=str, help="path to validation file")
parser.add_argument("--test_path", type=str, help="path to test file")
parser.add_argument("--out_path", type=str, help="path to generated output scores")
parser.add_argument("--section", type=int, help="section number, 1, 2 or 5")

args = parser.parse_args()

df = pd.read_csv(args.train_path, header=None)
df_val = pd.read_csv(args.val_path, header=None)
df_test = pd.read_csv(args.test_path, header=None)

if args.section == 1:
    test_pred = run_linear_regression(df, df_val, df_test)
    output_df = pd.concat({0: df_test[0], 1: test_pred}, axis=1)
    output_df.to_csv(
        os.path.join(args.out_path, "section_1_output.csv"), index=False, header=None
    )
    
elif args.section == 2:
    test_pred = run_ridge(df, df_val, df_test)
    output_df = pd.concat({0: df_test[0], 1: test_pred}, axis=1)
    output_df.to_csv(
        os.path.join(args.out_path, "section_2_output.csv"), index=False, header=None
    )
    
elif args.section == 5:
    test_pred = run_logistic_regression(df, df_val, df_test)
    print(test_pred)
    result = []
    for i in range(len(test_pred)):
        summ = 0
        classes = 0
        for j in range(8):
            if test_pred[i][j] > test_pred[i][classes]:
                classes = j
            summ += test_pred[i][j]

        if 1 - summ > test_pred[i][classes]:
            classes = 8
        result.append(classes + 1)
    print(result)

else:
    raise Exception("Incorrect section argument. Please select from 1, 2 or 5")
