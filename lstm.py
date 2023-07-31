import pandas as pd
import numpy as np
import json
import os
from os.path import exists
import math
import random
import re
import argparse


def main():
    suggestions = {}
    with open('./mytestdata/queryindex.json') as json_file:
        queryindex = json.load(json_file)
    for filename in os.listdir("./mytestdata/3screens"):
        if filename.endswith(".csv"):
            user = filename.replace('.csv','')
            print(user)
            suggestions[user] = []
            ratio = 0.7
            if user in ['D43D7EC3E0C2']:
                ratio = 0.85
            print('--------------',user,ratio,'----------')
            allindex = queryindex[filename.replace('.csv','')]
            splitindex = allindex[int(len(allindex)*ratio)]
            pred_index = allindex[int(len(allindex)*ratio):]
            print(splitindex)
            print(pred_index)
            data = pd.read_csv('./mytestdata/3screens/'+filename)
            train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
            print(len(train))
            print(len(validate))
            print(len(test))
            rows = []
            for row in train['target']:
                rows.append(row)
            with open('./mytestdata/lstm_test.txt', 'w') as f:
                for line in rows:
                    f.write(f"{line}\n")
            break

if __name__ == '__main__':
    main()  # execute this only when run directly, not when imported!   