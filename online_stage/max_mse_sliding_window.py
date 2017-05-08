import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join as opj
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count

dir_pattern = "{tag}_predictions"
file_pattern = "{label}_{tag}.txt"
test_data_pattern = "test/{label}/{label}_{tag}.csv"
results_dir = "./model_results"
scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
seq_len = 51
window_size = 25

tags = [item.rstrip("_predictions") for item in os.listdir(results_dir) if item.endswith("predictions")]
labels = ["_".join(item.split("_")[:2]) for item in os.listdir(opj(results_dir, "tag12_predictions/"))
          if item.endswith("txt")]

def MSE(test, prediction):
    return np.sum((test - prediction) ** 2) / len(test)

def max_mse_idx(label):
    """Searches for window with max MSE between prediction and actual data,
    using sliding window approach
    """
    label_mses = []
    for tag in tags:
        f = open(test_data_pattern.format(**locals()))
        data = np.array(list(map(lambda x: float(x.rstrip()), f.readlines())))
        f.close()
        test_data = scaler.fit_transform(data)[seq_len:]
        f = open(opj(results_dir, dir_pattern.format(**locals()),
                     file_pattern.format(**locals())))
        pred_data = np.array(list(map(lambda x: float(x.rstrip()),
                                      f.readlines())))
        f.close()
        mses = []
        for i in range(0, len(test_data) - window_size):
            test_data_window = test_data[i : i + window_size]
            pred_data_window = pred_data[i : i + window_size]
            mse = MSE(test_data_window, pred_data_window)
            mses.append(mse)
        label_mses.append(mses)
    df = pd.DataFrame(label_mses)
    df_sum = df.sum(axis = 0)
    max_so_far = float("-inf")
    idx = 0
    for i in range(len(df_sum)):
        if df_sum[i] > max_so_far:
            max_so_far = df_sum[i]
            idx = i
    return label, idx

results = {}

with Pool(cpu_count()) as p:
    result = p.map(max_mse_idx, labels)
for item in result:
    results[item[0]] = item[1]

f = open("results.csv", "w")
for label in sorted(results.keys()):
    data = pd.read_csv("./test/{}.csv".format(label))
    # seqlen first observations were used for initial prediction
    row = results[label] + seq_len
    label = label.rstrip("_test").lstrip("0")
    f.write("{},{}".format(label if label else "0", int(data.iloc[row, 0])))
f.close()






