# author Kingnight
# time 2022/4/30 11:35
import numpy as np
import pandas as pd
from donut import complete_timestamp, standardize_kpi, Donut, DonutTrainer, DonutPredictor
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from tfsnippet.utils import get_variables_as_dict, VariableSaver
import tensorflow.compat.v1 as tf
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
tf.disable_v2_behavior()

# path to the dataset
file_csv = "../cpu4.csv"

# Read the raw data.
data = pd.read_csv(file_csv)
timestamp = data["timestamp"]
values = data["value"]
labels = data["label"]
dataset_name = file_csv.split('.')[0]
print("Timestamps: {}".format(timestamp.shape[0]))