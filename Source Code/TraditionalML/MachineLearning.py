from autogluon.tabular import TabularDataset, TabularPredictor
from catboost import train
import numpy as np
import pandas as pd
import os
from sqlalchemy import false, true
from tqdm import tqdm

'''Configurations'''
TIME_SLICE = 48

TRAIN_TIME_SLICE = TIME_SLICE
TRAIN_SENSOR = ["attitude", "gravity", "rotation", "acceleration"]
TRAIN_VALID_RATIO = 0.25

TEST_TIME_SLICE = TIME_SLICE
TEST_SENSOR = ["attitude", "gravity", "rotation", "acceleration"]
TEST_VALID_RATIO = 0.25

TRAINING = true
'''End of Configurations'''

train_data_path = "..\\Baseline\\data\\preprocessed\\train_ts{ts}_sensor{sensor}_vr{vr}.npy".format(ts=TRAIN_TIME_SLICE, sensor=TRAIN_SENSOR, vr=TRAIN_VALID_RATIO)
test_data_path = "..\\Baseline\\data\\preprocessed\\test_ts{ts}_sensor{sensor}_vr{vr}.npy".format(ts=TEST_TIME_SLICE, sensor=TEST_SENSOR, vr=TEST_VALID_RATIO)

model_save_path = "agModels_ts{time_slice}_vr{mask_ratio}_Sensor{sensor}".format(time_slice = TRAIN_TIME_SLICE, mask_ratio = TRAIN_SENSOR, sensor = TRAIN_VALID_RATIO)
leaderboard_save_path = "leader_board_ts{time_slice}_vr{mask_ratio}_Sensor{sensor}.csv".format(time_slice = TEST_TIME_SLICE, mask_ratio = TEST_VALID_RATIO, sensor = TEST_SENSOR)

train_data = np.load(train_data_path)
test_data = np.load(test_data_path)

print("--->data size")
print("train data size:", train_data.shape)
print("test data size:", test_data.shape)

subsample_size = 100000
train_data = TabularDataset(pd.DataFrame(train_data)).sample(n=subsample_size, random_state=0)
test_data = TabularDataset(pd.DataFrame(test_data))

assert TEST_TIME_SLICE==TRAIN_TIME_SLICE
label = TEST_TIME_SLICE*12
print("label:", label)

if TRAINING:
    predictor = TabularPredictor(label=label, path=model_save_path, problem_type="multiclass").fit(train_data, ag_args_fit={'num_gpus': 0})

y_test = test_data[label]  # 提取标签列
test_data_nolab = test_data.drop(columns=[label])  # 删除标签列

predictor = TabularPredictor.load(model_save_path)
y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print("Results:  \n", perf)

leaderboard =predictor.leaderboard(test_data, silent=True)
print("leaderboard:  \n", leaderboard)

leaderboard.to_csv(leaderboard_save_path)