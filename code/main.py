#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from preprocessing import preprocess, upsample_downsample, feature_selection
from model import random_forest_with_cv

from IPython.display import set_matplotlib_formats
import seaborn as sns
import plotly.offline as pyo
import pickle

# Supress warnings and default INFO logging
import warnings
import logging

set_matplotlib_formats('pdf', 'png')
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

sns.set()
pyo.init_notebook_mode()


train_filepath = '/Users/pbhagwat/DEV/animall/animall-assignment/data/TrainingSet.csv'
test_filepath = '/Users/pbhagwat/DEV/animall/animall-assignment/data/TestingSet.csv'
submission_filepath = '/Users/pbhagwat/DEV/animall/animall-assignment/data/Submission.csv'
model_path = '/Users/pbhagwat/DEV/animall/animall-assignment/finalized_model.sav'


def predict_results(test_filepath, model_path):
    y_test_pred = pd.DataFrame()
    X, y = preprocess(train_filepath)
    fs, _, _ = feature_selection(X, y, 150)

    df_test = pd.read_csv(test_filepath)
    df_test = df_test.fillna(0)
    X_test = df_test.set_index('Sl No.')
    X_test = fs.transform(X_test)

    model = pickle.load(open(model_path, 'rb'))
    y_test_pred['Machine_State'] = model.predict(X_test)
    frame = [df_test[['Sl No.']], y_test_pred]
    result = pd.concat(frame, axis=1)
    result['Machine_State'].replace({0: 'Good', 1: 'Bad'}, inplace=True)
    result.to_csv(submission_filepath)
    print(result.head())
    return y_test_pred


def train(train_filepath, model_path):
    X, y = preprocess(train_filepath)
    fs, X_selected, y = feature_selection(X, y, 150)
    X_transformed, y_transformed = upsample_downsample(X_selected, y)

    # X_train, X_val, y_train, y_val = train_test_data(X_transformed, y_transformed)

    # random_forest(X_train, X_val, y_train, y_val)
    rf_model = random_forest_with_cv(X_transformed, y_transformed)

    pickle.dump(rf_model, open(model_path, 'wb'))


def main():
    # train(train_filepath, model_path)
    predict_results(test_filepath, model_path)


if __name__ == '__main__':
    main()
