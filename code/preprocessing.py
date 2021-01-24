import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split


def preprocess(filepath):
    df = pd.read_csv(filepath)
    df = df.fillna(0)

    # Drop the rows that belong to undefined class
    data = df[df['Machine_State'] != 0]

    # Encode the class labels
    data['Machine_State'].replace({"Good": 0, "Bad": 1}, inplace=True)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    X = data.drop('Machine_State', axis=1)
    y = data['Machine_State']

    return X, y


def upsample_downsample(X, y):
    # define oversampling strategy
    over_sample = RandomOverSampler(sampling_strategy=0.2)
    X_over, y_over = over_sample.fit_resample(X, y)

    # define under sampling strategy
    under_sample = RandomUnderSampler(sampling_strategy=0.5)
    X_under, y_under = under_sample.fit_resample(X_over, y_over)
    return X_under, y_under


def feature_selection(X, y, k=150):
    fs = SelectKBest(score_func=f_classif, k=k).fit(X,y)

    # apply feature selection
    X_selected = fs.transform(X)
    return fs, X_selected, y


def train_test_data(X, y, test_size=0.2, random_state=66, shuffle=True):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return X_train, X_val, y_train, y_val
