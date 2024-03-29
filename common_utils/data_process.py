import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    return train_data, test_data


def check_outliers(data):
    outliers = []
    for col in data.select_dtypes(include='float').columns:
        col_mean = data[col].mean()
        col_std = data[col].std()
        col_outliers = data[(data[col] > col_mean + 3 * col_std) | (data[col] < col_mean - 3 * col_std)]
        for idx in col_outliers.index:
            outliers.append((col, idx))

    return outliers


def handle_outliers(data, outliers):
    for outlier in outliers:
        col, idx = outlier
        val = data[col][idx]
        col_mean = data[col].mean()
        col_std = data[col].std()
        if val > col_mean + 3 * col_std:
            data[col][idx] = col_mean + 3 * col_std
        if val < col_mean - 3 * col_std:
            data[col][idx] = col_mean - 3 * col_std

    return data


def scale_data(data):
    sc = StandardScaler()
    num_cols = data.select_dtypes(include='float').columns
    data[num_cols] = sc.fit_transform(data[num_cols])

    return data


def encode_features(X):
    ls = []
    str_cols = X.select_dtypes(include=['object']).columns
    le_features = LabelEncoder()
    for col in str_cols:
        X[col] = le_features.fit_transform(X[col])
        ls.append((col, dict(zip(le_features.classes_, range(len(le_features.classes_))))))

    return ls


def encode_labels(y):
    le_labels = LabelEncoder()
    y_encoded = le_labels.fit_transform(y)

    return y_encoded


def decode_labels(le_labels, encoded_labels):
    decoded_labels = le_labels.inverse_transform(encoded_labels)

    return decoded_labels


def check_data_balance(y):
    label_dist = Counter(y)
    
    return label_dist




