import pandas as pd
from sklearn.model_selection import train_test_split

from privacyflow.configs import path_configs

from sklearn.preprocessing import *


def preprocess_heart_data():
    data = pd.read_csv(path_configs.HEART_DATA)

    # train test split
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

    # Scaling based on training data
    labels_train = [label for label in df_train['target']]
    df_train = df_train.drop(columns=['target'])
    columns = df_train.columns
    scaler = StandardScaler()
    df_train = scaler.fit_transform(df_train)
    df_train = pd.DataFrame(df_train, columns=columns)
    df_train['target'] = labels_train

    labels_test = [label for label in df_test['target']]
    df_test = df_test.drop(columns=['target'])
    columns = df_test.columns
    df_test = scaler.transform(df_test)
    df_test = pd.DataFrame(df_test, columns=columns)
    df_test['target'] = labels_test

    labels_val = [label for label in df_val['target']]
    df_val = df_val.drop(columns=['target'])
    columns = df_val.columns
    df_val = scaler.transform(df_val)
    df_val = pd.DataFrame(df_val, columns=columns)
    df_val['target'] = labels_val


    # save files
    df_train.to_csv(path_configs.HEART_DATA_TRAIN, index=False)
    df_val.to_csv(path_configs.HEART_DATA_VAL, index=False)
    df_test.to_csv(path_configs.HEART_DATA_TEST, index=False)

    df_total = pd.concat([df_train,df_val,df_test])
    df_total.to_csv(path_configs.HEART_DATA_TOTAL)
