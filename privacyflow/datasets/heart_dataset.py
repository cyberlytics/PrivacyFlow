import pandas as pd
import torch
from torch.utils.data import Dataset

from privacyflow.configs import path_configs


class HeartDataset(Dataset):
    def __init__(self, mode: str = "train", custom_df:pd.DataFrame = None):
        self.mode = mode

        if self.mode == "train":
            df = pd.read_csv(path_configs.HEART_DATA_TRAIN)
        elif self.mode == "val":
            df = pd.read_csv(path_configs.HEART_DATA_VAL)
        elif self.mode == "test":
            df = pd.read_csv(path_configs.HEART_DATA_TEST)
        else:
            df = custom_df

        self.df_labels = df['target']
        self.df_inputs = df.drop(columns=['target'])

    def __len__(self):
        return len(self.df_inputs.index)

    def __getitem__(self, idx):
        variables = torch.tensor(self.df_inputs.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.df_labels.iloc[idx])
        return variables, label


class MembershipInferenceDataset(Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df_labels = df['target']
        self.df_inputs = df.drop(columns=['target'])

    def __len__(self):
        return len(self.df_inputs.index)

    def __getitem__(self, idx):
        variables = torch.tensor(self.df_inputs.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.df_labels.iloc[idx])
        return variables, label
