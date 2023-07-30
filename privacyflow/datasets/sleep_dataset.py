import pandas as pd
import torch
from torch.utils.data import Dataset

from privacyflow.configs import path_configs


class SleepDataset(Dataset):
    def __init__(self, mode: str = "train"):
        self.num_classes = 3
        self.mode = mode

        if self.mode == "train":
            df = pd.read_csv(path_configs.SLEEP_DATA_PREP_TRAIN)
        elif self.mode == "val":
            df = pd.read_csv(path_configs.SLEEP_DATA_PREP_VAL)
        else:  # "test"
            df = pd.read_csv(path_configs.SLEEP_DATA_PREP_TEST)

        self.df_labels = df['Sleep Disorder']
        self.df_inputs = df.drop(columns=['Sleep Disorder'])

    def __len__(self):
        return len(self.df_inputs.index)

    def __getitem__(self, idx):
        variables = torch.tensor(self.df_inputs.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.df_labels.iloc[idx],dtype=torch.float32)
        return variables, label
