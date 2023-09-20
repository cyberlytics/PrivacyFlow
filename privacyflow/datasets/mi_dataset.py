import pandas as pd
import torch
from torch.utils.data import Dataset

class MembershipInferenceDataset(Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df_labels = df['target']
        self.df_inputs = df.drop(columns=['target'])

    def __len__(self):
        return len(self.df_inputs.index)

    def __getitem__(self, idx):
        variables = torch.tensor(self.df_inputs.iloc[idx], dtype=torch.float32)
        label = torch.tensor([self.df_labels.iloc[idx]],dtype=torch.float32)
        return variables, label
