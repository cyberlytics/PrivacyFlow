import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from privacyflow.configs import path_configs


class FacesDataset(Dataset):

    def __init__(self,
                 mode: str = "train",
                 transform=None,
                 img_folder_path: str = path_configs.IMG_FOLDER_PATH,
                 img_details: str = path_configs.IMG_INFO_PATH,
                 label_cols: list | str = 'all',
                 custom_range=None) -> None:
        self.mode = mode
        self.img_folder_path = img_folder_path
        self.img_details_path = img_details
        self.transform = transform

        self.label_df = pd.read_csv(img_details)
        columns = self.label_df.columns if label_cols == 'all' else [
                                                                        'image_id'] + label_cols  # if provided, select only given columns
        self.label_df = self.label_df[columns]
        for col in self.label_df.columns:  # change -1s to 0s
            self.label_df[col] = [0 if val == -1 else val for val in self.label_df[col]]

        # the dataset recommends a partitions into training, val and test
        # train -> images 1-162770
        # val -> 162771-182637
        # test -> 182638-202599
        if self.mode == "train":
            self.img_range = range(1, 162771)
        elif self.mode == "val":
            self.img_range = range(162771, 182638)
        elif self.mode == "test":
            self.img_range = range(182638, 202600)
        elif self.mode == "all":
            self.img_range = range(1, 202600)
        else:
            self.img_range = custom_range

    def __len__(self) -> int:
        return len(self.img_range)

    def __getitem__(self, idx):
        image_id = self.img_range[idx]
        image = self._load_image(image_id)
        label = self._load_label(image_id)
        return image, label

    def _load_image(self, img_id):
        img_filename = f"{self.img_folder_path}/{img_id:06d}.jpg"
        # img = torchvision.io.read_image(img_filename)
        img = Image.open(img_filename)
        if self.transform:
            img = self.transform(img)
        return img

    def _load_label(self, img_id):
        labels = self.label_df.loc[self.label_df['image_id'] == f"{img_id:06d}.jpg"]
        labels = labels.drop(columns=['image_id']).iloc[0]
        return torch.tensor(labels, dtype=torch.float32)


class FaceMIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_column_name:str = 'target'):
        self.df_labels = df[target_column_name]
        self.df_inputs = df.drop(columns=[target_column_name])

    def __len__(self):
        return len(self.df_labels.index)

    def __getitem__(self, idx):
        variables = torch.tensor(self.df_inputs.iloc[idx], dtype=torch.float32)
        label = torch.tensor(self.df_labels.iloc[idx])
        return variables, label