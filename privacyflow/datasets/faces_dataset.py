import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from privacyflow.configs import path_configs


class FacesDataset(Dataset):

    def __init__(self, mode: str = "train", transform=None, img_folder_path: str = path_configs.IMG_FOLDER_PATH,
                 img_details: str = path_configs.IMG_INFO_PATH) -> None:
        self.mode = mode
        self.img_folder_path = img_folder_path
        self.img_details_path = img_details
        self.transform = transform

        # the dataset recommends a partitions into training, val and test
        # train -> images 1-162770
        # val -> 162771-182637
        # test -> 182638-202599
        if self.mode == "train":
            self.img_range = range(1, 162771)
        elif self.mode == "val":
            self.img_range = range(162771, 182638)
        else:  # self.mode == "test"
            self.img_range = range(162771, 182638)

    def __len__(self) -> int:
        return len(self.img_range)

    def __getitem__(self, idx):
        image_id = self.img_range[idx]
        image = self._load_image(image_id)
        label = self._load_label(image_id)
        return image, label

    def _load_image(self, img_id):
        img_filename = f"{self.img_folder_path}/{img_id:06d}.jpg"
        return torchvision.io.read_image(img_filename)

    def _load_label(self, img_id):
        return ""
