from typing import List
import torchvision.transforms.functional
from PIL import Image
import numpy as np
from glob import glob
import os.path as osp

import torch


class TartanAirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataroot,
        parts: List[str]=["P001", "P003"],
        data_includes: List[str]=["image_left", "seg_left"],
    ):
        """

        :param dataroot: path to TartanAir dataroot
        :param parts: which subfolders to load from TartanAir
        :param data_includes: the type of data to include
        """
        self.dataroot = dataroot
        self.parts = parts
        self.data_includes = data_includes

        self.data = {di: [] for di in data_includes}

        for part in parts:
            for include in data_includes:
                if "pose" in include:
                    raise ValueError("Loading the robot pose is not implemented yet.")

                folder = osp.join(self.dataroot, part, include)
                files = glob(f"{folder}/*")
                self.data[include].extend(files)

    def __len__(self):
        first_key = list(self.data.keys())[0]
        return len(self.data[first_key])

    def __getitem__(self, idx):
        ret = {}
        for data_include, files in self.data.items():
            f = files[idx]
            if f.endswith(".npy"):
                data = np.load(f)
                data = torch.Tensor(data)
            elif f.endswith(".png"):
                img = Image.open(f)
                data = torchvision.transforms.functional.to_tensor(img)
            else:
                raise ValueError(f"{f} is not a valid extension")

            ret[data_include] = data

        return ret

if __name__ == "__main__":
    dataset = TartanAirDataset("/hdd/TartanAir")