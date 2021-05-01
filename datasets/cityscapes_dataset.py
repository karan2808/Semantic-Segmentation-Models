# parts of code borrowed from: https://github.com/fregu856/deeplabv3
import torch 
import torch.utils.data
import numpy as np 
import cv2
import os
import torchvision
from PIL import Image

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs  = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin/", "bielefeld/", "bonn/", "leverkusen/", "mainz/", "munich/"]

class DATASET_CITYSCAPES(torch.utils.data.Dataset):
    def __init__(self, cityscapes_path, split):
        if split == 'train':
            self.img_dir = cityscapes_path + '/leftImg8bit/train/'
            directories  = train_dirs
        if split == 'val':
            self.img_dir = cityscapes_path + '/leftImg8bit/val/'
            directories  = val_dirs
        if split == 'test':
            self.img_dir = cityscapes_path + '/leftImg8bit/test/'
            directories  = test_dirs
        
        self.lbl_dir = cityscapes_path + '/meta/label_imgs/'
        # set image dimensions
        self.img_h = 1024
        self.img_w = 2048

        # rescale to half the size (optional)
        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        for directory in directories:
            images_path = self.img_dir + directory

            # get the file names
            file_names = os.listdir(images_path)
            for file_name in file_names:
              # get image id and path 
              img_id   = file_name.split("_leftImg8bit.png")[0]
              img_path = images_path + file_name

              lbl_path = self.lbl_dir + img_id + ".png"

              example = {}
              example["img_path"] = img_path
              example["lbl_path"] = lbl_path
              example["img_id"]   = img_id
              self.examples.append(example)
        
        self.num_examples    = len(self.examples)
        if split == "train" or split == 'val':
            self.transforms_ = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.new_img_h, self.new_img_w)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # torchvision.transforms.RandomErasing(),
            ])
        else:
            self.transforms_ = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.new_img_h, self.new_img_w)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    def __getitem__(self, index):
        # get an example
        example = self.examples[index]
        
        img_path   = example['img_path']
        # print(img_path)
        img        = Image.open(img_path)
        # img        = img.astype(np.float32)
        img        = self.transforms_(img)
        # print(img.shape)
        # img        = img.permute(2, 0, 1)

        lbl_path = example["lbl_path"]
        # print(lbl_path)
        lbl_img  = cv2.imread(lbl_path, -1) # read as it is, shape is 1024, 2048

        lbl_img  = cv2.resize(lbl_img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)

        lbl_img  = torch.from_numpy(lbl_img)
        return (img, lbl_img)

    def __len__(self):
        return self.num_examples

class DATASET_CITYSCAPES_FOGGY(torch.utils.data.Dataset):
    def __init__(self, cityscapes_path, split, fog_scale):
        if split == 'train':
            self.img_dir = cityscapes_path + '/leftImg8bit_foggy/train/'
            directories  = train_dirs
        if split == 'val':
            self.img_dir = cityscapes_path + '/leftImg8bit_foggy/val/'
            directories  = val_dirs
        if split == 'test':
            self.img_dir = cityscapes_path + '/leftImg8bit_foggy/test/'
            directories  = test_dirs
        
        self.lbl_dir = cityscapes_path + '/meta/label_imgs/'
        # set image dimensions
        self.img_h = 1024
        self.img_w = 2048

        # rescale to half the size (optional)
        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        for directory in directories:
            images_path = self.img_dir + directory

            # get the file names
            file_names = os.listdir(images_path)
            for file_name in file_names:
              # get image id and path 
              img_id   = file_name.split("_leftImg8bit")[0]
              img_path = images_path + file_name

              if fog_scale != 0:
                  # print(file_name.split('_beta_')[1].split('.')[0])
                  img_fog_scale = file_name.split('_beta_')[1].split('.png')[0]
                  if img_fog_scale != str(fog_scale):
                      continue

              lbl_path = self.lbl_dir + img_id + ".png"

              example = {}
              example["img_path"] = img_path
              example["lbl_path"] = lbl_path
              example["img_id"]   = img_id
              self.examples.append(example)
        
        self.num_examples    = len(self.examples)
        if split == "train" or split == 'val':
            self.transforms_ = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.new_img_h, self.new_img_w)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # torchvision.transforms.RandomErasing(),
            ])
        else:
            self.transforms_ = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.new_img_h, self.new_img_w)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    def __getitem__(self, index):
        # get an example
        example = self.examples[index]
        
        img_path   = example['img_path']
        # print(img_path)
        img        = Image.open(img_path)
        # img        = img.astype(np.float32)
        img        = self.transforms_(img)
        # print(img.shape)
        # img        = img.permute(2, 0, 1)

        lbl_path = example["lbl_path"]
        # print(lbl_path)
        lbl_img  = cv2.imread(lbl_path, -1) # read as it is, shape is 1024, 2048

        lbl_img  = cv2.resize(lbl_img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)

        lbl_img  = torch.from_numpy(lbl_img)
        return (img, lbl_img)

    def __len__(self):
        return self.num_examples


