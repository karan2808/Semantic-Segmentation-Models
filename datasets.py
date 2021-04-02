# parts of code borrowed from: https://github.com/fregu856/deeplabv3
import torch 
import torch.utils.data
import numpy as np 
import cv2
import os

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs  = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_path):
        self.img_dir = cityscapes_path + '/leftImg8bit/train/'
        self.lbl_dir = cityscapes_path + '/gtFine/'

        # set image dimensions
        self.img_h = 1024
        self.img_w = 2048

        # rescale to half the size (optional)
        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        for train_dir in train_dirs:
            train_img_path = self.img_dir + train_dir
            
            # get the file names
            file_names = os.listdir(train_img_path)

            for file_name in file_names:
                # get the image id
                img_id   = file_name.split("_leftImg8bit.png")[0]
                # get image path and label path
                img_path = train_img_path + file_name
                lbl_path = self.lbl_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["lbl_path"] = lbl_path
                example["img_id"]   = img_id
                self.examples.append(example)

        self.num_examples    = len(self.examples)
        self.transforms      = None

    def __getitem__(self, index):
        # get an example
        example = self.examples[index]

        img_path = example["img_path"]
        img      = cv2.imread(img_path, -1)

        # resize the image and label with nearest neighbor inter, to preserve original data
        img      = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
        img      = img / 255.0
        img      = img - np.array([0.485, 0.456, 0.406])
        img      = img / np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img      = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img      = img.astype(np.float32)

        lbl_path = example["lbl_path"]
        lbl_img  = cv2.imread(lbl_path, -1) # read as it is, shape is 1024, 2048

        lbl_img  = cv2.resize(lbl_img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
        
        # apply augmentations..
        img      = torch.from_numpy(img)
        lbl_img  = torch.from_numpy(lbl_img)

        return (img, lbl_img)
    
    def __len__(self):
        return self.num_examples


class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_path):
        self.img_dir = cityscapes_path + '/leftImg8bit/train'
        self.lbl_dir = cityscapes_path + '/gtFine/'

        # set image dimensions
        self.img_h = 1024
        self.img_w = 2048

        # rescale to half the size (optional)
        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        for val_dir in val_dirs:
            val_img_path = self.img_dir + val_dir
            
            # get the file names
            file_names = os.listdir(val_img_path)

            for file_name in file_names:
                # get the image id
                img_id   = file_name.split("_leftImg8bit.png")[0]
                # get image path and label path
                img_path = val_img_path + file_name
                lbl_path = self.lbl_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["lbl_path"] = lbl_path
                example["img_id"]   = img_id
                self.examples.append(example)

        self.num_examples    = len(self.examples)
        self.transforms      = None

    def __getitem__(self, index):
        # get an example
        example = self.examples[index]

        img_path = example["img_path"]
        img      = cv2.imread(img_path, -1)

        # resize the image and label with nearest neighbor inter, to preserve original data
        img      = cv2.resize(img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
        img      = img / 255.0
        img      = img - np.array([0.485, 0.456, 0.406])
        img      = img / np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img      = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img      = img.astype(np.float32)

        lbl_path = example["lbl_path"]
        lbl_img  = cv2.imread(lbl_path, -1) # read as it is, shape is 1024, 2048

        lbl_img  = cv2.resize(lbl_img, (self.new_img_w, self.new_img_h), interpolation=cv2.INTER_NEAREST)
        
        # apply augmentations..
        img      = torch.from_numpy(img)
        lbl_img  = torch.from_numpy(lbl_img)

        return (img, lbl_img)
    
    def __len__(self):
        return self.num_examples