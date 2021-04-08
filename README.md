# Semantic Segmentation Models

This is a Pytorch implementation for training and testing different semantic segmentation models on the cityscapes dataset. 

> **Models Currently Supported**
>
> DeepLab V3


## ⚙️ Setup

Create a fresh eviornment using [Anaconda](https://www.anaconda.com/download/) distribution. You can then install the dependencies with:
```shell
conda install pytorch torchvision torchaudio -c pytorch
conda install opencv=4.1.2
conda install matplotlib
```
Recommended python version: 3.7

## 💾 Cityscapes Training Data

The dataset required for training and evalution can be downloaded using 
```shell
sh utils/download_dataset.sh
```
_YOUR_USER_NAME_ and _YOUR_PASSWORD_ in the script should be replaced with the users credentials. 

## References

@misc{FredrikGustafsson,
  author = {Fredrik Gustafsson},
  title = {deeplabv3},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://https://github.com/fregu856/deeplabv3}},
  commit = {415d983ec8a3e4ab6977b316d8f553371a415739}
}