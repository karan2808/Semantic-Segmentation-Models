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

Pre-process the dataset to convert label images to ids, using
```shell
python utils/preprocess_data.py PATH_TO_CITYSCAPES_DATASET
```

## References

1.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

2.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

3. **CityScapes**<br />
    Rich metadata: preceding and trailing video frames · stereo · GPS · vehicle odometry.<br />
    [[link]](https://www.cityscapes-dataset.com)

3. **DeepLab V3 Pytorch Implementation**<br />
    Parts of code have been adopted from this repository.<br />
    [[link]](https://github.com/fregu856/deeplabv3)