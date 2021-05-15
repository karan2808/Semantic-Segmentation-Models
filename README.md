# Semantic Segmentation Models for Foggy Weather.

This is a Pytorch implementation for training and testing different semantic segmentation models on the cityscapes dataset. We also use foggy weather data generated from the cityscpes dataset, to improve performance of baseline segmentation models in adverse weather. 

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
Foggy weather data can be downloaded using 
```shell
gdown https://drive.google.com/uc?id=1jIwFebKGmYuvpiN0EQ5KSYWmXjqStQac
# unzip -q -o leftImg8bit_trainvaltest_foggy.zip
# mv leftImg8bit_foggy cityscapes/
```
It should be extracted and placed in the same directory as unperturbed data. 

## 🖼️ Prediction on Images
You can predict the sementation labels on unperturbed, and foggy data at all scales using the demo script demo.py. Use the following commands,

```shell
mkdir demo
python demo.py
```
The predictions of class labels will be color coded and saved as images to the demo folder. 

## ⏳ Training
You can train on cityscapes good weather or foggy weather dataset using, 
```shell
mkdir saved_models
python train.py
```
The train command will also plot training and validation loss values at every epoch, and save the updated model at the end of each epoch to the saved_models directory. 


## 📊 Evaluation
Computing the class wise confusion matrix gives a good insight into the performance of a classification model. You can compute the confusion matrix for the model you trained using the command, 
```
python metrics/compute_confusion_mat.py --model_path 'path to your model' --fog_scale 'can be 0, 0.005, 0.01, 0.02' --dataset_path 'path to your dataset' '--compute_unperturbed'
```
compute_unperturbed flag should be used if you wish to compute the confusion matrix for the model trained on good weather data. 

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