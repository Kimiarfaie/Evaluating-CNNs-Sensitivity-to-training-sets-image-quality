# Dataset

In this project a subset of ImageNet1K is used to train the CNNs. 

1. You can download the dataset from kaggle:

https://www.kaggle.com/c/imagenet-object-localization-challenge

The dataset will have the following structure with txt file for synset mappings:

```bash
imagenet/ILSVRC/Data/CLS-LOC/train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ......
├── ......
imagenet/ILSVRC/Data/CLS-LOC/val/
├── n01440764
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ......
├── ......
```
2. You can use the one_makeimagenetsubset.m scirpt to create a subset of ImageNet containing 10,000 (100 images from 100 classes) training and 2000 (20 images from 100 classes) validation images. The script screates the subset following the original structure.

```bash
Dataset/
├── images
│   ├── val
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Defocus1.tiff
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Defocus2.tiff
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Focused.tiff
│   │   ├── ...
├── labels
│   ├── val
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Defocus1.txt
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Defocus2.txt
│   │   ├── car_18.0 mm_100ISO_0EV_Dist1_Focused.txt
│   │   ├── ...
├── eSFR Chart
│   ├── chart_18.0 mm_100ISO_+1EV_Defocus2.tiff
│   ├── chart_18.0 mm_1600ISO_+1EV_Defocus1.tiff
│   ├── chart_18.0 mm_100ISO_+1EV_Focused.tiff
│   ├── ...
├── annotations
│   ├── instances_val.json
├── data.yaml
```
