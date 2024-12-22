# Dataset

In this project a subset of ImageNet1K is used to train the CNNs. 

1. You can download the dataset from kaggle:

https://www.kaggle.com/c/imagenet-object-localization-challenge

The dataset will have the following structure with txt file for synset mappings:

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


2. You can use the one_makeimagenetsubset.m scirpt to create a subset of ImageNet containing 10,000 (100 images from 100 classes) training and 2000 (20 images from 100 classes) validation images. The script screates the subset following the original structure.

3. 
