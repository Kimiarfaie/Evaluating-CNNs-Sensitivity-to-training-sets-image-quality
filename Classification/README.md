# Training CNNs on the subset of ImageNet

The training implementation (main.py) is taken from https://github.com/pytorch/examples/tree/main/imagenet with some changes to visualize the results in tensorboard. 

## Requirements
1. Install PyTorch (pytorch.org)
2. pip install -r requirements.txt

## Training 

To train a model, run main.py, with the desired architecture, path to the dataset, and suitable hyperparameters. 
```python
python main.py pathtodataset --arch alexnet --id TrainingAlexnet --batch-size 8 --lr 0.01 --epochs 100 --gpu 0
```
## Testing 

To test a model run test.py with the path to the pretrained model, test folder, the architecture and the synset mapping file. 
```python
python test.py --model-path pathtomodel --test-folder pathtotestset --arch alexnet --id TestingAlexNet --synset-mapping pathtosynsetmapping--gpu 0
```