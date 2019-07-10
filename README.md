# SigNet : Writer Independent Offline Signature Verification
SigNet is a Siamese Convolutional Neural Network modeled to verify original and forged signatures offline. It takes just one genuine signature of a person and then all other signatures whether genuine or fraudulent can then be verified by it. This [paper](https://arxiv.org/abs/1707.02131) can be referred for better understanding.

#### Preparing the dataset
1. The model is trained on CEDAR dataset which can be downloaded from [here](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar)
2. Extract the files and then you will get the following file structure :
```
|-- signatures
|	|-- full_org
|	|	|-- original_1_1.jpg
|	|	|-- original_1_2.jpg
|	|	|-- ...(all original 24 signs of 24 signers i.e. 24x24 = 576 images)
|	|-- full_forg
|	|	|-- forgeries_1_1.jpg
|	|	|-- forgeries_1_2.jpg
|	|	|-- ...(all forged 24 signs of 24 signers i.e. 24x24 = 576 images)
```
3. The [generate_dataset.py](https://github.com/AtharvaKalsekar/SigNet/blob/master/generate_dataset.py) creates a pandas dataframe and the train and validation datasets are splitted here.

#### Base Network Architecture
This base network can be found in [base_network.py](https://github.com/AtharvaKalsekar/SigNet/blob/master/base_network.py)
```
Layer (type) Output Shape Param #
================================================================= 
input_4 (InputLayer) (None, 155, 220, 3) 0 
_________________________________________________________________ 
conv2d_5 (Conv2D) (None, 145, 210, 96) 34944 
_________________________________________________________________ 
batch_normalization_3 (Batch (None, 145, 210, 96) 384 
_________________________________________________________________ 
activation_5 (Activation) (None, 145, 210, 96) 0 
_________________________________________________________________ 
max_pooling2d_5 (MaxPooling2 (None, 48, 70, 96) 0 
_________________________________________________________________ 
dropout_4 (Dropout) (None, 48, 70, 96) 0 
_________________________________________________________________ 
conv2d_7 (Conv2D) (None, 46, 68, 384) 332160 
_________________________________________________________________ 
activation_7 (Activation) (None, 46, 68, 384) 0 
_________________________________________________________________ 
conv2d_8 (Conv2D) (None, 44, 66, 256) 884992 
_________________________________________________________________ 
activation_8 (Activation) (None, 44, 66, 256) 0 
_________________________________________________________________ 
max_pooling2d_6 (MaxPooling2 (None, 14, 22, 256) 0 
_________________________________________________________________ 
dropout_5 (Dropout) (None, 14, 22, 256) 0 
_________________________________________________________________ 
flatten_2 (Flatten) (None, 78848) 0 
_________________________________________________________________
dense_3 (Dense) (None, 1024) 80741376 
_________________________________________________________________ 
dropout_6 (Dropout) (None, 1024) 0 
_________________________________________________________________ 
dense_4 (Dense) (None, 128) 131200 
================================================================= 
Total params: 82,125,056 Trainable params: 82,124,864 Non-
trainable params: 192 
_________________________________________________________________
```
#### Training
The model is trained on **colab** using both keras and tansorflow which can be found here [signet_keras.ipynb](https://github.com/AtharvaKalsekar/SigNet/blob/master/signet_keras.ipynb) and [signet.ipynb](https://github.com/AtharvaKalsekar/SigNet/blob/master/signet.ipynb) respectively. The train accuracy of the model so far is **81.42%**. Better results can be achieved by augmenting the dataset with more examples.
