# SigNet : Writer Independent Offline Signature Verification
SigNet is a Siamese Convolutional Neural Network modeled to verify original and forged signatures offline. It takes just one genuine signature of a person and then all other signatures, whether genuine or fraudulent, can be verified by it. This [paper](https://arxiv.org/abs/1707.02131) can be referred for better understanding.

#### Preparing the dataset
1. The model is trained on CEDAR dataset which can be downloaded from [here](http://www.cedar.buffalo.edu/NIJ/data/signatures.rar).
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


#### Training
The model is trained on **colab** using both keras and tensorflow which can be found here [signet_keras.ipynb](https://github.com/AtharvaKalsekar/SigNet/blob/master/signet_keras.ipynb) and [signet_tf.ipynb](https://github.com/AtharvaKalsekar/SigNet/blob/master/signet_tf.ipynb) respectively. The train accuracy of the model so far is **81.42%**. Better results can be achieved by augmenting the dataset with more examples.

#### Loss Function
Contrastive loss was used for the training purpose alongside RMSprop optimizer.
```
def  contrastive_loss(y_true, y_pred):
	margin =  1
	sqaure_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * sqaure_pred + (1  - y_true)* margin_square)
```
> ***Note:** For training, the label for similar signatures is '1' and for dissimilar images it is '0'.*

#### Results
| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
| Original | Genuine | Forged |
|<img src="https://github.com/AtharvaKalsekar/SigNet/blob/master/Test%20Images/org.png"> |  <img src="https://github.com/AtharvaKalsekar/SigNet/blob/master/Test%20Images/org_1.png">  |<img src="https://github.com/AtharvaKalsekar/SigNet/blob/master/Test%20Images/forg.png"> |
Distance (compared to original) as output by model : | 0.12116826 | 1.43014560 |
Predicted Label:| 1 (similar)  | 0 (forged) |
