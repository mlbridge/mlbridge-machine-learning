# MLBridge - Machine Learning

This directory contains the the code for the training and evaluation of a binary 
classifier for alerting whether a person is querying a malicious domain. 

The `notebooks` directory contains the Jupyter Notebook where the training 
procedure can be observed. The `saved_models` directory contains the model that 
has achieved the maximum validation accuracy while training. The `python-code` 
directory contains code that helps the user to retrain the model via the 
`mlbridge-ui` app.

## Installation

Clone the repository:
```
git clone https://github.com/mlbridge/mlbridge-machine-learning.git
```

Go to the `mlbridge-machine-learning` directory and install the dependencies:
```
cd mlbridge-machine-learning
pip install -r requirements.txt
```

Install Elasticsearch by following the instructions from this 
[link](https://phoenixnap.com/kb/install-elasticsearch-ubuntu). Start the 
Elasticsearch server and then run the `training.py` app:
```
cd mlbridge-machine-learning/python-code
python3 training.py
```

## Training

The deep-learning model is trained on a COVID-19 Cyber Threat Coalition 
Blacklist for malicious domains that can be found 
[here](https://blacklist.cyberthreatcoalition.org/vetted/domain.txt) and on a 
list of benign domains from DomCop that can be found 
[here](https://www.domcop.com/top-10-million-domains). 

Currently, the pre-trained model has been trained on the top 500 domain names 
from both these datasets. The final version of the pre-trained model will be 
trained on the entirety of both the datasets.

The dataset was created by combining the malicious domains as well as the benign
domains. The dataset was split as follows: 
- Train Set: 80% of the dataset.
- Validation Set: 10 % of the dataset
- Test Set: 10% of the dataset

## TensorFlow Model Definition

The pre-trained deep learning model is a Convolutional Neural Net whose input is
a (16, 16, 1) shaped array and the output is a single value lying in between 0 
and 1. If the output value is less than 0.5 the domain name is considered benign
, else it is considered malicious. 

The model summary can be found below:


| Layer      | Output Shape          | Activation   | Number of Parameters |
|:----------:|:---------------------:|:------------:|:--------------------:|
| Input      | (None, 16, 16, 1 )    | -            |0                     |
| Conv2D     | (None, 15, 15, 16)    | Relu         |80                    |
| MaxPooling | (None, 7, 7, 16)      | -            |0                     |
| Conv2D     | (None, 6, 6, 16)      | Relu         |1040                  |
| MaxPooling | (None, 3, 3, 16)      | -            |0                     |
| Conv2D     | (None, 2, 2, 8 )      | Relu         |520                   |
| Flatten    | (None, 32)            | -            |0                     |
| Dense      | (None, 8 )            | Relu         |264                   |
| Dense      | (None, 1 )            | Sigmoid      |9                     |

## Accuracy 

The accuracy for the Train Set, Validation Set and Test Set is as follows:

| Metric   | Train Set   | Validation Set | Test Set |  
|----------|-------------|----------------|----------|
| Accuracy | 99.25 %     | 98.00 %        | 98.00 %  |

The training graphs, confusion matrices and other metrics can be found in the 
`training.ipynb` notebook in the `notebooks` directory.


