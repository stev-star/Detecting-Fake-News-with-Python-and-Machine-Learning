# Detecting-Fake-News-with-Python-and-Machine-Learning
## What is fake news ?
Fake news is to incorporate information that leads people to the wrong path. Nowadays fake news spreading like 
water and people share this information without verifying it.

The fake news on social media and various other media is wide spreading and is a matter of serious concern due 
to its ability to cause a lot of social and national damage with destructive impacts.

## importing the important libraries
```python
# importing important libralies
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

## read the data using read_csv
```python
df_fake=pd.read_csv("C:\\Users\\Documents\\news\\news.csv")
```
### - get the shape of the data
```python
df_fake.shape
```
### - get to know more about your data
```python
df_info()
```
## what is TfidfVectorizer
### Tf(Term Frequency): The number of times a word appears in a document divided by the total number of words in 
the document. Every document has its own term frequency.
### idf(Inverse Document Frequency): Words that occur many times in a document, but also occur many times in many others.
Inverse data frequency determines the weight of rare words across all documents in the corpus.

## what is PassiveAggressiveClassifier
Passive Aggressive Classifier is a classification algorithm that falls under the category of online learning in
machine learning. It works by responding as passive for correct classifications and responding as aggressive for 
any miscalculation.

