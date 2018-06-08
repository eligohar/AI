#!/usr/bin/python
import numpy as np
import pandas as pd
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_contrib.layers import SReLU
from keras.callbacks import BaseLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import os


import numpy as np
import pandas as pd
from __future__ import print_function

# To plot pretty figures
%matplotlib inline 
import matplotlib
import matplotlib.pyplot as plt
import hashlib

from sklearn.preprocessing import Imputer
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras_contrib.layers import SReLU
from keras.callbacks import BaseLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import os

# Reproducible random seed
seed = 1

# Create the output directories, if they don't exist
try:
    os.makedirs("logs")
except OSError:
    if not os.path.isdir("logs"):
        raise

try:
    os.makedirs("figs")
except OSError:
    if not os.path.isdir("figs"):
        raise
        

        
def load_credit_card_data(credit_card_path="data"):
    csv_path = os.path.join(credit_card_path, "creditcard.csv")
    return pd.read_csv(csv_path)  ## DataFrame Object Cotaining all the Data...


# Import and normalize the data
#data = pd.read_csv('data/creditcard.csv')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Credit_Card_images"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, CHAPTER_ID)

### User Defined Functions Section.....
# Saving the image at predefined location (Directory)..
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

    
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

data = load_credit_card_data()

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
data.head()
#data.describe()
# Standardize features by removing the mean and scaling to unit variance
data.iloc[:, 1:29] = StandardScaler().fit_transform(data.iloc[:, 1:29])
data.head() #to dispaly only first 5 instances
data.info()
data.describe()
print("Total Instances in our Credit card Data set:", len(data))
print("Total instances and total attributes (Input Features): ",data.shape)

### Displaying and generating "Transition_amount_vs_class_type" fig.
data.plot(kind="scatter", x="Transition Amount", y="Class Type", alpha=0.1)
save_fig("Transition_amount_vs_class_type")
data.hist(bins=50, figsize=(30,35))
save_fig("credit_card_fields")
plt.show()

### Discover and Visualize the training Data to Gain Insights (Deep Analysis of training data set)

data.plot(kind="scatter", x="Elapsed Time", y="Transition Amount", alpha=0.1)
plt.show()
#save_fig("V20_vs_ElapsedTime")

#print(card_data.shape)
#print(card_data.info())
#card_data.head()

# Task-1: Data Cleaning (filling out the missing values)....
imputer = Imputer(strategy="median") # to replace each attribute’s missing values with the median of that attribute
imputer.fit(data) # fit the imputer instance to the training data (card_data_features)
print(imputer.statistics_)

# transform the training set by replacing missing values by the learned medians
X = imputer.transform(data)
#print(X)
data_tr = pd.DataFrame(X, columns=data.columns) ## want to put it (plain Numpy array) back into a Pandas DataFrame, 



# Convert the data frame to its Numpy-array representation
data_matrix = data.as_matrix()
X = data_matrix[:, 1:29]
Y = data_matrix[:, 30]

# Estimate class weights since the dataset is unbalanced
class_weights = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], Y)))


## Splitting the Data Set into training and Test data set...
# Credit Card dataset does not have an identifier column, so we simply use row index as ID...
card_data_with_id = data.reset_index() # adds an `index` column for identifier.
start_train_set, start_test_set = split_train_test_by_id(card_data_with_id, 0.2, "index") # 80% for Training and 20% for Testing

print(len(start_train_set), "for training Data", len(start_test_set), "for testing Data")
print("Verification of total instances in Data set: ",len(start_train_set)+len(start_test_set))

# Create train/test indices to split data in train/test sets
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


# Define a model generator
def generate_model():
    _model = Sequential()
    _model.add(Dense(22, input_dim=28))
    _model.add(SReLU())
    _model.add(Dropout(0.2))
    _model.add(Dense(1, activation='sigmoid'))
    _model.compile(loss='binary_crossentropy', optimizer='adam')
    return _model
