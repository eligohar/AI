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
print(data_tr)