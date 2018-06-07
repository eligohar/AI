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