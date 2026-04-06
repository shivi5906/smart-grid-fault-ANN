import sys
print(sys.executable)
import numpy as np 

import pandas as pd
df = pd.read_csv("smart_grid_stability_augmented.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

irirs = load_iris()

scaler = MinMaxScaler() 
X = pd.DataFrame(irirs.data , columns= irirs.feature_names)

Y = pd.Series(irirs.target)


x = scaler.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(
    x , Y, test_size=0.2, random_state=42
)

