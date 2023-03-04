import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
dataset = sns.load_dataset('tips')
x=dataset.drop(columns=['sex','tip','smoker','day','time'])
y=dataset['tip']
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
