
import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib.pyplot as plt
covid=pd.read_csv("Covid_19_cases4.csv")
covid

covid.describe()

covid.info()

x=covid.drop("deaths",axis=1)
y=covid['deaths']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)
x_train

x_train.shape

x_test.shape

y_train.shape

y_test.shape

y_test

x_test

x

display(covid.drop_duplicates())

plt.plot(covid.cases,covid.deaths)
plt.xlabel('covid.cases')
plt.ylabel('covid.deaths')
plt.title('covid graph')
plt.show()
