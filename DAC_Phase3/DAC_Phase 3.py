import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#read the csv file.
covid=pd.read_csv("Covid_19_cases4.csv")
df=pd.DataFrame(covid)
X=list(df.iloc[:,0])
Y=list(df.iloc[:,1])

#plot the bar graph.
plt.bar(X,Y,color='g')
plt.title("covid graph")
plt.xlabel("covid.cases")
plt.ylabel("covid.deaths")
plt.show()

covid.describe()
covid.info()
x=covid.drop("deaths",axis=1)
y=covid['deaths']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)
x_train
x_train.shape
x_test.shape
y_train.shape
y_test.shape
display(covid.drop_duplicates())

#plot the graph.
plt.plot(covid.cases,covid.deaths)
plt.xlabel('covid.cases')
plt.ylabel('covid.deaths')
plt.title('covid graph')
plt.show()
