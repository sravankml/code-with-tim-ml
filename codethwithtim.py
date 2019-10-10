import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from matplotlib import style



data = pd.read_csv('student-mat.csv',sep=';')
data = data[['G1','G2','G3','studytime','failures','absences']]
predict = 'G3'
x = np.array(data.drop([predict],1)) 
y = np.array(data[predict])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
'''
#to find the best model we can itterate through and find the best model 
best =0

for _ in range(700):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
    linear = LinearRegression()
    linear.fit(x_train,y_train)
    #this will give the accuracy of the model
    acc = linear.score(x_test,y_test)
    print("Accuracy: " + str(acc))
    if best < acc:
        best=acc
        with open('studentgrades.pickle','wb') as f:
            pickle.dump(linear,f)
'''
# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
# import pdb; pdb.set_trace()
#To the coeifcent of the variabel 
print('coefficent',linear.coef_)

# To see the y interseft 
print('intercept',linear.intercept_)

#To see the prediction of the trained model we can check that using our x_test
prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x],x_train[x],y_train[x])

plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()