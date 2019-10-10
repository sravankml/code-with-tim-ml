# let start with knn fot the classfying the data 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.model_selection import  train_test_split

data = pd.read_csv('car.data')
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
clas = le.fit_transform(list(data['class']))

predict = 'class'
x = list(zip(buying,maint,door,persons,lug_boot,safety,clas))
y = list(clas)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print(acc)

#prediction start here 
predicted = model.predict(x_test)
name = ['unacc','acc','good','vgood']
for x in range(len(predicted)):
    print("predicted:",name[predicted[x]],"data",x_test[x],'Actual',name[y_test[x]])

