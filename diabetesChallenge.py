import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = pd.read_csv('Diabetes_XTrain.csv')
y_train = pd.read_csv('Diabetes_YTrain.csv')
x_test = pd.read_csv('Diabetes_XTest.csv')
x_train = x_train.values
y_train = y_train.values
y_train = y_train.reshape((-1,))
x_test = x_test.values

print(x_test.shape)

def calculateDist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(x,y,query,k=5):
    vars = []
    m = x.shape[0]

    for i in range(m):
        dist = calculateDist(x[i],query)
        vars.append((dist , y[i]))

    vars = sorted(vars)
    vars = vars[:k]

    vars = np.array(vars)
    new_vars = np.unique(vars[: , 1], return_counts=True)
    index = new_vars[1].argmax()
    return new_vars[0][index]

def drawImg(sample):
    img = sample.reshape((4,2))
    plt.imshow(img , cmap='gray')
    plt.show()

singleans = []
for j in range(x_test.shape[0]):
    prediction = int(knn(x_train,y_train,x_test[j]))
    singleans.append(prediction)
    # print(prediction)
answers = np.array(singleans)
answers = pd.DataFrame(answers , columns=['answers'])
answers.to_csv('answers.csv',index=False)
print(answers)
# drawImg(x_test[0])