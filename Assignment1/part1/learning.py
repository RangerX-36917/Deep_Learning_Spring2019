import numpy as np
import copy
import matplotlib.pyplot as plt
import perceptron as pt
mean = [-5, 5]
cov = [[5,0],[0,5]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.plot(x, y, 'x')
plt.axis('equal')
data1 = []
for i in range(100):
    data1.append([[x[i],y[i]],1])
mean = [5, 5]
cov = [[5,0],[0,5]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
data2 = []
for i in range(100):
    data2.append([[x[i],y[i]],-1])
dataTr = copy.copy(data1)
dataTr += data2
np.random.shuffle(dataTr)
dataTest = dataTr[:40]
dataTr = dataTr[40:]
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


_perceptron = pt.Perceptron(2)
vectors = []
labels = []
for i in range(len(dataTr)):
    vectors.append(dataTr[i][0])
    labels.append(dataTr[i][1])
#_perceptron.forward([0,1])
for i in range(200):
    _perceptron.train(vectors, labels)
print("w: ",_perceptron.w)

correct = 0
for i in range(len(dataTest)):
    result = _perceptron.forward(dataTest[i])
    #print("result: ", result)
    if(result == dataTest[i][1]):
        correct += 1
print(correct,'/',len(dataTest))
#_perceptron.forward([0,1])
#_perceptron.train(vectors, labels)
print("test: ", correct*100/len(dataTest),'%')
correct = 0
for i in range(len(dataTr)):
    result = _perceptron.forward(dataTr[i])
    #print("result: ", result)
    if(result == dataTr[i][1]):
        correct += 1
print(correct,'/',len(dataTr))
#_perceptron.forward([0,1])
#_perceptron.train(vectors, labels)
print("training:", correct*100/len(dataTr),'%')
