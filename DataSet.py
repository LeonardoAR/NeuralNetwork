import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------|
#-------------------create_data------------------------|
#---------just a function to create some data----------|
#------------------------------------------------------|
np.random.seed(0)
# how many feature sets(points) per how many classes we have
#each feature set is 2 descriptive features 'x' and 'y' to be represented in the plots
# in the case bellow we are making 3 classes of 100 feature sets each
def create_data(points, classes):
    X_data = np.zeros((points * classes, 2))    # o 'X' é grande porque é a convenção para data inputs(training data)
    y_data = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)    # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points)+np.random.randn(points)*0.2
        X_data[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y_data[ix] = class_number
    return X_data, y_data

#------------------------------------------------------|
#------creating a plot to see the data generated-------|
#------------------------------------------------------|

print("here")
X, y = create_data(100, 3)
#-----------plot sem cores-----------------------------|
plt.scatter(X[:, 0], X[:, 1])
plt.show()
#-------------plot com cores---------------------------|
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
