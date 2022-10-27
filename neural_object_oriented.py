#------------------------------------------------------
#----------------------objects-------------------------
# vamos começar a usar programação por objectos para tornar isto modular
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#np.random.seed(0)

nnfs.init()


# X = [[1.0, 2.0, 3.0, 2.5],  # standard in machine learning that the input feature sets is going to be denoted by 'X'
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

#X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weightsO = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biasesO = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weightsO)+self.biasesO

##################################antes de o DataSet#######################################
#layer1 = LayerDense(4, 5)   #n_inputs from layer 2 needs to be the same size of the outputs from layer 1
#layer2 = LayerDense(5, 2)

#layer1.forward(X)
#print(layer1.output)
#layer2.forward(layer1.output) # its going to give 3 exemples because  there are 3 inputs or X
#print(layer2.output)

#------------------------------------------------------|
#-----------------activation functions-----------------|
#------------------------------------------------------|            functions used for activation functions
#---------the weights and biases are passed through----|                → step function
#---------the activation function and this activation__|                → sigmoid function          -- better than step because of the granularity of the function
# _______or not, is whats going to go to the input of__|                → rectified linear function -- its faster than sigmoid
# ____________________the next layer___________________|
#↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓|
# rectified linear activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



##################################depois do DataSet##########################################
# o 4 do layer1 passa para um 2 porque os feature set do DataSet só têm 2 features
# layer1 = LayerDense(2, 5)
# activation1 = Activation_ReLU()
#
#
# layer1.forward(X)
#
# print(layer1.output)
# activation1.forward(layer1.output)
# print("second print")
# print(activation1.output)

class Activation_Softmax:
    def forward(self, inputs):
        #subtrair o max dos inputs é o overflow prevention
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1, keepdims=True)
        self.output = probabilities

# X, y = spiral_data(samples=100, classes=3)
# dense1 = LayerDense(2, 3)#
# activation1 = Activation_ReLU()
#
# dense2 = LayerDense(3, 3)
# activation2 = Activation_Softmax()
#
# dense1.forward(X)
# activation1.forward(dense1.output)
# dense2.forward(activation1.output)
# activation2.forward(dense2.output)
# print(activation2.output[:5])

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        dataBatch_loss = np.mean(sample_losses)
        return dataBatch_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #scaler values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #1hot
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likekihoods = -np.log(correct_confidences)
        return negative_log_likekihoods

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)#
activation1 = Activation_ReLU()

dense2 = LayerDense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])


loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

