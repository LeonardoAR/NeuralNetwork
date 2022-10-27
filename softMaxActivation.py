# #dantes estavamos a usar a rectified linear activation function mas não tinhamos o que fazer com o numeros negativos por isso vamos usar exponenciação
# import math
# layer_output = [4.8, 1.21, 2.385]
#
# #E = 2.71828182846
# E = math.e
# #print(E)
#
# #criar uma lista vazia para albergar o layer_output do fim da exponenciação
# exp_value = []
#
# #correr o layer_output e em cada casa dele pegar no numero e fazer a exponenciação
# for output in layer_output:
#     exp_value.append(E**output)
#
# print(exp_value)
#
# #
# #  a seguir á exponenciação o que temos de fazer é a normalização dos valores
# #
# #
# #
# norm_base = sum(exp_value)
# norm_values = []
#
# for value in exp_value:
#     norm_values.append(value / norm_base)
#
# print(norm_values)
# print(sum(norm_values))
# #---------------------------------------------
# #-------------------------------------------------
# #      a mesma coisa mas com numpy
# #------------------------------------------------
# #---------------------------------------------------
# import numpy as np
#
# layer_outputs = [4.8, 1.21, 2.385]
#
# exp_values = np.exp(layer_outputs)
#
# normal_values = exp_values / np.sum(exp_values)
#
# print("numpy output")
# print(normal_values)
# print(sum(normal_values))
#-----------------------------------------------------
#-----------------------------------------------------
#-----------------------------------------------------
#-----------------------------------------------------
#a mesma coisa mas para usar com conjuntos de inputs
import numpy as np
import nnfs

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
print(exp_values)

# axis=0 -> colum    axis=1 -> rows
#keepdims=True -> to get the same shape as the matrix from layer_outputs
normal_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(normal_values)
#-----------------------------------------------------
#-----------------------------------------------------um problema com exponenciação é o overflow(os numeros ficarem demasiado grandes)
#-----------------------------------------------------
#-----------------------------------------------------
#----------overflow prevention------------------------