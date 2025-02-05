#libraris 
import numpy as np 
#import class layers 
from layers.First_Layers import Layer


class NeuralNetwork : 
    
    def __init__(self):
         self.layers = []
         self.loss_list = []

    def add_layer(self, num_neuron , input_size):
        if not self.layers :
            self.layers.append(Layer(num_neuron,input_size))
        else :
            previus_outut_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neuron,previus_outut_size))
        
        
    def forward (self , inputs):
        for layers in self.layers :
            inputs = layers.forward(inputs)
        return inputs
    
    def backward (self ,loss_gradient , learning_rate ):
        for layers in reversed(self.layers) :
            loss_gradient =  Layer.backward(loss_gradient, learning_rate)
    
    
    def train (self , x ,y , epochs = 1000, learning_rate=0.1  ):
        for epochs in range(epochs):
            loss = 0  
            for i in range(len(x)):
                output = self.forward(x[i])
                loss +=  np.mean((y[i]-output) ** 2 )
                loss_gradient = 2 * ( output - y[i])
                self.backward(loss_gradient ,learning_rate)
            loss /= len(x)
            self.loss_list.append(loss)
            if epochs % 100 == 0:
                print(f"Epoch : {epochs} , loss {loss}") 
        
    def predict (self , x ):
        predictions = []
        for i in range(len(x)):
               predictions.append(self.forward(x[i]))
        return np.array(predictions)