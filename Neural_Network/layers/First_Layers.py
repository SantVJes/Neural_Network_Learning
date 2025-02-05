# Libraries
import numpy as np 

# Import the Neuron class from the corresponding module
from Neuronas.First_Neuron import Neuron

class Layer:
    """
    A layer in the neural network.

    Attributes:
        num_neurons (int): The maximum number of neurons in this layer.
        num_inputs (int): The number of inputs for the lptyayer in the neural network.
        neurons (list): A list containing neuron objects in the layer.
    """

    def __init__(self, num_neurons, num_inputs):
        """
        Initializes the layer with a given number of neurons.

        Parameters:
            num_neurons (int): Number of neurons in the layer.
            num_inputs (int): Number of inputs each neuron receives.
        """
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
    
    def forward(self, inputs):
        """
        Performs forward propagation for all neurons in the layer.

        Parameters:
            inputs (np.array): Input values for the layer.

        Returns:
            np.array: Output values from all neurons in the layer.
        """
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, d_outputs, learning_rate):
        """
        Performs backward propagation for all neurons in the layer.

        Parameters:
            d_outputs (np.array): Gradient of the loss with respect to the outputs.
            learning_rate (float): Step size for updating weights.

        Returns:
            np.array: Gradient with respect to the inputs.
        """
        d_inputs = np.zeros_like(self.neurons[0].inputs)
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs


