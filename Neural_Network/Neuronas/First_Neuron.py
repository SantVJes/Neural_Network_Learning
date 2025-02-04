#libraris
import numpy  as  np


class Neuron:
    """
    A simple artificial neuron with forward and backward propagation.
    
    Attributes:
        weight (np.array): Weights for the inputs.
        bias (float): Bias value for the neuron.
        output (float): Output value of the neuron.
        inputs (np.array): Stores the last input values.
        dweight (np.array): Gradient of the weight used in backpropagation.
        dbias (float): Gradient of the bias used in backpropagation.
    """
    
    def __init__(self, n_input):
        """
        Initializes the neuron with random weights and bias.
        
        Parameters:
            n_input (int): Number of inputs the neuron receives.
        """
        self.weight = np.random.rand(n_input)
        self.bias = np.random.rand()
        self.output = 0
        self.inputs = None
        self.dweight = np.zeros_like(self.weight)
        self.dbias = 0

    def activate(self, x):
        """
        Applies the sigmoid activation function.
        
        Parameters:
            x (float): Weighted sum of inputs.
        Returns:
            float: Activated output in the range (0,1).
        """
        return 1 / (1 + np.exp(-x))

    def derivate_activate(self, x):
        """
        Computes the derivative of the sigmoid function.
        
        Parameters:
            x (float): Activated value.
        Returns:
            float: Derivative of the sigmoid function.
        """
        return x * (1 - x)

    def forward(self, inputs):
        """
        Computes the forward propagation.
        
        Parameters:
            inputs (np.array): Input values to the neuron.
        Returns:
            float: Output after applying activation function.
        """
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weight) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output

    def backward(self, d_output, learning_rate):
        """
        Computes the backward propagation and updates weights and biases.
        
        Parameters:
            d_output (float): Derivative of the loss with respect to the output.
            learning_rate (float): Step size for updating weights.
        Returns:
            np.array: Gradient with respect to the inputs.
        """
        d_activation = d_output * self.derivate_activate(self.output)
        self.dweight = np.dot(self.inputs.T, d_activation)
        self.dbias = np.sum(d_activation)
        d_input = np.dot(d_activation, self.weight.T)

        # Update weights and bias
        self.weight -= learning_rate * self.dweight
        self.bias -= learning_rate * self.dbias
        
        return d_input


 
if __name__ == "__main__":
       Neuron = Neuron(3)