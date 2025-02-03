# My Neural Network Project

This repository includes an explanation of neural networks in Spanish and English. The explanation in English is simple, while the Spanish one is more detailed.

## Neuron 

An artificial neuron is inspired by the biological neuron. Its primary function is to receive multiple inputs, apply weights and biases, and compute an output. The Neuron class in this project contains activation functions and processes the outputs accordingly.

### Fuction the class Neuronal 

ubicacion de la clase :
Neural_Network/Neuronas/First_Neuron.py

La clase Neuron simula el comportamiento de una neurona real.

### Fuction Inicialización (__init__) Atributos
Atributos :

`Pesos (weight)`: Determinan la importancia de cada entrada.
`Sesgo (bias)`: Permite ajustar la salida de la neurona.
`Salida (output)`: Almacena el resultado después de aplicar la función de activación.
`Entradas (inputs)`: Guarda la entrada recibida.
`Gradientes (dweight, dbias)`: Se usan en el proceso de ajuste de los pesos durante el aprendizaje.

### Fuction Activate(self, x)
    
Esta función aplica la función de activación Sigmoide:

Convierte cualquier valor en un rango entre 0 y 1.
Se usa en redes neuronales para introducir no linealidad y decidir la activación de la neurona.

    ```bash
       def activate(self, x):
        """
        Applies the sigmoid activation function.
        
        Parameters:
            x (float): Weighted sum of inputs.
        Returns:
            float: Activated output in the range (0,1).
        """
        return 1 / (1 + np.exp(-x))
    ```
### Fuction derivate_Activate(self, x)
    
Calcula la derivada de la función Sigmoide, que es útil para la retropropagación.
La derivada de la sigmoide se calcula así:`σ (x)=σ(x)⋅(1−σ(x)) `
    
    ```bash
        def derivate_activate(self, x):
        """
        Computes the derivative of the sigmoid function.
        
        Parameters:
            x (float): Activated value.
        Returns:
            float: Derivative of the sigmoid function.
        """
        return x * (1 - x)
    ```
### Fuction forward(self, inputs)
    
Calcula la salida de la neurona en la fase de propagación hacia adelante (forward pass):

Multiplica las entradas por sus pesos.
Suma el sesgo.
Aplica la función de activación

    ```bash
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
    ```
### Fuction forward(self, inputs)
    
Realiza el ajuste de pesos mediante retropropagación:

Calcula la derivada del error con respecto a la salida.
Ajusta los pesos y el sesgo con descenso de gradiente.

    ```bash
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

    ```

## Layers 