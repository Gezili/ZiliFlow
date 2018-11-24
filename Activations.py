import numpy as np

class Activation:
    
    '''
    Perform multiplication in-place 
    '''
    def ReLU(arr):
        return np.maximum(arr, 0, arr)
        
class CostFunc:
    
    def Softmax(arr):
        
        #For numerical stability
        
        max_arr = np.max(arr)
        arr = np.exp(arr - max_arr)
        sum_arr = np.sum(arr)
        return arr/sum_arr
        
        #TODO support batch input

        

class Layer:
    
    def __init__(self, Neurons, Activation):
        
        self.Activation = Activation
        self.Neurons = np.zeros(Neurons)
        self.NeuronCount = Neurons
        
    '''
    Default initialization from 
    https://stats.stackexchange.com/questions/229885/whats-the-recommended-
    weight-initialization-strategy-when-using-the-elu-activat    
    '''
    
    def InitializeWeights(self, NeuronsPrevLayer):
        
        bound = np.sqrt(1.55/NeuronsPrevLayer)
        
        self.Weights = np.random.uniform(
                -bound, bound,
                [self.NeuronCount, NeuronsPrevLayer]
            )
        self.Biases = np.random.uniform(
                -bound, bound,
                [1, self.NeuronCount]
            )
        
class Graph:
    
    def __init__(self, InputDim, OutputDim, *args):
        
        self.InputDim = InputDim
        self.OutputDim = OutputDim
        
        self.Input = np.zeros(InputDim)
        self.Output = np.zeros(OutputDim)
        self.Graph = [self.Input]
        for i, arg in enumerate(args):
            #We need to make sure that the arguments are unique, otherwise Python will 
            #append too many 
            self.Graph.append(arg)
            if i >= 1:
                assert arg not in args[0:i]
                    
            
        self.Graph.append(self.Output)
        self.WeightInitializer()
            
    def WeightInitializer(self):

        self.Graph[1].InitializeWeights(self.InputDim) 
        for i in range(2, len(self.Graph) - 1):
            
            neurons_prev = self.Graph[i - 1].NeuronCount
            self.Graph[i].InitializeWeights(neurons_prev)
        
if __name__ == '__main__':
    
    l = Layer(12, Activation.ReLU)
    t = Layer(12, Activation.ReLU)
    p = Layer(12, Activation.ReLU)
    g = Graph(5, 5, l, t)