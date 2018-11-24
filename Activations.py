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
        
        print(NeuronsPrevLayer)
        bound = np.sqrt(1.55/NeuronsPrevLayer)
        
        self.Weights = np.random.uniform(
                -bound, bound,
                [self.NeuronCount, NeuronsPrevLayer]
            )
        print(self.Weights.shape)
        self.Biases = np.random.uniform(
                -bound, bound,
                [1, self.NeuronCount]
            )
        
        
    
class Graph:
    
    def __init__(self, InputDim, OutputDim, *args):
        
        self.InputDim = InputDim
        self.OutputDim = OutputDim
        
        self.Output = np.zeros(OutputDim)
        self.Graph = [self.Input]
        for arg in args:
            self.Graph.append(arg)
            
        self.Graph.append(self.Output)
        self.WeightInitializer()
        
            
    def WeightInitializer(self):
        
        for i in range(2, len(g.Graph) - 1):
            
            neurons_prev = self.Graph[i - 1].NeuronCount
            print(neurons_prev)
            self.Graph[i].InitializeWeights(neurons_prev)
            
        self.Graph[1].InitializeWeights(self.InputDim)
        
if __name__ == '__main__':
    
    l = Layer(64, Activation.ReLU)
    
    #g = Graph(5, 5, l, l, l, l, l, l, l)