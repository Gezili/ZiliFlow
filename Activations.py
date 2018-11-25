import numpy as np

class util:
    
    def MakeOneHot(arr, num_classes):
        
        assert num_classes > np.max(arr)
        one_hot_vec = np.zeros([len(arr), num_classes])
        one_hot_vec[np.arange(len(arr)), arr] = 1
        return one_hot_vec
       
class Loss:
    
    def L2(Pred, Label):
        return np.sum(np.square(Pred - Label))
        
    '''
    See https://deepnotes.io/softmax-crossentropy for resource used
    and gradation
    '''
        
    def CrossEntropy(Pred, Label, input_size = -1, grad = False):
        
        idx = Label.argmax()
        
        if not grad:
            return -np.log(Pred[idx])
        
        assert input_size is not -1
        grad = np.zeros([len(Pred), input_size])
        
        for i in range(input_size):
            
            grad[0:len(Pred), i] = Pred
        grad[idx] -= 1
        
        #Also return updates on Bias
        return grad, grad[:, 0]
        
    
class Activation:
    
    '''
    Perform multiplication in-place 
    '''
    def ReLU(arr, grad = False):
        
        #Set gradative to 0 at x = 0 for sparser matrix
        arr = np.maximum(arr, 0, arr)
        if grad:
            arr[arr > 0] = 1
            
        return arr
        
    def Linear(arr, input_size = -1, grad = False):
        
        if not grad:
            return arr
        #else:
        #   return np.
        
        
class CostFunc:
    
    def Softmax(arr, grad = False):
        
        #For numerical stability
        
        max_arr = np.expand_dims(np.max(arr, axis = 1), axis = 1)
        arr = np.exp(arr - max_arr)
        sum_arr = np.expand_dims(np.sum(arr, axis = 1), axis = 1)
        return arr/sum_arr

class Optimizer:
    
    def SGD(arr, CostFunc):
        pass

class Layer:
    
    def __init__(self, Neurons, Activation, Type):
        
        assert Type is 'Dense' or 'Output'
        
        self.Activation = Activation
        self.Neurons = np.zeros(Neurons)
        self.NeuronCount = Neurons
        self.Type = Type
        self.Output = np.zeros(Neurons)
        self.Biases = np.expand_dims(np.zeros(Neurons), axis = 0)
        
    '''
    Default initialization from 
    https://stats.stackexchange.com/questions/229885/whats-the-recommended-
    weight-initialization-strategy-when-using-the-elu-activat    
    '''
    
    def InitializeWeights(self, NeuronsPrevLayer):
        
        bound = np.sqrt(1.55/NeuronsPrevLayer)
            
        if self.Type is not 'Output':
            self.Biases = np.random.uniform(
                    -bound, bound,
                    [1, self.NeuronCount]
                )
        self.Weights = np.random.uniform(
                -bound, bound,
                [self.NeuronCount, NeuronsPrevLayer]
            )
        
class Graph:
    
    def __init__(self, InputDim, Output, *args):
        
        self.Output = Output
        
        self.InputDim = InputDim
        self.OutputDim = Output.NeuronCount
        
        #self.Input = np.zeros(InputDim)
        
        self.Graph = []
        for i, arg in enumerate(args):
            
            #We need to make sure that the arguments are unique, otherwise Python will 
            #point to the same object in multiple locations on the list
            self.Graph.append(arg)
            assert arg not in args[0:i]
            
        self.NumLayers = i + 2
            
        self.Graph.append(self.Output)
        self.WeightInitializer()
            
    def WeightInitializer(self):

        self.Graph[0].InitializeWeights(self.InputDim) 
        for i in range(1, self.NumLayers):
            
            neurons_prev = self.Graph[i - 1].NeuronCount
            self.Graph[i].InitializeWeights(neurons_prev)
            
    def RunInferenceStep(self, input):
        
        self.Input = input
        self.Graph[0].Output = self.Graph[0].Activation(np.dot(self.Graph[0].Weights, input) + self.Graph[0].Biases)
        
        for i in range(1, self.NumLayers):
            self.Graph[i].Output = self.Graph[i].Activation(np.dot(self.Graph[i].Weights,\
            self.Graph[i - 1].Output[0]) + self.Graph[i].Biases)
            
        return self.Graph[self.NumLayers - 1].Output
        
    def RunBackpropStep(self, label, learning_rate):
        
        output = self.Graph[self.NumLayers - 1].Output
        input = self.Input        
        grad_weights, grad_biases = Loss.CrossEntropy(output[0], label,\
        self.Graph[self.NumLayers - 1].Weights.shape[1], grad = True)
        
        out = self.Graph[self.NumLayers - 2].Output
        shape_output = self.Graph[self.NumLayers - 1].Weights.shape
        output = np.zeros(shape_output)
        
        for i in range(shape_output[0]):
            output[i, 0:shape_output[1]] = out
        
        self.Graph[self.NumLayers - 1].Weights = \
        self.Graph[self.NumLayers - 1].Weights -\
        grad_weights*output*learning_rate
        
        #self.Graph[self.NumLayers - 1].Biases = \
        #self.Graph[self.NumLayers - 1].Biases -\
        #grad_biases*learning_rate
        
if __name__ == '__main__':
    
    a = Layer(12, Activation.ReLU, 'Dense')
    b = Layer(12, Activation.ReLU, 'Dense')
    c = Layer(12, Activation.Linear, 'Linear ')
    output = Layer(5, CostFunc.Softmax, 'Output')
    
    g = Graph(5, output, a, b, c)
    
    g.RunInferenceStep(np.array([1, 2, 3, 4, 5]))
    g.RunInferenceStep(np.array([1, 2, 3, 4, 5]))
    g.RunBackpropStep(np.array([0, 0, 0, 1, 0]), 1e-4)
    