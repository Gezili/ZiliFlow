import numpy as np

class util:
    
    def MakeOneHot(arr, num_classes):
        
        assert num_classes > np.max(arr)
        one_hot_vec = np.zeros([len(arr), num_classes])
        one_hot_vec[np.arange(len(arr)), arr] = 1
        return one_hot_vec
        
    def NormalizeDimensions(Data):
        
        var = np.var(Data, axis = 0)
        mean = np.mean(Data, axis = 0)
        return (Data - mean)/var
       
class Loss:
    
    def L2(Pred, Label):
        return np.sum(np.square(Pred - Label))
        
    '''
    See https://deepnotes.io/softmax-crossentropy for resource used
    and gradients
    '''
        
    def CrossEntropyWithSoftmax(Pred, Label, input_size = -1, grad = False):
        
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
    def ReLU(arr, input_size = -1, grad = False):
        
        #Set gradative to 0 at x = 0 for sparser matrix
        arr = np.maximum(arr, 0, arr)
        if not grad:
            return arr
        else:
            arr[arr > 0] = 1
            grad = np.zeros([len(arr), input_size])
            for i in range(input_size):
                grad[0:len(arr), i] = arr
                
            return grad, grad[:, 0]
        
    def Linear(arr, input_size = -1, grad = False):
        
        if not grad:
            return arr
        else:
           return np.ones([6, input_size]), np.ones(6)
        
        
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
    
    def __init__(self, Neurons, Activation, Type, **kwargs):
        
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
        
        self.Graph.append(self.Output)
        self.NumLayers = len(self.Graph)
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
        
        for i in range(self.NumLayers - 1, 0, -1):
            
            output = self.Graph[i].Output
            #print(self.Graph[i].Type)
            input = self.Input

            if self.Graph[i].Type is 'Output':        
                grad_weights, grad_biases = Loss.CrossEntropyWithSoftmax(output[0], label,\
                self.Graph[i].Weights.shape[1], grad = True)
            else:
                #print(self.Graph[i].Weights.shape[1])
                grad_weights, grad_biases = self.Graph[i].Activation(grad_biases,\
                self.Graph[i].Weights.shape[1], grad = True)
                
                #print(grad_weights)
            
            if i is not 0:
                out = self.Graph[i - 1].Output
            else:
                out = self.Input
            
            shape_output = self.Graph[i].Weights.shape
            output = np.zeros(shape_output)
            
            for j in range(shape_output[0]):
                output[j, 0:shape_output[1]] = out
            
            self.Graph[i].Weights = \
            self.Graph[i].Weights -\
            grad_weights*output*learning_rate
        
            if self.Graph[i].Type is not 'Output':
                self.Graph[i].Biases = \
                self.Graph[i].Biases -\
                grad_biases*learning_rate
    
class KNN:
    
    #Distance metric can be defined individually as well
    def __init__(self, Data, Labels, K, DistanceMetric):
        
        assert type(Labels) is list
        assert len(Labels) is len(Data)
        assert type(K) is int
        
        if type(Data) is not np.ndarray:
            Data = np.array(Data)
        self.Data = Data
        self.K = K
        self.DistanceMetric = DistanceMetric
        
    def AddDataPoint(self, Datapoint, Label):
        
        if type(Datapoint) is not np.ndarray:
            Datapoint = np.array(Datapoint)
        assert len(Datapoint.shape) <= 2
        if len(Datapoint.shape) is 2:
            self.Data = self.Data.tolist()
            for datum in Datapoint:
                self.Data.append(datum.tolist())
            self.Data = np.array(self.Data)
        else:
            
            try:
                if not self.Data:  
                    self.Data = np.expand_dims(Datapoint, axis = 0)
            except:
                self.Data = np.append(self.Data, np.expand_dims(Datapoint, axis = 0), axis = 0)
                
    def Classification(self, to_classify)
    
        closest_pairs = np.ones(self.K)*np.inf
        labels = []
        for i in range(self.K):
            labels.append('')
        
        largest_dist = np.max(closest_pairs)
        
        for i, datum in enumerate(self.Data):
            
            if distance_metric is 'Euclidean' or 'L2':
                distance = DistanceMetric_L2(datum, to_classify)
            if distance < largest_dist:
                idx = np.argmax(closest_pairs)
                closest_pairs[np.argmax(idx)] = distance
                labels[idx] = self.Labels[i]
                largest_dist = np.max(closest_pairs)
                
        return max(labels,key = labels.count)
    

    def DistanceMetric_L2(Point, Classifier):
        return np.sqrt(np.sum((a - b)**2))

if __name__ == '__main__':
    
    a = Layer(6, Activation.ReLU, 'Dense')
    b = Layer(6, Activation.ReLU, 'Dense')
    c = Layer(6, Activation.Linear, 'Linear ')
    output = Layer(4, CostFunc.Softmax, 'Output')
    
    g = Graph(4, output, a, b, c)

    for i in range(10000):
        
        int = np.random.randint(0, 4)
        arr = np.zeros(4)
        arr[int] = 1
        
        g.RunInferenceStep(np.array(arr))
        g.RunBackpropStep(np.array(arr), 1e-3)
    
    print(g.Graph[g.NumLayers - 1].Output)