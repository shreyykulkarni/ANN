import numpy as np
import math 
from random import random 
import  matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(0)
 




class ANN:
    
    def __init__(self,activation=['Logistic','Logistic'], layers=[1,3,1]):
        assert(len(layers) == len(activation)+1)
        self.activation = activation;
        self.layers = layers;
        self.weights = []
        self.bias = []
        
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(self.layers[i+1],self.layers[i]))
            self.bias.append(np.random.randn(self.layers[i+1],1))
        
            
            
    @staticmethod
    def Activation_Func(x,name,deriv=False):
        if (name == 'Logistic'):
            if (deriv==True):
                return (np.exp(x)/(1+np.exp(x)))*(1-(np.exp(x)/(1+np.exp(x))))
            else:
                return np.exp(x)/(1+np.exp(x)) 
        
        elif (name == 'Tanh'):
            if (deriv==True):
                return 1 - ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))**2
            else:
                return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        
        elif (name == 'Relu'):
            if (deriv==True):
                x[x<0]=0
                x[x>0]=1
                return x
            else: 
                x[x<0]=0
                return x
                

    
        
        
    def FeedForward(self, x):
        input = np.copy(x).reshape(1,-1)
        output = []
        intermediate = [input]
        
        for i in range(len(self.weights)):
            output.append(self.weights[i].dot(input)+self.bias[i])
            #print(np.array(self.weights).shape)
            input = self.Activation_Func(output[-1],self.activation[i],deriv=False)
            intermediate.append(input)
        return(output,intermediate)
    
    
    
    
    def Backpropagation(self, y, output, intermediate):
        #The actual outputs
        deltas = [None] * len(self.weights)
        #Exclusive delta for last layer
        deltas[-1] = (y-intermediate[-1])*self.Activation_Func(output[-1],self.activation[-1],deriv=True)
        
        Dw = []
        Db = [] 

        for i in reversed(range(len(deltas)-1)):
            #Generalaized delta for the intermediate and input layers
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.Activation_Func(output[i],self.activation[i],deriv=True)) 
            #WHAT ABOUT NYU, I.E. 1/BATCH-SIZE
            batch_size = np.array(y).shape[0]
            Db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
            Dw = [d.dot(intermediate[i].T)/float(batch_size) for i,d in enumerate(deltas)]
            # Dw = np.ones(np.array(Dw).shape)*Beta*Dw + Dw
        return Dw, Db
    
    
    
    
    def MAPE(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        E = np.ones((y_true.shape[0],), dtype=float)*0.001
        M = np.absolute(y_true - y_pred) / np.absolute(y_true + E)
        return np.mean(M*100)/(len(y_true))
       

    
    def train(self,x,y,batch_size=100, epochs=10, lr=0.001):
        E=[]
        
        for e in range(epochs):
            i = 0
            y1=[]
            while i<len(y):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                i = i+batch_size

                output, intermediate = self.FeedForward(x_batch)
                Dw, Db = self.Backpropagation(y_batch, output, intermediate)
                
                self.weights = [w+lr*dweight for w,dweight in zip(self.weights, Dw)]
                self.bias = [w+lr*dbias for w,dbias in zip(self.bias, Db)]
                
                y1.append(intermediate[-1].T)
                
            y1 = np.array(y1).flatten()
            y_pred = np.transpose(y1)
            E.append(self.MAPE(y,y_pred))
        return (E, y_pred,self.weights,self.bias)
    



            
pi = int(math.pi)
Y_train=[]
X_train=[]
X_valid=[]

X_train = np.random.uniform(-1, 1, 1024)*(int(2*math.pi))
Y_train = np.sin(X_train)                                         
    
X_valid = np.random.uniform(-1, 1, 300)*(int(2*math.pi))
Y_valid = np.sin(X_valid)             



Neural = ANN(activation=['Tanh','Logistic'], layers=[1,5,1])
E, y_pred,Weig,bia = Neural.train(x=X_train,y=Y_train,epochs=1000, batch_size=64, lr = 0.004)
H,J = Neural.FeedForward(X_train)

# L=[]
# for i in range(len(E)):
#     L.append(i)
# plt.scatter(L,E,color='RED')
# plt.xlabel("Epochs")
# plt.ylabel("MAPE")
# print(np.mean(E))
# plt.show()


# plt.scatter(X_train.flatten(), Y_train.flatten(),color='BLUE')
# plt.scatter(X_train.flatten(), (J[-1].flatten()),color='RED')
# plt.xlabel("Output")
# plt.ylabel("Input")
# plt.legend(['Actual','Prediction'])
# print("Loss:")
# plt.show()


        



