import numpy as np
import math 
from random import random 
import pandas as pd
import  matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(0)
 


class ANN:
    
    def __init__(self,activation=['Logistic','Logistic'], layers=[4,10,1]):
        assert(len(layers) == len(activation)+1)
        self.activation = activation;
        self.layers = layers;
        self.weights = []
        self.bias = []
        
        A=[]
        B=[]
        for i in range(len(layers)-1):
            A.append(np.random.randn(self.layers[i+1],self.layers[i]))
            B.append(np.random.randn(self.layers[i+1],1))  
        self.weights=A
        self.bias=B
            
            
    @staticmethod
    def Activation_Func(x,name,deriv=False):
        if (name == 'Logistic'):
            if (deriv==True):
                return (math.e**(x)/(1+math.e**(x)))*(1-(math.e**(x)/(1+math.e**(x))))
            else:
                return 1 / (1 + math.e**(-x) )
        
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
        input = np.transpose(np.copy(x))
        output = []
        intermediate = [input]
       
        for i in range(len(self.weights)):
            output.append(np.dot(self.weights[i],input)+self.bias[i])
            input = self.Activation_Func(output[-1],self.activation[i],deriv=False)
            intermediate.append(input)
        return(output,intermediate)
    
    
    
    
    def Backpropagation(self, y, output, intermediate):
        deltas = [None] * len(self.weights)
        deltas[-1] = self.Activation_Func(output[-1],self.activation[-1],deriv=True)*(y.T-intermediate[-1])
        Dw = []
        Db = []
        Nyu = 1
        Lambda = 0.4
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.Activation_Func(output[i],self.activation[i],deriv=True)) 
            batch_size = np.array(y).shape[0]
            Dw = [d.dot(intermediate[i].T)*Nyu for i,d in enumerate(deltas)]
            Db = [d.dot(np.ones((batch_size,1)))*Nyu for d in deltas]
            Vw,Vb = self.SGD(Dw,Db)
            Sw,Sb = self.RMS_Prop(Dw, Db)
            W,B = self.ADAM(Vw, Sw, Vb, Sb)
            W += np.ones(np.array(self.weights).shape)*Lambda*(self.weights)
            B += np.ones(np.array(self.bias).shape)*Lambda*(self.bias)
        return W,B
    
    

    
    
    def SGD(self, Dw, Db):
        Dw1 = []
        Dw2 = []
        Db1 = []
        Db2 = []
        Beta=float(0.8)
        Dw1 = np.zeros(np.array(Dw).shape)*(1-Beta)
        Dw2 = np.ones(np.array(Dw1).shape)*(Beta)*Dw1 + np.ones(np.array(Dw).shape)*(1-Beta)*Dw
        Db1 = np.ones(np.array(Db).shape)*(1-Beta)*Db
        Db2 = np.ones(np.array(Db1).shape)*(Beta)*Db1 + np.ones(np.array(Db).shape)*(1-Beta)*Db
        Db2 += np.ones(np.array(Db).shape)*(1-Beta)*Db 
        Dw2 += np.ones(np.array(Dw).shape)*(1-Beta)*Dw
        return Dw2,Db2
        
    
    
    
    def RMS_Prop(self, Dw, Db):
        Dw1 = []
        Dw2 = []
        Db1 = []
        Db2 = []
        Beta=float(0.8)
        Dw1 = np.zeros(np.array(Dw).shape)*(1-Beta)
        Dw2 = np.ones(np.array(Dw1).shape)*(Beta)*Dw1 + np.ones(np.array(Dw).shape)*(1-Beta)*(np.square(Dw))
        Db1 = np.zeros(np.array(Db).shape)*(1-Beta)
        Db2 = np.ones(np.array(Db1).shape)*(Beta)*Db1 + np.ones(np.array(Db).shape)*(1-Beta)*(np.square(Db))
        Db2 = Db/((Db2)**(0.5))
        Dw2 = Dw/((Dw2)**(0.5))
        return Dw2,Db2
        
        
        
    def ADAM(self,Vw,Sw,Vb,Sb):
        W=[]
        B=[]
        W1=[]
        B1=[]
        Alpha=float(-0.01)
        W1 = Alpha*(Vw)/(Sw)
        B1 = Alpha*(Vb)/(Sb)
        # W = np.zeros(np.array(W1).shape)
        # B = np.zeros(np.array(B1).shape)
        # W += W1
        # B += B1
        return W1, B1
    
    
    
        
    def MAPE(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        E = np.ones((y_true.shape[0]), dtype=float)*0.001
        M = np.absolute(y_true - y_pred) / np.absolute(y_true + E)
        return (np.mean(M)/(len(y_true)))*100
       
        
       
       
    def RMS(self,y_true,y_pred):
        for i in range(len(y_true)):
            R=[]
            A=[]
            A = ((Y_train[i]-y_pred[i])**2)/(len(Y_train))
            R.append(A)
        return (np.mean(R))
    
    
    
    
    def train(self,x,y,xv,yv,batch_size=100, epochs=10, lr=0.004):
        E=[]
        Ev=[]
        rms=[]
        for e in range(epochs):
            print("Epoch number is:",e)
            i = 0
            j = 0
            y1=[]
            y2=[]
            u=[]
            v=[]
            while i<len(y):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                i = i+batch_size

                output, intermediate = self.FeedForward(x_batch)
                Dw, Db = self.Backpropagation(y_batch, output, intermediate)
                
                self.weights = [w+lr*dweight for w,dweight in zip(self.weights, Dw)]
                self.bias = [w+lr*dbias for w,dbias in zip(self.bias, Db)]
                
                y1.append(intermediate[-1].T)
                u.append(self.weights)
                v.append(self.bias)
            y1 = np.array(y1).flatten()
            w = u
            z = v
            y_pred = np.transpose(y1)
            E.append(self.MAPE(y,y_pred))
            rms.append(self.RMS(y,y_pred))
            
            
            while j<len(yv):
                
                x_batch = xv[j:j+batch_size]
                y_batch = yv[j:j+batch_size]
                output, intermediate = self.FeedForward_valid(x_batch,u[int(j/batch_size)],v[int(j/batch_size)])
                j = j+batch_size
                y2.append(intermediate[-1].T)
            y2 = np.array(y2).flatten()
            y2_pred = np.transpose(y2)
            Ev.append(self.MAPE(yv,y2_pred))
            
        return (E, y_pred,Ev,y2_pred,w,z,rms)
    
    

    
    
    
    def FeedForward_valid(self,x,weights1,bias1):
        input = np.transpose(np.copy(x))
        output = []
        intermediate = [input]
        for i in range(len(weights1)):
            output.append(np.dot(weights1[i],input)+bias1[i])
            input = self.Activation_Func(output[-1],self.activation[i],deriv=False)
            intermediate.append(input)
        return(output,intermediate)
    
    

    
    
    def Test(self,x,y,weight,bias):
        input = np.transpose(np.copy(x))
        output = []
        intermediate = [input]
        y1=[]
        Et=[]
        weight = weight[-1]
        bias = bias[-1]
        for i in range(len(weight)):
            output.append(np.dot(weight[i],input)+bias[i])
            input = self.Activation_Func(output[-1],self.activation[i],deriv=False)
            intermediate.append(input)
        y1.append(intermediate[-1].T)
        y1 = np.array(y1).flatten()
        y_pred = np.transpose(y1)
        Et.append(self.MAPE(y,y_pred))
        return(Et)

            
        

df= pd.read_excel(r'C:\Users\Atharv Kulkarni\Desktop\Semester 7\Deep Learning\Assignment 1\CCPP\CCPP\Dataset.xlsx')
X = df[['AT','V','AP','RH']]
Y = df[['PE']]

X=X.apply(lambda x:(x - x.min(axis=0))/(x.max(axis=0) - x.min(axis=0)))
Y=Y.apply(lambda y:(y - y.min(axis=0))/(y.max(axis=0) - y.min(axis=0)))
X=np.array(X.values)
Y=np.array(Y.values)

Split_in = (np.split(X, [int((23/37) * len(X)), int((31/37)* len(X))]))
X_train = Split_in[0]
X_valid= Split_in[1]
X_test = Split_in[2]

Split_out = (np.split(Y, [int((23/37) * len(Y)), int((31/37) * len(Y))]))
Y_train = Split_out[0]
Y_valid= Split_out[1]
Y_test = Split_out[2]
     



Neural = ANN(activation=['Logistic', 'Tanh'], layers=[4,2,1])
E, y_pred,E1,y_pred1,u,v,rms = Neural.train(x=X_train,y=Y_train,xv=X_valid,yv=Y_valid,epochs=10, batch_size=256, lr = 0.01)


E2 = Neural.Test(X_test,Y_test,u,v)


# #RMS
# I=[]
# for i in range(len(rms)):
#     I.append(i)
# plt.scatter(I,rms,color='TEAL',s=30)
# plt.xlabel("Epochs")
# plt.ylabel("RMS Error")
# plt.legend(['RMS'])
# plt.show





# L=[]
# for i in range(len(E)):
#     L.append(i)
# plt.scatter(L,E,color='blue')
# print(E,L)

# L1=[]
# for i in range(len(E1)):
#     L1.append(i)
# plt.scatter(L1,E1,color='mediumslateblue')
# print(E1,L1)

# L2=[]
# for i in range(len(E2)):
#     L2.append(i)
# plt.scatter(L2,E2,color='darkslateblue',s=100)
# print("MAPE test is:",E2,L2)


# plt.legend(['Training_Adam','Validaiton_Adam','Test_Adam'])
# plt.xlabel("Epochs")
# plt.ylabel("MAPE")
# plt.show()


        




