import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x)))
    
def relu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j]<0:
                x[i][j]=0
    return (x)
            
            
def initialize_params(X):
    W1=np.random.randn(4,X.shape[0])*0.01
    b1=np.zeros((4,1))
    W2=np.random.randn(1,4)*0.01
    b2=0
    
    params={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return params

def forward_prop(X,params):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    
    Z1=np.dot(W1,X)+b1
    A1=relu(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    cache={"Z1":Z1,"Z2":Z2,"A1":A1,"A2":A2}
    return cache

def compute_cost(A,Y):
    m=Y.shape[1]
    loss=np.dot(Y,np.log(A))+np.dot(Y,np.log(A))
    cost=-np.sum(loss)/m
    cost=np.squeeze(cost)
    return cost

def backprop(params,cache,Y,X):
    W1=params["W1"]
    W2=params["W2"]
    A1=cache["A1"]
    A2=cache["A2"]
    m=Y.shape[1]
    
    dZ2=A2-Y
    dW2=np.dot(dZ2,A2.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1=np.dot(W2.T,dZ2)*1
    dW1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m
    grads={"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
    return grads

def update_parameters(grads,params,learning_rate=0.01):
    W1=params["W1"]
    b1=params["b1"]
    W2=params["W2"]
    b2=params["b2"]
    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1=grads["db1"]
    db2=grads["db2"]
    
    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    
    params={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return params

def nn_model(X,Y,learning_rate=0.01,iterations=1000):
    parameters=initialize_params(X)
    
    for i in range(iterations):
        cache=forward_prop(X,parameters,relu,sigmoid)
        A2=cache["A2"]
        cost=compute_cost(A2,Y)
        grads=backprop(parameters, cache, Y, X)
        parameter=update_parameters(grads,parameters,learning_rate)
        if(iterations%100==0):
            print("Cost: ",cost)
        return parameter
def predict(X,parameters):
    cache=forward_prop(X,parameters)
    A2=cache["A2"]
    prediction= (A2 > 0.5).astype(int)
    return prediction

        
        
        