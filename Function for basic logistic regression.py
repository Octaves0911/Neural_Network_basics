import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialize_paramerters(shape):
    W=np.zeros((1,shape))
    b=0
    return W,b

def forward_prop(X,W,b):
    Z=np.dot(X,W)+ b
    A=sigmoid(Z)
    return A
def compute_cost(A,Y):
    cost=-np.sum(np.multiply(Y,np.log(A)))
    return cost

def back_prop(X,A,Y,m):
    dZ=A-Y
    dW=np.dot(dZ,X.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)
    return dW,db

def optimize(A,Y,parameter,grads,learning_rate=0.01,compute_cost, iterations=100):
    W=parameters["W"]
    b=parameters["b"]
    dW=grads["dW"]
    db=grads["db"]
    
    for i in range(iterations):
        cost=compute_cost(A,Y)
        W=W-learning_rate*dW
        b=b-learning_rate*db
        if iterations%100:
            print(cost)
    return W,b,dW,db

def predict(W,b,X):
    res=forward_prop(X,W,b)
    for i in range(A.shape[1]):
        if res[i]>0.5:
            res[i]=1
        else:
            res[i]=0
    return res
    