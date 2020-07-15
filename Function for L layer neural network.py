import numpy as np

def sigmoid(x):
    return(1/(1+np.exp(-x))),x

def relu(x):
    z=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i][j]<0):
                x[i][j]=0
    return x,z

def relu_derivative(dA,cache):
    Z=cache
    dZ=0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if(Z>0):
                dZ[i][j]=1
            else:
                dZ[i][j]=0
    dZ=dZ*dA
    return dZ 

def sigmoid_derivative(dA,cache):
    Z=cache
    dZ=np.multiply(dA,np.multiply(Z,(1-Z)))
    return dZ

def initialize_params(layer_dims):
    parameters={}
    L=len(layer_dims)
    for i in range(1,L):
        parameters["W"+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])
        parameters["b"+str(i)]=np.zeros((layer_dims[i],1))
    
    return parameters

def linear_forward(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    cache=(A_prev,W,b)
    return Z,cache

def linear_activation_forward(A_prev,b,W,activation):
    if(activation=='relu'):
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache= relu(Z)
    else:
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache= sigmoid(Z)
    cache=(linear_cache,activation_cache)
        
    return A,cache

def linear_model(X,parameters):
    A=X
    L=len(parameters)//2
    caches=[]
    
    for i in range(1,L):
        A_prev=A
        W=parameters["W"+str(i)]
        b=parameters["b"+str(i)]
        A,cache=linear_activation_forward(A,b,W,activation='relu')
        caches.append(cache)
    W=parameters["W"+str(L-1)]
    b=parameters["b"+str(L-1)]
    A,cache=linear_activation_forward(A,b,W,activation='sigmoid')
    caches.append(cache)

    return A,caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    logprobs=np.dot(Y,np.log(AL))+np.dot(1-Y,np.log(1-AL))
    cost=-np.sum(logprobs)/m
    return cost

def linear_backward(dZ,cache):
    m=cache[0].shape[1]
    linear_cache,activation_cache=cache
    dW=np.dot(dZ,cache[0])/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(cache[1].T,dZ)
    
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache=cache 
    if (activation=='relu'):
        dZ=relu_derivative(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    
    else:
        dZ=sigmoid_derivative(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    return dA_prev,dW,db

def backward_model(AL,Y,caches):
    m=Y.shape[1]
    L=len(caches)
    Y=Y.reshape(AL.shape)
    grads={}
    dAL= - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache=caches[-1]
    grads["A"+str(L-1)],grads["W"+str(L)],grads["b"+str(L)]=linear_activation_backward(dAL,current_cache,'sigmoid')
    
    for i in range (reversed(L-1)):
        current_cache=caches[i]
        dA_prev_t,dW_t,db_t=linear_activation_backward(grads["dA"+str(i+1)],current_cache,'relu')
        grads["dA"+str(i)]=dA_prev_t
        grads["dW"+str(i+1)]=dW_t
        grads["db"+str(i+1)]=db_t
        
    return grads
                    
def update_parametes(params,grads,learning_rate):
    L=len(params)//2
    for i in range(1,L+1):
        params["W"+str(i)]=params["W"+str(i)]-learning_rate*grads["dW"+str(i)]
        params["b"+str(i)]=params["b"+str(i)]-learning_rate*grads["db"+str(i)]
        
    return params
                    


    


        