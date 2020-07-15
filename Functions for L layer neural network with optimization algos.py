# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:41:00 2020

@author: amanm
"""
import numpy as np
def relu(x):
    m=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i][j]<0):
                x[i][j]=0
        return x,m
def sigmoid(x):
    m=x
    sig=1/(1+np.exp(-x))
    return sig,m

def relu_derivative(dA,Z):
    dZ=0
    for i in range(dA.shape[0]):
        for j in range(dA.shape[1]):
            if(dA[i][j]>0):
                dZ[i][j]=1
            else:
                dZ[i][j]=0
    return dZ

def sigmoid_derivative(dA,Z):
    der=np.multiply(dA,np.multiply(Z,(1-Z)))
    return der

def initialize_parameters(layer_dims):
    params={}
    L=len(layer_dims)
    for i in range(1,L):
        params["W"+str(i)]=np.random.randn(layer_dims[i+1],layer_dims[i])/np.sqrt(layer_dims[i-1])
        params["b"+str(i)]=np.zeros((layer_dims[i+1],1))
    return params

def linear_forward(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    cache=(A_prev,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    if(activation=='relu'):
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
    else:
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    
    cache=(linear_cache,activation_cache)
    return A,cache

def model_forward(X,params):
    L=len(params)//2
    caches=[]
    A=X
    for i in range(1,L):
        A_prev=A
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        A,cache=linear_activation_forward(A_prev,W,b)
        caches.append(cache)
    
    W=params["W"+str(L)]
    b=params["b"+str(L)]
    A,cache=linear_activation_forward(A,W,b)
    caches.append(cache)
    return A,caches

def compute_cost(A,Y):
    logprobs=np.multiply(Y,np.log(A))+np.multiply(1-Y,np.log(1-A))
    cost=-np.sum(logprobs)
    return cost
        
def backward_linear(dZ,cache):
    m=dZ.shape[1]
    A_prev,W,b=cache
    dA_prev=np.dot(W.T,dZ)
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    return dA_prev,dW,db
def backward_linear_activation(dA,params,cache,activation):
    linear_cache,activation_cache=cache
    if activation=='relu':
        dZ=relu_derivative(dA,activation_cache)
        dA_prev,dW,db=backward_linear(dZ,linear_cache)
    elif (activation=='sigmoid'):
        dZ=sigmoid_derivative(dA,activation_cache)
        dA_prev,dW,db=backward_linear(dZ,linear_cache)
    return dA_prev,dW,db

def model_backward(AL,Y,params,cache):
    grads={}
    L=len(params)//2
    dAL=np.divide(Y,AL)+np.divide((1-Y),(1-AL))
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=backward_linear_activation(dAL,params,cache,activation='sigmoid')
    
    for i in range(reversed(L-1)):
        dA_prev_temp,dW_temp,db_temp=backward_linear_activation(grads["dA"+str(i+1)],params,cache,activation='relu')
        grads["dA"+str(i)]=dA_prev_temp
        grads["dW"+str(i+1)]=dW_temp
        grads["db"+str(i+1)]=db_temp
        
    
    return grads

def update_parameters(params,grads,learning_rate):
    L=len(params)//2
    for i in range(1,L+1):
        params["W"+str(i)]=params["W"+str(i)]-learning_rate*grads["dW"+str(i)]
        params["db"+str(i)]=params["db"+str(i)]-learning_rate*grads["db"+str(i)]
    return params

def model(X,Y,learning_rate,epoch,batch_size,layer_dims):
    params={}
    m=Y.shape[1]
    costs=[]
    L=m//batch_size
    
    params=initialize_parameters(layer_dims)
    
    for i in range(epoch):
        for j in range(L):
            A,cache=model_forward(X[:j*batch_size(j+1)*batch_size:],params)
        
            cost=compute_cost(A,Y[:,j*batch_size:(j+1)*batch_size])
            costs.append(cost)
        
            grads=model_backward(A,Y[:,j*batch_size:(j+1)*batch_size],params,cache)
        
            params=update_parameters(params,grads,learning_rate=0.01)
    if (m%batch_size!=0):
        A,cache=model_forward(X[:,L*batch_size:m],params)
        cost=compute_cost(A,Y[:,L*batch_size:m])
        costs.append(cost)
        grads=model_backward(A,Y[:,L*batch_size:m],params,cache)
        params=update_parameters(params,grads,learning_rate=0.01)
        
    
    return params
        


## permutation=list(np.radom.permutation(m)){it will form a list of m elements randomly selected}
##shuffled_X=X[:,permutation]
##shuffled_Y=Y[:,permutation]
    

## MOMENTUM
def initialize_velocity(params):
    L=len(params)//2
    v={}
    for i in range(L):
        v["dW"+str(i+1)]=np.zeros((params["W"+ str(i+1)].shape))
        v["db"+str(i+1)]=np.zeros((params["b"+str(i+1)].shape))
    
    return v 


def velocity_value(v,grads,beta):
    L=len(grads)//2
    for i in range(1,L+1):
        v["dW"+str(i)]=beta*v["dW"+str(i)]+(1-beta)*grads["dW"+str(i)]
        v["db"+str(i)]=beta*v["db"+str(i)]+(1-beta)*grads["db"+str(i)]
    velocity=v
    return velocity

## in the model pass velocity in place of grads in backprop in order to use momentum
    
## RMSProp
    
def initialize_rms(params):
    L=len(params)//2
    s={}
    for i in range(1,L+1):
        s["dW"+str(i)]=np.zeros((params["W"+str(i)].shape))
        s["db"+str(i)]=np.zeros((params["db"+str(i)].shape))
    return s

def RMS_prop(s,grads,beta):
    L=len(s)//2
    for i in range(1,L+1):
        s["dW"+str(i)]=beta*s["dW"+str(i)]+(1-beta)*np.multiply(grads["dW"+str(i)],grads["dW"+str(i)])
        s["db"+str(i)]=beta*s["dW"+str(i)]+(1-beta)*np.multiply(grads["db"+str(i)],grads["dW"+str(i)])
    return s

##in the model pass s in place of grads in backprop in order to use rms prop

def update_parameters_with_rms_prop(s,params,grads,learning_rate,epsilon):
    L=len(params)//2
    for i in range(1,L+1):
        params["W"+str(i)]=params["W"+str(i)]+learning_rate*grads["dW"+str(i)]/np.sqrt(s["dW"+str(i)]+epsilon)
        params["b"+str(i)]=params["b"+str(i)]+learning_rate*grads["dW"+str(i)]/np.sqrt(s["dW"+str(i)]+epsilon)
        
    return params

def correct_optimization(s,v,t,beta1,beta2):
    L=len(s)//2
    for i in range(1,L+1):
        s["dW"+str(i)]=s["dW"+str(i)]/(1-beta1**t)
        v["dW"+str(i)]=s["dW"+str(i)]/(1-beta2**t)
        
    return s,v

#adam update
    
def update_parameters_using_adam(s,v,learning_rate,params,epsilon):
    L=len(params)//2
    for i in range(1,L+1):
        params["dW"+str(i)]=params["dW"+str(i)]-learning_rate*v["dW"+str(i)]/np.sqrt(s["dW"+str(i)]+epsilon)
        params["db"+str(i)]=params["db"+str(i)]-learning_rate*v["db"+str(i)]/np.sqrt(s["db"+str(i)]+epsilon)
        
    return params


    