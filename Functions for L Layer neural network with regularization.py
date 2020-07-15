# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:18:43 2020

@author: amanm
"""
import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x))),x

def relu(x):
    m=x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if(x[i][j]<0):
                x[i][j]=0
    return x,m

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

def initialize_parameters(layers_dim):
    params={}
    for i in range(1,len(layers_dim)):
        params["W"+str(i)]=np.random.randn(layers_dim[i],layers_dim[i-1])/(np.sqrt(layers_dim[i-1]))
        params["b"+str(i)]=np.zeros((layers_dim[i],1))
    
    return params

def linear_forward(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    cache=(A_prev,W,b)
    return Z,cache

def linear_activation_forward_regularization(A_prev,W,b,activation):
    if(activation=='relu'):
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
        
    elif activation=='sigmoid':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    cache=(linear_cache,activation_cache)
    return A,cache

def linear_model(X,params):
    L=len(params)//2
    A=X
    caches=[]
    for i in range(1,L):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        A_prev=A
        A,cache=linear_activation_forward_regularization(A,W,b,'relu')
        caches.append(cache)
        
    W=params["W"+str(L)]
    b=params["b"+str(L)]
    A,cache=linear_activation_forward_regularization(A,W,b,'sigmoid')
    caches.append(cache)
    return A,caches
        
    
def compute_cost_regularization(A,Y,params,lamda):
    l2_regularization=0
    m=Y.shape[1]
    logprobs=np.dot(Y,np.log(A))+np.dot(1-Y,np.log(1-A))
    for i in range(1,(len(params)//2)+1):
        l2_regularization+=params["W"+str(i)]
    l2_regularization=(l2_regularization*lamda)/(2*m)
    cost=-np.sum(logprobs)+l2_regularization
    return cost

def linear_backprop_regularization(dZ,cache,lamda):
    A_prev,W,b=cache
    m=b.shape[0]
    dA_prev=np.dot(W.T,dZ)
    dW=(np.dot(dZ,A_prev.T)+lamda*W)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    return dA_prev,dW,db

def linear_activation_back(dA,cache,lamda,activation):
    linear_cache,activation_cache=cache
    if(activation=='relu'):
        dZ=relu_derivative(dA,activation_cache)
        dA_prev,dW,db=linear_backprop_regularization(dZ,linear_cache)
    if(activation=='sigmoid'):
        dZ=sigmoid_derivative(dA,activation_cache)
        dA_prev,dW,db=linear_backprop_regularization(dZ,linear_cache)
    
    return dA_prev,dW,db

def model_back(AL,Y,cache,lamda):
    grads={}
    m=AL.shape[1]
    L=cache.shape[1]
    dAL=-np.sum(np.divide(Y,AL),-np.divide((1-Y),(1-AL)))
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_back(dAL,cache,lamda,'sigmoid')
    for i in range(reversed(L-1)):
        dA_temp,dW_temp,db_temp=linear_activation_back(grads["dA"+str(i+1)],cache,lamda,'relu')
        grads["dA"+str(i)]=dA_temp
        grads["dW"+str(i+1)]=dW_temp
        grads["db"+str(i+1)]=db_temp
    
    return grads

def update_parameters(grads,params,learning_rate):
    L=len(params)//2
    for i in range(1,L+1):
        params["W"+str(i)]=params["W"+str(i)]-learning_rate*grads["dW"+str(i)]
        params["b"+str(i)]=params["b"+str(i)]-learning_rate*grads["db"+str(i)]
    return params


def model(X,Y,layers_dim, iterations=1000,learning_rate=0.01 ):
    m=Y.shape[1]
    costs=[]
    parameters=initialize_parameters(layers_dim)
    
    for i in range(iterations):
        A,caches=linear_model(X[i],parameters)
        
        cost=compute_cost_regularization(A,Y[i],parameters,lamda=0.1)
        costs.append(cost)
        
        grads=model_back(A,Y[i],caches,lamda=0.01)
        
        parameters=update_parameters(grads,parameters,learning_rate=0.01)
        
    return parameters
        
    
        
        
        
#for dropouts 
#create a dropout matrix d for a particular layer suppose 3 
# d3=np.random.randn(A3.shape[1],A3.shape[0])
#d3=(d3<keep_prob).astype(int)
#A3=np.dot(A3,d3)
#A3/=keep_prob
#for backprop
#dA3=np.dot(W4.T,Z4)
#dA3=np.dot(dA3,D3)
#dA3/=keep_prob


            

    
        
    
    
    
        
    