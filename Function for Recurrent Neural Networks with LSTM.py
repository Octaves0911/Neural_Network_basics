# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:53:12 2020

@author: amanm
"""

import numpy as np

def softmax(x):
    return (np.exp(x)/np.sum(np.exp(x)))

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def rnn_cell_forward(x,a_prev,parameters):
    Waa=parameters['Waa']
    Wax=parameters['Wax']
    Wya=parameters['Wya']
    ba=parameters['ba']
    by=parameters['by']
    
    a_next=np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,x)+ba)
    y_curr= softmax(np.dot(Wya,a_next)+by)
    
    cache= (a_next,a_prev,x,parameters)
    
    return a_next,y_curr,cache

'''
#Test For the above function
np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_forward(xt_tmp,a_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)
'''

def rnn_forward(x,a0,parameters):
    nx,m,Tx=x.shape
    n_y,n_a=parameters['Wya'].shape
    
    caches=[]
    a_next=a0
    a=np.zeros((n_a,m,Tx))
    y=np.zeros((n_y,m,Tx))
    
    for i in range(Tx):
        xt=x[:,:,i]
        a_next,y_curr,cache=rnn_cell_forward(a_next,xt,parameters)
        a[:,:,i]=a_next
        y[:,:,i]=y_curr
        caches.append(cache)
        
    caches=(caches,x)
    return a,y,caches

'''
#To test the above function
np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))
'''





def lstm_cell_forward(xt,a_prev,c_prev, parameters):
    Wf=parameters['Wf']
    Wi=parameters['Wi']
    Wo=parameters['Wo']
    Wy=parameters['Wy']
    bi=parameters['bi']
    bo=parameters['bo']
    by=parameters['by']
    bf=parameters['bf']
    Wc=parameters['Wc']
    bc=parameters['bc']
    
    n_x,m=xt.shape
    n_y,n_a=Wy.shape
    concat=np.concatenate((a_prev,xt))
    
    ft= sigmoid(np.dot(Wf,concat)+ bf)
    it= sigmoid(np.dot(Wi,concat)+ bi)
    ot= sigmoid(np.dot(Wo,concat)+ bo)
    cct=np.tanh(np.dot(Wc,concat)+ bc)
    ct=ft*c_prev+it*cct
    a_next=ot*(np.tanh(ct))
    y_next=sigmoid(np.dot(Wy,a_next)+by)
    
    cache=(a_next,a_prev,c_prev,ct,xt,ct,it,ot,ft,parameters)
    
    return a_next,ct, y_next, cache

'''
#test for the above function
np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
c_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", a_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))
'''

def lstm_forward(x,a0,parameters):
    caches=[]
    Wy=parameters['Wy']
    n_y,n_a=Wy.shape
    n_x,m,T_x=x.shape
    print(x.shape)
    
    a=np.zeros((n_a,m,T_x))
    c=np.zeros((n_a,m,T_x))
    y=np.zeros((n_y,m,T_x))
    
    a_next=a0
    c_next=c[:,:,0]
    
    for i in range(T_x):
        x_next=x[:,:,i]
        print(x_next.shape)
        a_next,c_next, y_next, cache=lstm_cell_forward(x_next,a_next,c_next, parameters)
        a[:,:,i]=a_next
        c[:,:,i]=c_next
        y[:,:,i]=y_next
        
        caches.append(cache)
        
    caches=(caches,x)
    
    return a,y,c,caches


'''
np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi']= np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))
'''

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    dxt = np.dot(Wax.T,(da_next*(1-np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)))))
    dWax = np.dot(da_next*(1-np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba))),xt.T)
    da_prev = np.dot(np.transpose(Waa),(da_next*(1-np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)))))
    dWaa = np.dot(da_next*(1-np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba))),a_prev.T)
    dba = np.sum(da_next*(1-np.square(np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba))),axis=1,keepdims=True)
    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients

'''
np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)

da_next_tmp = np.random.randn(5,10)
gradients_tmp = rnn_cell_backward(da_next_tmp, cache_tmp)
print("gradients[\"dxt\"][1][2] =", gradients_tmp["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients_tmp["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients_tmp["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients_tmp["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients_tmp["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients_tmp["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients_tmp["dba"][4])
print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)
'''

