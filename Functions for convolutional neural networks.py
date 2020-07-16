import numpy as np

def zero_pad(x,pad):
    paded=np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
    return paded

def conv_single_step(A_prev,W,b):
    conv=np.multiply(A_prev,W)+b
    return conv

def conv_model_forward(A_prev,W,b,hparameters):
    pad=hparameters["pad"]
    stride=hparameters["stride"]
    m,n_H_prev,n_W_prev,n_C_prev=A_prev.shape
    f,n_C=W.shape[0],W.shape[-1]
    n_H=int(((n_H_prev+2*pad-f)/stride)+1)
    n_W=int(((n_H_prev+2*pad-f)/stride)+1)
    Z=np.zeros((m,n_H,n_W,n_C))
    A_pad=zero_pad(A_prev,pad)
    for i in range(m):
        a_slice=A_pad[i,:,:,:]
        for h in range (n_H):
            H_start=h*stride
            H_end=h*stride+f
            for w in range(n_W):
                W_start=W*stride
                W_end=W*stride+f
                for c in range(n_C):
                    W = W[:,:,:,c]
                    b=  b[:,:,:,c]
                    A_prev_slice=a_slice[H_start:H_end,W_start:W_end,:]
                    Z[i,h,w,c]=conv_single_step(A_prev_slice,W,b)
    cache=(A_prev,W,b,hparameters)
    return Z,cache

def pooling(A_prev,hparameters,mode='max'):
    m,n_H_prev,n_W_prev,n_C=A_prev.shape
    f=hparameters["f"]
    stride=hparameters["stride"]
    n_H=int(((n_H_prev-f)/stride)+1)
    n_W=int(((n_W_prev-f)/stride)+1)
    Z=np.zeros((m,n_H,n_W,n_C))
    for i in range (m):
        a_slice=A_prev[i,:,:,:]
        for h in range(n_H):
            H_start=h*stride
            H_end=h*stride+f
            for w in range (n_W):
                W_start=w*stride
                W_end=w*stride+f
                for c in range (n_C):
                    A_prev_slice=a_slice[H_start:H_end,W_start:W_end,c]
                    Z[i,h,w,c]=np.max(A_prev_slice)
                    
    cache=(A_prev,hparameters)
    return Z,cache

def conv_backward(dZ,cache):
    A_prev,W,b,hparameters=cache
    pad=hparameters["pad"]
    stride=hparameters["stride"]
    A_prev_pad=zero_pad((A_prev,pad))
    m,n_H,n_W,n_C_prev=A_prev.shape
    f,n_C=W.shape[0],W.shape[-1]
    dA_prev=np.zeros((A_prev.shape))
    dA_prev_pad=zero_pad(dA_prev,pad)
    dW=np.zeros((W.shape))
    db=np.zeros((b.shape))
    for i in range(m):
        a_slice=A_prev_pad[i]
        dA_pad_slice=dA_prev_pad[i]
        for h in range(n_H):
            H_start=h*stride
            H_end=h*stride+f
            for w in range(n_W):
                W_start=w*stride
                W_end=w*stride+f
                for c in range(n_C):
                    a_slice_prev=a_slice[H_start:H_end,W_start:W_end,:]
                    dA_pad_slice[H_start:H_end,W_start:W_end,:]+=np.multiply(W[:,:,:,c],dZ[i,h,w,c])
                    dW[:,:,:,c]+=a_slice_prev*dZ[i,h,w,c]
                    db[:,:,:,c]+=dZ[i,h,w,c]
        dA_pad_slice[i,:,:,:]=dA_pad_slice[pad:-pad,pad:,-pad,:]
        
    return dA_prev,dW,db

def create_mask(x):
    mask=(x==np.max(x))
    return mask

def distribute_values(dZ,shape):
    n_H,n_W=shape
    average=dZ/(n_H*n_W)
    a=np.ones((n_H,n_W))*average
    return a

def pool_back(dA,cache,mode='max'):
    A_prev,hparameters=cache
    m,n_H,n_W,n_C=dA.shape
    f=hparameters["f"]
    dA_prev=np.zeros((A_prev.shape))
    stride=hparameters["stride"]
    for i in range(m):
        a_slice=A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    H_start=h*stride
                    H_end=h*stride+f
                    W_start=w*stride
                    W_end=w*stride+f
                    if mode == 'max':
                        a_prev_slice=a_slice[H_start:H_end,W_start:W_end,c]
                        mask=create_mask(a_prev_slice)
                        dA_prev[H_start:H_end,W_start:W_end,c]+=mask*dA[i,h,w,c]
                    elif mode=='average':
                        dA1=dA[i,h,w,c]
                        shape=(f,f)
                        dA_prev[H_start:H_end,W_start:W_end,c]+=distribute_values(dA1,shape)
                        
    
    
                    
    