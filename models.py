##En este archivo se guardan todos los modelos

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential


def Upsample(filters,use_drop=False):
    init = tf.random_normal_initializer(0,0.02)
    layer = Sequential()
    layer.add(Conv2DTranspose(filters,4,2,padding="same",kernel_initializer=init))
    layer.add(BatchNormalization())
    if use_drop:
        layer.add(Dropout(0.5))
    layer.add(ReLU())
    return layer

def Downsample(filters,use_batch=True):
    init = tf.random_normal_initializer(0,0.02)
    layer = Sequential()
    layer.add(Conv2D(filters,4,2,padding="same",kernel_initializer=init))
    if use_batch:
        layer.add(BatchNormalization())
    layer.add(LeakyReLU())
    return layer

def ResidualBlock(filters,ni=0):
    inp = Input(shape=[None,None,filters])
    if not ni==0:
        ains = [Input(shape=[None,None,filters]) for i in range(ni)]
    
    x = inp
    xi = inp
    if not ni == 0:
        for i in range(ni):
            x = Concatenate()([x,ains[i]])
    
    l1 = Conv2DTranspose(filters,4,1,padding="same")
    ac1 = ReLU()
    x = l1(x)
    x = ac1(x)
    x = tf.add(x,xi)
    if not ni == 0:
        return Model([inp,ains],x)
    else:
        return Model(inp,x)
    
#Se pueden modificar las capas internas
#En particular, se pueden añadir mas bloques residuales
def Net():
    xi = Input(shape=[512,512,3])
    yt = Input(shape=[512,512,3])
    
    con = Concatenate()([xi,yt])
    
    down_stack = [
        Downsample(64), #256,256,64
        Downsample(128),#128,128,128
        Downsample(256),#64 ,64 ,256
        Downsample(512),#32 ,32 ,512
        Downsample(512),#16 ,16 ,512
    ]
    rs_stack = [
        ResidualBlock(512),
        ResidualBlock(512,1),
        ResidualBlock(512,2),
        ResidualBlock(512,3),
    ]
    
    up_stack = [
        Upsample(512),               #32 ,32 ,512
        Upsample(256),               #64 ,64 ,256
        Upsample(128),               #128,128,128
        Upsample(64),                #256,256,64
    ]
    last = Conv2DTranspose(3,4,2,padding="same")
    
    x = con
    s = []
    for layer in down_stack:
        x = layer(x)
        s.append(x)
    rsi = []
    for rs in rs_stack:
        if rsi == []:
            x = rs(x)
            rsi.append(x)
        else:
            x = rs([x,rsi])
            rsi.append(x)
    
    s = reversed(s[:-1])
    for layer,sk in zip(up_stack,s):
        x = layer(x)
        x = Concatenate()([x,sk])
    last = last(x)
    return Model([xi,yt],last)