#Leer con cuidado

import tensorflow as tf
import numpy as np

'''
---Funcion de entrenamiento---
img --> Imagen en alta resolucion       #bs,512,512,3
timg --> Imagen en baja resolucion      #bs,512,512,3
net --> Modelo de super resolucion
optimizer --> Optimizador
prod --> calculo de p(y)
'''
@tf.function()
def train_step(img,timg,net,optimizer,prod):
    im_shape = tf.shape(img)
    noise = tf.cast(tf.random.normal([im_shape[0],512,512,3]),tf.float32)
    with tf.GradientTape() as tape:
        #Prediccion del modelo
        pred = net([timg,np.sqrt(prod)*img+np.sqrt(1-prod)*noise])
        #Calculo del error
        loss = tf.reduce_mean(tf.abs(pred-noise))
        #Calculamos gradientes
        grads = tape.gradient(loss,net.trainable_variables)
    #Aplicamos gradientes
    optimizer.apply_gradients(zip(grads,net.trainable_variables))
    
    return loss

'''
img --> Imagen en baja resolucion
net --> Modelo de super resolucion
numbs --> Hiperparametros(entre 0 y 1)
n_steps --> Cuantos pasos de reduccion de ruido seran
'''
def inference(img,net,numbs,n_steps=100,get_frames=False):
    #Imagen inicial
    yt = tf.random.normal([1,512,512,3])
    if get_frames:
        hist = [yt]
    for i in reversed(range(0,n_steps)):
        #Seguimos lo visto en el papaer de SR3
        z = tf.random.normal([1,512,512,3]) if i > 0 else tf.zeros([1,512,512,3])
        pred = net([img,yt])
        a = (1/np.sqrt(numbs[i]))
        b = ((1-numbs[i])/np.sqrt(1-calc_p(numbs,i)))
        #Reducimos el ruido a la imagen yt para tener yt-1
        yt_1 = a*(yt-b*pred) + z*np.sqrt(1-numbs[i])
        if get_frames:
            hist.append(yt_1)
        yt = yt_1
    if get_frames:
        return hist,yt
    else:
        return yt
#Calculamos p(y) segun visto en el paper de SR3
def calc_p(arr,steps):
    prod = 0
    for i in range(steps):
        prod += (1/steps)*np.random.uniform(arr[i],arr[i+1])
    return prod

#Generamos hiperparametros, pueden ser diferentes
def generate_params(steps,e=1e-6):
    return np.array([abs(1-t/steps-e) for t in range(1,steps+2)])

def resize(img,width,height):
    return tf.image.resize(img,[width,height])

def open_image(path,train=False):
    img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(path)),tf.float32)[...,:3]
    sp = tf.shape(img)
    img = img/255
    #Imagen en alta resolucion
    img = resize(img,512,512)
    
    if train:
        #Obtenemos la imagen en baja resolucion y la aumentamos
        timg = resize(img,64,64)
        timg = resize(timg,512,512)
        
        #Las retornamos para el entrenamiento
        return img,timg
    else:
        return img,(sp[0],sp[1])
    
