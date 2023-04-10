import numpy as np
import math
import keras
from keras.layers import Input, Dense, Conv2D, merge, Convolution2D
from keras.models import Model, Sequential
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import h5py
import math
import glob
import tensorflow as tf
from keras import backend as K
#import cv2 as cv



epoch = 20
batch_size = 128

# ###############################
# Initial Parameters
# ###############################
c = 1
f1 = 9
f2 = 1
f3 = 5
n1 = 64
n2 = 32
n3 = 1
f_sub = 33


# ###############################
# Load Data and reshape to 33*33
# ###############################
print("#######################################--start loading--#######################################")
train = h5py.File('train_mscale.h5', 'r')
valid = h5py.File('test_mscale.h5', 'r')
test = h5py.File('test14_.h5', 'r')

x_train = train['data'][:]
y_train = train['label'][:]
x_valid = valid['data'][:]
y_valid = valid['label'][:]
x_test = test['data'][:]
y_test = test['label'][:]

train.close()
valid.close()
test.close()

# ------
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_valid = x_valid.astype('float32')
y_valid = y_valid.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train.reshape(x_train.shape[0], f_sub, f_sub, 1)
x_valid = x_valid.reshape(x_valid.shape[0], f_sub, f_sub, 1)
x_test = x_test.reshape(x_test.shape[0], f_sub, f_sub, 1)
y_train = y_train.reshape(y_train.shape[0], f_sub, f_sub, 1)
y_valid = y_valid.reshape(y_valid.shape[0], f_sub, f_sub, 1)
y_test = y_test.reshape(y_test.shape[0], f_sub, f_sub, 1)



#print number of Data patches
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_valid shape:', x_valid.shape)
print('y_valid shape:', y_valid.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'Valid samples')
print(x_test.shape[0], 'test samples')
print('type : ', type(x_train))
print('type : ', type(y_train))
print('type : ', type(x_valid))
print('type : ', type(y_valid))
print('type : ', type(x_test))
print('type : ', type(y_test))

print("#######################################--end loading--#######################################")


# #######################################
# Functions 
# #######################################
def psnr(im_gt, im_pre):
  max_pixel = 1.0
  return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(im_pre - im_gt))))

  
def ssim(y_true, y_pred):
    """structural similarity measurement system."""
    ## K1, K2 are two constants, much smaller than 1
    K1 = 0.04
    K2 = 0.06
    
    ## mean, std, correlation
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = (sig_x * sig_y) ** 0.5

    ## L, number of pixels, C1, C2, two constants
    L =  33
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim 
 
  

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 20
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


#################################### 
###  Model
#################################### 
input_shape = (f_sub, f_sub,1)
# x = Input(name='inputs', shape=input_shape, dtype='float32')
x = Input(shape=input_shape,)

# , kernel_initializer="he_normal"
c1 = Conv2D(n1, (f1,f1), activation='relu', padding='same')(x)
c2 = Conv2D(n2, (f2,f2), activation='relu', padding='same')(c1)
c3 = Conv2D(n3, (f3,f3), padding='same')(c2)

model = Model(input = x, output = c3)

model.summary()


#################################### 
### Compile and fit Model
####################################
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
model.compile(loss='MSE', metrics=[ssim , psnr], optimizer=adam)
# model.compile(loss='MSE', metrics=[gmsd], optimizer=adam)

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

print("#######################################--start training--#######################################")

history = model.fit(x_train, y_train,
	                epochs = epoch,
	                batch_size = batch_size,
	                callbacks = [lrate],
	                verbose = 1,
	                shuffle = True,
	                validation_data = (x_test, y_test))

print("#######################################--end training--#######################################")
print(history.history.keys())

# ###############################
# save model and weights
# ###############################
json_string = model.to_json()  
open('srcnn_model.json','w').write(json_string)  
model.save_weights('srcnn_model_weights.h5') 

# ###############################
# Summerize
# ###############################
#plot PSNR
plt.plot(history.history['metricLoss'])
plt.plot(history.history['val_metricLoss'])
plt.title('model loss')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

#plot SSIM
plt.plot(history.history['ssim'])
plt.plot(history.history['val_ssim'])
plt.title('model loss')
plt.ylabel('SSIM')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


# ###############################
# Evaluate with test dataset
# ###############################
alpha_pre = model.predict(x_test, verbose=0)



# #### plot test with predicted

i = 1244  # in range of len(x_test)
plt.figure(figsize = (33, 33))
plt.subplot(5, 5, 1)
plt.title("x_test", fontsize=20)
plt.imshow(x_test[i].squeeze())

plt.subplot(5, 5, 1+1)
plt.title("predicted", fontsize=20)
plt.imshow(alpha_pre[i].squeeze())


