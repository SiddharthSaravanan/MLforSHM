

#IMPORTING NECESSARY LIBRARIES
import matplotlib.pyplot as plt #import matplotlib
from matplotlib.pyplot import figure #import figure from matplotlib.pyplot
import numpy as np #import numpy
import tensorflow as tf #import tensorflow
import tensorflow_datasets as tfds #import tensorflow_datasets

from tensorflow import keras #import keras
import pandas as pd #import pandas
from scipy.fftpack import irfft, rfft #import rfft and irfft functions

print("importing datasets")

#IMPORTING THE DATASET pure.xlsx
df = pd.read_excel('pure.xlsx',header=None) #import pure dataset
pure_acc = df.to_numpy()

#IMPORTING THE DATASET noisy.xlsx
df = pd.read_excel('noisy.xlsx',header=None)#import noisy dataset
noisy_acc = df.to_numpy() #convert to numpy array

#IMPORTING THE DATASET test_pure.xlsx
df = pd.read_excel('test_pure.xlsx',header=None)#import test_pure dataset
test_pure_acc = df.to_numpy()#convert to numpy array

#IMPORTING THE DATASET test_noisy.xlsx
df = pd.read_excel('test_noisy.xlsx',header=None)#import test_noisy dataset
test_noisy_acc = df.to_numpy()#convert to numpy array

print("RANDOMIZING ORDER OF DATASET TO AVOID BIAS")
#RANDOMIZING ORDER OF DATASET TO AVOID BIAS
n = np.linspace(0,noisy_acc.shape[0]-1,noisy_acc.shape[0],dtype=int) #create array from 0 to 700
np.random.shuffle(n) #shauffle the ARRAY
temp_noisy = noisy_acc #save noisy dataset in temporary array
temp_pure = pure_acc #save pure dataset in temporary array

for i in n:
  noisy_acc[i] = temp_noisy[n[i]]#randomizing noisy datasets
  pure_acc[i] = temp_pure[n[i]]#randomizing pure dataset

print("DATA PREPROCESSING, CONVERTING TRAINING DATA TO FREQUENCY DOMAIN")

#DATA PREPROCESSING, CONVERTING TRAINING DATA TO FREQUENCY DOMAIN
noisy_acc_freq = np.zeros((noisy_acc.shape[0],noisy_acc.shape[1]),np.float64) #create array of zeros of same shape as noisy_acc
for i in range(noisy_acc.shape[0]): #looping in noisy_acc ARRAY
  noisy_acc_freq[i] = rfft(noisy_acc[i]) #converting all signals to freq domin

pure_acc_freq = np.zeros((pure_acc.shape[0],pure_acc.shape[1]),np.float64) #create array of zeros of same shape as pure_acc
for i in range(pure_acc.shape[0]): #looping in pure_acc ARRAY
  pure_acc_freq[i] = rfft(pure_acc[i])#converting all signals to freq domin

noisy_acc_freq = noisy_acc_freq.reshape(noisy_acc_freq.shape[0],noisy_acc_freq.shape[1],1)#RESHAPING THE NOISY ACCELEROMETER ARRAY

print("MODELING THE CNN FOR DENOISING")

#MODELING THE CNN FOR DENOISING
# linear activations are used as unlike regression it is a pattern to pattern matching
model1 = keras.Sequential([ #CREATING sequential model
   keras.layers.ZeroPadding1D(padding=3), #zeropadding layer
   keras.layers.Conv1D(16, 7, strides=1, activation='linear'), #convolutional layer filter size=7, no of filters=16
   keras.layers.ZeroPadding1D(padding=8), #zeropadding layer
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=32
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=32
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=32
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=16
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=16
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),#convolutional layer filter size=3, no of filters=16
   keras.layers.Flatten(), #flatten layer
   keras.layers.Dense(16, activation='linear'),#fully connected layer, size = 16
   keras.layers.Dense(pure_acc_freq.shape[1], activation=None)#fully connected layer size=size of signal
])

optim = tf.keras.optimizers.Adam(3e-4) #using adam optimizer, learning rate=3 x 10^-4

model1.compile(optimizer=optim,  loss = 'mse',metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')]) #defining loss function and optimizer functions

model1.fit(noisy_acc_freq, pure_acc_freq, epochs=100, batch_size=16) #training model1 for 100 epochs on batch size of 16

print("MODELING THE ANN FOR DENOISING")
#MODELING THE ANN FOR DENOISING
# linear activations are used as unlike regression it is a pattern to pattern matching
model2 = keras.Sequential([#CREATING sequential model
   keras.layers.Flatten(),#flatten layer
   keras.layers.Dense(4096, activation='linear'), #fully connected layer, size = 4096
   keras.layers.Dense(8192, activation='linear'),#fully connected layer, size = 8192
   keras.layers.Dense(4096, activation='linear'),#fully connected layer, size = 4096
   keras.layers.Dense(2048, activation='linear'),#fully connected layer, size = 2048
   keras.layers.Dense(pure_acc_freq.shape[1], activation=None)#fully connected layer, size = size of signal
])

optim = tf.keras.optimizers.SGD(1e-3)#using SGD optimizer, learning rate=1 x 10^-3
#the momentum aspect of Adam caused it to spiral out of control

model2.compile(optimizer=optim, loss = 'mse', metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])#defining loss function and optimizer functions

model2.fit(noisy_acc_freq, pure_acc_freq, epochs=100, batch_size=12)#training model2 for 100 epochs on batch size 12

print("SAVING THE MODEL")
#SAVING THE MODEL
model1.save('cnn1.h5')#SAVING MODEL1
model2.save('ann1.h5')#SAVING MODEL2
new_model = tf.keras.models.load_model('cnn1.h5') #loading CNN under the variable new_model

print("COMPARING THE PURE, NOISY AND DENOISED SIGNALS USING MATPLOTLIB")
#COMPARING THE PURE, NOISY AND DENOISED SIGNALS USING MATPLOTLIB
z = test_noisy_acc[1] #z has signal no. 1038 from the test_noisy dataset.
z= rfft(z) # converting z to freq DOMAIN
z = z.reshape(1,z.shape[0],1) #reshaping z ARRAY
y_denoised = new_model.predict(z) #denoising z using new_model
y_denoised = irfft(y_denoised) #converting z back to time DOMAIN
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k') #defining size and shape of the plot
x = np.linspace(start=0,stop=7,num=701) #creating array= {0,0.01,0.02....6.98,6.99,7}
y_noisy = test_noisy_acc[1]  #plotting the noisy SIGNAL
plt.plot(x,y_noisy)             # plotting NOISY SIGNAL
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')#defining size and shape of the plot
plt.plot(x,pure_acc[1])      #plotting PURE SIGNAL
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')#defining size and shape of the plot
plt.plot(x,y_denoised.reshape(701))     #plotting DENOISED SIGNAL

print("CALCULATING IDR BY CALCULATING RELATIVE DISPLACEMENTS BY DOUBLE INTEGRATION:pure")
#NOW THAT OUR ML ALGORITHM IS WORKING FINE, WE WILL MOVE ON TO EVALUATING IT USING [IO,LS,CP]
#CALCULATING IDR BY CALCULATING RELATIVE DISPLACEMENTS BY DOUBLE INTEGRATION. AND FINALLY CREATING [IO,LS,CP] ARRAY FOR FULL PURE #DATASET
time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

pure_classification = np.zeros((int(test_pure_acc.shape[0]*0.5),3),np.float64) #creatng array of zeros for classinfiying the pure signals

v = np.zeros(test_pure_acc.shape,np.float64) #velocity
disp = np.zeros(test_pure_acc.shape,np.float64) #displacement
floor_height = 2.75 #standard floor height in India = 2.75m


for i in range(0,test_pure_acc.shape[0]): #iterating over the test_pure acceleration signals
  for j in range(1,test_pure_acc.shape[1]):   #going over each sample
      v[i][j] = v[i][j-1] + (((test_pure_acc[i][j-1]+test_pure_acc[i][j])/2) * (time)) #integrating for velocity

for i in range(0,test_pure_acc.shape[0]):#iterating over the velocity
  for j in range(1,test_pure_acc.shape[1]):  #going over each sample
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time)) #integrating for displacement

for i in range(0,disp.shape[0],2): #going over adjacent displacement arrays
  idr = np.zeros(disp.shape[1],np.float64) #creating array of zeros for IDRs

  for j in range(disp.shape[1]): #going over adjacent displacement arrays
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height) #calculating IDRs

  '''
 if idr < 0.007       => Immediate Occupancy
 if idr 0.007 to 0.05 => Life Safety
 if idr >0.05         => Collapse prevention
 '''
  scores=np.array([0,0,0]) #scores
  '''
 io_score=0th index
 ls_score=1st index
 cp_score=2nd index
 '''
  for k in range (idr.shape[0]):#calculating scores for function/array of IDRs
    if idr[k]<0.007:#if IDR<0.007
      scores[0]+=1 #increment scores[0] by 1
    elif idr[k]>0.05:#if IDR>0.05
      scores[2]+=1 #increment scores[2] by 1
    else:#if 0.007<IDR<0.05
      scores[1]+=1#increment scores[2] by 1

  #most severe score is considered for labeling the dataset
  if scores[2]>0:#if scores[2]>0:
    scores = [0,0,1]#scores becomes [0,0,1]
  elif scores[1]>0:#if scores[1]>0 and scores[2]=0:
    scores = [0,1,0]#scores becomes [0,1,0]
  else: #if scores[0]>0 and scores[1]=0 and scores[2]=0
    scores = [1,0,0]#scores becomes [1,0,0]

  pure_classification[int(i/2)]=scores #adding scores to pure_classification array
  scores=np.array([0,0,0]) #resetting scores array

#FOR NOISY CLASSIFICATION

print("CALCULATING IDR BY CALCULATING RELATIVE DISPLACEMENTS BY DOUBLE INTEGRATION:noisy")
time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

noisy_classification = np.zeros((int(test_noisy_acc.shape[0]*0.5),3),np.float64)#creatng array of zeros for classinfiying the noisy signals

v = np.zeros(test_noisy_acc.shape,np.float64) #velocity
disp = np.zeros(test_noisy_acc.shape,np.float64) #displacement


for i in range(0,test_noisy_acc.shape[0]):#iterating over the test_noisy acceleration signals
  for j in range(1,test_noisy_acc.shape[1]):  #going over each sample
      v[i][j] = v[i][j-1] + (((test_noisy_acc[i][j-1]+test_noisy_acc[i][j])/2) * (time)) #integrating for velocity

for i in range(0,test_noisy_acc.shape[0]):#iterating over the velocity signals
  for j in range(1,test_noisy_acc.shape[1]):#going over each sample
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))#integrating for displacement

for i in range(0,disp.shape[0],2): #going over adjacent displacement arrays
  idr = np.zeros(disp.shape[1],np.float64)#creating array of zeros for IDRs

  for j in range(disp.shape[1]): #going over adjacent displacement arrays
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)#calculating IDRs

  '''
 if idr < 0.007       => Immediate Occupancy
 if idr 0.007 to 0.05 => Life Safety
 if idr >0.05         => Collapse prevention
 '''
  scores=np.array([0,0,0])#scores
  '''
 io_score=0th index
 ls_score=1st index
 cp_score=2nd index
 '''
  for k in range (idr.shape[0]):#calculating scores for function/array of IDRs
    if idr[k]<0.007:#if IDR<0.007
      scores[0]+=1 #increment scores[0] by 1
    elif idr[k]>0.05:#if IDR>0.05
      scores[2]+=1 #increment scores[2] by 1
    else:#if 0.007<IDR<0.05
      scores[1]+=1#increment scores[2] by 1

#most severe score is considered for labeling the dataset
  if scores[2]>0:#if scores[2]>0:
    scores = [0,0,1]#scores becomes [0,0,1]
  elif scores[1]>0:#if scores[1]>0 and scores[2]=0:
    scores = [0,1,0]#scores becomes [0,1,0]
  else: #if scores[0]>0 and scores[1]=0 and scores[2]=0
    scores = [1,0,0]#scores becomes [1,0,0]

  noisy_classification[int(i/2)]=scores#adding scores to noisy_classification array
  scores=np.array([0,0,0])#resetting scores array

#calculating accuracy without denoising.

b=0 # b=0
for i in range(pure_classification.shape[0]):#iterating over pure_classification
  if pure_classification[i][0] == noisy_classification[i][0] and pure_classification[i][1] == noisy_classification[i][1] and pure_classification[i][2] == noisy_classification[i][2] : #if pure and noisy classifications match
    b+=1 #increment b by 1

print("accuracy without denoising = ",end="") #priniting text
print(np.float64(b)/pure_classification.shape[0])#printing accuracy. divide b by the total number of classifications

print("denoising noisy test set")
#DENOISING TEST_NOISY_ACC
test_denoised_acc = np.zeros((test_noisy_acc.shape[0],test_noisy_acc.shape[1]),np.float64) # create empty array of zeros to store denoised signals of test_noisy
for i in range(test_noisy_acc.shape[0]): #iterate over test_noisy dataset
  z = rfft(test_noisy_acc[i]) #convert to freq DOMAIN
  z = z.reshape(1,701,1) #reshape the array containing the signal
  test_denoised_acc[i] = irfft(new_model.predict(z))#denoise the signal and convert it back to time domain an store it.


print("CALCULATING IDR BY CALCULATING RELATIVE DISPLACEMENTS BY DOUBLE INTEGRATION:denoised")
#GETTING CLASSIFICATION ARRAY FOR DENOISED SIGNALS
time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

denoised_classification = np.zeros((int(test_denoised_acc.shape[0]*0.5),3),np.float64)#creatng array of zeros for classinfiying the denoised signals

v = np.zeros(test_denoised_acc.shape,np.float64) #velocity
disp = np.zeros(test_denoised_acc.shape,np.float64) #displacement


for i in range(0,test_denoised_acc.shape[0]):#iterating over the test_denoised signals
  for j in range(1,test_denoised_acc.shape[1]):   #going over each sample
      v[i][j] = v[i][j-1] + (((test_denoised_acc[i][j-1]+test_denoised_acc[i][j])/2) * (time))#integrating for velocity

for i in range(0,test_denoised_acc.shape[0]):#iterating over the velocity signals
  for j in range(1,test_denoised_acc.shape[1]):   #going over each sample
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))#integrating for displacement

for i in range(0,disp.shape[0],2):#going over adjacent displacement array
  idr = np.zeros(disp.shape[1],np.float64)#creating array of zeros for IDRs

  for j in range(disp.shape[1]):#going over adjacent displacement arrays
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)#calculating IDRs

  '''
 if idr < 0.007       => Immediate Occupancy
 if idr 0.007 to 0.05 => Life Safety
 if idr >0.05         => Collapse prevention
 '''
  scores=np.array([0,0,0])#scores
  '''
 io_score=0th index
 ls_score=1st index
 cp_score=2nd index
 '''
  for k in range (idr.shape[0]):#calculating scores for function/array of IDRs
    if idr[k]<0.007:#if IDR<0.007
      scores[0]+=1 #increment scores[0] by 1
    elif idr[k]>0.05:#if IDR>0.05
      scores[2]+=1 #increment scores[2] by 1
    else:#if 0.007<IDR<0.05
      scores[1]+=1#increment scores[2] by 1

#most severe score is considered for labeling the dataset
  if scores[2]>0:#if scores[2]>0:
    scores = [0,0,1]#scores becomes [0,0,1]
  elif scores[1]>0:#if scores[1]>0 and scores[2]=0:
    scores = [0,1,0]#scores becomes [0,1,0]
  else: #if scores[0]>0 and scores[1]=0 and scores[2]=0
    scores = [1,0,0]#scores becomes [1,0,0]

  denoised_classification[int(i/2)]=scores#adding scores to denoised_classification array
  scores=np.array([0,0,0])#resetting scores array

#calculating accuracy with denoising.

b=0 # b=0
for i in range(pure_classification.shape[0]):#iterating over pure_classification
  if pure_classification[i][0] == denoised_classification[i][0] and pure_classification[i][1] == denoised_classification[i][1] and pure_classification[i][2] == denoised_classification[i][2] : #if pure and denoised classifications match
    b+=1 #increment b by 1

print("accuracy with denoising = ",end="") #priniting text
print(np.float64(b)/pure_classification.shape[0])#printing accuracy. divide b by the total number of classifications


print("DIRECT CLASSIFICATION USING ACCELEROMETER DATA")
##DIRECT CLASSIFICATION USING ACCELEROMETER DATA

#RELATIVE ACCELERATION CALCULATION
rel_acc_noisy = np.zeros((int(noisy_acc.shape[0]*0.5),noisy_acc.shape[1]),dtype=np.float64) # Creating array to store relative accelerations between noisy data
rel_acc_pure = np.zeros((int(pure_acc.shape[0]*0.5),pure_acc.shape[1]),dtype=np.float64) # Creating array to store relative accelerations between pure data

a=0 #a is assigned to 0

for i in range(rel_acc_pure.shape[0]):
  rel_acc_noisy[i] = noisy_acc[a+1] - noisy_acc[a] #evaluating the relative accelerations between adjacent noisy_acc
  rel_acc_pure[i]  = pure_acc[a+1] - pure_acc[a] #evaluating the relative accelerations between adjacent pure_acc
  a+=2 #incrementing a by 2 to get the next 2 accelerations
rel_acc_noisy = rel_acc_noisy.reshape(rel_acc_noisy.shape[0],rel_acc_noisy.shape[1],1) #reshaping the array

print("Getting the pure classifications from pure_acc")
### Getting the pure classifications from pure_acc
#For relative acceleration calculation we assume 2 simultaneous accelerometer-time signals

time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

pure_classification2 = np.zeros((int(pure_acc.shape[0]*0.5),3),np.float64) #creatng array of zeros for classinfiying the pure training signals

v = np.zeros(pure_acc.shape,np.float64) #velocity
disp = np.zeros(pure_acc.shape,np.float64) #displacement


for i in range(0,pure_acc.shape[0]): #iterating over the training pure signals
  for j in range(1,pure_acc.shape[1]):  #going over each sample
      v[i][j] = v[i][j-1] + (((pure_acc[i][j-1]+pure_acc[i][j])/2) * (time)) #integrating for velocity

for i in range(0,pure_acc.shape[0]): #iterating over velocities
  for j in range(1,pure_acc.shape[1]):   #double integration.
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))#integrating for displacement

for i in range(0,disp.shape[0],2):#going over adjacent displacement array
  idr = np.zeros(disp.shape[1],np.float64)#creating array of zeros for IDRs

  for j in range(disp.shape[1]):#going over adjacent displacement arrays
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)#calculating IDRs

  '''
 if idr < 0.007       => Immediate Occupancy
 if idr 0.007 to 0.05 => Life Safety
 if idr >0.05         => Collapse prevention
 '''
  scores=np.array([0,0,0])#scores
  '''
 io_score=0th index
 ls_score=1st index
 cp_score=2nd index
 '''
  for k in range (idr.shape[0]):#calculating scores for function/array of IDRs
    if idr[k]<0.007:#if IDR<0.007
      scores[0]+=1 #increment scores[0] by 1
    elif idr[k]>0.05:#if IDR>0.05
      scores[2]+=1 #increment scores[2] by 1
    else:#if 0.007<IDR<0.05
      scores[1]+=1#increment scores[2] by 1

  scores = np.floor(scores/(np.amax(scores))) #putting 1 in position that has maximum score and 0 in other positions
  pure_classification2[int(i/2)]=scores #adding scores to denoised_classification array
  scores=np.array([0,0,0])#resetting scores array

print("CNN FOR DIRECT CLASSIFICATION")
###CNN FOR DIRECT CLASSIFICATION
model3 = keras.Sequential([ #sequential model
   keras.layers.Conv1D(32, 32, strides=1, activation='relu'), #convolutional layer, kernal size=32, number of filters = 32
   keras.layers.Conv1D(32, 32, strides=1, activation='relu'),#convolutional layer, kernal size=32, number of filters = 32
   keras.layers.MaxPool1D(pool_size=8),#maxpool layer, pooling filtersize = 8
   keras.layers.Flatten(), #flatten layer
   keras.layers.Dense(512, activation='relu'),#fully connected layer, size = 512
   keras.layers.Dense(512, activation='relu'), #fully connected layer, size = 512
   keras.layers.Dense(3, activation='softmax')]) #fully connected layer, size = 3 to get output as one of the 3 classes

model3.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(),metrics=['accuracy']) # creating model using SGD optimizer and catagorical cross-entropy loss function.

model3.fit(rel_acc_noisy, pure_classification2, epochs=5, batch_size=16) # training the model. 5 epochs , batch size of 16

print("ANN FOR DIRECT CLASSIFICATION")
#ANN FOR DIRECT CLASSIFICATION
model4 = keras.Sequential([ #sequential model
   keras.layers.Flatten(),#flatten layer
   keras.layers.Dense(512, activation='relu'),#fully connected layer, size = 512
   keras.layers.Dense(512, activation='relu'),#fully connected layer, size = 512
   keras.layers.Dense(512, activation='relu'),#fully connected layer, size = 512
   keras.layers.Dense(3, activation='softmax')])#fully connected layer, size = 3 to get output as one of the 3 classes

model4.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(),metrics=['accuracy']) # creating model using SGD optimizer and catagorical cross-entropy loss function.

model4.fit(rel_acc_noisy, pure_classification2, epochs=5, batch_size=4)# training the model. 5 epochs , batch size of 4

print("final accuracy testing on unseen data!")
e = np.zeros((int(test_noisy_acc.shape[0]*0.5),test_noisy_acc.shape[1]),dtype=np.float64) #creating empty array  for storing test_noisy dataset's relative accelerations.
#e is the relative acceleration for the test_noisy_acc
a=0 # a   =   0

for i in range(e.shape[0]): # iterating over test_noisy dataset
  e[i] = test_noisy_acc[a+1] - test_noisy_acc[a] # evaluating relative acceletation and storing it in e
  a+=2 # increment a by 2 to get the next 2 accelerations.
e = e.reshape(e.shape[0],e.shape[1],1) # reshape the array e.

#EVALUATING MODELS
model3.evaluate(e,pure_classification) # evaluating the performance of the CNN on unseen data
