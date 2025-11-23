import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
from random import uniform

"""
A simple program to approximate Analytical Functions
with DNN & TF2
"""

####################################################
# generate training data

dim_x = 2
dim_y = 1
no_samples = 10000
filename = "mymodel"

####################### Training data ##############
## x coord
aPnts = np.empty([no_samples, dim_x])  
for iI in range(no_samples):
    for iJ in range(dim_x):
        aPnts[iI][iJ] = uniform(-1.0, 1.0)
data = aPnts #np.random.random((no_samples, dim_x))

## y value
aTres = np.empty([no_samples,])
for iI in range(no_samples):
    aTres[iI] = math.cos(0.5 * math.pi * aPnts[iI][0]) * math.cos(0.5 * math.pi * aPnts[iI][1])
labels = aTres #np.random.random((no_samples,dim_y ))

###################### Test data ###################

aPnts2 = np.empty([no_samples, dim_x])  
for iI in range(no_samples):
    for iJ in range(dim_x):
        aPnts2[iI][iJ] = uniform(-1.0, 1.0)
data2 = aPnts2 #np.random.random((no_samples, dim_x))

## y value
aTres2 = np.empty([no_samples,])
for iI in range(no_samples):
    aTres2[iI] = math.cos(0.5 * math.pi * aPnts2[iI][0]) * math.cos(0.5 * math.pi * aPnts2[iI][1])
labels2 = aTres2 #np.random.random((no_samples,dim_y ))

####################################################

######### Model ####################################
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(dim_x,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add an output layer with 10 output units:
layers.Dense(dim_y)])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

#fit model
model.fit(data, labels, epochs=10, batch_size=32)


######## Test Accuracy #############################

test_loss, test_acc = model.evaluate(data2,  labels2, verbose=2)
print('\nTest accuracy:', test_acc)

######## Predict individual values #################
predictions = model.predict(data2)
x = data2[0]

x1 = x[0,]
y1 = x[1,]
print(" point to test")
print (x[0,], "  ",x[1,])
#x.shape
#print(x.shape, data2.shape)

## Analytical solution:
res = math.cos(0.5 * math.pi * x1) * math.cos(0.5 * math.pi * y1)
print("NN prediction: " , predictions[0], "Analytical solution", res, "difference" ,abs(predictions[0]-res))

#####################################################
# Save weights to a TensorFlow Checkpoint file
print("store the model")
model.save_weights(filename)

# this requires a model with the same architecture.
print("reload the model")
model.load_weights(filename)

predictions = model.predict(data2)
res = math.cos(0.5 * math.pi * x1) * math.cos(0.5 * math.pi * y1)
print("NN prediction: " , predictions[0], "Analytical solution", res, "difference" ,abs(predictions[0]-res))

#####################################################
#https://www.tensorflow.org/guide/keras/train_and_evaluate#using_the_gradienttape_a_first_end-to-end_example

######  A second model with Gradient Tape ###########
inputs = keras.Input(shape=(dim_x,), name='x-coord')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(dim_y, name='predictions')(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# Instantiate a loss function.
loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

# Prepare the training dataset.
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 10
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    # Open a GradientTape to record the operations run
    # during the forward pass, which enables autodifferentiation.
    with tf.GradientTape() as tape:

      # Run the forward pass of the layer.
      # The operations that the layer applies
      # to its inputs are going to be recorded
      # on the GradientTape.
      logits = model2(x_batch_train, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model2.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model2.trainable_weights))

    # Log every 200 batches.
    if step % 200 == 0:
        print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
        print('Seen so far: %s samples' % ((step + 1) * 64))
        
        
######## Predict individual values #################
predictions2 = model2.predict(data2)
x = data2[0]

x1 = x[0,]
y1 = x[1,]
print(" point to test")
print (x[0,], "  ",x[1,])
#x.shape
#print(x.shape, data2.shape)

## Analytical solution:
res = math.cos(0.5 * math.pi * x1) * math.cos(0.5 * math.pi * y1)
print("NN prediction: " , predictions2[0], "Analytical solution", res, "difference" ,abs(predictions2[0]-res))
