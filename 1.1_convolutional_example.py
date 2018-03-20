from util import readDatabase, showConfusionMatrix
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam


xTrain, yTrain, xTest, yTestCategorical, yTest = readDatabase()
print(xTrain.shape)
print(xTest.shape)
print(yTestCategorical.shape)

# Network parameters
firstConvLayerDepth = 2
numberOfNeurons = 200

# Training hyper-parameters
learningRate = 0.001
noOfEpochs = 20
batchSize = 32

numberOfClasses = yTrain.shape[1]
featureSize = xTrain.shape[1]

# Program parameters
verbose = 1
showPlot = True

# Network architecture

model = Sequential()
model.add(Conv2D(firstConvLayerDepth, kernel_size=(5, 5),
                 activation='relu',
                 strides=(2, 2),
                 padding='same',
                 input_shape=(32, 32, 3)))
# output is 16x16

model.add(Flatten())
model.add(Dense(numberOfNeurons, activation='relu'))
model.add(Dense(numberOfClasses, activation='softmax'))

sgd = Adam(lr=learningRate)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=xTrain,
          y=yTrain,
          epochs=noOfEpochs,
          batch_size=batchSize,
          verbose=verbose)

predictedValues = model.predict(xTest, batch_size=1)
showConfusionMatrix(yTest, predictedValues)
