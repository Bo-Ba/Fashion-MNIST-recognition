import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()

imgRows, imgCols = 28, 28

trainImages = trainImages / 255.0
testImages = testImages / 255.0

inputShape = (imgRows, imgCols, 1)

trainImages = trainImages.reshape(trainImages.shape[0], imgRows, imgCols, 1)
testImages = testImages.reshape(testImages.shape[0], imgRows, imgCols, 1)

trainLabels = keras.utils.to_categorical(trainLabels)
testLabels = keras.utils.to_categorical(testLabels)

model = tf.keras.Sequential([
    keras.layers.Conv2D(15, (3, 3), padding='same', activation='relu', input_shape=inputShape),
    keras.layers.Conv2D(15, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(30, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(30, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(735, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

stop = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

history = model.fit(trainImages, trainLabels, epochs=100, validation_split=0.05, callbacks=[stop], batch_size=64)

test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(testImages)

np.argmax(predictions[0])


def drawCurves(history):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'], "r--")
    plt.plot(history.history['val_' + 'accuracy'], "g--")
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0.7, 1.00))
    plt.legend(['train', 'test'], loc='best')

    plt.show()


def drawCorrectPercentage(netPredictions):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    predictions = np.argmax(netPredictions, axis=1)
    true = np.argmax(testLabels, axis=1)
    predictions = np.vstack((predictions, np.equal(predictions, true)))
    trueNumbers = np.zeros(10)

    for column in predictions.transpose():
        trueNumbers[column[0]] += column[1]
    trueNumbers /= 10

    plt.figure(figsize=(10, 4))
    plt.barh(classes, trueNumbers, color='green')
    plt.xlabel("Correctness percentage")
    plt.yticks(classes)
    for index, value in enumerate(trueNumbers):
        plt.text(value, index, str(value))
    plt.show()


drawCurves(history)
drawCorrectPercentage(predictions)
