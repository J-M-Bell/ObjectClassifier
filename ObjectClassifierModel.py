import numpy as np
import pandas as pd
from keras import Input, activations
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Data Preparation and Processing
data = cifar10.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

#Target transformation
output_encoder = LabelBinarizer()
y_train = output_encoder.fit_transform(y_train)
y_test = output_encoder.transform(y_test)

 # Shapes of data variables
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

#Model Creation

model = Sequential([
    Conv2D(filters=32, kernel_size=(5,5), activation=activations.relu),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=(7,7), activation=activations.relu),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=192, activation=activations.relu),
    Dense(units=10, activation=activations.softmax)
])
# print(model.summary())


callbacks = []
epochs = 50
batch_size = 4000

checkpoint = ModelCheckpoint(f"conv_weights/Conv2D Model 1/{epochs}e-{batch_size}bs" + "-{loss:.4f}" + ".keras", monitor='loss', verbose=1,
                                         save_best_only=True, mode='min')
callbacks = [checkpoint]
model.load_weights("/conv_weights/Conv2D Model 1/50e-4000bs-0.8914.keras")
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False, callbacks=callbacks, verbose=1, validation_split=0.1)
