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

X_train = X_train / 255
X_test = X_test / 255

# X_train, X_validation, y_train, y_validation = train_test_split(X_trainset, y_trainset, test_size=0.2, random_state=0)

# Shapes of data variables
# print(X_trainset.shape)
# print(X_test.shape)
# print(y_trainset.shape)
# print(y_test.shape)
# print(X_train.shape)
# print(X_validation.shape)
# print(y_train.shape)
# print(y_validation.shape)

#Target transformation
output_encoder = LabelBinarizer()
y_train = output_encoder.fit_transform(y_train)
# y_validation = output_encoder.transform(y_validation)
y_test = output_encoder.transform(y_test)

print(X_train.shape)
# print(X_validation.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_validation.shape)
# print(y_test.shape)


# print(X_train.shape[1:4])
# print(y_train.shape[1])

#Testing
print(X_train.shape[1:2])
#Model Creation
# model = Sequential([
#     Input(shape=X_train.shape[1:4]),
#     Flatten(),
#     Dense(units=2, activation=activations.relu),
#     Dense(units=2000, activation=activations.relu),
#     Dense(units=4, activation=activations.relu),
#     Dense(units=1500, activation=activations.relu),
#     Dense(units=6, activation=activations.relu),
#     Dense(units=1000, activation=activations.relu),
#     Dense(units=20, activation=activations.relu),
#     Dense(units=1000, activation=activations.relu),
#     Dense(units=y_train.shape[1], activation=activations.softmax)
# ])

model = Sequential([
    Input(shape=X_train.shape[1:4]),
    Flatten(),
    Dense(units=2, activation=activations.relu),
    Dense(units=2000, activation=activations.relu),
    Dense(units=4, activation=activations.relu),
    Dense(units=1500, activation=activations.relu),
    Dense(units=6, activation=activations.relu),
    Dense(units=1000, activation=activations.relu),
    Dense(units=20, activation=activations.relu),
    Dense(units=1000, activation=activations.relu),
    Dense(units=y_train.shape[1], activation=activations.softmax)
])
print(model.summary())


callbacks = []
checkpoint = ModelCheckpoint("weights/2-2000-4-1500-6-1000-20-1000/100-4000-{loss:.4f}" + ".keras", monitor='loss', verbose=1,
                                         save_best_only=True, mode='min')
callbacks = [checkpoint]
# model.load_weights("weights/Test 3 (20)/100-4000-2.0240.keras")
model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=4000, shuffle=False, callbacks=callbacks, verbose=1, validation_split=0.1)
