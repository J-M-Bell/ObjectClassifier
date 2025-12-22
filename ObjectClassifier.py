import numpy as np
import pandas as pd
from keras import Input, activations
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Data Preparation and Processing
data = cifar10.load_data()
(X_trainset, y_trainset), (X_test, y_test) = cifar10.load_data()

X_trainset = X_trainset / 255
X_test = X_test / 255

X_train, X_validation, y_train, y_validation = train_test_split(X_trainset, y_trainset, test_size=0.2, random_state=0)

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
y_validation = output_encoder.transform(y_validation)
y_test = output_encoder.transform(y_test)

# print(X_train.shape)
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
model = Sequential([
    Input(shape=X_train.shape[1:4]),
    Flatten(),
    Dense(units=4, activation=activations.sigmoid),
    Dense(units=y_train.shape[1])
])
print(model.summary())


callbacks = []
checkpoint = ModelCheckpoint("weights/Test 2 (4)/5-4000-{loss:.4f}" + ".weights.h5", monitor='loss', verbose=1,
                                         save_best_only=True, save_weights_only=True, mode='min')
callbacks = [checkpoint]
model.load_weights("weights/Test 2 (4)/5-4000-5.3198.weights.h5")
model.compile(optimizer='adam', loss=CategoricalCrossentropy())
model.fit(X_train, y_train, epochs=5, batch_size=4000, shuffle=False, callbacks=callbacks, verbose=1)
