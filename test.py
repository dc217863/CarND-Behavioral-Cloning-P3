import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print('lines', lines)

images = []
measurements = []
correction = 0.2

for line in lines:
    # print('line', line)
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
    image_right = cv2.imread(line[2])

    images.append(image_center)
    measurements.append(steering_center)

    if steering_center > 0.0:
        print('steering center = ', steering_center)
        images.append(image_left)
        images.append(image_right)
        # images.extend([image_center, image_left, image_right])

        measurements.append(steering_left)
        measurements.append(steering_right)
        # measurements.append([steering_center, steering_left, steering_right])

    # flip
    # images.extend([cv2.flip(image_center, 1), cv2.flip(image_left, 1), cv2.flip(image_right, 1)])
    # measurements.append([steering_center * -1.0, steering_left * -1.0, steering_right * -1.0])

X_train = np.array(images)
y_train = np.array(measurements)

print('shape', X_train.shape[:])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=X_train.shape[1:]))
model.add(Lambda(lambda x: x / 255.0) - 0.5, input_shape=X_train.shape[1:])

#LeNet
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
print('model summary:', model.summary())
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
