import csv
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, SpatialDropout2D

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_data():
    """
    Load training data and split it into training and validation set
    """
    all_data = []

    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            all_data.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(all_data, test_size=0.2)
    return train_samples, validation_samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # using matplotlib as rgb is used in drive.py!
                center_image = mpimg.imread(batch_sample[0])
                # print("image size: ", center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def build_model():
    """
    NVIDIA model
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=INPUT_SHAPE))
    model.add(Cropping2D(cropping=((60, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


def main():
    """
    Load train/validation data set and train the model
    """
    train_samples, validation_samples = load_data()
    # print('train', train_samples)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = build_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 4)

if __name__ == '__main__':
    main()
