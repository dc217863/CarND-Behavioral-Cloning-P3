import csv
import matplotlib.pyplot as plt
import cv2
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

    training_angles = [float(row[3]) for row in all_data]
    plot_histogram(training_angles)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(all_data, test_size=0.2)
    return train_samples, validation_samples


def plot_histogram(angles):
    """
    plot histogram of all steering angles in training data
    :param angles: steering angles
    :return:
    """
    plt.hist(angles, bins=50)
    plt.xlabel('Steering angles')
    plt.ylabel('Quantity')
    plt.title('Steering angle distribution in training data')
    plt.show()


def generator(samples, mode, batch_size=64):
    num_samples = len(samples)
    if mode == 'train':
        cameras = ['center', 'left', 'right']
    else:
        cameras = ['center']

    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for cam in cameras:
                    if mode == 'train':
                        augment = np.random.choice(['flip', 'brighten', 'shift', 'none'])
                    else:
                        augment = 'none'

                    if cam == 'center':
                        image = cv2.imread('./data/' + batch_sample[0])
                        angle = float(batch_sample[3])
                    elif cam == 'left':
                        image = cv2.imread('./data/' + batch_sample[1])
                        angle = float(batch_sample[3]) + 0.2
                    elif cam == 'right':
                        image = cv2.imread('./data/' + batch_sample[2])
                        angle = float(batch_sample[3]) - 0.2

                    # convert to rgb
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image, angle = augment_image(augment, image, angle)

                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def augment_image(augment, image, angle):
    """
    randomly augment image
    :param augment: one of 'flip', 'brighten', 'shift', 'none'
    :param image: the image to be augmented
    :param angle: steering angle
    :return: image, angle
    """
    if augment == 'flip':
        image = cv2.flip(image, 1)
        angle *= -1.0

    elif augment == 'brighten':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        image[:, :, 2] = image[:, :, 2] * random_bright
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    elif augment == 'shift':
        # Translation in x direction
        trans_x = np.random.randint(0, 100) - 50
        # Correct angle
        angle += trans_x * 0.004
        # Translation in y direction
        trans_y = np.random.randint(0, 40) - 20
        # Create the translation matrix
        trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        rows, cols = image.shape[:2]
        image = cv2.warpAffine(image, trans_matrix, (cols, rows))

    return image, angle


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

    # compile and train the model using the generator function
    train_generator = generator(train_samples, 'train', batch_size=64)
    validation_generator = generator(validation_samples, 'valid', batch_size=64)

    model = build_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 15)

    # Save it to a file and show message again
    model.save('model.h5')
    model.summary()

if __name__ == '__main__':
    main()
