import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('./data/IMG/' + 'center_2016_12_01_13_33_04_891.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flipping
img_flip_orig = image
img_flipped = cv2.flip(image, 1)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.25))
f.tight_layout()
ax1.imshow(img_flip_orig)
ax1.set_title('Original Image', fontsize=12)
ax2.imshow(img_flipped)
ax2.set_title('Flipped image', fontsize=12)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# Brightening
img_brighten_orig = image
image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
random_bright = .25 + np.random.uniform()
image[:, :, 2] = image[:, :, 2] * random_bright
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
img_brighten = image

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.25))
f.tight_layout()
ax1.imshow(img_brighten_orig)
ax1.set_title('Original Image', fontsize=12)
ax2.imshow(img_brighten)
ax2.set_title('Brightness changed', fontsize=12)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# Shifting
# Translation in x direction
trans_x = np.random.randint(0, 100) - 50
# Translation in y direction
trans_y = np.random.randint(0, 40) - 20

# Create the translation matrix
trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
rows, cols = image.shape[:2]

img_shift_orig = image
img_shifted = cv2.warpAffine(image, trans_matrix, (cols, rows))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.25))
# f.tight_layout()
ax1.imshow(img_shift_orig)
ax1.set_title('Original Image', fontsize=12)
ax2.imshow(img_shifted)
ax2.set_title('Shifted Image', fontsize=12)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
