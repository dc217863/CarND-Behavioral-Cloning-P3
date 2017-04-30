#**Behavioral Cloning** 

##Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "NVidia Model Visualization"
[image2]: ./examples/model_summary.png "Model Summary"
[image3]: ./examples/st_angle_hist.png "Steering Angle Histogram"
[image4]: ./examples/crop_example.jpg "Crop Example"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for communicating with the simulator. It receives the images for the neural net as input and sends the steering wheel angle and throttle as output.
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model uses the Nvidia CNN as it delivered the best results. The Nvidia CNN was also trained using the center, left and right camera images from a car albeit on a real car compared to the simulation used in this project.

Like in the paper from NVidia, the weights of the network is trained to minimize the mean squared error between the steering command output by the network and the simulator trained steering angles along with the adjusted steering command for off-center and flipped images.

![NVidia Model Visualization][image1]

Before the NVidia model, a lambda layer was used to normalize the images. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing. This was followed by a cropping function to concentrate the iamge on the actual lane and cut out the scenery and the visible dashboard in the images.

After cropping, the image has a shape of (80, 320, 3). There would be a possibility to resize this image to (66, 200, 3) as was done by NVidia as future work.

The model includes ELU layers after each convolutional or fully connected layer to introduce non-linearity. Using the standard RELU's as activation function has the disadvantage, that they can turn "dead", which means that they are never activated because the pre-activation value is always negative. The so called "Exponential Linear Unit" solves this problem, as for negative values it is a function bounded by a fixed value. In experiments, ELU units seem to learn faster and are preferred in many situations nowadays.

For regularization, L1 or L2 is not used, instead I focussed on Dropout. Dropout with drop rates of 0.5 were used in the model. A high drop rate was possible as the number of images used for the training are considered enough to pull this through. A high drop rate was used to avoid overfitting to the lake track.

In general, the NVIDIA model consists of strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution
with a 3×3 kernel size in the last two convolutional layers.The last neuron predicts the steering angle with no activation function at the end.

Summary of model:
![Model Summary][image2]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually. It uses moving averages of the parameters (momentum) to allow a larger effective step size. The algorithm will converge to this step size without fine tuning.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Only the data provided by Udacity was used to train the model. For each epoch, the data set was shuffled, and for each batch at the end, too.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVidia model as it was trained on a similar approach.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80 %) and validation set (20 %). 

The final step was to run the simulator to see how well the car was driving around track one. THe car was unable to make the curves even after a number of recovery laps in the training mode. Using the Udacity data, the same model was able to drive around the lake track without problems.

####2. Creation of the Training Set & Training Process

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as the validation loss stayed constant or even increased in the subsequent epochs.

####3. Preprocessing of images

Before pre-processing, the steering angle distribution of the training data looks like this:

![Steering Angle Historgram][image3]

This shows the need for further data. Four different techniques were used for data augmentation and pre-processing. Flipping, brightness change, cropping and shift techniques were implemented.

Flipping mirrors the image around the y-axis. This is important, because the first track has a bias towards left turns, so the network could develop such a bias, too. Of course, the steering angle also has to be inverted, which is done by multiplying it with a negative value of -1.

For the brightness change, the color space is first converted from RGB to HSV. Then, a random factor for the change is created, which is in the range of 0.5 to 1.5. So some images will be darkened, and some brightened. This factor is used for multiplication with the value channel, and then the image is converted back to RGB.

Finally, random shifting is also used. In x direction, the image is shift up to 50 pixel left or right. This also influences the angle, so a correction factor of 0.004 per shifted pixel is added. In y direction, the shift is -20 to 20 randomly.

Cropping was used in the model itself. Here is an example of an image fed into the model after cropping.

![Cropping Example][image4]

###Results

The vehicle is able to drive along the lake course lap without leaving the lane. 

###Future Prospects

To further the implementation for the second course and experiment further with the model and also the pre-processing of the images. Detection of lane boundaries could be an interesting idea here. Also, grayscaling of the images was not pursued during this project. 