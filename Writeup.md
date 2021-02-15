# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/augimage.png "Augmented Images"
[image2]: ./examples/explore.png "Exploration"
[image3]: ./examples/graph.png "Accuracy vs Epoch"
[image4]: ./TestImages/40Limit.png "Traffic Sign 1"
[image5]: ./TestImages/no_enter.png "Traffic Sign 2"
[image6]: ./TestImages/roundabout.png "Traffic Sign 3"
[image7]: ./TestImages/stop.png "Traffic Sign 4"
[image8]: ./TestImages/turn_left.png "Traffic Sign 5"
[image9]: ./examples/softmax.png "Softmax Graph"
[image9]: ./examples/grayscale.jpg "Grayscale"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is randomly sampled images wtih their class label and corrsponding sign name

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I thought feature would contrast more in grayscale


![alt text][image10]Here is an example of a traffic sign image before and after grayscaling.


As a last step, I normalized the image data because it is easier for optimiser to run on data that is zero mean and normalised

I decided to generate additional data because additional augemented data such as rotated, translated etc will improve the classifier and make it recognise non conventional image inuputs

To add more data to the the data set, I used the ImageDataGeneration class of the [Keras library,](https://keras.io/api/preprocessing/image/) which automatically rotates, zooms and shifts images

Here is an example of an original image and an augmented image:

![alt text][image1]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Grayscale image						| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Dropout layer	      	| 0.5 probability                  				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Dropout layer	      	| 0.5 prob                        				|
|Flatten layer          | output 1600									|
| Fully connected		|1600->240   									|
| Dropout layer	      	| 0.5 probability                  				|
| Fully connected		|240->84    									|
| Dropout layer	      	| 0.5 probability                  				|
| Fully connected		|840->43    									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate of 0.0005, epoch of 30, and batch size of 128. However, because of the augmented data generated by ImageDataGenerator, my effective batch size per epoch was around 5000.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.963 
* test set accuracy of 0.958

Here is a graph of accuracy vs epoch:
![alt text][image3]

I chose the LaNet architecture because we had implemented it before in class.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of digit recognition
The second image might be difficult to classify because of blurriness of the text
The third image should not be an issue
The fourth image might be difficult to classify because the sign is not in the center
The fifth image should not be an issue



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 40    	|Speed limit 20									|
| Do not enter       	|No passing 									|
| Stop Sign      		| Stop sign   									| 
|Roundabout         	|Roundabout 									|
| Turn Left				| Turn Left										|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is lesser than test accuracy of 95%, but that is probably because the do not enter sign wasnt part of the database
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
![alt text][image9]. 
The only image it had lower confidence was the 40 speed limmit, which it misclassified. This is probably because while it recognised it as a speed sign, there were multiple speed signs present, and couldnt determine which limit the sign was for.