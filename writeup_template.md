# **Traffic Sign Classifier Project** 

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

[image1]: Histograms.png "Visualization"
[image2]: Original-Image.gif "Image prior to pre-processing"
[image3]: After-Equalization.gif "Image after pre-processing"
[image4]: Internet%20Sample/1.jpg "Traffic Sign 1"
[image5]: Internet%20Sample/2.jpg "Traffic Sign 2"
[image6]: Internet%20Sample/3.jpg "Traffic Sign 3"
[image7]: Internet%20Sample/4.jpg "Traffic Sign 4"
[image8]: Internet%20Sample/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is a link to my [project code (in HTML format)](https://github.com/MartinKan/Uda-P2-TrafficSign/blob/master/Traffic_Sign_Classifier.html) and [ipynb format](https://github.com/MartinKan/Uda-P2-TrafficSign/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a set of 3 histograms showing the distribution of traffic sign classes across the training, validation and test data sets.

![Traffic Sign Histograms][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to perform histogram equalization on each of the images in the data set to improve their contrast levels by first converting them from the RGB image space to the YUV image space, which would in turn allow me to equalize those images using their Y channels.  I then converted the images back into RGB format for further processing. 

I experimented with other pre-processing methods (equalizing on the grayscale image space instead of the YUV image space and retaining the YUV image space after pre-processing) but did not find any meaningful differences in the accuracy rate.  I therefore decided to stick with the RGB image space after pre-processing.

Here is an example of a traffic sign image before and after histogram equalization.

Before equalization:

![Before equalization][image2]

After equalization:

![After equalization][image3]

As a last step, I normalized the entire data set by subtracting each image by the mean value of the data set and then dividing the result by the data set's standard deviation.  This has the effect of centering the values of the images around 0 which makes it easier to reduce the loss rate later on. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5 (L1) 	| 1x1 stride, valid padding, outputs 28x28x12 	|
| Batch Norm and RELU	|												|
| Convolution 5x5 (L2) 	| 1x1 stride, valid padding, outputs 24x24x32 	|
| Batch Norm and RELU	|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x32 				|
| Convolution 5x5 (L3) 	| 1x1 stride, valid padding, outputs 8x8x96 	|
| Batch Norm and RELU	|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x96   				|
| Flatten			    | outputs 1,536 								|
| Fully connected (L4)	| Random weights, zero bias, outputs 768  		|
| Batch Norm, RELU & Dropout	| 												|
| Fully connected (L5)	| Random weights, zero bias, outputs 384     	|
| Batch Norm, RELU & Dropout	| 												|
| Fully connected (L6)	| Random weights, zero bias, outputs 192   		|
| Batch Norm, RELU & Dropout	| 												|
| Fully connected (L7)	| Random weights, zero bias, outputs 96  		|
| Batch Norm, RELU & Dropout	| 												|
| Fully connected (L8)	| Random weights, zero bias, outputs 43    		|
| Logits & Softmax		|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I retained most the settings that were used in the tutorial including batch size (128), optimizer (AdamOptimzer) and learning rate (0.001).  The one major change I made in my code was to increase the number of epochs to 30.  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000 (100%)
* validation set accuracy of 0.985 (98.5%)
* test set accuracy of 0.960 (96.0%)

These results were calculated in the section of the code entitled "Train, Validate and Test the Model"

* What was the first architecture that was tried and why was it chosen?

I started off using the LeNet-5 architecture initially because it was canvassed in detail in the tutorial and was recommended by the course video as a good starting point.

* What were some problems with the initial architecture?

The accuracy rate was not high enough - the validation accuracy score would average at around mid 80.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I adapted from the LeNet-5 architecture and created a deeper network consisting of 8 layers and used batch normalization (for each layer) and dropout (for 3 of the fully connected layers) to improve the accuray rate (i.e. the adjustments were made due to underfitting)

* Which parameters were tuned? How were they adjusted and why?

I made each convolutional layer deeper by using extra filters (roughly double of that in the tutorial code).  This had the effect of bumping up the accuracy rate of the overall network (but made it slower at the same time).  Otherwise, I used the same set of parameters for the convolutional layers and pooling layers (e.g. kernel size and strides) in my code as those that were used in the tutorial code.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I found that a deeper network with more layers had the single biggest impact on the accuracy rate than anything else.  This is probably because having additional layers in a network allowed for more features to be identified accurately than a network with less layers.  The other design choices that I made (convolutional layer, max pooling, batch normalization and dropout) obviously played their part as well in creating an accurate model.  For example, a convolutional layer is essential for feature matching (e.g. lines and shapes) which makes it a good choice for this question as it involves the classification of images.  And adding a dropout layer helps to improve the accuracy rate of the model by forcing the nodes to learn the multiple characteristics of the neural network. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6] 
![Traffic Sign 4][image7] ![Traffic Sign 5][image8]


The first image might be difficult to classify because the red ring bordering the sign looks faded.

The second image might be difficult to classify because the program may confuse the figure at the center with something else.

The third image might be difficult to classify because the sign is tilted at an angle.

The fourth image might be difficult to classify because there are no horizontal or vertical lines in the sign.

The fifth image might be difficult to classify because the sign is tilted at an angle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Speed limit (60km/h) 					    | Speed limit (60km/h) 	   						| 
| Right-of-way at the next intersection    	| Right-of-way at the next intersection 		|
| Speed limit (30km/h)						| Speed limit (30km/h)							|
| Priority Road      						| Priority Road					 				|
| Turn left ahead							| Turn left ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.0%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a Speed limit (60km/h) sign (probability of 1.00), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (60km/h) 							| 
| 0.00     				| No passing 									|
| 0.00					| Slippery road									|
| 0.00	      			| Speed limit (50km/h)			 				|
| 0.00				    | Speed limit (80km/h)     						|


For the second image, the model is absolutely sure that this is a Right-of-way at the next intersection sign (probability of 1.00), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection 		| 
| 0.00     				| Beware of ice/snow 							|
| 0.00					| General caution								|
| 0.00	      			| Turn right ahead		    	 				|
| 0.00				    | Road work     								|

For the third image, the model is absolutely sure that this is a Speed limit (30km/h) sign (probability of 1.00), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)					 		| 
| 0.00     				| Speed limit (80km/h) 							|
| 0.00					| Speed limit (50km/h)							|
| 0.00	      			| Wild animals crossing		     				|
| 0.00				    | Speed limit (70km/h) 							|

For the fourth image, the model is absolutely sure that this is a Priority road sign (probability of 1.00), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road 								| 
| 0.00     				| End of all speed and passing limits			|
| 0.00					| End of no passing								|
| 0.00	      			| Traffic signals		    	 				|
| 0.00				    | No entry	     								|

For the fifth image, the model is absolutely sure that this is a Turn left ahead sign (probability of 1.00), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn left ahead						 		| 
| 0.00     				| Ahead only		 							|
| 0.00					| Beware of ice/snow							|
| 0.00	      			| No entry				    	 				|
| 0.00				    | Bumpy road     								|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


