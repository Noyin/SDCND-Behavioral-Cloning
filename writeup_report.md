
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/normal.png "Normal Image"
[image3]: ./examples/flipped.png "Flipped Image"
[image4]: ./examples/cropped.png "Cropped Image"
[image5]: ./mse_vs_epoch.png "MSE VS Epoch"

---
My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* video.mp4 showing car driving in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Model architecture**

My model adopts the LENET-5 architecture which consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 139-150) 

The model uses ELU to introduce nonlinearity (code line 142, line 144), and the data is normalized in the model using a Keras lambda layer (code line 140). 

**Handling overfitting in the model**

The model contains a dropout layer in order to reduce overfitting (model.py lines 148). 

The model is trained and validated on dataset recorded from driving one lap around track one. Early stopping callback is implemented to ensure that the model is not overfitting (code line 160). The model is tested by running it through the simulator and ensuring that the vehicle stays on the track.

**Model parameter tuning**

The model uses an adam optimizer, so the learning rate is not tuned manually (model.py line 165).

**Training data**

Appropriate training data is collected to keep the vehicle driving on the road. I used a combination of center lane driving and dataset augmentation to train the model.

For details about how I created the training data, see the next section. 

**Model Architecture and Training Strategy**

**1. Solution Design Approach**

The overall strategy for deriving a model architecture is to first of all split the dataset into training set and validation set. Then start by training a simple model on the training set and validating on the validation set and iteratively increase the model's complexity until it performs well on the validation set and it is able to drive autonomously around the track without leaving the road.

The dataset is randomly split into a ratio of 70:30 (16074:6894) for training and validation set respectively.

The first test model architecture contains layers:

| Layer         		|     Description	        					  |
|:---------------------:|:-----------------------------------------------:|
| Flatten        		| input:160x320x3 RBG image   	                  |
| Dense  				| output 1        								  |


The second test model architecture contains layers:

| Layer         		|     Description	        					  |
|:---------------------:|:-----------------------------------------------:|
| Lambda        		| Normalization, input:160x320x3 RBG image   	  |
| Flatten        		| input:160x320x3 RBG image   	                  |
| Dense  				| output 1        								  |


The third test model architecture contains layers:

| Layer         		|     Description	        					  |
|:---------------------:|:-----------------------------------------------:|
| Lambda        		| Normalization, input:160x320x3 RBG image   	  |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:156x316x6 |
| Max pooling	      	| 2x2 stride,  outputs 78x158x6 				  |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:74x154x16 |
| Max pooling	      	| 2x2 stride,  outputs 37x77x16                   |
| Dense         		| outputs 1X120       						      |
| Dense         		| outputs 1x84        						      |
| Dense  				| output 1        								  |


The fourth and final test model architecture contains layers:

| Layer         		|     Description	        					  |
|:---------------------:|:-----------------------------------------------:|
| Lambda        		| Normalization, input:160x320x3 RBG image   	  |
| Cropping        		| Top:75 , bottom:25  	                          |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:156x316x6 |
| Max pooling	      	| 2x2 stride,  outputs 78x158x6 				  |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:74x154x16 |
| Max pooling	      	| 2x2 stride,  outputs 37x77x16                   |
| Dense         		| outputs 1X120       						      |
| Dropout				| 0.75								              |
| Dense         		| outputs 1x84        						      |
| Dense  				| output 1        								  |

A dropout layer was added to the final model to reduce overfitting. A preprocessing layer (Croppin2D) was added as well to help the model train better and focus on the lanes.

**2. Final Model Architecture**

The final model architecture (model.py lines 139-150) consists of a convolution neural network with the following layers:

| Layer         		|     Description	        					  |
|:---------------------:|:-----------------------------------------------:|
| Lambda        		| Normalization, input:160x320x3 RBG image   	  |
| Cropping        		| Top:75 , bottom:25  	                          |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:156x316x6 |
| Max pooling	      	| 2x2 stride,  outputs 78x158x6 				  |
| Convolution 5x5    	| 1x1 stride, valid padding,ELU,outputs:74x154x16 |
| Max pooling	      	| 2x2 stride,  outputs 37x77x16                   |
| Dense         		| outputs 1X120       						      |
| Dropout				| 0.75								              |
| Dense         		| outputs 1x84        						      |
| Dense  				| output 1        								  |


[//]: <Here is a visualization of the architecture:>
[//]: <[alt text][image1]>

**3. Creation of the Training Set & Training Process**

To capture good driving behavior, I recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To help the model recover if it was not on the center of the lane, I included images from both the left and right cameras.

I drove fast on straight sections of the track and I drove slow at curved sections of the track.This helped to create a balance in the data set between straight and curved sections.

To augment the data set, I also flipped images and angles so that the model is not be biased to driving to the left .For example, here is an image that has been flipped:

![alt text][image3]

After the collection process, I had 3828 data points. After including both images from the left and right cameras and  applying augmentation , I had 22,968 data points in total.

I finally randomly shuffled the data set and put 30.00% of the data into a validation set. 

I then preprocessed this data by normalizing and then cropping each image in both the training and validation set.

Here is an example of an image that has been cropped:

![alt text][image4]

*note the images are first normalized before cropping is applied. The image above is for visualization purpose.

I used this training data for training the model. The validation set helped determine if the model is over or under fitting. The ideal number of epochs is 5 Any value of epochs above 5 shows that the model overfits. I used an adam optimizer so that manually training the learning rate would not be necessary.
