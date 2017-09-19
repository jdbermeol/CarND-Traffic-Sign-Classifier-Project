# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the data set (see below for links to the project dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/1.png "Class distribution on training set"
[image2]: ./writeup_imgs/2.png "Class distribution on validation set"
[image3]: ./writeup_imgs/3.png "Class distribution on test set"
[image4]: ./writeup_imgs/4.png "Sample images"
[image5]: ./writeup_imgs/5.png "Transformed sample images"
[image6]: ./writeup_imgs/6.png "gayscale"
[image7]: ./writeup_imgs/7.png "Softmax new image 1"
[image8]: ./writeup_imgs/8.png "Softmax new image 2"
[image9]: ./writeup_imgs/9.png "Softmax new image 3"
[image10]: ./writeup_imgs/10.png "Softmax new image 4"
[image11]: ./writeup_imgs/11.png "Softmax new image 5"
[image12]: ./writeup_imgs/12.png "Conv 1 output"
[image13]: ./writeup_imgs/13.png "Conv 2 output"
[image14]: ./new_images/70.jpg "Speed limit (70km/h)"
[image15]: ./new_images/bicycle.jpg "Bicycles crossing"
[image16]: ./new_images/bumpy_road.jpg "Bumpy road"
[image17]: ./new_images/no_pass.jpg "No passing"
[image18]: ./new_images/stop_sign.jpg "Stop"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! And here is a link to my [project code](https://github.com/jdbermeol/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and pandas methods rather than hardcoding results manually.

I used numpy methods to calculate summary statistics of the traffic
signs dataset:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I include class distribution on each of our datasets(train, validation, and test). Also, I display some of the sign.

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Augmenting the training data

I decided to generate additional data because deep learning architectures improve as more data you have, and to balance class distribution. Samples are scale ([.9,1.1] ratio), rotation ([-15,+15] degrees), translated([-2, 2] pixel on each direction). All this transformation should not affect final classification. Here are some samples of the generated data, and the new distribution of classes in the training dataset.

![alt text][image5]

#### Pre-processing

Based on this [article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), most relevant data pre-processing is normalization and greyscale transformation.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 grayscale image                               | 
| pre-process                 | 32x32x1 grayscale image                               | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x32     |
| RELU                    |                                                |
| Max pooling              | 2x2 pool size, 1x1 stride,  outputs 28x28x32                 |
| Dropout              | 0.9 keep probability                 |
| Convolution 5x5         | 1x1 stride, valid padding, outputs 24 24 32     |
| RELU                    |                                                |
| Max pooling              | 2x2 pool size, 1x1 stride,  outputs 28x28x32                 |
| Dropout              | 0.9 keep probability                 |
| Fully connected        | outputs 100                                            |
| Fully connected        | outputs 43                                            |
| Softmax                |                                            |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After a few experiments, I used Adam optimizer to train the model, with a learning rate of 0.001, batch size of 128, during ten epocs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.382%
* validation set accuracy of 95.034%
* test set accuracy of 94.553%

* What architecture was chosen? I used the final architecture describe in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
* Why did you believe it would be relevant to the traffic sign application? It is the best submission for a competition on traffic sign classification; also it follows the same intuitions behind LeNet architecture.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Final accuracy results are excellent, however as we have a huge imbalance both in validation and test set, it is not sure if we are not overfitting or if we only learn to correctly clisify a few classes.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)              | Speed limit (20km/h)           | 
| Bicycles crossing                 | Bicycles crossing                        |
| Bumpy road                    | Bumpy road                              |
| No passing                  | No passing                                     |
| Stop            | Stop                                  |


The model was able to correctly guess four of the five traffic signs, which gives an accuracy of 80%. It compares below to the test set accuracy, even thought five images are very few, the discrepancy could be related to the class imbalance issue. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 35th and 36th cell of the Ipython notebook.

For the first image, the model is sure that this is a speed limit (20km/h) (probability of ~1), and the image does not contain that sign. The top five soft max odds were

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .999998689                     | Speed limit (20km/h)                              | 
| ~0                     | Speed limit (70km/h)                                         |
| ~0                    | Speed limit (30km/h)                                            |
| ~0                      | Speed limit (80km/h)                                     |
| ~0                    | Speed limit (50km/h)                                  |

![alt text][image7]

Second image

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Bicycles crossing                              | 
| ~0                     | Wild animals crossing                                         |
| ~0                    | Beware of ice/snow                                           |
| ~0                      | Road work                                     |
| ~0                    | Children crossing                                 |

![alt text][image8]

Third image

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Bumpy road                              | 
| ~0                     | Beware of ice/snow                                         |
| ~0                    | Road work                                           |
| ~0                      | No passing                                     |
| ~0                    | Slippery road                                 |

![alt text][image9]

Fourth image

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | No passing                              | 
| ~0                     | End of no passing                                         |
| ~0                    | Priority road                                           |
| ~0                      | Roundabout mandatory                                     |
| ~0                    | No entry                                 |

![alt text][image10]

Last image

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Stop                              | 
| ~0                     | Speed limit (30km/h)                                         |
| ~0                    | No entry                                           |
| ~0                      | Speed limit (50km/h)                                     |
| ~0                    | Yield                                 |

![alt text][image11]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications

![alt text][image12]

![alt text][image13]

At the first Convolution output, it seems model learn edges from the image and them in the second layer was trying to learn symbols in the sign. However, those symbols appear to have a problem with the form of the numbers(there is no clear difference between a seven and a 2).
