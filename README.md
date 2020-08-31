# The full project is in the subfolder MMAI894 - Please note this was written using Tensorflow V1, 2020 V2 was released code was note updated
# Indoor Scene Prediction #

Indoor Scene Recognition, also known as MIT67 image recognition dataset, consists of a total of 15,520 images of indoor environments across 67 categories. The images were collected from different online sources across the global such as Google, Flickr, and the famous LabelMe dataset. The different indoor categories vary dramatically across all spectrum from indoor airports to hair salon.  Moreover, the number of images per category also varies across, but it was guaranteed to contain at least 100 images per category. All the images have a bare minimum resolution of 200 pixel in the smallest axis, nevertheless images were in different sizes. With 67 classes, the dataset posed a challenge in variability and distinguishability between classes. Furthermore, there were very few distinctive attributes in comparison with similar category of images. For example, bedroom, children room, and nursery presented few objects that helped to distinguish from each other.  

![Image Sample](https://github.com/UWMonkey/CNN/blob/master/image/image_sample.png)

Many of the images and classes are highly characteristic and clear, where anchor objects can be easily detected. However, many images are not easy to distinguish, like library and bookstore, or movie theater and concert hall. 
As demonstrated concert halls and movie theaters both exhibit many rows of seats, and a big stage with a similar color setup. Finding distinguishable features between these two class would be a difficult task for convolutional neural network to classify images as it preserves the global spatial structure. Indoor Scene Recognition not only suffers from high inter-class similar problem but also suffer from high intra-class variability. 
![inter Sample](https://github.com/UWMonkey/CNN/blob/master/image/inter.png)
![intar Sample](https://github.com/UWMonkey/CNN/blob/master/image/intra.PNG)

### Data Pre-Processing 

Image preprocessing in the context of Convolution Neural Networks is as important as data preprocessing is for other data science projects involving textual or numerical data. Similar to our approach to model development, the data was pre-processed in its entirety using the TensorFlow library for Python. The following steps were taken as part of image pre-processing for all developed models contained within this report:

#### Step 1: Reading Images 
There is a total of 15,620 images spread across 67 folders. Each folder corresponds with an image category, such as an airport, bathroom, daycare etc., with all pictures associated with a particular category being placed in the respective folder. The following figure shows the main image directory.

1.	Each folder was one-hot encoded, and therefore was read as a number from 0 to 66 that was used as the class for all the images inside that folder. For the images, we only read the path of each image and associated its folder number with it as its class or category. 
2.	Image dataset was shuffled before feeding into the next step.

#### Step 2: Image Parsing 
1.	Images are read from their corresponding paths. 
2.	Decoded into .jpeg or .jpg extension.
3.	Resized according to model requirements.  
4.	Normalized the image array by a standard TensorFlow function. 

#### 3: Image Pipeline 
As part of the last step of image pre-processing, an image pipeline was built with a variable used to define the batch size of images to be read during each step of model training. This pipeline makes sure that the next batch is already read before the completion of training for the previous batch.  
The following diagram depicts an example of a TensorFlow image pipeline: 
![pipeline](https://github.com/UWMonkey/CNN/blob/master/image/pipe.png)


### Customized Model

Architecture:

The design of the image classification model task was divided into two steps process, base model and finetuned model. The base model was firstly designed to test the challenges presented by the dataset as well as getting hands on experience with building a convolution neural network. 
Base Model:

The base model was designed with 3 convolution layers each followed by a pooling layer and finally shaped with 3 fully connected layers. For simplicity, all the convolution layers will be using relatively small receptive fields of 3*3 kernel. As for the pooling layer, it will be a 2*2 max pooling with strides of 1. Finally, it is shaped into three 128 neurons fully connected layers, with the final layer being the soft-max layer. Rectification nonlinearity (ReLU) activation was applied to all hidden layer. The model did not incorporate any dropout regularization in the fully connected layer. However, it did incorporate normalization layer after each max pooling layer. 
![m1](https://github.com/UWMonkey/CNN/blob/master/image/model1.png)
![m1loss](https://github.com/UWMonkey/CNN/blob/master/image/model1loss.png)

#### First fine-tuned model:

The accuracy was not increasing over time and the loss was not decreasing over time for the base model, therefore a more complex architect is needed. Increasing the number of filters on each convolution layers, increasing the number convolution layers and increasing number of neurons on the fully connected layer were the three adjustments on top of the base model. The first finetune model was constructed with 5 convolution layers along with 5 pooling layers. Comparing the base model, there was a significant increase in the number of filters in each convolution layer. The first convolution layer uses a reception fields of 7*7 kernel and 64 filters. The second and third convolution layers continue double the filter size to 128 and 256 respectively while uses a reception fields of 7*7 kernel size. Lastly, the fourth and fifth convolution layers uses 512 filters with the same kernel size. The activation function is the same ReLu function as base model, while increase the neurons in the fully connected layers to be 2048. 
![m2](https://github.com/UWMonkey/CNN/blob/master/image/model2.png)
![m2loss](https://github.com/UWMonkey/CNN/blob/master/image/model2loss.png)
![m2loss](https://github.com/UWMonkey/CNN/blob/master/image/model2acc.png)

#### Second Fine-tuned model:

The second finetune model, simply add 1 convolution layer before each max pooling layer to the first finetune model. As the result it was similar architect as the famous model design “VGG16”. The second finetune model used 2X convolution layer before each 2*2 max pooling layer. The number of filters was left the same as the first finetune model with 64,128,256, and 512. However, the kernel size was reverted to a smaller reception field of 3*3 kernel.  The neurons in each of the three fully connected layers was doubled to 4096.

![m3](https://github.com/UWMonkey/CNN/blob/master/image/model3.PNG)
![m3acc](https://github.com/UWMonkey/CNN/blob/master/image/model3acc.png)


#### Third Fine-tuned Model:

Comparing to the second fine-tune model, the third fine-tune model changes the 3*3 kernel size in each convolution layers to be 7*7. There was noticeable performance increase when the same model uses larger reception field kernel size. Unlike outdoor images, the anchor objects are recognizable by smaller size pixels, whereas indoor scene images need larger pixel size to extract the correct features. A kernel of size 3*3 able to pick up 9-pixel square. By contract a 7*7 kernel size able to pick up 49-pixcel square. Although smaller kernels are picking up more details, however, it may not pick up certain edges. 
![m4](https://github.com/UWMonkey/CNN/blob/master/image/mode4.PNG)
![m4loss](https://github.com/UWMonkey/CNN/blob/master/image/mode4loss.png)
![m4acc](https://github.com/UWMonkey/CNN/blob/master/image/mode4acc.png)


Sample of tensorboard used 
![tensor](https://github.com/UWMonkey/CNN/blob/master/image/example%20tensorboard.png)

