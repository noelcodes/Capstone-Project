# [Home Page](https://noelcodes.github.io/)
# Capstone-Project
### Title: Enhance shopping experience thru product image recognition
#### Technique : Image Classification using Keras + Tensorflow Object Detection API + OpenCV + Faster-RCNN
![alt text](https://i.imgur.com/0y1SB7u.jpg)

#### Here's a VIDEO of the final product where I showcase during Meet and Greet hiring event. 
[![LIVE DEMO](https://github.com/noelcodes/Capstone-Project/blob/master/ezgif.com-video-to-gif%20(1).gif)](https://youtu.be/CwzLjc1-kj8)


### Introduction:
Many of us do shopping regularly. Wouldn't it be awesome if you could pick up your phone, switch on the camera app, scan items around the shopping mall, to recognize the product of interest and displays its price and discounts in real-time? My project is about image classification of 12x household products and classified according to its labels then tag this labels to its advertising slogan. The finished project had to locate the product items thru an image or video stream on webcam, draw a bounding box around them, to display "made-up" prices and discounts.

![alt text](https://i.imgur.com/rKorkmi.jpg)  

### Overview
I will break this up into 4 parts:
1) Build CNN from scratch, Image classification using Keras. This trains a model to classify all 20,000 images of household product into its labels.
2) Build CNN via Transfer Learning + Tensorflow Object Detection API. This locate the object and draws the bounding box around it.
3) OpenCV for video streaming on webcam.
4) Put all the above together + Demo.

![alt text](https://i.imgur.com/dj6ZReY.jpg)

### Part 1) Image classification using Keras
Reference to : https://github.com/noelcodes/Capstone-Project/blob/master/CNN_train-12Classes.ipynb

Before I jump into the coding. There is a few concept you had to know.
- Images are matrix of pixel values. 

![alt text](https://i.imgur.com/qFIfB2z.gif)  
- A greyscale (black and white) has one number for each pixel which describes the brightness of that pixel. The numbers range from 0=black to 1=white to indicate how bright that part of the photo should be. A standard size for a photo is 1000 x 667 pixels is height x width, represented in matrix (1000 x 667 x 1), where 1 means greyscale. 
- For a colour photograph, each pixel is described by three numbers, ranging from 0 to 1, that describe the redness, blueness and greenness of that pixel. The photo is described by a 3D array (a tensor) of size 1000 x 667 x 3.
![alt text](https://i.imgur.com/4VaJwVv.png)    
- Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. I'll be using Keras to do most of the layer construction.

### Image data generation and augmentation
Keras provides the ImageDataGenerator class that defines the configuration for image data preparation and augmentation. 

```
train_batches = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True).flow_from_directory(train_path, target_size = (128, 128), 
                                                           classes=['clock','massage_chair','microwave','swing','stools','rice_cooker',
                                                                  'frying_pan','chopper', 'pots','scissors','wine_cooler','wine_glass'],
                                                                   batch_size = 20)
                                                                   
valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(valid_path, target_size = (128, 128), 
                                                         classes=['clock','massage_chair','microwave','swing','stools','rice_cooker',
                                                                  'frying_pan','chopper', 'pots','scissors','wine_cooler','wine_glass'],
                                                         batch_size = 20)
```

ImageDataGenerator() is a keras object which generates batches of tensor image data. For CNN, an input must be a 4-D tensor [batch_size, width, height, channels], so each image is a 3-D sub-tensor. ImageDataGenerator() will turn images to matrix in backend. You can also augment the images in order to increase the number of images for training. Meaning, if take for example zoom_range=0.2, you have just doubled your number of images with the new set zoomed by 20%. This is very useful if your train image is less than 200. You may flip, shear, rescale and many more written in the documentation. [https://keras.io/preprocessing/image/]

```
# Default values in ImageDataGenerator() class
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0)
```


flow_from_directory() is where you tell keras where is your images, located at which path. Also for training all images must be of the same size - WIDTH and HEIGHT, therefore, Target_size defines the size of image in matrix form. 'Classes' turns the labels to one-hot-encoded based. chopper=[0,0,1,0...0], clock=[1,0,0...0]. Batch_size defines how many images to feed into the network at ones, and if you recalled previously, this is the 4th Dimension in Tensor. 

### Convolution Neural Network (CNN)
CNN is a technique use to classifiy images via a series of deep learning layers.

```
# Below is a Simplified version of CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))  
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())  
classifier.add(Dense(units = 128, activation = 'relu')) 
classifier.add(Dense(units = 12, activation = 'softmax'))  
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```
Question: How do I design my model?
Answer: 1) borrow architecture from other models (github + youtube + blogs + DS online course) 
        2) Experience with my previous models. (based on Accuracy score + Confusion Matrix)
        3) trial-and-error (no gridsearch)

Below I will explain what each layer is doing:

- Sequential() tells keras that we will be buliding sequential layers for our model. [https://keras.io/getting-started/sequential-model-guide/].
- Conv2D() is our 1st layer. 32 is the number of filters. You can use other numbers. But I usually see from other practitioners, they usually used multiples of 32 (example 64, 128, 256, etc..). (3,3) is the filter matrix size. Input_shape 128 x 128, is the same as previously mentioned about Target Size. Next is the '3' meaning we are dealing with RGB. And lastly 'relu' the activation method.

![alt text](https://i.imgur.com/u29H1cH.gif)    

So what is happening here in this Conv2D line, there are 32 sets of different filters, each filter size 3x3 matrix, are strided across a 128x128 colored image, to perform dot multiplication. The output of this Math operation is a bunch of matrix called Feature Maps. 

![alt text](https://i.imgur.com/LbjwPFV.gif)  

The purpose is to extract the features of the images, example the lines, curves, shape, and many more.

Below are what fitlers looked like to have a better idea. These filter images are taken from a blog from Francois Chollet, you should definetly check it out. [https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html]

![alt text](https://i.imgur.com/PeMOau0.jpg)    

- ReLU activations function outputs a 0 if the input is less than 0, and otherwise output is the raw input itself. That is, if the input is greater than 0, the output is equal to the input. ReLUs' machinery is more like a real neuron in your body. Research has shown that ReLUs result in much faster training for large networks. Most frameworks like TensorFlow and TFLearn make it simple to use ReLUs on the the hidden layers, so you won't need to implement them yourself.

- MaxPooling2D(pool_size = (2, 2)). Next the matrix are fed into the Maxpool layer [https://keras.io/layers/pooling/]
The same concept as before applies but this time its a 2x2 matrix, stride across the Feature Map matrix, taking the MAX value in that matrix. The purpose is to down-sampled the matrix, so that it is easier to computate later. Although it reducing the size but the important features is retained. 

![alt text](https://i.imgur.com/rw1tnpm.jpg)    

- Flatten() layer, basically as the name suggested, flatten the tensor matrix (was a 3D matrix) to a 1D. So that it can be fed into the dense layer a row at a time.

![alt text](https://i.imgur.com/ujndZCj.jpg)      

- Dense() Layers. The 2x dense forms the Fully Connected Layer. With the input layer having 128 nodes and the Output layer stating the number of classes we want to predict, where in our class is 12 class. ReLU activations will once again be used for all layers. The output dense layer softmax activation (for purposes of probabilistic classification). 

- Softmax activaton: I choose softmax at last node because it is usually used for multi-classification, probabiliies sum will be 1, so the high value will have the higher probability than other values. Softmax Activation function highlights the largest input and suppresses all the significantly smaller ones. Yes, the final output is in a form of probability, since the network is predicting based on the features. So in simplest explaination, when model sees 4x wheels, some windows, boot and headlights, its probably a CAR. 

![alt text](https://i.imgur.com/SA4cJoF.png)

##### Summary of my final CNN

![alt text](https://i.imgur.com/EGmKvar.jpg)     

- The above is the summary of our model. The Param here means the number of trainable nodes in each layer. Then below are the total of trainable nodes. The larger the number, the longer it takes to do training. I have a GTX1060 GPU, it took me about 3hrs to train 20,000 of 12x classes into its labels. You should use cloud if you do not have a GPU of the similar grade as mine. I have been using AWS EC2 (Amazon Web Services)

![alt text](https://i.imgur.com/It5siXG.jpg)

![alt text](https://i.imgur.com/U63wZCr.jpg)

As soon as the accuracy hover around close to 1, and the loss hover near 0, the training is sufficent. Training can stop.

#### Test metrics:
This 1st diagram shows the confusion metrics. Each category, we test the model with 10 images. In the diagram, the objective is to get a diagonal number all 10. As you can see, most of the items are classified correctly, except for pot. Pot are sometimes predicted as a Clock or a Rice_cooker. Reason? CNN takes the features for prediction, pot and clock and rice_cooker have certain feature that are really similar, such as its curve lines and shape. To overcome this, a better network design is needed, or training for longer period of time, or add more images for training. 
![alt text](https://i.imgur.com/PIb1wBn.jpg)  

![alt text](https://i.imgur.com/PPpyX85.jpg)  

##### Testing the saved model. Live demo. Recorded on video.
Reference to : https://github.com/noelcodes/Capstone-Project/blob/master/CNN_train-12Classes.ipynb/Demo-12Classes.ipynb

Now that the model is trained. I will to test it using images that the model has never seen before.

[![LIVE DEMO](https://i.imgur.com/bJixWTi.jpg)](https://youtu.be/r_jMNsYlsbY)

### Part 2: Tensorflow Object Detection API
Reference to: https://github.com/noelcodes/Capstone-Project/blob/master/object_detection_noel.ipynb

In Part 1, I have bulit CNN to classify 12x classes. I have a week left before project deadline. I can choose to stop here, or explore other techniques. I feel that I should try out Object Detection.

The real world is very busy with multiple activities happening at the same time.  I want not just classifying 1x single object, rather multiple overlapping objects with different backgrounds. And in order to see the relationship between different objects, in a single photo, we need to draw a bounding boxes around them. Therefore, our CNN had to predict, many different labels as well as their X Y coordinates of its respective bounding box, all at once.

This is actually a very complex operation, which requires tons of coding. I left 1 week to deadline, so not enough time. Then I came across Google's open sourse API called Tensorflow Object Detection. I have applied this API into this static image. As you can see, it performs rather well, identifying the table, bottle, wine glass, and the person.

![alt text](https://i.imgur.com/DBui9Xl.jpg) 

The API also made it easy for me to train custom images on pre-trained models. I choose Faster-RCNN beacause of its accuracy. I wanted to use the keras model which was trained in Part 1, but the API did not like H5 format as weights, so I did not use it. In order to train custom images on Faster-RCNN, a couple of extra steps needed to do. I will explain in details in Part 4.


##### API Installation
If you are lazy to read documentations and just want to get it up and running, you are in luck. there's a no bullshit, just follow installation guide.
I have a Windows 10, GTX1060 GPU, 16GB DDR3, Intel i&-7700HQ 2.8GHz. You don't need to be the same specs as mine, just windows 10 and a decent GPU will still works. Software side, I'm using VS2015, Nvdia 9.0 and Cudnn 7. Install them if you want to use GPU, which is w...way faster.
```
1) Download and install [Anaconda 5.1 For Windows Installer, choose Python 3.6] (https://www.anaconda.com/download/)
2) Open Anaconda prompt. Start -> Anaconda 3 -> Anaconda prompt -> type cd C:\
3) C:\>mkdir C:\tensorflow1
4) C:\>cd C:\tensorflow1
5) C:\tensorflow1>
6) Download and unzip into C:\tensorflow1> : [Tensorflow Models](https://github.com/tensorflow/models/archive/master.zip)
7) Download and unzip into C:\tensorflow1\models\research\object_detection>: [Object detection codes(https://github.com/noelcodes/Capstone-Project/archive/master.zip)
8) C:\tensorflow1> conda create -n tensorflow1 pip python=3.5
9) C:\tensorflow1> activate tensorflow1
10) (tensorflow1) C:\tensorflow1> pip install --ignore-installed --upgrade tensorflow-gpu
11) (tensorflow1) C:\tensorflow1> conda install -c anaconda protobuf
12) (tensorflow1) C:\tensorflow1> pip install pillow
13) (tensorflow1) C:\tensorflow1> pip install lxml
14) (tensorflow1) C:\tensorflow1> pip install Cython
15) (tensorflow1) C:\tensorflow1> pip install jupyter
16) (tensorflow1) C:\tensorflow1> pip install matplotlib
17) (tensorflow1) C:\tensorflow1> pip install pandas
18) (tensorflow1) C:\tensorflow1> pip install opencv-python
19) (tensorflow1) C:\tensorflow1> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
20) (tensorflow1) C:\tensorflow1> cd tensorflow1\models\research
21) (tensorflow1) C:\tensorflow1\models\research> protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
22) (tensorflow1) C:\tensorflow1\models\research> python setup.py build
23) (tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

##### Let's do object detection on static image.
- Go to path models/research/object_detection/test_images/ , paste your image there and rename image1.jpg.
- Open models/research/object_detection/object_detection_tutorial.ipynb in jupyter notebook.
- Run it, warning it takes a while.
- Tar Tar...!! Magic!!
- If it doesn't work, copy the codes in this github, object_detection_noel.ipynb , paste on exact same folder. Run it!

### Part 3: Now let's use OpenCV to stream videos.
Reference to: https://github.com/noelcodes/Capstone-Project/blob/master/object_detection/Object_detection_webcam.py

Videos are actually streams of images. This is where OpenCV comes in.
Run "object_detection_webcam.py" in terminal instead of notebook for stability.

```
import cv2
cap = cv2.VideoCapture(0)
    while True:
      ret, image_np = cap.read()
      
      ###  Image Prediction model put here. ### 
      
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
```
Calling cv2 from OpenCV, VideoCapture(0) means switch ON webcam. If you have another camera, try change to VideoCapture(1). Next cap.read() separate the video into 2 variable, a True/False (ret) meaning whether there is video or not, and the other variable is a static image (image_np). The idea is to put this static image into Object Detection API in Part2 for prediction. Since this is in video streaming, we are feeding streams of images for multiple classification continuously. Is this fun??

![alt text](https://i.imgur.com/cX1WlVR.jpg)

### Part 4: Put everything together.
Reference to: https://github.com/noelcodes/Capstone-Project/blob/master/object_detection/Object_detection_webcam.py
After the steps below, it will replace the Inference Graph. 

If you have not spotted, for Part 2 & 3, the model I was using is a pre-trained model. Where is my 12x classes of household products which I trained in Part 1? Well, I'm not going to use the model in Part 1 created via Keras, because accuracy is only 85%. Not bad, but not good enough. The API also do not like Keras's H5 weights. That's why I do Part 2 and 3, where tensorflow object detection api allows fine tune the existing pre-train model. There are a list of model selection from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Some of the most relevant system types today are Faster R-CNN, R-FCN, Multibox Single Shot Detector (SSD) and YOLO (You Only Look Once).

![alt text](https://i.imgur.com/AK9BXTu.jpg)

Originally I used Multibox Single Shot Detector (SSD), but accuracy is not good, even thought it is fast. Here's I have choosen Faster R-CNN, because based on the chart above, although it is computational expensive, but it can achieve high accuracy. As you can see that the architecture is extremely complex. 

![alt text](https://i.imgur.com/tezXUpC.png)

#### Image praparation
Unfortunately, in order to train the Faster R-CNN model using custom images, the model needs all .xml file containing the label data for each image , a csv file containing summary of xml and labelmap file containing the label details. Below are the steps to create this.

##### Installation
```
1) Clone https://github.com/tzutalin/labelImg
2) Anaconda Prompt and go to the labelImg directory
3) conda install pyqt=5
4) pip install sip
5) pyrcc5 -o resources.py resources.qrc
6) python labelImg.py
```
##### Create .xml files
```
Click 'Open Dir'
Click 'Create RectBox'
Click and release left mouse to select a region to annotate the rect box.
Type label name into label box (only need to do once for each class)
Click Save, same name as image.
Repeat this for every images.
```

##### Create csv file.
```
Anaconda Prompt
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```

##### Generate TFRecord file
Open the generate_tfrecord.py file in a text editor. Then change the label to desired class names. Add more if you want.
```
    if row_label == 'clock':
        return 1
    elif row_label == 'chopper':
        return 2
    elif row_label == 'scissors':
        return 3
    else:
        None
```        
Then do this.
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

##### Create labelmap.pbtxt
Open a text editor, create a file name labelmap.pbtxt, copy paste below example, save in C:\tensorflow1\models\research\object_detection\training folder. 
```
item {
  id: 1
  name: 'clock'
}

item {
  id: 2
  name: 'chopper'
}

item {
  id: 3
  name: 'scissors'
}

...... many more you can type in yourself.
```
##### Train
From the \object_detection directory
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```
While training, open another prompt, do this.
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training
```
Copy paste the URL into browser to view real time result in Tensorboard.
Ctrl+C to terminate training as soon as the graph flatten close to 0.
![alt text](https://i.imgur.com/fh3db1N.jpg)

##### After many hours of trainig, Export Inference Graph.
Goto training folder, pick the largest checkpointnumber, replace on XXXX below with this checkpoint number.
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

##### Let's do webcam
Tweak the dictionary, replace with advertising mock-up price and promotions. Example
```
category_index2 = {1: {'id': 1, 'name': 'chopper $39'}, 2: {'id': 2, 'name': 'Clock buy 2 get 1'}, 3: {'id': 3, 'name': 'wine_glass Xmas special'}, 4: {'id': 4, 'name': 'microwave gift for your mum'}, 5: {'id': 5, 'name': 'scissors $2'}, 6: {'id': 6, 'name': 'frying_pan New year special'}}

```
Run it in terminal.
```
(tensorflow1) C:\tensorflow1\models\research\object_detection>python Object_detection_webcam.py
```

##### Demo on 12x household products.

[![LIVE DEMO](https://github.com/noelcodes/Capstone-Project/blob/master/ezgif.com-video-to-gif%20(1).gif)](https://youtu.be/CwzLjc1-kj8)

You are looking at Fast RCNN model on my custom image and labels and some mock-up advertising prices and promotion.

## Conclusion:
I have tried train these images via keras method on my own model design. The result is not perfect with just 80% accuracy. Most likely it is due to the simple artchitecture design. Given more time, I should be able to create a better CNN design. However, it is only thru this exercise, I learnt about what each layers in Keras are doing, and appreciated it. Then of course the more logical method is to train the images via transfer learning, i.e using a pre-trained model. I'm using the faster R-CNN model to train my images, and the result is very good. In order to spice up this project, I have also try out their tensorflow's object detection api. This helps to locate the XY coordinates, draw the bounding boxes, and perform multiple classification. Then with the help of OpenCV, treating an video as streams of images for recongnition (while loop), the project becomes very interesting. Just a simple tagging the labels to an advertising slogon, give us an idea the "use case" for such technique. The same technique can be use in Health care, to detect cancer cells in a tissue.  Or in Education to teach children different species of animals. Or in Semiconductor where I come from, to detect defects in a wafer.  Self driving car and surveillance. And of course back to my project, Advertising. Thank you for your time. 

![alt text](https://i.imgur.com/J6QpCZw.jpg)

I am still new to this, so if you spotted any mistake, pls highlight to me. Thank you.
