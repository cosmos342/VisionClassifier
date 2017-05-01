


# VisionClassifier
* VisionClassifier to classify images(currently for dog vs cats redux kaggle kernel) </br>
* Uses VCG16 imagenet implementation of keras. Removes top FC layers and classification layer.</br>
* Builds a TopModel on top of the VCG16(with one Fully connected layer of 1024 neurons) and changes the classification to 2 classes(cats or dogs) </br>
* Given a train/validation directory with sub-folders with images for each class, picks up images and creates dog and cat classification.
* Given a test directory with images in one sub-folder picks up images for prediction.</br>
* Initially run prediction on the VCG16 modified bottom model to get train/validation/test VCG16 convolutional(CNN) feature maps and store the feature maps. </br>
* Subsequenty load the CNN feature maps and train the topModel and do prediction.</br>
* Current Validation accuracy for kaggle dogs vs cats redux kernel competition about 93% with handful of epochs. </br>
* Latest update. Tried the following 3 models and bettered performance.</br>
* First important change is adding a Lambda layer as the input layer of VGG16(this layer does not add any weights) </b>
* Lambda layer takes the input image and subtracts the mean per channel specified by VGG16 folks and also changes the channels from RGB to BGR as VGG was trained for BGR.
  * Simple Model </br>
    * Takes the VGG 16 output of the final layer for each image(train/valid/test) and feeds that as the input for a simple dense layer with 2 outputs and softmax activation to classify as dog or cat. This provided accuracy of over 96% and put the kaggle submission in top 63% </br>
  * Convolution Model </br>
    * This removes the top FC layers of the VGG model. Runs the prediction on the VGG16 model gets last convolution layer output as input features for the top model.</br>
    * The top Layer consists of 2 Fully connected layers, same as that of the VGG16(4K neurons each) but each with BatchNormalization layer. Then a final classification is Fully connected layer of 2 elements with softmax. Trained this with learning rate of .01 for a few epochs(about 5) and then reduce the learning rate to 0.00001 and trained for few more epochs and the validation accuracy improved to 98%. Kaggle submission was in the top 30% range.
   * Augmented Convolution Model </br>
     * This is the same as the convolution model except that the input data to the VGG model is augmented with (rotation range etc) as can be seen in the VisionClassifer.py file when the --model option is specified as "aug". This model validation accuracy also came at around 98% and did not improve on Convolution Model. May be it could improve with more amount of training which i didn't get to try. 
     * Also tried with L2 and L1 regularizers on this model. With regularizers initially the loss was high but it converged and could get to 98% accuracy within the few epochs tried. At this point may be some other model like vgg19 or inception model could be tried to see accuracy can be further improved.

# Credits:
* Keras examples

# Future TODO:</br>
* Further improve accuracy of dogs vs cats redux kernel.</br>
* Make it generic to use other pre-trained deep models such as RESNET,INCEPTION etc.</br>
