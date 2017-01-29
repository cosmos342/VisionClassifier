# VisionClassifier
VisionClassifier to classify images(currently for dog vs cats redux kaggle kernel)
Uses VCG16 imagenet implementation of keras. Removes top FC layers and classification layer.
Builds a TopModel on top of the VCG16(with one Fully connected layer of 1024 neurons) and changes the classification
to 2 classes(cats or dogs)
Given a train/validation directory with folders within with images for each class, picks up images
and creates dog and cat classification.
Given a test directory with unknown folder picks up images for prediction.
Initially run to get train/validation/test VCG16 convolutional(CNN) feature maps and store the feature maps.
Subsequenty load the CNN feature maps and train the topModel and do prediction.
Current Test accuracy for kaggle dogs vs cats redux kernel competition about 93% with handful of epochs.
Future todo:
Further improve accuracy of dogs vs cats redux kernel.
Implement option to use other trained models such as resnet,inception etc.
