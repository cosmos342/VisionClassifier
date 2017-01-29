# prepare data augmentation configuration
import os
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt
import argparse

#import pdb
#pdb.set_trace()
mpath =  os.getcwd()
train_data_dir= mpath + '/data/train'
validation_data_dir= mpath + '/data/valid'
test_data_dir= mpath + '/data/test/'
saved_dir = mpath + '/saved/'
from VisionModel import *

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
	print("end of imshow ")

def plots_idx(idx, titles=None):
    plots([image.load_img(test_data_dir + test_images[i]) for i in idx], titles=titles)


parser = argparse.ArgumentParser(description='Vision Model parser')

parser.add_argument('--tf-features', type=str,default="load",
                    help='load or fresh trained features')
parser.add_argument('--topmodel', type=str,default="load",
                    help='load or fresh trained features')
parser.add_argument('--batchnorm', type=str,default="yes",
                    help='load or fresh trained features')
parser.add_argument('--numepoch', type=int,default=3,
                    help='load or fresh trained features')
parser.add_argument('--lr', type=float,default=0.01,
                    help='load or fresh trained features')

args = parser.parse_args()

tf_features = args.tf_features
load_topmodel = args.topmodel
batch_norm = args.batchnorm
num_epoch = args.numepoch
learn_rate = args.lr

trainedModel = VisionTrainedModel(train_data_dir,validation_data_dir,test_data_dir,saved_dir)
trainedModel.createDataGen()

if(tf_features == "fresh"):
	print("trainedModel predicting ")
	trainedModel.predictGenerator()
	trainedModel.saveFeatures()
else:
	print("trainedModel loading features")
	trainedModel.loadFeatures()

train_output, val_output, test_output = trainedModel.getFeatures()
train_labels = trainedModel.getTrainLabels()
val_labels = trainedModel.getValLabels()

topModel = VisionTopModel()


if(load_topmodel == "load"):
	topModel.loadModel(saved_dir , 'dogs1.h5')
#	topModel.setLearningRate(learn_rate)
else:
	topModel.flatten(trainedModel.getFeaturesShape())

	if(batch_norm == "yes"):
		topModel.addFC(1024)
	else:
		topModel.addFC(1024,batch_norm=False)

	topModel.addClassificationLayer(trainedModel.getNumClasses())

topModel.compileModel(lr=learn_rate)
topModel.fit(train_output, train_labels,val_output,val_labels,nbepoch=num_epoch, batch=32)
topModel.saveModel(saved_dir , 'dogs1.h5')

pred = topModel.predict(test_output)

'''
test_gen = trainedModel.getTestGen()
test_images = test_gen.filenames
test_pred = pred[:,0]

idx = [ i for i in range(10)]
plots_idx(idx, test_pred[idx])
'''


# you can do predict generator.
#test_label = topModel.predict(self,test_input)
