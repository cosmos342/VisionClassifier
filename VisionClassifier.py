# prepare data augmentation configuration
import os
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt
import argparse
import bcolz

import pdb
pdb.set_trace()
mpath =  os.getcwd()
train_data_dir= mpath + '/data/train'
validation_data_dir= mpath + '/data/valid'
test_data_dir= mpath + '/data/test/'
saved_dir = mpath + '/saved/'
results_dir = mpath + '/results/'

from VisionModel import *

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]


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
parser.add_argument('--testonly', type=bool,default=False,
                    help='test only no training')

args = parser.parse_args()

tf_features = args.tf_features
load_topmodel = args.topmodel
batch_norm = args.batchnorm
num_epoch = args.numepoch
learn_rate = args.lr
test_only = args.testonly

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
val_gen = trainedModel.getValGen()
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

if(test_only == False):
	history = topModel.fit(train_output, train_labels,val_output,val_labels,nbepoch=num_epoch, batch=32)
	topModel.saveModel(saved_dir , 'dogs1.h5')

'''
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''


pred = topModel.predict(test_output)

test_gen = trainedModel.getTestGen()
test_images = test_gen.filenames
# test_pred = pred[:,0]
save_array(results_dir + 'test_preds.dat', pred)
save_array(results_dir + 'filenames.dat', test_images)
#Grab the dog prediction column
isdog = pred[:,1]
print "Raw Predictions: " + str(isdog[:5])
print "Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)])
print "Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)])
print "#dog Predictions: " + str(len(isdog[isdog >= 0.5]))

isdog = isdog.clip(min=0.03, max=0.97)
ids = np.array([int(f[8:f.find('.')]) for f in test_images])

subm = np.stack([ids,isdog], axis=1)
subm[:5]

submission_file_name = 'submission1.csv'
np.savetxt(results_dir + submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')

'''
idx = [ i for i in range(10)]
plots_idx(idx, test_pred[idx])
'''
