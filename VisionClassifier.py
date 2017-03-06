## prepare data augmentation configuration
import os
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt
import argparse

#import pdb
#pdb.set_trace()
mpath =  os.getcwd()
train_data_dir= mpath + '/data/dogscats/train'
#train_data_dir= mpath + '/data/dogscats/sample/train'
validation_data_dir= mpath + '/data/dogscats/valid'
#validation_data_dir= mpath + '/data/dogscats/sample/valid'
#test_data_dir= mpath + '/dogscats/test/'
test_data_dir= mpath + '/data/dogscats/test'
saved_dir = mpath + '/saved/'
results_dir = mpath + '/results/'

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
parser.add_argument('--numepoch', type=int,default=1,
                    help='load or fresh trained features')
parser.add_argument('--lr', type=float,default=0.001,
                    help='load or fresh trained features')
parser.add_argument('--testonly', type=bool,default=False,
                    help='test only no training')
parser.add_argument('--model',type=str,default="simple",
		    help='specify which type of model to finetune')
parser.add_argument('--regularizer',type=str,default="none",
                    help='specify l2 or l1 weight regularizer')
			


args = parser.parse_args()

tf_features = args.tf_features
load_topmodel = args.topmodel
batch_norm = args.batchnorm
num_epoch = args.numepoch
learn_rate = args.lr
test_only = args.testonly
model_type = args.model
regularizer = args.regularizer

print("create trained model")

def createTrainedModel():
	return VisionTrainedModel(train_data_dir,validation_data_dir,test_data_dir,saved_dir,model_type)

def createAugmentedTrainedModel():
	return VisionTrainedModel(train_data_dir,validation_data_dir,test_data_dir,saved_dir,model_type,rotation_range=10, width_shift_range=0.05, width_zoom_range=0.05, zoom_range=0.05, channel_shift_range=10, height_shift_range=0.05, shear_range=0.05, horizontal_flip=True)


if(model_type == "aug"):
	trainedModel = createAugmentedTrainedModel()
else:
	trainedModel = createTrainedModel()

print("done creating trained model ")

trainedModel.createDataGen()
train_labels = trainedModel.getTrainLabels()
val_labels = trainedModel.getValLabels()
#print("nuking gens")
#trainedModel.nukeGens()

if(tf_features == "fresh"):
	print("trainedModel predicting ")
	trainedModel.predictGenerator()
	#trainedModel.predict()
	trainedModel.saveFeatures()
else:
	print("trainedModel loading features")
	trainedModel.loadFeatures()

train_output, val_output, test_output = trainedModel.getFeatures()

topModel = VisionTopModel(2,model_type,trainedModel.getFeaturesShape(),regularizer)

if(load_topmodel == "load"):
	topModel.loadModel(saved_dir , 'dogs1.h5')
#	topModel.setLearningRate(learn_rate)
else:
	#topModel.flatten(trainedModel.getFeaturesShape())
	topModel.fineTune()

	#if(batch_norm == "yes"):
	#	topModel.addFC(1024)
	#else:
	#	topModel.addFC(1024,batch_norm=False)

	#topModel.addClassificationLayer(trainedModel.getNumClasses())

topModel.compileModel(lr=learn_rate)

if(test_only == False):
	for i in range(3):
		history = topModel.fit(train_output, train_labels,val_output,val_labels,nbepoch=num_epoch, batch=100)
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

print("predicting topmodel")
pred = topModel.predict(test_output)

test_gen = trainedModel.getTestGen()
test_images = test_gen.filenames
# test_pred = pred[:,0]
#save_array(results_dir + 'test_preds.dat', pred)
#save_array(results_dir + 'filenames.dat', test_images)
#Grab the dog prediction column
isdog = pred[:,1]
#print "Raw Predictions: " + str(isdog[:5])
#print "Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)])
#print "Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)])
#print "#dog Predictions: " + str(len(isdog[isdog >= 0.5]))

isdog = isdog.clip(min=0.03, max=0.97)
ids = np.array([int(f[8:f.find('.')]) for f in test_images])

subm = np.stack([ids,isdog], axis=1)
#subm[:5]

submission_file_name = 'submission' + model_type + '.csv'
np.savetxt(results_dir + submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')

'''
idx = [ i for i in range(10)]
plots_idx(idx, test_pred[idx])
'''
