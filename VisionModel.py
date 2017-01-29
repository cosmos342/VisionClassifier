from keras.applications.vgg16 import  VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2

MODEL_VGG16 = 1
MODEL_VGG19 = 2
MODEL_RESNET = 3
MODEL_INCEPTION = 4
MODEL_TOP = 5

def oneHot(x):
	return to_categorical(x)

class VisionModel:
	def __init__(self,train_dir,val_dir,test_dir,saved_dir):
		self.train_datagen = image.ImageDataGenerator(
        				rescale=1./255,
        				shear_range=0.2,
        				zoom_range=0.2,
        				horizontal_flip=True)

		self.val_datagen = image.ImageDataGenerator(rescale=1./255)
		self.test_datagen = image.ImageDataGenerator(rescale=1./255)

		self.train_dir =train_dir
		self.test_dir  =test_dir
		self.val_dir   = val_dir
		self.saved_dir = saved_dir
	

	def createDataGen(self,height=224,width=224,batch=32,entropy=None):

		self.train_gen = self.train_datagen.flow_from_directory(
        					self.train_dir,
        					target_size=(height, width),
        					batch_size=batch,
						class_mode=entropy,
						shuffle=False)
		self.num_classes = self.train_gen.nb_class

		print("createTrainGen")
        					

		self.val_gen = self.val_datagen.flow_from_directory(
        						self.val_dir,
        						target_size=(height, width),
        						batch_size=batch,
        						class_mode=entropy,
							shuffle=False)
		print("createValGen")

		self.test_gen = self.test_datagen.flow_from_directory(
        						self.test_dir,
        						target_size=(height, width),
        						batch_size=batch,
							class_mode=entropy,
							shuffle=False)

		print("createTestGen")

	def getNumClasses(self):
		return self.num_classes

	def getTrainLabels(self):
		return oneHot(self.train_gen.classes)

	def getValLabels(self):
		return oneHot(self.val_gen.classes)

	def getTestGen(self):
		return self.test_gen

        def getValGen(self):
                return self.val_gen

        def getTrainGen(self):
                return self.train_gen

		


class VisionTrainedModel(VisionModel):

	def __init__(self,train_dir,val_dir,test_dir,saved_dir,type=MODEL_VGG16,keep_top=False):
		VisionModel.__init__(self,train_dir,val_dir,test_dir,saved_dir)
		self.type = type
		if(self.type == MODEL_VGG16):
			self.model = VGG16(keep_top,weights='imagenet')
		else:
			print("ERROR: only VCG16 model is supported")

	def predictGenerator(self):
		self.train_X = self.model.predict_generator(self.train_gen, 23000)
		self.val_X =  self.model.predict_generator(self.val_gen,2000)
		# need to know test size
		self.test_X =  self.model.predict_generator(self.test_gen,12500)

	def saveFeatures(self):
		if(self.train_X is not None ):
                	np.save(open(self.saved_dir + '/bottleneck_features_train.npy', 'w'), self.train_X)
		if(self.val_X is not None ):
                	np.save(open(self.saved_dir + '/bottleneck_features_val.npy', 'w'), self.val_X)
		if(self.test_X is not None ):
                	np.save(open(self.saved_dir + '/bottleneck_features_test.npy', 'w'), self.test_X)

	def loadFeatures(self):
		self.train_X = np.load(open(self.saved_dir + '/bottleneck_features_train.npy'))
		self.val_X = np.load(open(self.saved_dir + '/bottleneck_features_val.npy'))
		self.test_X = np.load(open(self.saved_dir + '/bottleneck_features_test.npy'))

	def getFeatures(self):
		return self.train_X, self.val_X, self.test_X

	def getFeaturesShape(self):
		if(self.train_X is not None):
			return self.train_X.shape[1:]
		else:
			print("error model has not been trained ")
			return None


class VisionTopModel:
	def __init__(self):
		self.model = Sequential();

	def flatten(self,shape):
		self.model.add(Flatten(input_shape=shape))

	def addConv(self,num_filter,nb_row,nb_col):
		# self.model.add(Convolution2D(filters, length, width, border='same',activation='relu'))
		self.model.add(Convolution2D(num_filter, nb_row,nb_col, border_mode='same'))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))

	def addFC(self,num_neurons, batch_norm=True):
		if(batch_norm == False):
			print("addFC without batch norm ")
			self.model.add(Dense(num_neurons,activation='relu', W_regularizer=l2(0.01)))
		else:
			print("addFC add batch norm ")
			self.model.add(Dense(num_neurons))
			self.model.add(BatchNormalization())
			self.model.add(Activation('relu'))

	def disableWeightUpdate(self):
		for layer in self.model.layers:
			layer.trainable=False

	def addBatchNormalizationLayer(self):
		self.model.add(BatchNormalization(axis=1))

	def setLearningRate(self,rate):
		self.model.optimizer.lr=rate
		

	def addClassificationLayer(self,num_classes):
		self.num_classes = num_classes
		self.model.add(Dense(self.num_classes,activation='softmax'))

    	def compileModel(self, lr=0.01):
		# to do add binary cross entropy
		self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	def saveModel(self,mypath,filename):
		self.model.save(mypath+filename)

	def loadModel(self,mypath,filename):
		print("loading vision model")
		self.model = load_model(mypath+filename)

	def fit(self,train_input,train_labels,val_input,val_labels,batch=32,nbepoch=1):

		self.model.fit(train_input, train_labels,nb_epoch=nbepoch, batch_size=batch,
          		validation_data=(val_input, val_labels))

	def predict(self,test_input,batch=32):

		return self.model.predict(test_input,batch_size=batch)


	def fit_generator(self,train_input,train_labels,val_input,val_labels,batch_size=32,nb_epoch=1):
		if(self.model is not None):
			self.visionModel.fit_generator(
				self.train_gen,
        			nb_train_samples,
        			nb_epoch,
        			self.val_gen,	
				nb_val_samples);
		else:
			print("Error: model not created\n");
	

	def createTestGen(self,test_data_dir,height=224,width=224,batch=32,entropy=None):
		self.test_gen = self.test_datagen.flow_from_directory(
        						test_data_dir,
        						target_size=(height, width),
        						batch_size=batch,
							class_mode=entropy,
							shuffle=False)

