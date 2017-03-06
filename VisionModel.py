import vgg16; reload(vgg16)
from vgg16 import VGG16
#from keras.applications.vgg16 import  VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Flatten,Dense, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
import bcolz

MODEL_VGG16 = 1
MODEL_VGG19 = 2
MODEL_RESNET = 3
MODEL_INCEPTION = 4
MODEL_TOP = 5

#vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def vgg_preprocess(x,num):
	print("vgg process start")
	my_mean = np.tile(vgg_mean,(num,1))
	my_mean = my_mean.reshape((num,3,1,1))
	x = x - my_mean 
	print("vgg process end")

    	return x[:, ::-1] # reverse axis rgb->bgr

def concat_data(data_gen):
	return np.concatenate([data_gen.next() for i in range(data_gen.nb_sample)])
	#for i in range(data_gen.nb_sample):
	#	ndata = data_gen.next()


def oneHot(x):
	return to_categorical(x)

class VisionModel:
	def __init__(self,train_dir,val_dir,test_dir,saved_dir,
		width_shift_range=0, width_zoom_range=0, zoom_range=0, 
		channel_shift_range=0, height_shift_range=0, shear_range=0, 
		horizontal_flip=False):
		self.train_datagen = image.ImageDataGenerator()
		#self.train_datagen = image.ImageDataGenerator(
        	# 			rescale=1./255,
        	#			shear_range=0.2,
        	#			zoom_range=0.2,
        	#			horizontal_flip=True)
		self.val_datagen = image.ImageDataGenerator()
		self.test_datagen = image.ImageDataGenerator()

		#self.val_datagen = image.ImageDataGenerator(rescale=1./255)
		#self.test_datagen = image.ImageDataGenerator(rescale=1./255)

		self.train_dir =train_dir
		self.test_dir  =test_dir
		self.val_dir   = val_dir
		self.saved_dir = saved_dir
	

	def createDataGen(self,height=224,width=224,batch=1,entropy=None):

		print("createTrainGen train")
		self.train_gen = self.train_datagen.flow_from_directory(
        					self.train_dir,
        					target_size=(height, width),
        					batch_size=batch,
						class_mode=entropy,
						shuffle=False)
		self.num_classes = self.train_gen.nb_class
		#self.train_imgs = concat_data(self.train_gen)
		#save_array(self.saved_dir+'train_data.bc',self.train_imgs)
		#self.train_imgs = load_array(self.saved_dir+'train_data.bc')
		#self.train_imgs = vgg_preprocess(self.train_imgs,self.train_gen.nb_sample)


        					
		print("createTrainGen val")

		self.val_gen = self.val_datagen.flow_from_directory(
        						self.val_dir,
        						target_size=(height, width),
        						batch_size=batch,
        						class_mode=entropy,
							shuffle=False)
		#self.val_imgs = concat_data(self.val_gen)
		#self.val_imgs = load_array(self.saved_dir+'val_data.bc')
		#save_array(self.saved_dir+'val_data.bc',self.val_imgs)
		#self.val_imgs = vgg_preprocess(self.val_imgs,self.val_gen.nb_sample)

		print("createTestGen")

		self.test_gen = self.test_datagen.flow_from_directory(
        						self.test_dir,
        						target_size=(height, width),
        						batch_size=batch,
							class_mode=entropy,
							shuffle=False)
#
		#self.test_imgs = concat_data(self.test_gen)
		#self.test_imgs = load_array(self.saved_dir+'test_data.bc')
		#save_array(self.saved_dir+'test_data.bc',self.test_imgs)
		#print("done createTesGen saving ")
		#self.test_imgs = vgg_preprocess(self.test_imgs,self.test_gen.nb_sample)

		print("createTestGenFinito")

	def getNumClasses(self):
		return self.num_classes

	def getTrainLabels(self):
		return oneHot(self.train_gen.classes)

	def getValLabels(self):
		return oneHot(self.val_gen.classes)

	def nukeGens(self):
		self.val_gen = []
		self.train_gen = []
		self.test_gen = []

	def getTestGen(self):
		return self.test_gen

        def getValGen(self):
                return self.val_gen

        def getTrainGen(self):
                return self.train_gen

		


class VisionTrainedModel(VisionModel):

	def __init__(self,train_dir,val_dir,test_dir,saved_dir,model_type,
		type=MODEL_VGG16,rotation_range=0, 
		width_shift_range=0, width_zoom_range=0, zoom_range=0, 
		channel_shift_range=0, height_shift_range=0, shear_range=0, 
		horizontal_flip=False):
		VisionModel.__init__(self,train_dir,val_dir,test_dir,saved_dir,
		width_shift_range=0, width_zoom_range=0, zoom_range=0, 
		channel_shift_range=0, height_shift_range=0, shear_range=0, 
		horizontal_flip=False)
		self.type = type
		self.model_type = model_type
		if(self.type == MODEL_VGG16):
			if(self.model_type == "simple"):
				self.model = VGG16(include_top=True,weights='imagenet')
			else:
				self.model = VGG16(include_top=False,weights='imagenet')
		else:
			print("ERROR: only VCG16 model is supported")

	def predictGenerator(self):
		self.train_X = self.model.predict_generator(self.train_gen, self.train_gen.nb_sample)
		self.val_X =  self.model.predict_generator(self.val_gen,self.val_gen.nb_sample)
		# need to know test size
		self.test_X =  self.model.predict_generator(self.test_gen,self.test_gen.nb_sample)

	def predict(self):
		print("trainedModel Predict train")
		self.train_X = self.model.predict(self.train_imgs,batch_size=8)
		print("trainedModel Predict val")
		self.val_X = self.model.predict(self.val_imgs,batch_size=8)
		print("trainedModel Predict test")
		self.test_X = self.model.predict(self.test_imgs,batch_size=8)

	def saveFeatures(self):
		if(self.train_X is not None ):
			print("trainedModel save train features")
                	np.save(open(self.saved_dir + '/bottleneck_features_train' + self.model_type + '.npy', 'w'), self.train_X)
		if(self.val_X is not None ):
			print("trainedModel save val features")
                	np.save(open(self.saved_dir + '/bottleneck_features_val'+ self.model_type +'.npy' , 'w'), self.val_X)
		if(self.test_X is not None ):
			print("trainedModel save test features")
               		np.save(open(self.saved_dir + '/bottleneck_features_test' + self.model_type + '.npy', 'w'), self.test_X)


	def loadFeatures(self):
		self.train_X = np.load(open(self.saved_dir + '/bottleneck_features_train' + self.model_type + '.npy'))
		self.val_X = np.load(open(self.saved_dir + '/bottleneck_features_val' + self.model_type + '.npy'))
	
		self.test_X = np.load(open(self.saved_dir + '/bottleneck_features_test' + self.model_type + '.npy'))
	
	def getFeatures(self):
		return self.train_X, self.val_X, self.test_X

	def getFeaturesShape(self):
		if(self.train_X is not None):
			return self.train_X.shape[1:]
		else:
			print("error model has not been trained ")
			return None


class VisionTopModel:
	def __init__(self,num_classes,model_type,input_shape,regularizer):
		self.model = Sequential();
		self.num_classes = num_classes
		self.input_shape = input_shape
		self.model_type = model_type
		self.regularizer = regularizer

	def fineTune(self):
		if(self.model_type == "simple"):
			self.model = Sequential([ Dense(self.num_classes, activation='softmax', input_shape=self.input_shape) ])
		else:
			self.flatten()
			self.addFC(2,4096, batch_norm=True)
			self.addClassificationLayer()
		print("TopModel add dense ");

	def flatten(self):
		self.model.add(Flatten(input_shape=self.input_shape))

	def addConv(self,num_filter,nb_row,nb_col):
		# self.model.add(Convolution2D(filters, length, width, border='same',activation='relu'))
		self.model.add(Convolution2D(num_filter, nb_row,nb_col, border_mode='same'))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))

	def addFC(self,num_layers,num_neurons, batch_norm=True):
		if(batch_norm == False):
			print("addFC without batch norm ")
			self.model.add(Dense(num_neurons,activation='relu', W_regularizer=l2(0.01)))
		else:
			print("addFC add batch norm ")
			for i in range(num_layers):
				if(self.regularizer == "none"):
					self.model.add(Dense(num_neurons,activation='relu'))
				elif(self.regularizer == "l2"):
					self.model.add(Dense(num_neurons,activation='relu',W_regularizer=l2(0.01)))
				else:
					self.model.add(Dense(num_neurons,activation='relu',W_regularizer=l1(0.01)))
				self.model.add(BatchNormalization())

	def disableWeightUpdate(self):
		for layer in self.model.layers:
			layer.trainable=False

	def addBatchNormalizationLayer(self):
		self.model.add(BatchNormalization(axis=1))

	def setLearningRate(self,rate):
		self.model.optimizer.lr=rate
		
	def fineTuneVCG16TopLayer(self,num_out):
		#self.model.add(Dense(num_out,activation='softmax',input_shape=(1000,)))
		self.model = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
		print("TopModel add dense ");

	def addClassificationLayer(self):
		if(self.regularizer == "none"):
			self.model.add(Dense(self.num_classes,activation='softmax'))
		elif(self.regularizer == "l2"):
			self.model.add(Dense(self.num_classes,activation='softmax',W_regularizer=l2(0.01)))
		else:
			self.model.add(Dense(self.num_classes,activation='softmax',W_regularizer=l1(0.01)))

    	def compileModel(self, lr=0.01):
		# to do add binary cross entropy
		self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	def saveModel(self,mypath,filename):
		self.model.save(mypath+ self.model_type + filename)

	def loadModel(self,mypath,filename):
		print("loading vision model")
		self.model = load_model(mypath+self.model_type + filename)

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

