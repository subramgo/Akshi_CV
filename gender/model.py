from keras.layers import Input, Conv2D, Dense,MaxPooling2D, Flatten, Activation,Dense, Dropout, BatchNormalization,GlobalAveragePooling2D
from keras.models import Model
from keras.backend import tf as ktf
from keras import optimizers
from keras.callbacks import History, EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
import keras.backend as K


"""

"""
class runModel():
	"""

	"""
	def __init__(self, **config_dict):
		self.model = config_dict['model']
		self.train_generator = config_dict['train_generator']
		self.eval_generator = config_dict['eval_generator']
		self.callbacks = None
		self.config_dict = config_dict

	def _callbacks(self):

		self.callbacks = [EarlyStopping(monitor='acc', min_delta=self.config_dict['early_stop_th'], patience=5, 
			verbose=0, mode='auto')
		, 
		#ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', 
		#	epsilon=0.0001, cooldown=0, min_lr=0)
		#, 
		CSVLogger(self.config_dict['log_path'], separator=',', append=False)
		#,
		
		#ModelCheckpoint(self.config_dict['check_path'], monitor='val_loss', verbose=0, save_best_only=True, 
		#	save_weights_only=False, mode='auto', period=1)
		]

	def _save(self):
		self.model.save_weights(self.config_dict['weights_path'])


	def _train(self):

		self._callbacks()

		self.model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ["accuracy"])

		hist = self.model.fit_generator(self.train_generator, steps_per_epoch=self.config_dict['steps_per_epoch'],verbose = 2,callbacks =self. callbacks,
		epochs=self.config_dict['epochs'], validation_data=self.eval_generator,validation_steps=self.config_dict['validation_steps'])

		self._save()

	def process(self):
		self._train()





class Models():
	def __init__(self):
		self.model = None
		return

	@staticmethod
	def age_gender_model(input_shape, nb_classes):
		"""
		"""
		x_input = Input(input_shape)
		# Conv Layer 1
		x = Conv2D(filters = 96, kernel_size = (5,5), strides = (1,1), \
		           padding = "valid", name = 'conv-1',kernel_initializer='glorot_uniform')(x_input)

		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
		x = BatchNormalization()(x)

		# Conv Layer 2
		x = Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), 
		           padding = "valid",name= 'conv-2',kernel_initializer='glorot_uniform')(x)
		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
		x = BatchNormalization()(x)

		# Conv Layer 3
		x = Conv2D(filters = 512, kernel_size = (5,5), strides = (1,1), 
		           padding = "valid",name= 'conv-4',kernel_initializer='glorot_uniform')(x)
		x = Activation("relu")(x)
		x = MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
		x = BatchNormalization()(x)

		# Conv Layer 4
		x = Conv2D(filters = 1024, kernel_size = (1,1), strides = (1,1), 
		           padding = "valid",name= 'conv-3',kernel_initializer='glorot_uniform')(x)
		x = Activation("relu")(x)

		    
		x = Flatten()(x)
		x = Dense(1024, activation = "relu",name='dense-1')(x)
		x = Dropout(rate = 0.5)(x)
		x = Dense(512, activation = "relu",name='dense-2')(x)
		x = Dropout(rate = 0.5)(x)
		x = Dense(512, activation ="relu",name='dense-3')(x)
		x = Dropout(rate = 0.5)(x)

		predictions = Dense(nb_classes, activation="softmax",name="softmax")(x)

		return Model(inputs = x_input, outputs = predictions)





