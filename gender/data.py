"""Data preparation code for Gender-Age Models

LabelGenerator :

H5Generator :

"""

import glob
import yaml
import pandas as pd 
import logging
import random
import glob
import pandas as pd
import os
import numpy as np
import h5py
from PIL import Image
from keras.utils import np_utils as _np_utils



age_filter = {'(0, 2)' :0, '(4, 6)':1 , '(8, 12)':2, '(15, 20)':3,'(25, 32)':4,'(38, 43)':5, '(48, 53)':6, '(60, 100)':7}
gen_filter = {'m':0,'f':1}

logger = logging.getLogger(__name__)


class LabelGenerator():
	"""Generate a label file for Adience data set 
	"""
	def __init__(self, label_path, out_file, image_path):
		self.label_path = label_path   # Input label files
		self.out_file     = out_file
		self.label_files  = glob.glob(self.label_path)
		self.image_path = image_path
		logger.info("Reading label files " + ','.join(self.label_files))

		df_from_each_file = (pd.read_table(f, sep='\t') for f in self.label_files)
		self.labels_df   = pd.concat(df_from_each_file, ignore_index=True)
		logger.info("Read label files :" + ",".join(self.label_files))
		self._apply_filters()

	def _apply_filters(self):
		logger.info("Filter age and gender labels")
		self.labels_df = self.labels_df[self.labels_df['age'].isin(age_filter.keys())]
		self.labels_df = self.labels_df[self.labels_df['gender'].isin(gen_filter.keys())]

	def process(self):
		self.labels_df['image_prefix'] = self.labels_df['face_id'].apply(lambda x : 'coarse_tilt_aligned_face.' + str(x) + ".")
		self.labels_df['image_loc'] 	= self.image_path + self.labels_df['user_id'] + '/' +  self.labels_df['image_prefix'] + self.labels_df['original_image']

		# Get the corresponding class integer from filter dict
		self.labels_df['gender_class'] = self.labels_df['gender'].apply(lambda x: gen_filter[x])
		self.labels_df['age_class']     = self.labels_df['age'].apply(lambda x: age_filter[x]) 
		
		self.labels_df['type'] = self.labels_df.apply(lambda row: self._get_type(), axis =1 )
		# Save the new labels files
		self.labels_df[['face_id', 'original_image','age','gender','image_loc','age_class','gender_class','type']].to_csv(self.out_file, index = False)
		#
		logger.info("Finished creating csv file")

	def _get_type(self):
		eval_percentage = 0.2
		type_ = ''

		if random.random() > eval_percentage:
			type_ = "train"
		else:
			type_ = "eval"
		return type_




class H5Generator():
	"""

	""" 
	def __init__(self, csv_file_path, hdf5_path, class_col, directory_col, type_col, image_w, image_h):
		self.csv_file_path = csv_file_path
		self.hdf5_path     = hdf5_path
		self.class_col = class_col
		self.directory_col = directory_col
		self.type_col = type_col

		self.image_w = image_w
		self.image_h = image_h

		self.labels_df = None
		self.train_df  = None
		self.eval_df   = None

		self.hdf5_file = None

		self._load_csv()

	def _load_csv(self):
		self.labels_df = pd.read_csv(self.csv_file_path)
		self.train_df  = self.labels_df[self.labels_df['type'] == 'train']
		self.eval_df   = self.labels_df[self.labels_df['type'] == 'eval'] 

	def process(self):
		self._createH5()
		self._populateh5()


	def _createH5(self):
		total_images = self.labels_df.shape[0] * 5 # As we add five crops for each image
		train_images = self.train_df.shape[0] * 5
		eval_images = self.eval_df.shape[0] *5
		
		logger.info("Train {} Eval {} Total {}".format(train_images, eval_images, total_images))
		logger.info("Create h5 file @ " + self.hdf5_path)

		train_shape = (train_images, self.image_w, self.image_h, 3)
		eval_shape  = (eval_images, self.image_w, self.image_h, 3)

		self.hdf5_file = h5py.File(self.hdf5_path, mode = 'w')

		self.hdf5_file.create_dataset("train_images", train_shape, np.int8)
		self.hdf5_file.create_dataset("train_labels", (train_images,), np.int8)

		self.hdf5_file.create_dataset("eval_images", eval_shape, np.int8)
		self.hdf5_file.create_dataset("eval_labels", (eval_images,), np.int8)

		iindex = 0
		for index, row in self.train_df.iterrows():
			for i in range(5):
				self.hdf5_file["train_labels"][iindex] = row[self.class_col]
				iindex+=1

		iindex = 0
		for index, row in self.eval_df.iterrows():
			for i in range(5):
				self.hdf5_file["eval_labels"][iindex] = row[self.class_col]
				iindex+=1

		#self.hdf5_file["train_labels"][...] =  self.train_df[self.class_col]
		#self.hdf5_file["eval_labels"][...]  =  self.eval_df[self.class_col]
		logger.info("Finished creating empty h5 file")



	def _random_crop(self, img, random_crop_size):
		assert img.shape[2] == 3
		height, width = img.shape[0], img.shape[1]
		dy, dx = random_crop_size
		x = np.random.randint(0, width - dx + 1)
		y = np.random.randint(0, height - dy + 1)
		return img[y:(y+dy), x:(x+dx), :]

	def _populateh5(self):
		images_location_list = self.labels_df[self.directory_col].tolist()
		type_list = self.labels_df[self.type_col].tolist()
		i = 0
		train_i = 0
		eval_i = 0

		for (_type, path) in zip(type_list, images_location_list):
			if i % 100 == 0 and i > 0:
				logger.info("Train data {}/{}".format(i, self.labels_df.shape[0]))

			image = Image.open(path)
			image = image.resize((self.image_w, self.image_h), Image.ANTIALIAS)
			image_array = np.array(image.getdata(), np.uint8).reshape(image.size[1], image.size[0], 3)

			orig_image = np.copy(image_array)
			crops =  []
			# 5 Random 227,227 crops
			for j in range(5):
				random_c = self._random_crop(image_array, (227, 227))
				if _type == 'train':
					self.hdf5_file["train_images"][train_i, ...] = random_c[None]
					train_i+=1
				if _type == 'eval':
					self.hdf5_file["eval_images"][eval_i, ...] = random_c[None]
					eval_i+=1
				i+=1

		self.hdf5_file.close()
		logger.info("Finished writing {} images".format(i))



def _generatorFactory(filepath,x_label='train_images',y_label='train_labels'):
    """
        Produces a generator function.
        Give a filepath and column labels:
            HDF5 data is loaded from the filepath
            x and y labels need to match the HDF5 file's columns
    """
    def _generator(dimensions,nbclasses,batchsize):
        while 1:
            with h5py.File(filepath, "r") as f:
                filesize = len(f[y_label])
                n_entries = 0
                while n_entries < (filesize - batchsize):
                    x_train= f[x_label][n_entries : n_entries + batchsize]
                    x_train= np.reshape(x_train, (batchsize, dimensions[0], dimensions[1], 3)).astype('float32')

                    y_train = f[y_label][n_entries:n_entries+batchsize]
                    # data-specific formatting should be done elsewhere later, even onecoding
                    # if dimensions is needed, can be gotten from x_train.shape
                    y_train_onecoding = _np_utils.to_categorical(y_train, nbclasses)

                    n_entries += batchsize

                    # Shuffle
                    p = np.random.permutation(len(y_train_onecoding))
                    yield (x_train[p], y_train_onecoding[p])
                f.close()
    
    return _generator





        
       