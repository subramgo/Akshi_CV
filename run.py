from gender.model import Models, runModel
from gender.data import _generatorFactory, LabelGenerator, H5Generator
import yaml 
import logging

logging.basicConfig(level=logging.INFO)



# create csv file
def create_csv():
	with open('config.yml', 'r')  as  yaml_file:
		cfg = yaml.load(yaml_file)
	label_cfg = cfg['labelgenerator']

	label_obj = LabelGenerator(label_cfg['label_path'], label_cfg['out_file'], label_cfg['image_path'])
	label_obj.process()


# create h5 file
def create_h5():
	with open('config.yml', 'r')  as  yaml_file:
		cfg = yaml.load(yaml_file)
	label_cfg = cfg['h5generator']
	h5_obj = H5Generator(label_cfg['csv_file_path'], label_cfg['hdf5_path'], label_cfg['class_col'] 
		, label_cfg['directory_col'], label_cfg['type_col'], label_cfg['image_w'],label_cfg['image_h'])

	h5_obj.process()


# train the model
def model_pararms():
	with open('config.yml', 'r') as yaml_file:
		cfg = yaml.load(yaml_file)

	model_cfg = cfg['model']
	model = Models.age_gender_model(model_cfg['input_shape'], model_cfg['nb_classes'])

	model_cfg['model'] = model 

	_adience_train_factory = _generatorFactory(model_cfg['h5_input'], x_label='train_images', y_label='train_labels')
	train_generator = _adience_train_factory(dimensions=model_cfg['target_size'] ,nbclasses=model_cfg['nb_classes'],batchsize=model_cfg['batch_size'])

	_adience_eval_factory = _generatorFactory(model_cfg['h5_input'], x_label='eval_images', y_label='eval_labels')
	validation_generator = _adience_eval_factory(dimensions=model_cfg['target_size'] ,nbclasses=model_cfg['nb_classes'],batchsize=model_cfg['batch_size'])

	model_cfg['train_generator'] = train_generator
	model_cfg['eval_generator'] = validation_generator

	return model_cfg



def train_model():
	model_cfg = model_pararms()
	run_obj = runModel(**model_cfg)
	run_obj.process()


if __name__ == "__main__":
	#create_csv()
	#create_h5()
	train_model()













# test the model