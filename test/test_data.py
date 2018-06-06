from gender.data import LabelGenerator, H5Generator
import yaml
import unittest
import os

import logging
logging.basicConfig(level=logging.INFO)

class dataTestCase(unittest.TestCase):
	""" Test for gender.LabelGenerator """

	def test_is_file_created(self):
		with open('config.yml', 'r')  as  yaml_file:
			cfg = yaml.load(yaml_file)
		label_cfg = cfg['labelgenerator']

		label_obj = LabelGenerator(label_cfg['label_path'], label_cfg['out_file'], label_cfg['image_path'])
		label_obj.process()

		self.assertTrue(os.path.exists(label_cfg['out_file']) and os.path.getsize(label_cfg['out_file']) > 0)

	def test_h5_file_creation(self):
		with open('config.yml', 'r')  as  yaml_file:
			cfg = yaml.load(yaml_file)
		label_cfg = cfg['h5generator']
		h5_obj = H5Generator(label_cfg['csv_file_path'], label_cfg['hdf5_path'], label_cfg['class_col'] 
			, label_cfg['directory_col'], label_cfg['type_col'], label_cfg['image_w'],label_cfg['image_h'])

		h5_obj.process()


if __name__ == '__main__':
	unittest.main()





