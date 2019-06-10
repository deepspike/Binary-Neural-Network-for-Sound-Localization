import torch
import numpy as np
import tensorflow as tf

from torch.utils.data import Dataset
from tensorpack import *
from collections import OrderedDict

class SLDataFlow(DataFlow):
	"""Create tensorpack dataflow for SL dataset"""
	def __init__(self, list_IDs):
		self.list_IDs = list_IDs

	def __len__(self):
		return len(self.list_IDs)
		
	def __iter__(self):
		for file_name in self.list_IDs:
			X, Y, I = torch.load('Data/' + file_name + '.pt')

			yield [X, Y.astype('float32'), I]

class SLDataset(Dataset):
	""" Sound Localization Generator"""
	def __init__(self, list_IDs):
		self.list_IDs = list_IDs

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# load data and get label
		X, Y, I = torch.load('Data/' + self.list_IDs[index] + '.pt')

		return X, Y.astype('float32'), I

def angular_distance_compute(a1, a2):
	dist_error = (180 - torch.abs(torch.abs(a1 - a2) - 180)).type(torch.cuda.FloatTensor)

	return torch.mean(dist_error)

def angular_distance_compute_tf(a1, a2):
	dist_error = (180 - tf.abs(tf.abs(a1 - a2) - 180))

	return dist_error

def testing(model, dataLoader, device):   
	model.eval() # Put the model in test mode 

	running_mae = 0
	for i_batch, (Xte, Yte, Ite) in enumerate(dataLoader):
		# Transfer to GPU
		inputs, target, source_indicator = Xte.type(torch.FloatTensor).to(device),\
							Yte.type(torch.FloatTensor).to(device),\
							Ite.type(torch.FloatTensor).to(device)

		# forward pass
		x1, x2, x3, x4, y_pred = model.forward(inputs)

		# error analysis
		single_source_mask = source_indicator.eq(1.0).squeeze()
		loc_pred = torch.masked_select(torch.argmax(y_pred, dim=1), single_source_mask)
		loc_target = torch.masked_select(torch.argmax(target, dim=1), single_source_mask)
		running_mae += angular_distance_compute(loc_pred, loc_target)
				
	mae = running_mae.item() / (i_batch+1)

	return mae

def state_dict_data_parallel(state_dict):
	"""# remove 'module.' of for model trained with dataParallel """

	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' 
		new_state_dict[name] = v

	return new_state_dict

def get_data(File_IDs, batch_size):
	ds = SLDataFlow(File_IDs)
	ds = BatchData(ds, batch_size, remainder=False)

	return ds