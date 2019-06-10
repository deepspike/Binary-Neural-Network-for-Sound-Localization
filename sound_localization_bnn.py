import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from torch.utils import data
from utils import SLDataset, angular_distance_compute
from models.sl_bnn import SL_BNN

def testing(model, dataLoader, device):   
	model.eval() # Put the model in test mode 

	running_mae = 0
	for i_batch, (Xte, Yte, Ite) in enumerate(dataLoader,1):
		# Transfer to GPU
		inputs, target, source_indicator = Xte.type(torch.FloatTensor).to(device),\
							Yte.type(torch.FloatTensor).to(device),\
							Ite.type(torch.FloatTensor).to(device)

		# forward pass
		y_pred = model.forward(inputs)

		# error analysis
		single_source_mask = source_indicator.eq(1.0).squeeze()
		loc_pred = torch.masked_select(torch.argmax(y_pred, dim=1), single_source_mask)
		loc_target = torch.masked_select(torch.argmax(target, dim=1), single_source_mask)
		mae = angular_distance_compute(loc_pred, loc_target)
			
		running_mae += mae
		
	mae = running_mae.item() / (i_batch)

	return mae

if __name__ == '__main__':        
	# CUDA configuration 
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	torch.multiprocessing.set_start_method("spawn")
	# Parameters
	params = {'batch_size': 512,
				'shuffle': True,
				'num_workers': 10}
	num_epochs = 100

	# Datasets
	partition = torch.load('sample_IDs.pt')

	# Data Generators
	training_set = SLDataset(partition['train'])
	training_generator = data.DataLoader(training_set, **params)

	testing_set = SLDataset(partition['test'])
	testing_generator = data.DataLoader(testing_set, **params)

	# Models and training configuration 
	model = SL_BNN()
	model = model.to(device)
	if device == 'cuda':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True

	criterion = torch.nn.MSELoss(reduction='mean')
	optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

	global best_mae 
	best_mae = 360

	for epoch in range(num_epochs):
		# Training Loop
		model.train()
		scheduler.step()
		running_loss = 0.0
		since = time.time()

		for i_batch, (Xtr, Ytr, Itr) in enumerate(training_generator):
			#print(i_batch)
			# Transfer to GPU
			inputs, target, source_indicator = Xtr.type(torch.FloatTensor).to(device),\
												Ytr.type(torch.FloatTensor).to(device),\
												Itr.type(torch.FloatTensor).to(device)

			# Model computation and weight update
			y_pred = model.forward(inputs)
			loss = criterion(y_pred, target)

			optimizer.zero_grad()
			loss.backward()

			# for p in list(model.parameters()):
			# 	if hasattr(p,'org'):
			# 		p.data.copy_(p.org)

			optimizer.step()
			running_loss += loss.item()

			# for p in list(model.parameters()):
			# 	if hasattr(p,'org'):
			# 		p.org.copy_(p.data.clamp_(-1,1))
								
		epoch_loss = running_loss/i_batch

		# Testing Stage
		mae_test = testing(model, testing_generator, device)

		if mae_test <= best_mae:
			print("Saving the model.")\

			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')

			state = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss,
					'mae': mae_test,
			}
			torch.save(state, 'checkpoint/sl_bnn.pt')
			best_mae = mae_test

		time_elapsed = time.time() - since

		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60, time_elapsed % 60))
		# print('Loss: {:4f}, Train MAE: {:4f}, Test MAE: {:4f}'.format(epoch_loss, mae_train, mae_test))
		print('Loss: {:4f}, Test MAE: {:4f}'.format(epoch_loss, mae_test))
