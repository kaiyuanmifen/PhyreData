import math
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from torchvision.utils import save_image
import phyre
import torch
import torch.nn.functional as F
random.seed(0)



####save the data 
import os
import h5py

def save_dict_h5py(array_dict, fname):
	"""Save dictionary containing numpy arrays to h5py file."""

	# Ensure directory exists
	directory = os.path.dirname(fname)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with h5py.File(fname, 'w') as hf:
		for key in array_dict.keys():
			hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
	"""Restore dictionary containing numpy arrays from h5py file."""
	array_dict = dict()
	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			array_dict[key] = hf[key][:]
	return array_dict


def save_list_dict_h5py(array_dict, fname):
	"""Save list of dictionaries containing numpy arrays to h5py file."""

	# Ensure directory exists
	directory = os.path.dirname(fname)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with h5py.File(fname, 'w') as hf:
		for i in range(len(array_dict)):
			grp = hf.create_group(str(i))
			for key in array_dict[i].keys():
				grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
	"""Restore list of dictionaries containing numpy arrays from h5py file."""
	array_dict = list()
	with h5py.File(fname, 'r') as hf:
		for i, grp in enumerate(hf.keys()):
			array_dict.append(dict())
			for key in hf[grp].keys():
				array_dict[i][key] = hf[grp][key][:]
	return array_dict


####to save the data 

def ExtractData(tasks,action_tier):
	replay_buffer = []
	i=0
	simulator = phyre.initialize_simulator(tasks, action_tier)#build the simulator
	AllSolved=[]
	for task_index in range(len(tasks)):###repeating different actions on the same task a couple of time to increase sample size
		 # The simulator takes an index into simulator.task_ids.
		for repeats in range(50):
			actions = simulator.build_discrete_action_space(max_actions=100)
			
			action = random.choice(actions)#randomly choose actions
			#print('A random action:', action)
			# print("task index")
			# print(task_index)

			simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)

			Solved=int(simulation.status.is_solved())#####if the  task is solved
			
			#print("simulation here")
			#print('Does', action, 'solve task', tasks[task_index], '?', simulation.status.is_solved())

			if (i<10) or (Solved==1 and (np.sum(AllSolved)/len(AllSolved))<=0.5 ) or (Solved==0 and (np.sum(AllSolved)/len(AllSolved))>0.5 ): 
				##########balancing +1 and -ve in reasonig task
				AllSolved.append(Solved)
				if simulation.images is not None:####for certain reason , some simulation does not give images
					replay_buffer.append({
								'Initial_frame': [],
								'End_frame': [],
								'action':[],
								'Solved':[],         
							})
					img_initial=torch.from_numpy(phyre.observations_to_float_rgb(simulation.images[0])).permute(2,0,1)

					img_initial=F.interpolate(img_initial.unsqueeze(0),(50,50)).squeeze(0)###make it smaller and easier to save

					img_end=torch.from_numpy(phyre.observations_to_float_rgb(simulation.images[-1])).permute(2,0,1)

					img_end=F.interpolate(img_end.unsqueeze(0),(50,50)).squeeze(0)###make it smaller and easier to save


					replay_buffer[i]['Initial_frame'].append(img_initial.to("cpu").numpy())
					replay_buffer[i]['End_frame'].append(img_end.to("cpu").numpy())
					replay_buffer[i]['action'].append(action)
					replay_buffer[i]['Solved'].append(Solved)

					i=i+1
					print(i)

				# if random.random()<0.1:
				# 	save_image(img_initial,'ExampleImages/SaveEm_initial'+str(i)+'.png')
				# 	save_image(img_end,'ExampleImages/SaveEm_end'+str(i)+'.png')
	print("number of tasks:",len(AllSolved))
	print("number solved:",np.sum(AllSolved))

	return replay_buffer





print('All eval setups:', *phyre.MAIN_EVAL_SETUPS)
# For now, let's select cross template for ball tier.
eval_setup = 'ball_cross_template'




train_tasks, dev_tasks, test_tasks =(),(),()

for fold_id in range(5):
	train,dev,test=phyre.get_fold(eval_setup, fold_id)
	train_tasks=train_tasks+train
	dev_tasks=dev_tasks+dev
	test_tasks=test_tasks+test    

print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:',
	len(dev_tasks), '\n test:', len(test_tasks))





print(*dev_tasks[:10], sep=', ')



action_tier = phyre.eval_setup_to_action_tier(eval_setup)
print('Action tier for', eval_setup, 'is', action_tier)

# Create the simulator from the tasks and tier.

SaveDir= "../data/phyre2framedata/"
import os
if not os.path.exists(SaveDir):
	os.makedirs(SaveDir)
###train data
tasks = train_tasks
replay_buffer=ExtractData(tasks,action_tier)

save_list_dict_h5py(replay_buffer, SaveDir+"/train.h5")
print("train data saved")




###validation data
tasks = dev_tasks
replay_buffer=ExtractData(tasks,action_tier)



save_list_dict_h5py(replay_buffer, SaveDir+"/validation.h5")
print("validation data saved")

# Miega=load_list_dict_h5py("../data/phyre2framedata/"+"validation.h5")
# print("data loaded")




###test data
tasks = test_tasks
replay_buffer=ExtractData(tasks,action_tier)


save_list_dict_h5py(replay_buffer, SaveDir+"/test.h5")
print("test data saved")

