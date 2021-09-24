import math
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from torchvision.utils import save_image
import phyre
import torch
random.seed(0)

print('All eval setups:', *phyre.MAIN_EVAL_SETUPS)
# For now, let's select cross template for ball tier.
eval_setup = 'ball_within_template'


fold_id = 0  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:',
	len(dev_tasks), '\n test:', len(test_tasks))


print(*dev_tasks[:10], sep=', ')



action_tier = phyre.eval_setup_to_action_tier(eval_setup)
print('Action tier for', eval_setup, 'is', action_tier)


tasks = dev_tasks[200:250]

# Create the simulator from the tasks and tier.
simulator = phyre.initialize_simulator(tasks, action_tier)



######save example images

task_index = 0  # Note, this is a integer index of task within simulator.task_ids.
task_id = simulator.task_ids[task_index]
initial_scene = simulator.initial_scenes[task_index]
print('Initial scene shape=%s dtype=%s' % (initial_scene.shape, initial_scene.dtype))
print(phyre.observations_to_float_rgb(initial_scene).shape)

img=torch.from_numpy(phyre.observations_to_float_rgb(initial_scene)).permute(2,0,1)
print(img.shape)
save_image(img,'ExampleImages/Initials.png')


# plt.imshow(phyre.observations_to_float_rgb(initial_scene))
# plt.title(f'Task {task_id}');



print('Dimension of the action space:', simulator.action_space_dim)



# Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)



# Three statuses could be returned.
print('Action solves task:', phyre.SimulationStatus.SOLVED)
print('Action does not solve task:', phyre.SimulationStatus.NOT_SOLVED)
print('Action is an invalid input on task (e.g., occludes a task object):',
      phyre.SimulationStatus.INVALID_INPUT)
# May call is_* methods on the status to check the status.
print()
print('Result of taking action', action, 'on task', tasks[task_index], 'is:',
      simulation.status)
print('Does', action, 'solve task', tasks[task_index], '?', simulation.status.is_solved())
print('Is', action, 'an invalid action on task', tasks[task_index], '?',
      simulation.status.is_invalid())




print('Number of observations returned by simulator:', len(simulation.images))

num_across = 5
height = int(math.ceil(len(simulation.images) / num_across))
# fig, axs = plt.subplots(height, num_across, figsize=(20, 15))
# fig.tight_layout()
# plt.subplots_adjust(hspace=0.2, wspace=0.2)

# We can visualize the simulation at each timestep.
for i in  range(len(simulation.images)):
    # Convert the simulation observation to images.
    img=torch.from_numpy(phyre.observations_to_float_rgb(simulation.images[i])).permute(2,0,1)
    save_image(img,f'ExampleImages/Simu{i}.png')


print("action")
print(action)






####save the images and action
actions = simulator.build_discrete_action_space(max_actions=100)
print('A random action:', actions[0])


###simulation

task_index = 0  # The simulator takes an index into simulator.task_ids.
action = random.choice(actions)
simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True)

replay_buffer.append({
                    'Initial_frame': [],
                    'End_frame': [],
                    'action':action,         
                })


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

