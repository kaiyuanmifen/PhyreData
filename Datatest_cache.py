import functools
import multiprocessing
import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import phyre


tier = 'ball'
eval_setup = 'ball_cross_template'
fold_id = 0
random.seed(0)


train, dev, test = phyre.get_fold(eval_setup, fold_id)

cache = phyre.get_default_100k_cache(tier)

print('cache.action_array() and shape:', cache.action_array.shape,
      cache.action_array)


task_id = random.choice(train)
print('Randomly selected task:', task_id)
statuses = cache.load_simulation_states(task_id)
print('Cached simulation status of actions on task', task_id, ':',
      statuses.shape, statuses)
print('Share of SOLVED statuses:', (statuses == phyre.SimulationStatus.SOLVED).mean())



cached_status = phyre.simulation_cache.INVALID
while cached_status == 0:  # Let's make sure we chose a valid action.
    action_index = random.randint(0, len(cache))
    action = cache.action_array[action_index]
    # Get the status for this action from the cache.
    cached_status = statuses[action_index]

# Now let's create a simulator for this task to simulate the action.
simulator = phyre.initialize_simulator([task_id], tier)
simulation = simulator.simulate_action(0,
                                                     action,
                                                     need_images=True)

# Let's compare.
print('Cached status is:', cached_status)
print('Simulated status is:', simulation.status)
print('Simulator considers task solved?', simulation.status.is_solved())



img = phyre.vis.observations_to_float_rgb(simulation.images[-1])