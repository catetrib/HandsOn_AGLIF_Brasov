from Genetic_optimization_250523 import runsimulation
from joblib import Parallel, delayed
from neuron_model_utility import compute_block_functions,print_figure,testAccuracy_optimized_neuron
import time
t = time.time()


neuron_list=['neuron191113002_S75']#,'18n21011','18n21013','18n21017','18n20001','18n20007']#,#,

num_cores = len(neuron_list)#multiprocessing.cpu_count();

Ltt = Parallel(n_jobs=num_cores, verbose=50)(delayed(runsimulation)(neuron_list[i]) for i in range(len(neuron_list)))


for i in range(len(neuron_list)):
    compute_block_functions(neuron_list[i])
    
    monod_active=False
    print_figure(neuron_list[i],monod_active)
    
    monod_active=True
    print_figure(neuron_list[i],monod_active)
    
    testAccuracy_optimized_neuron(neuron_list[i])
    
elapsed = time.time() - t