import configparser
from functools import partial

from constraints import Constraints, constraint_distances, constraint_max_charge, constraint_max_num_changes
from evolution import *
from logger import FileLogger
from utils import *

# PARSING CONFIG
config = configparser.ConfigParser()
config.read('config.ini')

# задаём некоторые константы из config
pdb_file = config['PDB']['File']
descriptors = [float(x) for x in config['PDB']['Descriptors'].split()]
cros_prob = float(config['PARAMS']['CrosProb'])
mut_prob = float(config['PARAMS']['MutProb'])
mut_num = int(config['PARAMS']['MutNum'])
eval_param = float(config['PARAMS']['EvalParam'])
pop_size = int(config['PARAMS']['PopSize'])
weights = [float(x) for x in config['PARAMS']['Weights'].split()]
compute_lmb_inf = config['COMPUTING']['ComputeLambdaInf']
compute_lmb_ouf = config['COMPUTING']['ComputeLambdaOuf']
computed_proteins_path = config['COMPUTING']['ComputedProteinsFileName']
result_file_name = config['COMPUTING']['ResultFileName']
positionsset = {int(x) for x in config['PARAMS']['PositionsSet'].split()}
stop_step = int(config['PARAMS']['StopStep'])
attempts = int(config['PARAMS']['Attempts'])
use_computed = config.getboolean('PARAMS', 'UseComputed')
logger = FileLogger(result_file_name)

# GENERATING CONSTRAINTS
constraints = Constraints()

coordinates = read_coordinates(pdb_file)
sequence = read_sequence(pdb_file)

# функции ограничений
f1 = partial(constraint_distances, min_distance=5.0, coords=coordinates, positions_set=positionsset)
f2 = partial(constraint_max_charge, max_charge=7)
f3 = partial(constraint_max_num_changes, max_num_changes=mut_num)

constraints.add(f1)
constraints.add(f2)
constraints.add(f3)

# COMPUTING
population = ProteinEvolution(population=None, mut_prob=mut_prob, mut_num=mut_num, cros_prob=cros_prob,
                              input_file=compute_lmb_inf, output_file=compute_lmb_ouf, save_file=computed_proteins_path,
                              logger=logger, checker=constraints, weights=weights, positionsset=positionsset)
population.load_computed_proteins()
max_num_changes = population.generate_population(default_sequence=sequence, default_descriptors=descriptors,
                               pop_size=pop_size, from_computed=use_computed)

ini_step = 1
sets, pulls, probs, consts = read_replacements('sites')
iteration, step = max_num_changes + 1, 0

best_protein = population.get_best_protein()
# основной цикл эволюции
while step < stop_step:
    logger(f"Iteration: {iteration}\n")

    population.crossover(attempts=attempts)
    population.mutation(attempts=attempts, iteration=iteration, sets=sets, pulls=pulls, probs=probs, consts=consts)
    population.compute()

#    population.selection(eval_param=0.05, save_n_best=3)
    population.selection(eval_param=eval_param, save_n_best=3)

    cur_best_protein = population.get_best_protein()
    if population.fitness(best_protein.descriptor) \
            < population.fitness(cur_best_protein.descriptor):
        best_protein = cur_best_protein
        step = 0
    else:
        step += 1

    logger(f"Current population:\n")
    population.print_current_population()
    logger(f"The best value: {' '.join(map(str, best_protein.descriptor))}\n"
           f"Step/Stop {step}/{stop_step}\n")
    logger("\n")

    iteration += 1
