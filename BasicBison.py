# ~~~~~~~~~~~~~~~~~ Bison Algorithm. ~~~~~~~~~~~~~~~~~
# Swarm optimization algorithm developed by kazikova@utb.cz in 2017

import benchmark
import testing
import random
import numpy

# ~ Basic problem definition ~
# Can be overridden by PROBLEM DEFINITION in compare.py. If so, there is no need to change the parameters below:
dimension = 10
low_bound = -100
up_bound = 100
objf = benchmark.cec2017
func_num = 1  # There are 30 functions in CEC2017. This defines the number of the currently solved one.

# ~ Herd definition ~
population = 50
elite_group_size = 20  # recommended 20
swarm_group_size = 40  # recommended 40
overstep = 3.5  # Determines how many times can swarming bison overstep the center (0=no movement, 1=max to the center)

# Parameter recommendations are in paper: "Kazikova, A., Pluhacek, M., & Senkerik, R. (2018).
#       Tuning Of The Bison Algorithm Control Parameter. In ECMS (pp. 156-162)."

center_computation = 'ranked'  # possible values: arithmetic / weighted / ranked. Shows non-significant differences.
neighbourhood = abs(up_bound - low_bound) / 15  # distribution of the running group

# ~ Global variables ~
max_evaluation = 10000 * dimension
max_iteration = round((max_evaluation - population) / population)
bisons = numpy.zeros((population, dimension), dtype=numpy.double)
bisons_fitness = numpy.zeros(population, dtype=numpy.double)
convergence_curve = {}
evaluations = 0
run_direction = numpy.zeros(dimension, dtype=numpy.double)
center = numpy.zeros(dimension, dtype=numpy.double)
savefilename = 'results/'  # Where do we save the results?
boundary_politics = 'hypersphere'  # border strategies were compared at paper:
# "Kazíková, A., Komínková Oplatková, Z., Pluháček, M., & Šenkeřík, R. (2019).
#       Border strategies of the bison algorithm. In Proceedings of the 33rd International ECMS Conference on
#       Modelling and Simulation (ECMS 2019). European Council for Modelling and Simulation."
# -> Hypersphere rocks. Other options include: 'bounce', 'random', 'stay'

# ~ Run Support Mechanism ~
# This improvement of the Basic Bison Algorithm significantly raises the utilization of explored solutions.
# When a bison from the running group finds PROMISING SOLUTION, the swarming group tries to swarm closer to the
# successful runner in NEXT FEW ITERATIONS. (Preliminary recommended value is 2.)

# The Run Support Strategy cite as:
# "Kazikova, A., Pluhacek, M., Kadavy, T., & Senkerik, R. (2018, September).
#       Introducing the Run Support Strategy for the Bison Algorithm.
#       In International Conference on Advanced Engineering Theory and Applications (pp. 272-282). Springer, Cham."

run_support = 2  # number of iterations for swarmers to explore the area around the promising solution
successful_runners = -2


def set_global_parameters(problem_definition, test_flags):
    global dimension;
    global low_bound;
    global up_bound;
    global objf;
    global max_evaluation;
    global max_iteration;
    global population;
    global swarm_group_size;
    global elite_group_size;
    global bisons;
    global bisons_fitness;
    global run_direction;
    global neighbourhood;
    global func_num;
    global savefilename;
    global center;
    global run_support;
    global overstep;
    global boundary_politics;

    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound = problem_definition['up_bound']
    objf = problem_definition['function']
    func_num = problem_definition['func_num']
    savefilename = problem_definition['filename']
    run_support = problem_definition['iterations_to_enhance_run']
    overstep = problem_definition['overstep']
    boundary_politics = problem_definition['boundary_politics']
    population = problem_definition['population']
    swarm_group_size = problem_definition['swarm']
    elite_group_size = problem_definition['elity']

    neighbourhood = abs(up_bound - low_bound) / 15  # 15
    if test_flags['complexity_computation']:
        max_evaluation = 200000
    else:
        max_evaluation = benchmark.get_max_fes(dimension, objf)
    max_iteration = round((max_evaluation - population) / population)
    bisons = numpy.zeros((population, dimension), dtype=numpy.double)
    bisons_fitness = numpy.zeros(population)
    run_direction = numpy.zeros(dimension, dtype=numpy.double)
    center = numpy.zeros(dimension, dtype=numpy.double)
    run_direction = [random.choice([-1, 1]) * random.uniform(neighbourhood / 3, neighbourhood) for i in
                     range(dimension)]


def fitness(position):
    global evaluations
    evaluations += 1
    return objf(position, dimension, func_num)


def bisons_init():
    global convergence_curve
    global bisons
    global run_direction
    global bisons_fitness
    global center

    # position bisons in the swarming group randomly and sort them by fitness value
    for x in range(swarm_group_size):
        bisons[x] = [random.uniform(low_bound, up_bound) for i in range(dimension)]
        bisons_fitness[x] = fitness(bisons[x])
    bisons[:swarm_group_size] = bisons[bisons_fitness[:swarm_group_size].argsort()]
    bisons_fitness[:swarm_group_size].sort()

    # position running bisons around the best solution
    for x in range(swarm_group_size, population):
        bisons[x] = [bisons[0][i] + random.uniform(-neighbourhood, neighbourhood) for i in range(dimension)]
        check_bounds(bisons[x])
        bisons_fitness[x] = fitness(bisons[x])

    # copy better runners into the swarming group and toss the worse swarming solutions
    sorting_indices = bisons_fitness.argsort()
    bisons[:swarm_group_size] = bisons[sorting_indices[:swarm_group_size]]
    bisons_fitness[:swarm_group_size] = bisons_fitness[sorting_indices[:swarm_group_size]]

    # initiate the run direction vector and results array
    run_direction = [random.choice([-1, 1]) * random.uniform(neighbourhood / 3, neighbourhood) for i in
                     range(dimension)]
    convergence_curve = {'best': [], 'median': [], 'worst': [], 'evaluation': [], 'errors': []}


def bisons_move(iteration):
    global bisons;
    global run_direction;
    global bisons_fitness;
    global center
    global successful_runners

    # subtle alternation of the run direction vector in each iteration
    run_direction = [run_direction[x] * random.uniform(0.9, 1.1) for x in range(dimension)]

    # The Run Support Strategy of the Bison Algorithm works as follows:
    #   If runners find a promising solution, swarmers swarm towards the promising solution
    #   for next few iterations defined by the run support parameter.
    #   Otherwise swarming group swarms towards its center as usual.

    for x, item in enumerate(bisons):
        current = numpy.array(bisons[x])
        if x < swarm_group_size:
            if successful_runners > 0:
                swarm(current, 0.95, 1.05)
            else:
                swarm(current, 0, overstep)
            current_fitness = fitness(current)
            if current_fitness < bisons_fitness[x]:
                bisons[x] = current
                bisons_fitness[x] = current_fitness
        if x >= swarm_group_size:
            run(current, x)
            bisons[x] = current
            bisons_fitness[x] = fitness(current)

    # Sort the swarming group
    sorting_indices = bisons_fitness.argsort()
    bisons[:swarm_group_size] = bisons[sorting_indices[:swarm_group_size]]
    bisons_fitness[:swarm_group_size] = bisons_fitness[sorting_indices[:swarm_group_size]]
    update_convergence_curve()

    # Check if runners found a promising solution and set appropriate center for next movement
    successful_runners -= 1
    for better in range(swarm_group_size, population):
        if sorting_indices[better] < swarm_group_size:
            successful_runners = run_support
            center = numpy.copy(bisons[better])
    if successful_runners <= 0:
        center = compute_center()


def swarm(bison, from_=0, to_=overstep):
    direction = numpy.zeros(dimension, dtype=numpy.double)
    for x in range(dimension):
        direction[x] = center[x] - bison[x]
        bison[x] += direction[x] * random.uniform(from_, to_)
    check_bounds(bison)


def run(bison, x):
    for d in range(dimension):
        bison[d] += run_direction[d]
    check_bounds(bison)
    return bison


def check_bounds(bison):
    global run_direction
    size = up_bound - low_bound
    # Boundary strategies were compared in paper:
    # Kazíková, A., Komínková Oplatková, Z., Pluháček, M., & Šenkeřík, R. (2019).
    #    Border strategies of the bison algorithm. In Proceedings of the 33rd International ECMS Conference on Modelling
    #    and Simulation (ECMS 2019). European Council for Modelling and Simulation.
    # 1] STANDARD BOUNDARY POLITICS: HYPERSPHERE
    if boundary_politics == "hypersphere":
        for x in range(dimension):
            if bison[x] > up_bound:
                bison[x] = low_bound + (abs(bison[x] - up_bound) % size)
            elif bison[x] < low_bound:
                bison[x] = up_bound - (abs(bison[x] - low_bound) % size)

    # 2] BOUNCE BACK IN CROSSED DIMENSION aka Reflection
    elif boundary_politics == "bounce":
        for x in range(dimension):
            if bison[x] < low_bound:
                bison[x] = low_bound + (abs(low_bound - bison[x]))
                run_direction[x] *= -1
            if bison[x] > up_bound:
                bison[x] = up_bound - (abs(bison[x] - up_bound))
                run_direction[x] *= -1

    # 3] RANDOM POSITION
    elif boundary_politics == "random":
        for x in range(dimension):
            if bison[x] < low_bound or bison[x] > up_bound:
                bison[x] = random.uniform(low_bound, up_bound)

    # 4] STAY ON BORDERS + change movement vector in the other direction aka Clip and Flip
    elif boundary_politics == "stay":
        for x in range(dimension):
            if bison[x] < low_bound or bison[x] > up_bound:
                bison[x] = numpy.clip(bison[x], low_bound, up_bound)
                run_direction[x] *= -1


def update_convergence_curve():
    global convergence_curve
    convergence_curve['best'].append(bisons_fitness[0])
    convergence_curve['median'].append(bisons_fitness[int(round(population / 2))])
    convergence_curve['worst'].append(bisons_fitness[population - 1])
    convergence_curve['evaluation'].append(evaluations)


def reset_run():
    global bisons;
    global convergence_curve;
    global evaluations;
    global center;
    center = numpy.zeros(dimension)
    evaluations = 0
    bisons = numpy.zeros((population, dimension), dtype=numpy.double)
    convergence_curve.clear()


def compute_center():
    center = numpy.zeros(dimension, dtype=numpy.double)
    bison_weight = numpy.ones(elite_group_size)
    all_weights = sum(bisons_fitness[:elite_group_size])

    # There are many ways to compute center. Their impact, however, did not prove to be significant.
    # Defaultly, we use the ranked center computation.
    if center_computation == "arithmetic":
        all_weights = elite_group_size
    elif center_computation == "weighted":
        for x in range(elite_group_size):
            bison_weight[x] = all_weights - bisons_fitness[x]
        all_weights = sum(bison_weight)
    elif center_computation == "ranked":
        for x in range(elite_group_size):
            bison_weight[x] = (elite_group_size - x) * 10
        all_weights = sum(bison_weight)
    elif center_computation == "median":
        for dim in range(dimension):
            center[dim] = numpy.median(bisons[:elite_group_size, dim])
        return center
    if all_weights == 0:
        all_weights = elite_group_size
        bison_weight = numpy.ones(elite_group_size)

    for d in range(dimension):
        for x in range(elite_group_size):
            center[d] += (bison_weight[x] * bisons[x][d]) / all_weights
    return center


def bison_algorithm(number_of_runs, problem_definition, test):
    global bisons;
    global convergence_curve
    set_global_parameters(problem_definition, test)

    solution_score = 0.0
    solution = numpy.zeros(dimension, dtype=numpy.double)
    statistics = numpy.zeros(number_of_runs)
    all_errors = numpy.zeros(
        (number_of_runs, len(benchmark.when_to_record_results(dimension, problem_definition['function']))))
    save_elites = []
    save_swarmers = []
    save_runners = []

    for i in range(number_of_runs):

        reset_run()
        bisons_init()
        if i == 0:
            solution = numpy.array(bisons[0])
            solution_score = bisons_fitness[0]
        save_errors_at = benchmark.when_to_record_results(dimension, problem_definition['function'])

        for x in range(max_iteration):

            if test['movement_in_2d'] and x < 50:
                testing.plot_contour(savefilename, bisons, center, low_bound, up_bound, x, elite_group_size,
                                     swarm_group_size)

            bisons_move(x)

            if test['error_values'] and len(save_errors_at) > 0 and evaluations >= save_errors_at[0]:
                convergence_curve['errors'].append(bisons_fitness[0] - benchmark.known_optimum_value(func_num))
                save_errors_at.pop(0)
            if test['convergence']:
                testing.save_progress(convergence_curve)
            if test['cumulative_movement']:
                for b in range(elite_group_size):
                    save_elites.append(numpy.copy(bisons[b]))
                for b in range(elite_group_size, swarm_group_size):
                    save_swarmers.append(numpy.copy(bisons[b]))
                for b in range(swarm_group_size, population):
                    save_runners.append(numpy.copy(bisons[b]))
                if x == 50:
                    testing.plot_cumulative_movement(save_elites, save_swarmers, save_runners, low_bound, up_bound, x)

        if test['error_values']:
            all_errors[i] = numpy.array(convergence_curve['errors'])
        if test['statistics']:
            statistics[i] = bisons_fitness[0]
        if solution_score > bisons_fitness[0]:
            solution = bisons[0]
            solution_score = bisons_fitness[0]
        print("Bison Algorithm %s: %s, %s evaluations, %s iterations" %
              (i, bisons_fitness[0], evaluations, max_iteration))

    print("Best solution: %s" % solution)
    if test['statistics']:
        statistics = testing.evaluate_all_statistics(statistics)
        print("Statistics of bisons: %s" % statistics)
    if test['error_values']:
        filename = str(savefilename) + str(func_num) + '_' + str(dimension) + '.csv'
        testing.save_errors_to_file(all_errors, filename)

    return solution_score, solution
