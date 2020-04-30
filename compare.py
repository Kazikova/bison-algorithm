import BasicBison
import benchmark
import PSO
import CS
import FFA
import BAT
import ctypes
import numpy
import time

problem = {
    'dimension': 10,
    'low_bound': -100,
    'up_bound': 100,
    'function': benchmark.cec2017,
    'func_num': 1,
    'population': 50,
    'swarm': 40,
    'elity': 20,
    'overstep': 3.5,
    'iterations_to_enhance_run': 0,  # run support parameter = number of iterations for run support strategy
    'filename': 'results/cec2017/',  # path where to save results
    'boundary_politics': 'hypersphere',
    'multirun': 1
}

test_flags = {
    'error_values': True,  # standard IEEE testing
    'convergence': False,
    'statistics': False,
    'movement_in_2d': False,
    'cumulative_movement': False,
    'complexity_computation': False
}


def optimize(func, dim, optimization_algorithm, number_of_runs=51):
    global problem
    problem['func_num'] = func
    problem['dimension'] = dim

    if optimization_algorithm['pso']:
        print("Now dealing with PSO %sD %sF" % (dim, func))
        stats_of_pso, best_pso = PSO.PSO(number_of_runs, problem, test_flags)
        print("PSO %sD %sF: %s" % (dim, func, best_pso))

    if optimization_algorithm['bison']:
        print("Now dealing with Bison %sD %sF" % (dim, func))
        improvements, best_bison = BasicBison.bison_algorithm(number_of_runs, problem, test_flags)
        print("Bison %sD %sF: %s" % (dim, func, best_bison))

    if optimization_algorithm['cs']:
        print("Now dealing with CS %sD %sF" % (dim, func))
        stats_of_cs, best_cs = CS.CS(number_of_runs, problem, test_flags)
        print("CS %sD %sF: %s" % (dim, func, best_cs))

    if optimization_algorithm['bat']:
        print("Now dealing with BAT %sD %sF" % (dim, func))
        stats_of_bat, best_bat = BAT.BAT(number_of_runs, problem, test_flags)
        print("BAT %sD %sF: %s" % (dim, func, best_bat))

    if optimization_algorithm['ffa']:
        print("Now dealing with FFA %sD %sF" % (dim, func))
        stats_of_ffa, best_ffa = FFA.FFA(number_of_runs, problem, test_flags)
        print("FFA %sD %sF: %s" % (dim, func, best_ffa))
    print("Yay!~")


def close_library():
    handle = benchmark.dll_15._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)
    handle = benchmark.dll_17._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)
    handle = benchmark.dll_20._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)


def test_movement(test_scenario=1):
    problem['dimension'] = 2
    problem['filename'] = 'results/movement/basic bison/'
    problem['overstep'] = 3.5
    problem['population'] = 50
    problem['swarm'] = 40
    problem['elity'] = 20
    problem['iterations_to_enhance_run'] = 2

    if test_scenario == 1:
        problem['low_bound'] = -100
        problem['up_bound'] = 100
        problem['function'] = benchmark.rastrigin
    else:
        problem['function'] = benchmark.schwefel
        problem['low_bound'] = -514
        problem['up_bound'] = 514
    test_flags = {
        'convergence': False,
        'statistics': False,
        'movement_in_2d': True,
        'cumulative_movement': False,
        'error_values': False,
        'complexity_computation': False
    }
    BasicBison.bison_algorithm(1, problem, test_flags)

# command optimization from command line with this function:
# test_concrete_problem(int(sys.argv[1]), int(sys.argv[2]))

optimization_algorithm = {
    'bison': True,
    'pso': False,
    'cs': False,
    'ffa': False,
    'bat': False
}

# For movement example in 2 dimensions use function: test_movement(2)
# test_movement(2)

for x in range(1, 30):
    optimize(x, 10, optimization_algorithm, 51)

close_library()
