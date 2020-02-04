import matplotlib.pyplot as plt
import benchmark
import numpy as np
import csv


def save_progress(convergence):
    # plt.plot(convergence['evaluation'], convergence['median'], 'r.')
    # plt.plot(convergence['evaluation'], convergence['worst'], 'b.')
    plt.plot(convergence['evaluation'], convergence['best'], 'k')


def plot_saved_progress(dimension = "", center = ""):
    plt.ylabel('best result quality')
    plt.xlabel('fitness evaluations')
    plt.title('%s %s %s' % (benchmark.name_of_function, dimension, center))
    plt.show()


def plot_cumulative_movement(filename, elite_pos, swarm_pos, run_pos, low_bound=-100, up_bound=100, iteration=1):
    fig = plt.figure()
    name_of_function = benchmark.name_of_function
    X, Y, Z = define_objective_function(name_of_function, low_bound, up_bound)

    for i in range(0, len(run_pos)):
        running, = plt.plot(run_pos[i][0], run_pos[i][1], 'k.')

    for i in range(0, len(swarm_pos)):
        swarming, = plt.plot(swarm_pos[i][0], swarm_pos[i][1], 'c.')

    for i in range(0, len(elite_pos)):
        elites, = plt.plot(elite_pos[i][0], elite_pos[i][1], 'r.')

    plt.legend([elites, swarming, running], ["Elites", "Swarming", "Running"], loc=1)

    plt.contour(X, Y, Z)
    plt.axis([low_bound, up_bound, low_bound, up_bound])
    plt.title('Iteration %s' % iteration)
    name = str(filename) + '_cumulative_' + str(iteration) + '.svg'
    fig.tight_layout()
    fig.savefig(name)
    plt.close(fig)


# use by: testing.plot_contour(filename, bisons, low_bound, up_bound, 0, number_of_elite_bisons, number_of_swarming_bisons)
def plot_contour(filename, positions, center, low_bound=-100, up_bound=100, iteration=1, number_of_elite=20, number_of_swarm=40, center_computation="", beta=""):
    fig = plt.figure()
    name_of_function = benchmark.name_of_function
    X, Y, Z = define_objective_function(name_of_function, low_bound, up_bound)

    for i in range(0, len(positions)):
        if i >= number_of_elite and i < number_of_swarm:
            swarming, = plt.plot(positions[i][0], positions[i][1], 'cs')
        elif i < number_of_elite:
            elites, = plt.plot(positions[i][0], positions[i][1], 'rD')
        else: # if i > number_of_swarm:
            running, = plt.plot(positions[i][0], positions[i][1], 'ko')
    center_point, = plt.plot(center[0], center[1], 'bX')

    plt.legend([elites, swarming, running, center_point], ["Elites", "Swarming", "Running", "Center"], loc=2)
    plt.contour(X, Y, Z)
    plt.axis([low_bound, up_bound, low_bound, up_bound])
    plt.title('Iteration %s' % iteration)
    fig.tight_layout()
    name = filename + '_' + str(iteration) + '.svg'
    # name2 = filename + str(iteration) + '.svg'
    fig.savefig(name)
    plt.close(fig)


def define_objective_function(name_of_function, low_bound, up_bound):
    x = np.linspace(low_bound, up_bound)
    y = np.linspace(low_bound, up_bound)
    X, Y = np.meshgrid(x, y)

    if name_of_function == 'De Jong 1':
        Z = X**2 + Y**2;
    if name_of_function == 'Rastrigin':
        Z = 20 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y);
    if name_of_function == 'Schwefel':
        Z = -X*np.sin(np.sqrt(abs(X)))-Y*np.sin(np.sqrt(abs(Y)));
    if name_of_function == 'Rosenbrock':
        Z = (1. - X) ** 2 + 100. * (Y - X * X) ** 2
    if name_of_function == 'Easom':
        Z = -np.cos(X) * np.cos(Y) * np.exp( -(X - np.pi)**2 - (Y - np.pi)**2 )
    return [X, Y, Z]


def save_population_to_table(population, population_fitness, iteration):
    table = []
    for x in range(len(population_fitness)):
        table.append([population_fitness[x], population[x]])
    # write it
    with open('diversity/population_diversity'+str(iteration)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]

def save_errors_to_file(errors, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in errors]


def save_statistics(fitness, statistics):
    statistics.append(fitness[0])
    return statistics


def evaluate_all_statistics(statistics):
    solution = {}
    solution['min'] = min(statistics)
    solution['avg min'] = np.average(statistics)
    solution['std'] = np.std(statistics)
    return solution


def save_statistics_to_file(statistics, filename=""):

    all_runs = len(statistics['min'])

    result = {}
    result['best'] = min(statistics['min'])
    result['min'] = sum(statistics['min'])/all_runs
    result['max'] = sum(statistics['max'])/all_runs
    result['median'] = sum(statistics['median'])/all_runs
    result['average'] = sum(statistics['average'])/all_runs
    result['last_population_deviation'] = sum(statistics['deviation'])/all_runs

    variance = 0
    for x in range(len(statistics['min'])):
        variance += (statistics['min'][x] - result['min']) ** 2 / (len(statistics['min']) - 1)
    result['best_result_deviation'] = np.sqrt(variance)

    with open('statistics_'+str(filename)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in result]

    print(result)


def save_statistics_to_file(statistics, filename=""):
    with open('statistics_'+str(filename)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in statistics]


def evaluate_average_convergence(convergence, evaluations, label_text = " ", line_style = "-"):
    number_of_runs = len(convergence)
    max_iteration = len(convergence[0])

    average_convergence = np.zeros(max_iteration)

    if label_text=="weighted":
        line_style = ":"
    elif label_text=="ranking":
        line_style="-."

    for iteration in range(max_iteration):
        for run in range(number_of_runs):
            average_convergence[iteration] += convergence[run][iteration]/number_of_runs

    # plt.plot(np.arange(0, max_iteration, 1), average_convergence)
    plt.plot(evaluations, average_convergence, label=label_text, linestyle=line_style)
    plt.legend()

    return

