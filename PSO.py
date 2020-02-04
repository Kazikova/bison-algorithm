# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016
within the EvoloPy optimization library
@author: Hossam Faris
-> Modified by Anezka Kazikova to fit the uniform template in 2018
"""

import random
import numpy
import math
import time
import testing


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = {'best': [], 'median': [], 'worst': [], 'evaluation': []}
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0


def PSO(number_of_runs, problem_definition, test_flags):

    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound  = problem_definition['up_bound']
    objf      = problem_definition['function']

    test_convergence = test_flags['convergence']
    test_error_values = test_flags['error_values']
    test_statistics = test_flags['statistics']
    func_num = problem_definition['func_num']

    statistics = numpy.zeros(number_of_runs)

    # PSO parameters

    Vmax = 6 # navrhovana zmena 40. puvodne bylo 6
    # idealne 20% celkoveho rozsahu
    PopSize = 50
    wMax = 0.9
    wMin = 0.2 # 0.4 navrhovana zmena
    c1 = 2
    c2 = 2
    max_evaluation = 10000 * dimension
    max_iteration = round((max_evaluation) / PopSize)
    s = solution()
    average_convergence_curve = numpy.zeros((number_of_runs, max_iteration))
    all_errors = numpy.zeros((number_of_runs, 14))

    result = numpy.zeros(dimension)
    result_score = float("inf")

    for runs in range(number_of_runs):

        save_errors_at = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        ######################## Initializations
        values = numpy.zeros(PopSize)
        evaluations = 0
        vel = numpy.zeros((PopSize, dimension))

        pBestScore = numpy.zeros(PopSize)
        pBestScore.fill(float("inf"))

        pBest = numpy.zeros((PopSize, dimension))
        gBest = numpy.zeros(dimension)

        gBestScore = float("inf")

        pos = numpy.random.uniform(0, 1, (PopSize, dimension)) * (up_bound - low_bound) + low_bound

        convergence_curve = numpy.zeros(max_iteration)
        convergence_errors = []
        evaluation_curve = numpy.zeros(max_iteration)


        ############################################
        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for l in range(0, max_iteration):
            for i in range(0, PopSize):
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)


                pos[i, :] = numpy.clip(pos[i, :], low_bound, up_bound)
                # zmenit na random a na 40
                # pos[i, :] = numpy.random.uniform(0, 1, (PopSize, dimension)) * (up_bound - low_bound) + low_bound

                # Calculate objective function for each particle
                fitness = objf(pos[i, :], dimension, func_num)
                values[i] = fitness
                evaluations += 1

                if (pBestScore[i] > fitness):
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :]

                if (gBestScore > fitness):
                    gBestScore = fitness
                    gBest = pos[i, :]

            # Update the W of PSO
            w = wMax - l * ((wMax - wMin) / max_iteration);

            for i in range(0, PopSize):
                for j in range(0, dimension):
                    r1 = random.random()
                    r2 = random.random()
                    vel[i, j] = w * vel[i, j] + c1 * r1 * (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                    if (vel[i, j] > Vmax):
                        vel[i, j] = Vmax

                    if (vel[i, j] < -Vmax):
                        vel[i, j] = -Vmax

                    pos[i, j] = pos[i, j] + vel[i, j]
            if test_convergence:
                convergence_curve[l] = gBestScore
                evaluation_curve[l] = evaluations

            if test_error_values and evaluations >= save_errors_at[0] * max_evaluation:
                convergence_errors.append(gBestScore - (func_num*100))
                save_errors_at.pop(0)

        if test_error_values:
            all_errors[runs] = numpy.array(convergence_errors)


        print(['PSO ' + str(runs) + ': [' + str(gBestScore) + '] Evaluations: ' + str(evaluations) + ' Iterations: ' + str(max_iteration)])
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence['best'] = convergence_curve
        s.convergence['evaluation'] = evaluation_curve

        # testing.save_progress(s.convergence)
        s.optimizer = "PSO"
        s.objfname = objf.__name__

        if test_convergence:
            average_convergence_curve[runs] = convergence_curve
        if test_statistics:
            statistics[runs] = gBestScore

        if result_score > gBestScore:
            result_score = gBestScore
            result = gBest


    if test_error_values:
        filename = 'cec2015/pso_' + str(func_num) + '_' + str(dimension) + '.csv'
        testing.save_errors_to_file(all_errors, filename)



    # testing.plot_saved_progress(dimension)
    if test_convergence:
        testing.evaluate_average_convergence(average_convergence_curve, evaluation_curve, "PSO", ":")
    if test_statistics:
        statistics = testing.evaluate_all_statistics(statistics)

    return statistics, result
