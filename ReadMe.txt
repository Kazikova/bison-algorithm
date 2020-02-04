Bison Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
basic version with the Run Support Strategy

created by Anezka Kazikova in 2017/2018 
feel free to contact me at kazikova@utb.cz
Tomas Bata University in Zlin, Czech Republic


Citation: Kazikova A., Pluhacek M., Kadavy T., Senkerik R. (2020) Introducing the Run Support Strategy for the Bison Algorithm. In: Zelinka I., Brandstetter P., Trong Dao T., Hoang Duy V., Kim S. (eds) AETA 2018 - Recent Advances in Electrical Engineering and Related Sciences: Theory and Application. AETA 2018. Lecture Notes in Electrical Engineering, vol 554. Springer, Cham


FILES

> BasicBison.py > source code of the Bison Algorithm with the Run Support Strategy. The main function is the bison_algorithm(number_of_runs, problem_definition, test_flags).
> benchmark.py > here you can write your objective functions to optimize. It includes implementation of the IEEE CEC 2015 and IEEE CEC 2017 benchmark libraries, and some other functions like Easom, Schwefel, etc.
> compare.py > main executable code. Can compare more algorithms, or run only one. Define the problem in array 'problem' and compared optimization algorithms in array 'optimization_algorithm'.
> testing.py > support functions for movement visualization and saving files.
> PSO.py > Particle Swarm Optimization algorithm for comparison. Based on the EvoloPy library, modified for the use of this code.
> CS.py > Cuckoo Search algorithm for comparison. Based on the EvoloPy library, modified for the use of this code.

If PSO and CS used, please, cite: Faris, Hossam & Aljarah, Ibrahim & Mirjalili, Seyedali & Castillo, Pedro & Merelo Guervós, Juan. (2016). EvoloPy: An Open-Source Nature-Inspired Optimization Framework in Python. 10.5220/0006048201710177. 



HOW TO USE THE ALGORITHM FOR OPTIMIZATION?
> 1. Unzip the input zips to use the IEEE CEC Benchmarks

In compare.py:
> define optimized problem 
> pick optimization algorithm(s)
> choose, what you want to test in test_flags
> for optimization us function optimize(x, dimension, optimization_algorithm)
	where x = number of tested function in CEC benchmark
	dimension = dimensionality (CEC has only 10D, 30D, 50D, 100D)
	optimization_algorithm copy as is, defines which algorithm to use
> for pictures of 2D movement, use function test_movement()


VERSION OF PYTHON AND LIBRARIES
> Developed for Python 3.6.0
> With libraries: 
>	NumPy 1.12.0, 
>	MatPlotLib 2.0.0


With the wish of many great optimization successes,

Yours sincerely,
Anezka Kazikova
