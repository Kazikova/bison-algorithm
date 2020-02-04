import math
import ctypes
import numpy

global name_of_function

dll_17 = ctypes.cdll.LoadLibrary('CEC2017 x64.dll')
# CEC2017 x32.dll for 32bit systems
# CEC2017 x64.dll for 64bit systems
dll_15 = ctypes.cdll.LoadLibrary('CEC2015 x64.dll')
# CEC2015 x32.dll for 32bit systems
# CEC2015 x64.dll for 64bit systems

def cec2017( position, dimension, func_num ):
    global name_of_function
    name_of_function = 'Cec 2017'

    fun = dll_17.call_function
    fun.restype = ctypes.c_double
    fun.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    result = fun(position, dimension, func_num)

    return result


def cec2015( position, dimension, func_num ):
    global name_of_function
    name_of_function = 'Cec 2015'

    fun = dll_15.call_function
    fun.restype = ctypes.c_double
    fun.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    result = fun(position, dimension, func_num)

    return result


def rastrigin( position, *args ):
    global name_of_function
    name_of_function = 'Rastrigin'
    result = 10 * len(position)
    for x in position:
        # x += 2
        result += (x)**2 - 10*numpy.cos(2*math.pi*x)
    return result


def dejong1( position, dim=0, func=0 ):
    global name_of_function
    name_of_function = 'De Jong 1'
    result = 0
    for x in position:
        result += (x)**2
    return result


def schwefel( position, *args  ):
    global name_of_function
    name_of_function = 'Schwefel'
    alpha = 418.982887
    result = 0.0
    for i in range(len(position)):
        result -= position[i] * math.sin(math.sqrt(math.fabs(position[i])))
    return result + alpha * len(position)



def rosenbrock( position, *args  ):
    global name_of_function
    name_of_function = 'Rosenbrock'
    result = 0.0
    for x in range(1, len(position)):
        result += 100 * (position[x] - position[x-1] ** 2 ) ** 2 + ( 1 - position[x-1] ) ** 2
    return result


def easom( position, *args  ):
    global name_of_function
    name_of_function = 'Easom'
    result = 0.0
    for x in range(1, len(position)):
        x1 = position[x-1]
        x2 = position[x]
        exponent = - (x1 - numpy.pi)**2 - (x2 - numpy.pi)**2
        result += -numpy.cos(x1) * numpy.cos(x2) * numpy.e**exponent
    return result
