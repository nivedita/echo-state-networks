from scipy import optimize
import timeit

def minimize(value):
    x=value[0]
    y=value[1]

    return abs(2*x-y)


def testDifferentialEvolution():
    bounds = [(0,2),(1,2)]
    result = optimize.differential_evolution(minimize,bounds=bounds)
    print(result)

def basinHopping():
    initial = [0.5,0.5]
    result = optimize.basinhopping(minimize, initial)
    print(result)


print(timeit.timeit("testDifferentialEvolution()", number=1, setup="from __main__ import testDifferentialEvolution"))
print(timeit.timeit("basinHopping()", number=1, setup="from __main__ import basinHopping"))