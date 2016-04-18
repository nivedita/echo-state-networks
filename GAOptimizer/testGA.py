import numpy as np
from GAOptimizer import Optimizer as opt, MutationOperators as mOpr, CrossoverOperators as cOpr
from performance import ErrorMetrics as metrics
from decimal import Decimal

class DoubleMinimizer:
    def evaluate(self, parameter):
        x = parameter[0, 0]
        y = parameter[0, 1]

        # Minimize/Solve
        # #x^2 + y^2 = 25
        # fun = np.square(x) + np.square(y)
        # error = abs(25-fun)

        # Minimize/Solve
        #x + y = 100
        fun = x + y
        error = abs(100-fun)

        return error

class SingleMinimizer:
    def evaluate(self, parameter):
        x = parameter[0, 0]

        # Minimize/Solve
        #x^2 = 25
        fun = np.square(x)
        error = abs(25-fun)

        return error


optimzer = opt.Optimizer(populationSize=100, parameterBounds=[(0,1000),(10,200)], fitnessObj=DoubleMinimizer(),
                        mutationOperator=mOpr.SimpleAndUniformHybrid(), crossoverOperator=cOpr.SinglePointAndLineHybridCrossover(),
                        mutationRate=0.1, maxGeneration=1000, elitismRate=0.3)


# optimzer = opt.Optimizer(populationSize=100, parameterBounds=[(0,1200)], fitnessObj=SingleMinimizer(),
#                         mutationOperator=mOpr.UniformMutationOperator(), crossoverOperator=cOpr.LineCrossover(),
#                         mutationRate=0.4, maxGeneration=100, elitismRate=0.3)

optimzer.optimize()
solution, minValue = optimzer.getOptimalParameters()
print(solution, minValue)

print("The best candidates in the population pool are:")
bestCandidates = optimzer.getBestParameters(100)
for item in bestCandidates:
    x = Decimal(item[0][0,0])
    y = Decimal(item[0][0,1])
    error = Decimal(item[1])
    print(x,y,error)