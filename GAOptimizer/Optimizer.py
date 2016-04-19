import numpy as np
import operator

class Optimizer:

    def __init__(self, populationSize, parameterBounds,
                 fitnessObj, crossoverOperator, mutationOperator,
                 maxGeneration = 100, mutationRate = 0.01,
                 elitismRate = 0.1):

        self.popSize = populationSize
        self.parameterBounds = parameterBounds
        self.fitnessObj = fitnessObj
        self.crossoverOperator = crossoverOperator
        self.mutationOperator = mutationOperator
        self.maxGeneration = maxGeneration
        self.mutationRate = mutationRate
        self.elitismRate = elitismRate

        # Data structures to store the population
        self.population = [] # List of tuples ie. (parameter tuple, fitness)

        # Roulette wheel
        self.rouletteWheel = []

        # Others
        self.parameterSize = len(parameterBounds)


    def __initialize__(self):

        # Initialize the population with randomly generated chromosomes
        for i in range(self.popSize):
            chromosome = []
            for bound in self.parameterBounds:
                low, high = bound
                value = (high - low)*np.random.random() + low
                chromosome.append(value)

            # Change it to 1 X G tuple
            chromosome = np.array(chromosome).reshape(1,self.parameterSize)

            # Calculate the fitness value
            fitness, others = self.fitnessObj.evaluate(chromosome)

            # Add it to the population
            self.population.append((chromosome, fitness, others))

        # Sort em based on the fitness value - for minimization
        self.population = sorted(self.population, key=operator.itemgetter(1))

    def __select1best__(self, random):
        for item in self.rouletteWheel:
            if item[1] >= random:
                return item[0]

    def __addToPopulation(self, population, popUnique, chromosome, fitness, others):
        chromosomeTuple = tuple(chromosome.tolist())
        if(chromosomeTuple not in popUnique):
            population.append((chromosome, fitness, others))
            popUnique.append(chromosomeTuple)


    def __select2best__(self):
        # Generate a random number(uniform distribution) and select the first one whose
        # fitness is greater than the random

        best1 = np.zeros((1, self.parameterSize))
        best2 = np.zeros((1, self.parameterSize))

        while(np.array_equal(best1,best2)):
            random = np.random.random(2)
            best1 = self.__select1best__(random[0])
            best2 = self.__select1best__(random[1])
        return best1, best2


    def __createRoulleteWheel(self, pop):
        # Since this is a minimization, subtract it from the max fitness in this generation
        max = pop[len(pop)-1][1]
        population = []
        for item in pop:
            fitness = max - item[1]
            population.append((item[0],fitness, item[2]))

        # As this is already sorted, now just have to normalize
        # Find the sum
        sum = 0.0
        for item in population:
            sum += item[1]

        previousFitness = 0.0
        self.rouletteWheel = []
        for item in population:
            fitness = (item[1]/sum) + previousFitness
            previousFitness = fitness
            self.rouletteWheel.append((item[0], fitness, item[2]))

    def optimize(self):

        # Step 1 - Initialize the population
        self.__initialize__()

        # Run for each generation
        for i in range(self.maxGeneration):

            # Create the roulette wheel
            self.__createRoulleteWheel(list(self.population))

            # Retain the best few from the previous population
            retainIndex = int(self.popSize * self.elitismRate)
            newPopulation = list(self.population[:retainIndex])

            # List to maintain the unique parameters
            popUnique = []
            for item in newPopulation:
                chromosome = tuple(item[0].tolist())
                popUnique.append(chromosome)

            # Repeat until the new population is filled
            while len(newPopulation) < self.popSize:

                # Step 2 - Select 2 best candidates from population
                best1, best2 = self.__select2best__()

                # Step 3 - Cross over
                offspring1, offspring2 = self.crossoverOperator(best1, best2)

                fitnessOffspring1, othersOffspring1 = self.fitnessObj.evaluate(offspring1)
                fitnessOffspring2, othersOffspring2 = self.fitnessObj.evaluate(offspring2)

                # Step 4 - Mutate
                mutated1 = self.mutationOperator(np.copy(best1), self.mutationRate, self.parameterBounds)
                mutated2 = self.mutationOperator(np.copy(best2), self.mutationRate, self.parameterBounds)
                fitnessMutated1, othersMutated1 = self.fitnessObj.evaluate(mutated1)
                fitnessMutated2, othersMutated2 = self.fitnessObj.evaluate(mutated2)

                # Step 5 - Add them to the population
                self.__addToPopulation(newPopulation, popUnique, offspring1, fitnessOffspring1, othersOffspring1)
                self.__addToPopulation(newPopulation, popUnique, offspring2, fitnessOffspring2, othersOffspring2)
                self.__addToPopulation(newPopulation, popUnique, mutated1, fitnessMutated1, othersMutated1)
                self.__addToPopulation(newPopulation, popUnique, mutated2, fitnessMutated2, othersMutated2)

                # Step 6 - Sort the population based on fitness
            newPopulation = sorted(newPopulation, key=operator.itemgetter(1))
            self.population = []
            self.population = list(newPopulation)

            # Step 7 - Retain only the good chromosomes
            self.population = self.population[:self.popSize]

            # Display the best
            print("Iteration "+str(i+1)+" Best: "+str(self.population[0]))

    def getOptimalParameters(self):
        return self.population[0]

    def getBestParameters(self, n):
        return self.population[:n]
