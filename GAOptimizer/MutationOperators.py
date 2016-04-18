import numpy as np

class SimpleMutationOperator(object):
    def __call__(self, chromosome, mutationRate, parameterBounds):

        # Randomly mutate one of the dimensions
        mutationPoint = np.random.randint(0,chromosome.shape[1])

        # Boundary check
        lowBound, highBound = parameterBounds[mutationPoint]

        currentValue = chromosome[0,mutationPoint]
        if(np.random.random() >= 0.5):
            chromosome[0,mutationPoint] = currentValue + (mutationRate * np.random.random() * (highBound-currentValue))
        else:
            chromosome[0,mutationPoint] = currentValue - (mutationRate * np.random.random() * (currentValue-lowBound))

        return chromosome

class UniformMutationOperator(object):
    def __call__(self, chromosome, mutationRate, parameterBounds):

        # Randomly mutate one of the dimensions
        mutationPoint = np.random.randint(0,chromosome.shape[1])

        # Just randomly assign a new value
        low, high = parameterBounds[mutationPoint]
        value = low + (high-low) * np.random.random()
        chromosome[0,mutationPoint] = value
        return chromosome

class SimpleAndUniformHybrid(object):
    def __call__(self, chromosome, mutationRate, parameterBounds):
        random = np.random.random()
        if(random >= 0.5):
            return SimpleMutationOperator().__call__(chromosome, mutationRate, parameterBounds)
        else:
            return UniformMutationOperator().__call__(chromosome, mutationRate, parameterBounds)