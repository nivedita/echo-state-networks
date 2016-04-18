import numpy as np

class SinglePointCrossover(object):
    def __call__(self, parent1, parent2):

        # Randomly generate the crossover point
        crossPoint = np.random.randint(0,parent1.shape[1])

        # Create offsprings from chromosome by mixing the genes
        offspring1 = np.zeros(parent1.shape)
        offspring2 = np.zeros(parent1.shape)
        offspring1[0, :crossPoint] = parent1[0, :crossPoint]
        offspring1[0, crossPoint:] = parent2[0, crossPoint:]
        offspring2[0, :crossPoint] = parent2[0, :crossPoint]
        offspring2[0, crossPoint:] = parent1[0, crossPoint:]

        return offspring1, offspring2

class TwoPointCrossover(object):
    def __call__(self, parent1, parent2):

        # Randomly generate two crossover points
        crosspoints = sorted(np.random.randint(0,parent1.shape[1],2).tolist())
        crossPoint1 = crosspoints[0]
        crossPoint2 = crosspoints[1]

        # Create offsprings from chromosome by mixing the genes
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)
        offspring1[0, crossPoint1:crossPoint2] = parent2[0, crossPoint1:crossPoint2]
        offspring2[0, crossPoint1:crossPoint2] = parent1[0, crossPoint1:crossPoint2]

        return offspring1, offspring2

class LineCrossover(object):
    def __call__(self, parent1, parent2):

        # Randomly generate two crossover points
        alpha = np.random.rand(parent1.shape[0], parent1.shape[1])

        # Create offsprings from chromosome by mixing the genes
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        return offspring1, offspring2

class TwoPointAndLineHybridCrossover(object):
    def __call__(self, parent1, parent2):

        random = np.random.random()
        if(random >= 0.5):
            return LineCrossover().__call__(parent1, parent2)
        else:
            return TwoPointCrossover().__call__(parent1, parent2)


class SinglePointAndLineHybridCrossover(object):
    def __call__(self, parent1, parent2):

        random = np.random.random()
        if(random >= 0.5):
            return LineCrossover().__call__(parent1, parent2)
        else:
            return SinglePointCrossover().__call__(parent1, parent2)