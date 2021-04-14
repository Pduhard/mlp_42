# import Protodeep as ptd
from GAModel import GAModel
"""
[{
    unit_range = [40, 60]
    fa = ['relu'....]
    init = ['random'...]
    w_reg = [''...]
    # b_reg = [''...]
    # out_reg = [''...]
    # use_bias = boolean
}, {}, {}]
"""


class Genetic():

    def __init__(self, constraints, dataset, population_size=20,
                 mutation_rate=0.05, generation=10):
        self.constraints = constraints
        self.dataset = dataset
        self.input_shape = dataset.features.shape[1:]
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = generation
        self.init_pop()

    # def new_rand_entity(self, constraints)
    def init_pop(self):
        self.population = [
            GAModel(self.constraints, self.input_shape) for i in range(self.population_size)
        ]
        print('hallo')
        quit()

    def evaluate(self, entity):
        pass

    def fit_pop(self):
        pass
        # for p in population:
        #     p.fitness = score(p)

    def cross_pop(self):
        pass

    def mutate_pop(self):
        pass


if __name__ == '__main__':
    print('hello')
