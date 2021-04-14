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
        self.best_entity = None

    # def new_rand_entity(self, constraints)
    def init(self):
        self.population = [
            GAModel(self.constraints, self.input_shape) for i in range(self.population_size)
        ]

    def evaluate(self):
        return 

    def fit(self):

        self.bestscore = self.best_entity.fitness if self.best_entity else -1
        for entity in self.population:
            fitness = entity.fit()
            if fitness > self.bestscore:
                wefjk nwef ergn p
                egjr
                 wefjkgr pjwe
                  gjew
                   pgjpw
                   egjr p
                   iwgr p
                   iejwg p
                   jerg 
                   pjegr

    def find_model(self):
        self.init()
        for g in range(self.generation):
            self.fit()
            self.cross()
            self.mutate()
        return self.best_entity.model
        # for p in population:
        #     p.fitness = score(p)

    def cross(self):
        pass

    def mutate(self):
        pass


if __name__ == '__main__':
    print('hello')
