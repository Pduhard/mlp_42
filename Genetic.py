# import Protodeep as ptd
from GAModel import GAModel
from Preprocessing.Split import Split
from random import random

"""
[{
    unit_range = [40, 60]
    fa = ['relu'....]
    init = ['random'...]
    w_reg = [''...]]
}, {}, {}]
"""


class Genetic():

    def __init__(self, constraints, dataset, population_size=50,
                 mutation_rate=0.1, generation=20):
        self.constraints = constraints
        self.dataset = dataset
        self.input_shape = dataset.features.shape[1:]
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = generation
        ((self.x_train, self.y_train), (self.x_test, self.y_test)) = Split.train_test_split(
            dataset.features, dataset.targets)
        self.best_entity = None

    # def new_rand_entity(self, constraints)
    def init(self):
        self.population = [
            GAModel(self.constraints, self.input_shape, self.dataset) for i in range(self.population_size)
        ]

    def evaluate(self):
        return

    def fit(self):
        bestscore = self.best_entity.fitness if self.best_entity else -1

        for entity in self.population:
            fitness = entity.fit(self.x_train, self.y_train, self.x_test, self.y_test)
            if fitness > bestscore:
                bestscore = fitness
                self.best_entity = entity
        # print(self.best_entity.evaluate(self.x_train, self.y_train, self.x_test, self.y_test))

    def find_model(self):
        self.init()
        for g in range(self.generation):
            self.fit()
            self.cross()
            # self.mutate()

        # self.best_entity.model.summary()
        print(f"Best model :\n{self.best_entity.model_attr}")
        print(f"Loss: {self.best_entity.evaluate(self.x_train, self.y_train, self.x_test, self.y_test)}")
        return self.best_entity.models[0]

    def create_pool(self):
        fit_sum = sum([entity.fitness for entity in self.population])
        current = 0
        pool = []
        for entity in self.population:
            current += entity.fitness
            pool.append({
                'entity': entity,
                'fs': current,
                'fitness': entity.fitness
            })
        return pool, fit_sum

    def select_one_parent(self, pool, fit_sum):
        
        rnd = random() * fit_sum
        for p in pool:
            if p['fs'] > rnd:
                return p['entity']

        raise Exception("Unable to find a parent")

    def select_parents(self, pool, fit_sum):
        a, b = None, None
        while a is b:
            rnda = random() * fit_sum
            rndb = random() * fit_sum
            a, b = None, None
            for p in pool:
                if a is None and p['fs'] > rnda:
                    a = p['entity']
                if b is None and p['fs'] > rndb:
                    b = p['entity']
        return a, b

    def cross(self):
        pool, fit_sum = self.create_pool()
        new_population = []
        print(pool)
        for i in range(self.population_size):
            # a, b = self.select_parents(pool, fit_sum)
            # new_population.append(a.cross(b, self.mutation_rate))
            parent = self.select_one_parent(pool, fit_sum)
            new_population.append(parent.cross(parent, self.mutation_rate))
        self.population = new_population

    # def mutate(self):
    #     pass


if __name__ == '__main__':
    print('hello')
