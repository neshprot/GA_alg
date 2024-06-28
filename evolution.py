import os
import random
import time
from abc import abstractmethod, ABC
from copy import copy
import numpy as np

from data import Protein, Gene

Pull = "ARNDVHGQEILKMPSYTWFV"  # список 20 существующих аминокислот

class Evolution(ABC):
    def __init__(self):
        self._population = None
        self._mut_prob = None
        self._cros_prob = None

    @property
    def population(self):
        return self._population

    @property
    def mut_prob(self):
        return self._mut_prob

    @property
    def cros_prob(self):
        return self._cros_prob

    @abstractmethod
    def mutation(self, *args, **kwargs):
        pass

    @abstractmethod
    def crossover(self, *args, **kwargs):
        pass

    @abstractmethod
    def selection(self, *args, **kwargs):
        pass


class BaseFunction(ABC):
    def __init__(self):
        self._input_file: str = None
        self._output_file: str = None
        self._save_file: str = None
        self._computed = None

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_computing(self, *args, **kwargs):
        pass


# класс с основными функциями эфолюции
class ProteinEvolution(Evolution, BaseFunction):
    def __init__(self, population, mut_prob, mut_num, cros_prob, input_file,
                 output_file, save_file, logger, checker=None,
                 weights=1.0, positionsset={}):
        super().__init__()
        self._population = population
        self._mut_prob = mut_prob
        self._mut_num = mut_num
        self._cros_prob = cros_prob
        self._input_file = input_file
        self._output_file = output_file
        self._save_file = save_file
        self._computed = dict()
        self._checker = checker
        self._logger = logger
        self._weights = weights
        self._positionsset = positionsset

    # mutation function
    def mutation(self, attempts=1, iteration=1, sets={}, pulls={}, probs={}, consts={}):
        """

        :param attempts: число попыток инциниализации на один protein
        :return:
        """

        new_population = []     # список белков в новой популяции
        num_of_changed = 0      # кол-во измененных белков

        # перебор белков в популяции
        for protein in self._population:
            new_protein = copy(protein)
            # условие возникновения мутации(с вероятностью mut_prob)
            if random.random() < self._mut_prob:
                attempt = 0
                num_of_changed += 1
                num_mut = 0
                mutations = []
                while attempt < attempts and num_mut <= self._mut_num and num_mut < iteration:
                    position = random.choice(tuple(self._positionsset))
                    if position in mutations:
                        continue
                    old_gene = new_protein.genes[position - 1]

                    # no changes for CHARGED and TRP
                    set_name = 'Set1'
                    for name, sites in sets.items():
                        if old_gene.value in sites:
                            set_name = name
                            continue
                    pull = random.choices(probs[set_name][0], weights=probs[set_name][1][:-1])[0]
                    if random.random() > probs[set_name][1][-1]:
                        new_value = old_gene.value
                    else:
                        new_value = random.choice(pulls[pull])
                    if int(position) in consts:
                        if new_value in pulls[consts[int(position)][0]]:
                            continue

                    new_gene = Gene(value=new_value)
                    new_protein.update_gene(position - 1, new_gene)

                    # проверка стабильности белка
                    if self.is_stable_protein(new_protein) and new_protein.num_changes <= iteration:
                        mutations.append(position)
                        num_mut += 1
                    else:
                        # Restore old gene
                        new_protein.update_gene(position - 1, old_gene)
                    attempt += 1
            new_population.append(new_protein)

        self._logger(f"Mutation: I will try to change {num_of_changed} proteins... {num_of_changed} proteins changed\n")

        self._population = new_population

    # crossover function
    def crossover(self, attempts=1):
        new_population = []
        for_cross = []  # белки для кроссовера

        for protein in self._population:
            new_protein = copy(protein)
            # условие на кроссинговер
            if random.random() < self._cros_prob:
                for_cross.append(new_protein)
            else:
                new_population.append(new_protein)

        # проверка на четное число белков в списке на кроссовер
        if len(for_cross) % 2 == 1:
            new_population.append(for_cross.pop())

        random.shuffle(for_cross)

        need = 0
        real = 0

        pair_cros_prob = 0.5  # crossover pair probability
        # цикл кроссовера(перемешивания генов белков)
        for protein1, protein2 in zip(for_cross[0:-1:2], for_cross[1::2]):
            need += 2
            new_protein1, new_protein2 = protein1, protein2
            for attempt in range(attempts):
                attempt_protein1, attempt_protein2 = copy(new_protein1), copy(new_protein2)
                mut_num = 0
                for i, (gene1, gene2) in enumerate(zip(attempt_protein1.genes, attempt_protein2.genes)):
                    if mut_num > self._mut_num:
                        continue
                    if random.random() < pair_cros_prob:
                        new_gene1 = Gene(value=gene2.value)
                        new_gene2 = Gene(value=gene1.value)
                        attempt_protein1.update_gene(i, new_gene1)
                        attempt_protein2.update_gene(i, new_gene2)
                        mut_num += 1

                if self.is_stable_protein(attempt_protein1) and self.is_stable_protein(attempt_protein2):
                    new_protein1 = attempt_protein1
                    new_protein2 = attempt_protein2
                    real += 2
                    break

            new_population.append(new_protein1)
            new_population.append(new_protein2)

        self._logger(f"Crossover: I will try to change {need} proteins... {real} proteins changed\n")

        self._population = new_population

    # selection function
    def selection(self, eval_param, save_n_best):
        def distribution(p, m):
            def evaluate(p: float, n: int) -> float:
                return p * pow(1 - p, n - 1)

            vs = []
            for j in range(1, m + 1):
                v = 0
                for j in range(1, j + 1):
                    v += evaluate(p, j)
                vs.append(v)
            return vs

        new_population = []
        pop_size = len(self._population)

        population = sorted(self._population, key=lambda x: self.fitness(x.descriptor),
                            reverse=True)  # increasing value

        for i in range(save_n_best):
            protein = copy(population[i])
            new_population.append(protein)

        q = distribution(eval_param, pop_size)
        for _ in range(pop_size - save_n_best):
            n, r = 0, random.uniform(0, q[-1])
            while r > q[n]:
                n += 1
            protein = copy(population[n])
            new_population.append(protein)

        new_population = sorted(new_population, key=lambda x: self.fitness(x.descriptor))[
                         0:pop_size]
        random.shuffle(new_population)

        self._population = new_population

    # функция подсчета value
    def compute(self):
        proteins_for_computing = []

        # Find existing calcs
        for protein in self._population:
            if protein.sequence not in self._computed:
                proteins_for_computing.append(protein)
        # Print to output file
        with open(".tempfile", "w") as ouf:
            for protein in proteins_for_computing:
                for idx, g1, g2 in protein.get_differences():
                    ouf.write(f"{g1}/{idx}/{g2} ")
                ouf.write("\n")
        os.rename(".tempfile", self._output_file)

        with open(self._output_file, "r") as inf, open(self._input_file, 'w') as ouf:
            for line in inf:
                ouf.write(f'{random.random()} {random.random()} {random.random()}\n')
        # Wait results
        while not os.path.exists(self._input_file):
            time.sleep(1)

        # Read results and save
        with open(self._input_file) as inf:
            descriptors = []
            for line in inf.readlines():
                values = line.split()
                descriptors.append([float(value) for value in values])
            for i, protein in enumerate(proteins_for_computing):
                self.save_computing(protein.sequence, descriptors[i])

        # Write values to proteins
        for protein in self._population:
            values = self._computed[protein.sequence]
            protein.set_descriptor(values)

        # Remove out/inp filess
        os.remove(self._output_file)
        os.remove(self._input_file)

    def save_computing(self, sequence, descriptors):
        if sequence not in self._computed:
            self._computed[sequence] = descriptors
            with open(self._save_file, 'a') as f:
                f.write(f"{sequence} {' '.join(map(str, descriptors))}\n")

    # функция проверки стабильности белка(выполнения constraints)
    def is_stable_protein(self, protein):
        if self._checker is not None:
            return self._checker.check(protein)
        return True

    # функция выбора лучшего белка в популяции
    def get_best_protein(self):
        best_protein = max(self.population, key=lambda x: self.fitness(x.descriptor))
        return best_protein

    # функция создания первой популяции
    def generate_population(self, default_sequence, default_descriptors, pop_size,
                            from_computed=True):
        population = []
        max_num_changes = 0

        self.save_computing(default_sequence, default_descriptors)
        if from_computed:
            for sequence, value in self._computed.items():
                protein = Protein.create_protein(sequence, default_sequence, descriptor=value)
                population.append(protein)
                if max_num_changes < protein.num_changes:
                    max_num_changes = protein.num_changes

            population = sorted(population, key=lambda x: self.fitness(x.descriptor),
                                reverse=True)[:pop_size]

        while len(population) < pop_size:
            protein = Protein.create_protein(default_sequence, default_sequence, descriptor=default_descriptors)
            population.append(protein)

        self._population = population
        return max_num_changes

    def load_computed_proteins(self):
        if os.path.exists(self._save_file):
            with open(self._save_file, "r") as inf:
                for line in inf.readlines():
                    values = line.split()
                    self._computed[values[0]] = [float(x) for x in values[1::]]

    def print_current_population(self):
        for protein in self._population:
            self._logger(f"{protein.sequence}, descriptors {' '.join(map(str, protein.descriptor))}"
                         f" num of changes {protein.num_changes}\n")
            for idx, g1, g2 in protein.get_differences():
                self._logger(f"{g1}/{idx}/{g2} ")
            self._logger("\n")

    def fitness(self, values):
        fit = np.dot(np.array(values), np.array(self._weights))
        return fit

    def initial_step(self):
        ini_step = 1
        for protein in self._population:
            muts = len(protein.get_differences())
            if muts > ini_step:
                ini_step = muts
        return ini_step
