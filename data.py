from itertools import count
from typing import List, Tuple

AMINOACIDS = "RHKDESTNQCGPAVILMFYW"
NON_CHARGED = "STNQCGPAVILMFYW"
CHARGED = "RHKDE"
POSITIVE_CHARGE = "RHK"
NEGATIVE_CHARGE = "DE"
POLAR = "STCNQKRYDEH"
NONPOLAR = "GAVILPMFW"
BULKY = "IML"


class Gene:
    def __init__(self, value) -> None:
        self.__value = value

    @property
    def charged(self):
        return self.value in CHARGED

    @property
    def charge(self):
        if self.value in POSITIVE_CHARGE:
            return 1
        if self.value in NEGATIVE_CHARGE:
            return -1
        return 0

    @property
    def polared(self):
        return self.value in POLAR

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, x):
        self.__value = x

    def __copy__(self):
        copy = Gene(self.__value)

        return copy

    def __eq__(self, other: 'Gene'):
        return self.__value == other.__value


class Protein:
    @classmethod
    def create_protein(cls, sequence, origin_sequence, descriptor):
        protein = Protein(sequence, origin_sequence)

        protein.__descriptor = descriptor

        # Calc charge
        protein.__charge = 0
        for gene in protein.__genes:
            protein.__charge += gene.charge

        # Calc changes
        protein.__num_changes = 0
        for x1, x2 in zip(sequence, origin_sequence):
            if x1 != x2:
                protein.__num_changes += 1

        return protein

    def __init__(self, sequence, origin_sequence) -> None:
        self.__genes = [Gene(x) for x in sequence]
        self.__origin_genes = [Gene(x) for x in origin_sequence]
        self.__num_changes = None
        self.__descriptor = None
        self.__charge = None

    @property
    def charge(self):
        return self.__charge

    @property
    def sequence(self):
        return ''.join([str(x.value) for x in self.__genes])

    @property
    def origin_sequence(self):
        return ''.join([str(x.value) for x in self.__origin_genes])

    @property
    def genes(self):
        return self.__genes

    @property
    def num_changes(self):
        return self.__num_changes

    @property
    def descriptor(self):
        return self.__descriptor

    def set_descriptor(self, new_value):
        self.__descriptor = new_value

    def update_gene(self, idx, gene):
        """
        Обновляет значение (value) в гене.

        :param idx: позиция гена в sequence. 0 <= idx < len(sequqence)
        :param gene: новый ген
        :return:
        """
        if gene.value != self.__genes[idx].value:
            # Update num changes
            if self.__genes[idx].value != self.__origin_genes[idx].value:
                self.__num_changes -= 1

            # Update charge
            self.__charge -= self.__genes[idx].charge

            # Change gene
            self.__genes[idx] = gene

            # Update charge
            self.__charge += self.__genes[idx].charge

            # Update num changes
            if self.__genes[idx].value != self.__origin_genes[idx].value:
                self.__num_changes += 1

            # Update value

            self.__descriptor = None

    def get_differences(self) -> List[Tuple[int, Gene, Gene]]:
        differences = []
        for idx, g1, g2 in zip(count(1), self.origin_sequence, self.sequence):
            if g1 != g2:
                differences.append((idx, g1, g2))
        return differences

    def get_gene(self, idx):
        """
        Возвращает копию гена по индексу
        :param idx: индекс гена, 0 <= idx < len(genes)
        :return:
        """
        return self.__genes[idx].__copy__()

    def __copy__(self):
        copy = Protein(self.sequence, self.origin_sequence)
        copy.__descriptor = self.__descriptor
        copy.__charge = self.__charge
        copy.__num_changes = self.__num_changes

        return copy

    def __eq__(self, other: 'Protein'):
        return self.sequence == other.sequence
