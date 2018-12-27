import numpy as np
import random
from time import perf_counter as timer

class TravelingSalesman_GeneticAlgorithm():
    """
    Traveling Salesman Genetic Algorithm
    """
    def __init__(self, points_to_visit, index_home):
        """
        points_to_visit, ndarray of points to visit
        index_home,      index of starting point in points_to_visit
        population_size, the population size
        """
        self.points_to_visit  = points_to_visit
        self.index_home       = index_home
        self.population_size  = 40
        self.elite_size       = 12
        self.crossover_chance = 0.9
        self.mutation_chance  = 0.2

        self.population = []
        self.distance   = []
        self.fitness    = []
        self.roulette   = []

        self.parents    = []
        self.elite_pop  = []

        self.initialize_population()

    def run_algorithm(self, generations):
        for _ in range(generations):
            self.selection()
            self.crossover()
            self.mutate()

    def calculate_distance(self, gene):
        dist = 0
        for i in range(len(gene) - 1):
            dist += np.linalg.norm(gene[i]-gene[i+1])
        return dist

    def initialize_population(self):
        home = self.points_to_visit[self.index_home]
        gene_sequence = self.points_to_visit.tolist()
        del gene_sequence[self.index_home]

        for _ in range(self.population_size):
            gene = np.array(gene_sequence[0:])
            np.random.shuffle(gene)
            temp = gene.tolist()
            gene = np.array([home] + temp)
            # Record data
            self.population.append(gene)

        # print(self.population)
        # print("Pop size:", len(self.population))

    def selection(self):
        """
        Selection
        """
        # Calculate distance
        del self.distance[0:]
        total_dist = 0
        for gene in self.population:
            dist = 0
            for i in range(len(gene) - 1):
                dist += np.linalg.norm(gene[i]-gene[i+1])
            total_dist += dist
            self.distance.append(dist)

        # Calculate fitness
        del self.fitness[0:]
        for i in range(len(self.population)):
            self.fitness.append(1/(self.population_size-1)*(1 - self.distance[i]/total_dist))

        # Create roulette
        del self.roulette[0:]
        chance = 0
        for i in self.fitness:
            chance += i
            self.roulette.append(chance)

        # Locate best n elites
        del self.elite_pop[0:]
        temp_fitness = self.fitness[0:]
        temp_elite_list = []
        for _ in range(self.elite_size):
            # Find index of elite individual
            max_elite_fit = max(temp_fitness)
            elite_index = self.fitness.index(max_elite_fit)
            # Clone elite
            temp_elite_list.append(self.population[elite_index])
            temp_fitness[elite_index] = -999

        # Roulette selection of parents
        while True:
            roll_a, roll_b = random.random(), random.random()
            parent_a_index, parent_b_index = 0, 0
            for element in self.roulette:
                if roll_a > element:
                    parent_a_index += 1
                else:
                    break
            for element in self.roulette:
                if roll_b > element:
                    parent_b_index += 1
                else:
                    break
            if parent_a_index != parent_b_index:
                break

        # Clone 2 selected parents
        self.parents.append(self.population[parent_a_index])
        self.parents.append(self.population[parent_b_index])
        # Clone elites
        for individual in temp_elite_list:
            self.elite_pop.append(individual)

        # print("Average dist", total_dist/self.population_size)
        # print("Distance:", self.distance)
        # print("Fitness :", self.fitness)
        # print("Roulette:", self.roulette)
        # print("Elite 0 :", self.elite_pop[0])
        print("Elite dist:", self.calculate_distance(self.elite_pop[0]))

    def crossover(self):
        """
        Crossover
        """
        # Clear previous population
        del self.population[0:]
        # Crossover genes
        a = self.parents[0].tolist()
        b = self.parents[1].tolist()
        for _ in range(int((self.population_size-self.elite_size)/2)):
            if random.random() <= self.crossover_chance:
                seq_len = len(a)-1
                while True:
                    cross_0, cross_1 = random.randint(1, seq_len), random.randint(1, seq_len)
                    if cross_0 < cross_1:
                        break
                # Partially-Mapped Crossover (PMX)
                # child_a
                gene_ax = a[0:cross_0]
                gene_ay = a[cross_1:]
                gene_bx = [segment_b for segment_b in b if segment_b not in gene_ax+gene_ay]
                child_a = np.array(gene_ax + gene_bx + gene_ay)
                self.population.append(child_a)
                # child_b
                gene_bj = b[0:cross_0] 
                gene_bk = b[cross_1:]
                gene_aj = [segment_a for segment_a in a if segment_a not in gene_bj+gene_bk]
                child_b = np.array(gene_bj + gene_aj + gene_bk)
                self.population.append(child_b)
            else:
                temp_a, temp_b = a[1:], b[1:]
                random.shuffle(temp_a)
                random.shuffle(temp_b)
                self.population.append(np.array([a[0]] + temp_a))
                self.population.append(np.array([b[0]] + temp_b))

                # self.population.append(np.array(a))
                # self.population.append(np.array(b))

        # print(a, self.calculate_distance(np.array(a)))
        # print(b, self.calculate_distance(np.array(b)))

    def mutate(self):
        """
        Mutate progeny
        """
        if random.random() < self.mutation_chance:
            seq_len = len(self.population[0])-1
            for i in range(len(self.population)):
                temp_gene = self.population[i].tolist()
                while True:
                    mut_0, mut_1 = random.randint(1, seq_len), random.randint(1, seq_len)
                    # Mutation at least 2 pieces long (adding +1)
                    if mut_0+1 < mut_1:
                        break
                # # Segment shuffle
                # gene_ax = temp_gene[0:mut_0]
                # gene_bx = temp_gene[mut_1:]
                # segment = temp_gene[mut_0:mut_1]
                # random.shuffle(segment)
                # # Mutate gene
                # self.population[i] = np.array(gene_ax + segment + gene_bx)
                roll = random.random()
                if roll < 0.5:
                    # inversion operator
                    gene_ax = temp_gene[0:mut_0]
                    gene_bx = temp_gene[mut_1:]
                    segment = temp_gene[mut_0:mut_1]
                    segment = segment[::-1]
                    # Mutate gene
                    self.population[i] = np.array(gene_ax + segment + gene_bx)
                else:
                    # shift operator
                    shift = random.randint(1, len(temp_gene))
                    self.population[i] = np.array([temp_gene[0]] + temp_gene[shift:] + temp_gene[:shift])

        # Add elite individuals back to population
        self.population += self.elite_pop

def initalize_binary_ndarray():
    """
    Initializes with a binary ndarray
    Coordinates is denoted as (row, column)
    For conventional cartesian coordinates (x, y), rotate the array by 90 degress clockwise 

    Returns a nx2 ndarray of all points to visit:
    [[row, column]], [row, column], ...]
    """
    # Test matrix
    matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 1, 0, 0, 0, 0, 1]])

    # rows = 10
    # cols = 10
    # matrix = np.random.randint(2, size=(rows, cols))

    points_to_visit = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1:
                points_to_visit.append([i, j])
    total_points = sum(np.sum(matrix, axis=0))
    points_to_visit = np.array(points_to_visit)

    print("Number to visit: {}".format(total_points))

    return points_to_visit

def starting_point(points_to_visit):
    """
    points_to_visit, ndarray [[rows, cols], [rows, cols], ...]

    Returns the index of the point that is closest to the origin (0, 0)
    """
    origin = np.array([0, 0])
    temp_dist = [np.linalg.norm(points_to_visit[i]-origin) for i in range(len(points_to_visit))]
    min_dist = min(temp_dist)
    index_home = temp_dist.index(min_dist)

    # print("Starting at {}: {}".format(index_home, points_to_visit[index_home]))
    # print("Distance: {}".format(np.linalg.norm(points_to_visit[index_home]-origin)))

    return index_home

def main():
    points_to_visit = initalize_binary_ndarray()
    start_index = starting_point(points_to_visit)

    TSP_GA = TravelingSalesman_GeneticAlgorithm(points_to_visit, start_index)
    TSP_GA.run_algorithm(1000)

if __name__ == '__main__':
    start_time = timer()
    main()
    print("Finished in: {} secs".format(round(timer() - start_time, 7)))
