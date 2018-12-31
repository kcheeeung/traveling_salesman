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
        self.population_size  = 24
        self.elite_size       = 3
        self.crossover_chance = 0.9
        self.mutation_chance  = 0.1
        self.sequence_length  = None
        
        self.population = []
        self.best_gene  = None
        self.best_dist  = None

        self.initialize_population()
        self.set_best_gene()
        print("Best relative value:", self.best_dist)

    def run_algorithm(self, generations):
        for _ in range(generations):
            self.selection()
            self.crossover()
            self.mutate()
            self.compare_best_gene()
        print("Best relative value:", self.best_dist)

    # def calculate_distance(self, gene):
    #     dist = 0
    #     for i in range(self.sequence_length-1):
    #         # dist += np.linalg.norm(gene[i]-gene[i+1])
    #         x = gene[i][0]-gene[i+1][0]
    #         y = gene[i][1]-gene[i+1][1]
    #         dist += (x*x + y*y)**0.5
    #     return dist

    def initialize_population(self):
        home = self.points_to_visit[self.index_home]
        gene_sequence = self.points_to_visit.tolist()
        self.sequence_length = len(gene_sequence)
        del gene_sequence[self.index_home]

        for _ in range(self.population_size):
            gene = np.array(gene_sequence[0:])
            np.random.shuffle(gene)
            temp = gene.tolist()
            gene = np.array([home] + temp)
            # Record data
            self.population.append(gene)

    def set_best_gene(self):
        distance = []
        for gene in self.population:
            dist = 0
            for i in range(self.sequence_length-1):
                x = gene[i][0]-gene[i+1][0]
                y = gene[i][1]-gene[i+1][1]
                dist += x*x + y*y
            distance.append(dist)
        best_dist  = min(distance)
        best_index = distance.index(best_dist)
        # Record best gene and distance
        self.best_dist = best_dist
        self.best_gene = self.population[best_index]

    def compare_best_gene(self):
        distance = []
        for gene in self.population:
            dist = 0
            for i in range(self.sequence_length-1):
                x = gene[i][0]-gene[i+1][0]
                y = gene[i][1]-gene[i+1][1]
                dist += x*x + y*y
            distance.append(dist)
        current_best_dist  = min(distance)
        current_best_index = distance.index(current_best_dist)
        # Compare current gen with all time best
        if current_best_dist < self.best_dist:
            self.best_dist = current_best_dist
            self.best_gene = self.population[current_best_index]

    def selection(self):
        """
        Selection
        """
        # Calculate relative distance (does not square root)
        distance = []
        total_dist = 0
        for gene in self.population:
            dist = 0
            for i in range(self.sequence_length-1):
                x = gene[i][0]-gene[i+1][0]
                y = gene[i][1]-gene[i+1][1]
                dist += x*x + y*y
            total_dist += dist
            distance.append(dist)

        # Calculate fitness and create roulette
        fitness  = []
        roulette = []
        chance = 0
        for i in range(len(self.population)):
            # Fitness
            fitness.append(1/(self.population_size-1)*(1 - distance[i]/total_dist))
            # Roulette
            chance += fitness[i]
            roulette.append(chance)

        # Add all time best
        temp_parents = []
        temp_parents.append(self.best_gene)

        # Locate current best n elites
        for _ in range(self.elite_size):
            # Find index of elite individual
            max_elite_fit = max(fitness)
            elite_index   = fitness.index(max_elite_fit)
            # Clone elite (creates 1 original + 2 mutated per n elites)
            elite_individual = self.population[elite_index]
            temp_parents.append(elite_individual)
            # Mutation
            temp_elite = elite_individual.tolist()
            seq_len = len(temp_elite)-1
            while True:
                mut_0, mut_1 = random.randint(1, seq_len), random.randint(1, seq_len)
                # At least 2 pieces long (adding +1)
                if mut_0+1 < mut_1:
                    break
            # Inversion operator
            gene_ax = temp_elite[0:mut_0]
            gene_bx = temp_elite[mut_1:]
            segment = temp_elite[mut_0:mut_1]
            segment = segment[::-1]
            temp_parents.append(np.array(gene_ax + segment + gene_bx))
            # Shift operator
            shift_gene = temp_elite[1:]
            shift = random.randint(1, len(shift_gene)-1)
            temp_parents.append(np.array([temp_elite[0]]+shift_gene[shift:]+shift_gene[:shift]))
            # "Remove elite" to find next n elite
            fitness[elite_index] = -999

        # Roulette selection for remaining spots
        queue_indices = []
        for i in range(self.population_size - len(temp_parents)):
            while True:
                roll = random.random()
                parent_index = 0
                for element in roulette:
                    if roll > element:
                        parent_index += 1
                    else:
                        break
                if parent_index not in queue_indices:
                    queue_indices.append(parent_index)
                    break
            temp_parents.append(self.population[queue_indices[i]])

        # Clear previous population
        del self.population[0:]
        self.population += temp_parents
        # print(len(self.population))
        # print("Average dist", total_dist/self.population_size)

    def crossover(self):
        """
        Crossover
        """
        # Crossover genes
        queue_indices = [i for i in range(len(self.population)) if random.random() < self.crossover_chance]
        random.shuffle(queue_indices)
        for i in range(0, len(queue_indices)-1, 2):
            a = self.population[queue_indices[i]].tolist()
            b = self.population[queue_indices[i+1]].tolist()
            seq_len = self.sequence_length-1
            while True:
                cross_0, cross_1 = random.randint(1, seq_len), random.randint(1, seq_len)
                if cross_0+1 < cross_1:
                    break
            # Partially-Mapped Crossover (PMX)
            # child_a
            gene_ax = a[0:cross_0]
            gene_ay = a[cross_1:]
            gene_bx = [segment_b for segment_b in b if segment_b not in gene_ax+gene_ay]
            self.population[i]   = np.array(gene_ax + gene_bx + gene_ay)
            # # child_b
            gene_bj = b[0:cross_0] 
            gene_bk = b[cross_1:]
            gene_aj = [segment_a for segment_a in a if segment_a not in gene_bj+gene_bk]
            self.population[i+1] = np.array(gene_bj + gene_aj + gene_bk)
            
            # print(a)
            # print(self.population[i].tolist())
            # print(b)
            # print(self.population[i+1].tolist())
            # print("\n")

    def mutate(self):
        """
        Mutate progeny
        """
        for i in range(len(self.population)):
            if random.random() < self.mutation_chance:
                temp_gene = self.population[i].tolist()
                while True:
                    mut_0, mut_1 = random.randint(1, self.sequence_length), random.randint(1, self.sequence_length)
                    # Mutation at least 2 pieces long (adding +1)
                    if mut_0+1 < mut_1:
                        break
                # Mutate
                if random.random() < 0.5:
                    # Inversion operator
                    gene_ax = temp_gene[0:mut_0]
                    gene_bx = temp_gene[mut_1:]
                    segment = temp_gene[mut_0:mut_1]
                    segment = segment[::-1]
                    self.population[i] = np.array(gene_ax + segment + gene_bx)
                else:
                    # Shift operator
                    shift_gene = temp_gene[1:]
                    shift = random.randint(1, len(shift_gene)-1)
                    self.population[i] = np.array([temp_gene[0]] + shift_gene[shift:] + shift_gene[:shift])

def initalize_binary_ndarray():
    """
    Initializes with a binary ndarray
    Coordinates is denoted as (row, column)
    For conventional cartesian coordinates (x, y), rotate the array by 90 degress clockwise 

    Returns a nx2 ndarray of all points to visit:
    [[row, column]], [row, column], ...]
    """
    # Test matrix
    matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 1, 0, 0, 1, 0, 0]])

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
    print("Finished in: {} secs".format(round(timer() - start_time, 5)))
