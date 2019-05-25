# Traveling Salesman Problem
The traveling salesman problem was first proposed in the 1800s by the mathematician W.R. Hamilton and Thomas Kirkman. This NP-hard problem asks, for a given set of nodes, what is the optimal path to visit all nodes exactly once, minimizing the distance and returning back to the home position. In this implementation, I employ the use of a heuristic search algorithm premised on the evolutionary ideas of natural selection and genetics for a one-way traveling salesman.
![TSP-GA Results](/images/Figure_1.png)

## Clone and test for yourself!
#### Dependencies: matplotlib.pyplot, numpy, random, time

    cd traveling_salesman/
    python traveling_salesman.py

## Genetic Algorithm
Genetic algorithms are meta-heuristics inspired by the process of natural selection and can be utilized to find near-optimum solutions of many combinatorial problems.

### Initialization
Candidate solutions are randomly generated for a given n-sized population.

### Selection
A roulette wheel selection and an elite strategy of best-individual to survive is used the select the "parents" for the next generation.

#### Roulette Wheel Selection
The roulette wheel selection algorithm is illustrated as follows.

![roulette algorithm](/images/roulette_formula.png)

Where `f_i` is the value for fitness in an individual.

#### Elite Selection
The best n elites are always carried over to the next generation.

### Crossover
The crossover operator is used to generate new offspring. The genes of the previous generation of parents are randomly grouped into pairs to produce two offspring.

#### Partially-Matched Crossover (PMX)
Schematic of PMX is as follows: `S1|S2|S3`, where S1 and S3 are from `Parent_A` and S2 from `Parent_B`.

Break points are randomly chosen. However, because order matters, notice `Parent_B` would contribute `1|2`, but segments from `Parent_A` already contain `1|2`. To solve this, `Parent_B` contributes the pieces not in segments of `Parent_A`, namely `5|4`.

              Gene                    Child_A
    Parent_A  [0|1|2|3|4|5|6|7|8] \   
                       ? ?         -> [0|1|2|3|5|4|6|7|8]
    Parent_B  [8|7|5|3|1|2|4|6|0] /            ^ ^
                   ^       ^

### Mutate
The mutation operator has a low probability to partially modify the genes, allowing the algorithm to approach a near-optimal solution.

#### Inversion Operator
Random break points are chosen and inverts selected segment.

    [0|1|2|3|4|5|6|7|8] -> [0|1|2|5|4|3|6|7|8]
           ^ ^ ^                  ^ ^ ^ 

#### Shift Operator
A random number is generated and the gene is shifted by that number.

    [0|1|2|3|4|5|6|7|8] -> [7|8|0|1|2|5|4|3|6]
     ^                          ^
