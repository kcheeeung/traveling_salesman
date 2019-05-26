# Traveling Salesman Problem
The traveling salesman problem was first proposed in the 1800s by the mathematician W.R. Hamilton and Thomas Kirkman. This NP-hard problem asks, for a given set of nodes, what is the optimal path to visit all nodes exactly once, minimizing the distance and returning back to the home position. In this implementation, I employ the use of a heuristic search algorithm premised on the evolutionary ideas of natural selection and genetics for a one-way traveling salesman.
![TSP-GA Results](/images/Figure_1.png)

## Clone and test for yourself!

    cd traveling_salesman/
    javac -d out/ TSP.java
    java -cp out/ TSP

## Genetic Algorithm
Genetic algorithms are meta-heuristics inspired by the process of natural selection and can be utilized to find near-optimum solutions of many combinatorial problems.

### Initialization
Candidate solutions are randomly generated for a given n-sized population.

### Selection
A roulette wheel selection and an elite strategy of best-individual to survive is used the select the "parents" for the next generation.

#### Roulette Wheel Selection
The roulette wheel selection algorithm is illustrated as follows.

![roulette algorithm](/images/roulette_formula.png)

f<sub>i</sub> being the fitness value for an individual.

#### Elite Selection
The best n elites are always carried over to the next generation.

### Crossover
The crossover operator is used to generate new offspring. The genes of the previous generation of parents are randomly grouped into pairs to produce two offspring.

#### Partially-Matched Crossover (PMX)
Schematic of PMX is as follows: `S1 + S2 + S3`, where `Parent_A` contributes S1 and S3 and `Parent_B` contributes S2.

Break points are randomly chosen. In this example, let the break point be at the index junctions 3-4 and 5-6.
For `Parent_A`, this leaves S1 as `0|2|1|3` and S3 as `6|7|8`.
For `Parent_B`, the indices show that S2 would be `1|2`, but this would be a duplicate! To solve this, the sequence is iterated over itself, whilst maintaining order, to find the missing pieces. This results in S2 being `5|4`.

              Gene                    Child_A
    Parent_A  [0|2|1|3|4|5|6|7|8]  -> [0|2|1|3] + [5|4] + [6|7|8]
                      ^   ^           Child_B
    Parent_B  [8|7|5|3|1|2|4|6|0]  -> [8|7|5|3] + [2|1] + [4|6|0]
                      ^   ^

### Mutate
The mutation operator has a low probability to partially modify the genes, allowing the algorithm to approach a near-optimal solution.

#### Inversion Operator
Random break points are chosen and the selected portion is inverted

    [0|1|2] + [3|4|5] + [6|7|8] -> [0|1|2] + [5|4|3] + [6|7|8]
               ---->                          <---- 

#### Shift Operator
A random number is generated and the gene is shifted by that number.

    [0|1|2|3|4|5|6|7|8] -> [8|0|1|2|3|4|5|6|7]
     ^                        ^
