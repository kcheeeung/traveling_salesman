import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * TSP
 */
public class TSP {
    public static int numToVisit;
    public static Point[] pointsToVisit;

    private static final int RANDOMSEED = 100;
    private static final Random random = new Random(RANDOMSEED);
    private static final int POPULATION_SIZE = 10;
    private static final int ELITE_SIZE = 2;
    private static final double CROSSOVER_CHANCE = 0.9;
    private static final double MUTATION_CHANCE = 0.1;
    private static final double ROULETTE_RATIO = 1.0 / (POPULATION_SIZE - 1.0);

    private int[] refIndexPopulation;
    private int[] refIndexToVisit;
    private Individual[] population;
    private Individual best;
    
    private ArrayHeapMinPQ<Individual> minheap;
    private double[] roulette;
    private float[] history;
    private float startDist;

    /**
     * Utilizes a genetic algorithm in order to solve the traveling salesman problem
     * Implements a dummy point (0, 0) to solve a one-way variant of the TSP problem
     * @param arr int[][] consisting of binary points to visit
     * @param iterations number of generations to iterate over
     * @param historyOn whether to record the best's history
     */
    public TSP(int[][] arr, int iterations, boolean historyOn) {
        long start = clock();
        minheap = new ArrayHeapMinPQ<>();
        population = new Individual[POPULATION_SIZE];
        roulette = new double[POPULATION_SIZE];
        best = new Individual("dummy"); // creates a dummy "best node"

        refIndexPopulation = new int[POPULATION_SIZE];
        for (int i = 0; i < POPULATION_SIZE; i++) {
            refIndexPopulation[i] = i;
        }

        initialize(arr);
        startingPopulation();
        populationToMinHeap();

        history = new float[iterations];
        for (int i = 0; i < iterations; i++) {
            selection();
            crossover();
            mutate();
            
            if (historyOn) {
                history[i] = best.getDistance();
            }

            minheap.clear();
            populationToMinHeap();
        }
        elapsedTime(start);
    }

    /**
     * Initializes all starting parameters.
     * Uses and ArrayList and converts into Point[] for faster implementation
     * @param arr input array
     */
    private void initialize(int[][] arr) {
        ArrayList<Point> temp = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                if (arr[i][j] > 0) {
                    temp.add(new Point(i, j));
                }
            }
        }
        numToVisit = temp.size();
        // Converts Arraylist<Point> to Point[]
        pointsToVisit = new Point[numToVisit];
        for (int i = 0; i < numToVisit; i++) {
            pointsToVisit[i] = temp.get(i);
        }
        // Reference index
        refIndexToVisit = new int[numToVisit];
        for (int i = 0; i < numToVisit; i++) {
            refIndexToVisit[i] = i;
        }
    }

    /**
     * Creates the seed population for the simulation
     */
    private void startingPopulation() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            shuffleArray(refIndexToVisit);
            population[i] = new Individual(refIndexToVisit);
            minheap.add(population[i], population[i].getDistance());
        }
        startDist = minheap.getSmallest().getDistance();
        System.out.println("Start distance: " + startDist);
    }

    /**
     * Performs a selection for the next generation.
     * The all-time best is guaranteed to be passed on, along with the N top Individuals.
     * Progeny variants of the top N are mutated.
     * The final remaining spots are selected through roulette fitness selection.
     */
    private void selection() {
        Individual[] newPop = new Individual[POPULATION_SIZE];
        // Select best
        Individual currentBest = minheap.getSmallest();
        
        if (currentBest.getDistance() < best.getDistance()) {
            best = currentBest;
        }
        newPop[0] = new Individual(best);
        int size = 1;
        // Select elites
        for (int i = 0; i < ELITE_SIZE; i++) {
            Individual p = minheap.removeSmallest();
            // Original
            newPop[size] = new Individual(p);
            // Invert
            newPop[size + 1] = new Individual(p);
            newPop[size + 1].invertSequence(random);
            // Shift
            newPop[size + 2] = new Individual(p);
            newPop[size + 2].shiftSequence(random);
            size += 3;
        }
        // Creates the roulette and selects remaining 
        float rouletteTotalDist = 0;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            rouletteTotalDist += population[i].getDistance();            
        }
        double chance = 0;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            chance += ROULETTE_RATIO * (1.0 - (population[i].getDistance() / rouletteTotalDist));
            roulette[i] = chance;
        }
        for (int i = size; i < POPULATION_SIZE; i++) {
            double roll = random.nextDouble();
            for (int j = 0; j < roulette.length; j++) {
                if (roll < roulette[j]) {
                    newPop[i] = new Individual(population[j].getSequence());
                    break;
                }
            }
        }
        population = newPop;
    }

    /**
     * Performs partially-matched crossover on two pairs
     */
    private void crossover() {
        shuffleArray(refIndexPopulation);
        Individual targetA, targetB;
        for (int i = 0; i < POPULATION_SIZE; i += 2) {
            if (random.nextDouble() < CROSSOVER_CHANCE) {
                targetA = population[refIndexPopulation[i]];
                targetB = population[refIndexPopulation[i + 1]];
                targetA.cross(random, targetA, targetB);
            }
        }
    }

    /**
     * Performs are mutation on a given Individual.
     * Random mutation helps allow the population to escape the local minimum.
     */
    private void mutate() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            if (random.nextDouble() < MUTATION_CHANCE) {
                if (random.nextDouble() < 0.5) {
                    population[i].invertSequence(random);
                } else {
                    population[i].shiftSequence(random);
                }
            }
        }
    }

    /** 
     * Shuffles the entire array 
     */
    private void shuffleArray(int[] array) {
        int index;
        for (int i = array.length - 1; i > 0; i--) {
            index = random.nextInt(i + 1);
            if (index != i) {
                array[index] ^= array[i];
                array[i] ^= array[index];
                array[index] ^= array[i];
            }
        }
    }

    /** 
     * Adds entire population to minheap
     */
    private void populationToMinHeap() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            minheap.add(population[i], population[i].getDistance());
        }
    }

    /** 
     * Returns the current clock time 
     */
    private long clock() {
        return System.currentTimeMillis();
    }

    /** 
     * Prints the time elapsed since the startTime
     */
    private void elapsedTime(long startTime) {
        System.out.println((clock() - startTime) / 1000.0f + " secs");
    }

    /** 
     * Prints the best stats
     */
    public void printBest() {
        System.out.print("Best distance: ");
        System.out.println(best.getDistance());
        System.out.printf("Improved by: %.2f%%", startDist / best.getDistance() * 100 - 100);
        // for (Point p : getVisitSequence()) {
        //     System.out.print(p + " ");
        // }
    }

    public Individual getBest() {
        return best;
    }

    public List<Point> getVisitSequence() {
        int[] seq = best.getSequence();
        List<Point> result = new ArrayList<>(seq.length);
        for (int i = 0; i < numToVisit; i++) {
            result.add(pointsToVisit[seq[i]]);
        }
        return result;
    }

    public void writeHistory() {
        try {
            StringBuilder result = new StringBuilder();
            BufferedWriter out = new BufferedWriter(new FileWriter("savefile.txt"));
            for (int i = 0; i < history.length; i++) {
                result.append(history[i]);
                result.append("\n");
            }
            out.write(result.toString());
            out.close();
        } catch (Exception e) {
            System.out.println("Write failed");
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        int size = 50;
        int[][] array = new int[size][size];
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                if (random.nextDouble() < 0.5) {
                    array[i][j] = 1;                    
                }
            }
        }

        /* Starting Point */
        TSP tsp = new TSP(array, 1000, false);
        tsp.printBest();
        // tsp.writeHistory();
    }
}
