import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

/**
 * Individual
 */
public class Individual {
    private static final Point dummy = new Point(0, 0);
    private static Point[] pointsToVisit;
    private static HashSet<Integer> set = new HashSet<>();

    private int[] sequence;
    private float distance;

    /**
     * Creates a new individual based on the input sequence
     * @param seq input sequence
     */
    public Individual(int[] seq) {
        sequence = new int[seq.length];
        for (int i = 0; i < sequence.length; i++) {
            sequence[i] = seq[i];
        }
        calcDistance();
    }

    /**
     * Makes a new copy of the individual
     * @param other iput Individual
     */
    public Individual(Individual other) {
        sequence = new int[other.sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            sequence[i] = other.sequence[i];
        }
        distance = other.distance;
    }

    /**
     * Makes a dummy copy just for initialization
     * @param dummy any string
     */
    public Individual(String dummy) {
        sequence = null;
        distance = Float.MAX_VALUE;
    }

    /**
     * Calculate distance of the sequence
     */
    private void calcDistance() {
        pointsToVisit = TSP.pointsToVisit;
        distance = Point.distance(dummy, pointsToVisit[sequence[0]]);
        for (int i = 1; i < sequence.length - 1; i++) {
            distance += Point.distance(pointsToVisit[sequence[i]], pointsToVisit[sequence[i + 1]]);
        }
    }

    /**
     * Calculate distance for a given target's sequence
     * @param target target's sequence to be calculated
     * @return returns the distance
     */
    private float calcDistance(Individual target) {
        pointsToVisit = TSP.pointsToVisit;
        float dist = Point.distance(dummy, pointsToVisit[target.sequence[0]]);
        for (int i = 1; i < target.sequence.length - 1; i++) {
            dist += Point.distance(pointsToVisit[target.sequence[i]], pointsToVisit[target.sequence[i + 1]]);
        }
        return dist;
    }
    
    /**
     * Returns a random number between A and B inclusive 
     * @param random random instance
     * @param a index a
     * @param b index b
     * @return returns a random number between a and b inclusive
     */
    private int uniform(Random random, int a, int b) {
        return a + random.nextInt((b - a) + 1);
    }

    /**
     * Randomly inverts a portion of the array from index A to B inclusive
     * @param random random instance
     */
    public void invertSequence(Random random) {
        int midpoint = sequence.length / 2;
        int endPoint = sequence.length - 1;
        int start = uniform(random, 1, midpoint);
        int end = uniform(random, midpoint, endPoint);
        // Invert
        int temp;
        int offset = 0;
        for (int i = start; i <= start + (end - start) / 2; i++) {
            temp = sequence[i];
            sequence[i] = sequence[end - offset];
            sequence[end - offset] = temp;
            offset++;
        }
        calcDistance();
    }

    /**
     * Shifts array by k indices
     * @param random random instance
     */
    public void shiftSequence(Random random) {
        int length = sequence.length;
        int k = random.nextInt(length);
        // Shift
        int[] temp = new int[length];
        for (int i = 0; i < length; i++) {
            temp[(i + k) % length] = sequence[i];
        }
        sequence = temp;
        calcDistance();
    }

    /**
     * Performs partially-matched crossover on A and B
     * @param random random instance
     * @param a target a
     * @param b target b
     */
    public void cross(Random random, Individual a, Individual b) {
        // Reset
        set.clear();
        int midpoint = a.sequence.length / 2;
        int endPoint = a.sequence.length - 1;
        int start = uniform(random, 1, midpoint);
        int end = uniform(random, midpoint + 1, endPoint);
        // Copy for targetA
        int[] copyA = new int[a.sequence.length];
        for (int i = 0; i < a.sequence.length; i++) {
            copyA[i] = a.sequence[i];
        }
        // crossA
        for (int i = 0; i < start; i++) {
            set.add(a.sequence[i]);
        }
        for (int i = end; i < a.sequence.length; i++) {
            set.add(a.sequence[i]);
        }
        int j = 0;
        for (int i = start; i < end; i++) {
            while (j < b.sequence.length) {
                int item = b.sequence[j];
                if (!set.contains(item)) {
                    a.sequence[i] = item;
                    set.add(item);
                    break;
                }
                j++;
            }
        }
        // Reset
        set.clear();
        // crossB
        for (int i = 0; i < start; i++) {
            set.add(b.sequence[i]);
        }
        for (int i = end; i < b.sequence.length; i++) {
            set.add(b.sequence[i]);
        }
        j = 0;
        for (int i = start; i < end; i++) {
            while (j < copyA.length) {
                int item = copyA[j];
                if (!set.contains(item)) {
                    b.sequence[i] = item;
                    set.add(item);
                    break;
                }
                j++;
            }
        }
        // Recalculate distance
        float dist = calcDistance(a);
        a.distance = dist;
        dist = calcDistance(b);
        b.distance = dist;
    }

    /**
     * @return the sequence
     */
    public int[] getSequence() {
        return sequence;
    }

    /**
     * @return the distance
     */
    public float getDistance() {
        return distance;
    }

    /**
     * Prints the sequence
     */
    public void printSequence() {
        StringBuilder temp = new StringBuilder(sequence.length * 2);
        for (int i : sequence) {
            temp.append(i);
            temp.append(" ");
        }
        System.out.println(temp.toString());
    }

    /**
     * Prints the distance
     */
    public void printDistance() {
        System.out.println(distance);
    }

    @Override
    public String toString() {
        StringBuilder temp = new StringBuilder(sequence.length * 2);
        for (int i : sequence) {
            temp.append(i);
            temp.append(" ");
        }
        return temp.toString();
    }
    
    @Override
    public boolean equals(Object o) {
        if (o == null || o.getClass() != this.getClass()) {
            return false;
        } else {
            return Arrays.equals(((Individual) o).sequence, sequence);
        }
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(sequence);
    }
}
