import java.util.Objects;

/**
 * Point
 */
public class Point {
    private int x;
    private int y;
    
    /**
     * Creates a new point
     * @param x x coordinate
     * @param y y coordinate
     */
    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Calculates the Euclidean distance between Point A and Point B
     * @param a Point A
     * @param b Point B
     * @return the distance
     */
    public static float distance(Point a, Point b) {
        int x = a.x - b.x;
        int y = a.y - b.y;
        return x * x + y * y;
    }

    /**
     * Prints the itself
     */
    public void print() {
        System.out.println(this);
    }

    @Override
    public String toString() {
        return "(" + x + ", " + y + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || o.getClass() != this.getClass()) {
            return false;
        } else {
            return x == ((Point) o).x && y == ((Point) o).y;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}
