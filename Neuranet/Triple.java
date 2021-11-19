package Neuranet;

/**
 * Class that represents a triplet of objects
 * of any type.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Triple<X, Y, Z> {
    /** The first object. */
    public final X x;
    /** The second object. */
    public final Y y;
    /** The third object. */
    public final Z z;
    
    /**
     * Constructs tuple object.
     * @param f the first object.
     * @param s the second object.
     */
    public Triple(X x, Y y, Z z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}
