package Neuranet;

/**
 * Class that represents a pair of objects
 * of any type.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Tuple<F, S> {
    /** The first object. */
    public final F f;
    /** The second object. */
    public final S s;
    
    /**
     * Constructs tuple object.
     * @param f the first object.
     * @param s the second object.
     */
    public Tuple(F f, S s) {
        this.f = f;
        this.s = s;
    }
}
