package Neuranet.RuntimeExceptions;

/**
 * Class representing matrix modification error.
 * Called when inputted value does not fit the
 * matrix's dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class InvalidMatrixArrayValue extends RuntimeException {
    /**
     * Constructs an InvalidMatrixOperation object.
     */
    public InvalidMatrixArrayValue(int expectedLength, int inputtedLength, String rowOrColumn) {
        super("Cannot set matrix " + rowOrColumn + " of length " + expectedLength + " to array of length " + inputtedLength + ".");
    }
}
