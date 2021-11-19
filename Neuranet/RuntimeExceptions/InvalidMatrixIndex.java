package Neuranet.RuntimeExceptions;

import Neuranet.Matrix;
import Neuranet.Matrix2D;

/**
 * Class representing matrix index out of bounds error.
 * Called when inputted index does not fit the
 * matrix's dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class InvalidMatrixIndex extends RuntimeException {
    /**
     * Constructs an InvalidMatrixIndex object.
     */
    public InvalidMatrixIndex(Matrix a, int...dims) {
        super("Invalid matrix entry index [" + dims[0] + ", " + dims[1] + (dims.length > 2 ? ", " + dims[2] : "") + "] for matrix of dimensions (not indexes) " + a.getDimensions() + ".");
    }
}