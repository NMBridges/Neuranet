package Neuranet.RuntimeExceptions;

import Neuranet.Matrix;

/**
 * Class representing matrix operation error.
 * Called when matrices do not have compatible
 * dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class InvalidMatrixOperation extends RuntimeException {
    /**
     * Constructs an InvalidMatrixOperation object.
     */
    public InvalidMatrixOperation(Matrix a, Matrix b, String operation) {
        super("Matrices of dimension " + a.getDimensions() + " and "
            + b.getDimensions() + " do not have compatible dimensions for " + operation + ".");
    }
}
