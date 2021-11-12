package Neuranet.RuntimeExceptions;

import Neuranet.Matrix;

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
    public InvalidMatrixIndex(Matrix a, int row, int col) {
        super("Invalid matrix entry index [" + row + ", " + col + "]. Valid indices for given matrix are: [0 <= row < " + a.getRowCount() + ", 0 <= column < " + a.getColumnCount() + "].");
    }
}