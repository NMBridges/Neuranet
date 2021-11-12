package Neuranet.RuntimeExceptions;

/**
 * Class representing an invalid Dataset format provided in a file.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class InvalidDatasetFormat extends RuntimeException {
    /**
     * Constructs an InvalidDatasetFormat object.
     */
    public InvalidDatasetFormat(int lineNumber, String fileName, String line) {
        super("Invalid dataset provided on line " + lineNumber + " of file " + fileName + ": " + line
            + ". Format must be {i_1} {i_2} ... {i_m} | {o_1} ... {o_n} with each "
            + "input i_x and output o_x being a valid integer or double.");
    }
}
