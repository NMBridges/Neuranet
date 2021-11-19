package Neuranet.RuntimeExceptions;

/**
 * Exception that represents an invalid CNN filter.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class InvalidCNNFilter extends RuntimeException {
    /**
     * Creates a RuntimeException with the specified message.
     * @param message the message to display.
     */
    public InvalidCNNFilter(String message) {
        super(message);
    }
}
