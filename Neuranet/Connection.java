package Neuranet;

/**
 * Class that represents a transformation of
 * data from one neural network layer to
 * another.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Connection {
    private int inNodeCount;
    private int outNodeCount;
    private Matrix weights;
    private Matrix biases;

    /**
     * Default no-arg constructor that creates a blank
     * connection with no inputs or outputs.
     */
    public Connection() {
        this(0,0);
    }

    /**
     * Two-arg constructor that creates a connection
     * with the specified number of inputs and outputs.
     * @param inNodeCount The number of input nodes.
     * @param outNodeCount The number of output nodes.
     */
    public Connection(int inNodeCount, int outNodeCount) {
        //this(inNodeCount, outNodeCount, new Matrix(outNodeCount, inNodeCount), new Matrix(outNodeCount, 1));
        this(inNodeCount, outNodeCount, Matrix.random(outNodeCount, inNodeCount, -1.0, 1.0), Matrix.random(outNodeCount, 1, -1.0, 1.0));
    }

    /**
     * Full-arg constructor that creates a connection with
     * the specified number of inputs and outputs, as well
     * as the specified weights and biases.
     * @param inNodeCount
     * @param outNodeCount
     * @param weights
     * @param biases
     */
    public Connection(int inNodeCount, int outNodeCount, Matrix weights, Matrix biases) {
        this.inNodeCount = inNodeCount;
        this.outNodeCount = outNodeCount;

        /** Validates that the weights are the correct dimensions. If not, weights are reset to 0.0. */
        if (weights.getRowCount() == outNodeCount && weights.getColumnCount() == inNodeCount) {
            this.weights = new Matrix(weights);
        } else {
            this.weights = new Matrix(new double[outNodeCount][inNodeCount]);
        }
        /** Validates that the biases are the correct dimension. If not, the biases are reset to 0.0. */
        if (biases.getRowCount() == outNodeCount && biases.getColumnCount() == 1) {
            this.biases = new Matrix(biases);
        } else {
            this.biases = new Matrix(new double[outNodeCount][1]);
        }
    }

    /**
     * Computes the output node values of a connection from weights and biases,
     * given a set of input parameters.
     * @param inputs The set of input values to compute, as a Matrix.
     * @return The output node values, as a Matrix.
     */
    public Matrix compute(Matrix inputs) {
        /** Ensures compatibility. */
        if (inputs.getRowCount() != inNodeCount || inputs.getColumnCount() != 1) {
            return null;
        }
        Matrix output = Matrix.add(Matrix.multiply(weights, inputs), biases);
        if (output.getRowCount() != outNodeCount) {
            return null;
        }
        return activate(output);
    }

    /**
     * Puts a matrix through an activation function (sigmoid).
     * @param input
     * @return
     */
    public static Matrix activate(Matrix input) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();

        Matrix activatedMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                /** ------- ReLU ------- */
                // activatedMatrix.set(row, col, Math.max(0.0, input.get(row, col)));
                
                /** ------ Sigmoid ----- */
                activatedMatrix.set(row, col, 1.0 / (1.0 + Math.exp(-input.get(row, col))));
            }
        }
        return activatedMatrix;
    }

    /**
     * Returns the number of input nodes.
     * @return the number of input nodes.
     */
    public int getInNodeCount() {
        return inNodeCount;
    }

    /**
     * Returns the number of output nodes.
     * @return the number of output nodes.
     */
    public int getOutNodeCount() {
        return outNodeCount;
    }

    /**
     * Convers the state of the Connection into a readable String.
     */
    @Override
    public String toString() {
        String out = "\nWeights:" + weights + "\nBiases:" + biases;
        return out;
    }
}
