package Neuranet;

import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Interface that contains many common functions found in
 * different types of neural networks.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public interface Network {
    /**
     * Computes the 'loss' of the current weights and biases;
     * in other words, how inaccurate the results were from
     * expected.
     * @param expectedOutput The true output of the inputs.
     * @param output The model's predicted output based on the inputs.
     * @return The loss of the weights and biases.
     */
    static double loss(Matrix2D expectedOutput, Matrix2D output) throws InvalidMatrixOperation {
        return Matrix2D.sumEntries(Matrix2D.pow(Matrix2D.subtract(expectedOutput, output), 2.0)) / (output.getRowCount() * output.getColumnCount());
    }

    /**
     * Puts the double through a sigmoid function
     * and returns the output.
     * @param in the input to sigmoid-ify.
     * @return the sigmoid-ified double.
     */
    static double sigmoid(double in) {
        return 1.0 / (1.0 + Math.exp(-in));
    }

    /**
     * Puts the double through a ReLU function
     * and returns the output.
     * @param in the double to linearify.
     * @return the output of the ReLU function.
     */
    static double reLU(double in) {
        return Math.max(0.0, in);
    }

    /**
     * Puts a matrix through an activation function (sigmoid).
     * @param input the matrix to 'activate'.
     * @param activationType the activation function type to use.
     * @return the activated matrix.
     */
    static Matrix2D activate(Matrix2D input, Activation activationType) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();

        double maxValue = 0.0001;
        Matrix2D activatedMatrix = new Matrix2D(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                switch(activationType) {
                    case SIGMOID:
                        activatedMatrix.set(row, col, sigmoid(input.get(row, col)));
                        break;
                    case RELU:
                        activatedMatrix.set(row, col, reLU(input.get(row, col)));
                        break;
                    case RELU_NORMALIZED:
                        activatedMatrix.set(row, col, reLU(input.get(row, col)));
                        if (input.get(row, col) > maxValue) {
                            maxValue = input.get(row, col);
                        }
                        break;
                    default:
                        activatedMatrix.set(row, col, input.get(row, col));
                }
            }
        }
        /** Normalizes the ReLU function. */
        if (activationType == Activation.RELU_NORMALIZED) {
            return Matrix2D.divide(activatedMatrix, maxValue);
        }
        return activatedMatrix;
    }

    /**
     * Puts a matrix through an activation function (sigmoid).
     * @param input the matrix to 'activate'.
     * @param activationType the activation function type to use.
     * @return the activated matrix.
     */
    static Matrix3D activate(Matrix3D input, Activation activationType) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();
        int layerCount = input.getLayerCount();

        double maxValue = 0.0001;
        Matrix3D activatedMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    switch(activationType) {
                        case SIGMOID:
                            activatedMatrix.set(row, col, lay, sigmoid(input.get(row, col, lay)));
                            break;
                        case RELU:
                            activatedMatrix.set(row, col, lay, reLU(input.get(row, col, lay)));
                            break;
                        case RELU_NORMALIZED:
                            activatedMatrix.set(row, col, lay, reLU(input.get(row, col, lay)));
                            if (input.get(row, col, lay) > maxValue) {
                                maxValue = input.get(row, col, lay);
                            }
                            break;
                        default:
                            activatedMatrix.set(row, col, lay, input.get(row, col, lay));
                    }
                }
            }
        }
        /** Normalizes the ReLU function. */
        if (activationType == Activation.RELU_NORMALIZED) {
            return Matrix3D.divide(activatedMatrix, maxValue);
        }
        return activatedMatrix;
    }

    /**
     * Finds the derivative of an activation function (sigmoid) for a given matrix.
     * @param input the matrix at to find the derivative of the 'activate' function.
     * @param activationType the activation function type to use.
     * @return the derivative matrix of the activation function.
     */
    static Matrix2D activateDerivative(Matrix2D input, Activation activationType) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();

        Matrix2D derivativeMatrix = new Matrix2D(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                switch(activationType) {
                    case SIGMOID:
                        derivativeMatrix.set(row, col, sigmoid(input.get(row, col)) * (1 - sigmoid(input.get(row, col))));
                        break;
                    case RELU:
                        derivativeMatrix.set(row, col, reLU(input.get(row, col)) > 0 ? 1.0 : 0.0);
                        break;
                    case RELU_NORMALIZED:
                        derivativeMatrix.set(row, col, reLU(input.get(row, col)) > 0 ? 1.0 : 0.0);
                        break;
                    default:
                        derivativeMatrix.set(row, col, 0.0);
                }
            }
        }
        return derivativeMatrix;
    }

    /**
     * Finds the derivative of an activation function (sigmoid) for a given matrix.
     * @param input the matrix at to find the derivative of the 'activate' function.
     * @param activationType the activation function type to use.
     * @return the derivative matrix of the activation function.
     */
    static Matrix3D activateDerivative(Matrix3D input, Activation activationType) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();
        int layerCount = input.getLayerCount();

        Matrix3D derivativeMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    switch(activationType) {
                        case SIGMOID:
                            derivativeMatrix.set(row, col, lay, sigmoid(input.get(row, col, lay)) * (1 - sigmoid(input.get(row, col, lay))));
                            break;
                        case RELU:
                            derivativeMatrix.set(row, col, lay, reLU(input.get(row, col, lay)) > 0 ? 1.0 : 0.0);
                            break;
                        case RELU_NORMALIZED:
                            derivativeMatrix.set(row, col, lay, reLU(input.get(row, col, lay)) > 0 ? 1.0 : 0.0);
                            break;
                        default:
                            derivativeMatrix.set(row, col, lay, 0.0);
                    }
                }
            }
        }
        return derivativeMatrix;
    }
}
