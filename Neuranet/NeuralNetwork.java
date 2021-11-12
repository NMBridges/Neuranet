package Neuranet;

import java.util.Arrays;

import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Class that represents a neural network.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class NeuralNetwork {
    /** The the weights of the neural network. */
    private Matrix[] weights;
    /** The biases of the neural network. */
    private Matrix[] biases;
    /** The type of activation function that the networks uses. */
    private Activation activationType;
    
    /**
     * Default no-args constructor that creates a network object.
     */
    public NeuralNetwork() {
        weights = new Matrix[0];
        biases = new Matrix[0];
        activationType = Activation.SIGMOID;
    }

    /**
     * Creates a neural network with the given number of nodes.
     * The activation function will default to SIGMOID.
     * @param nodeCounts The numbers of nodes per layer in the network.
     */
    public NeuralNetwork(int[] nodeCounts) {
        this(nodeCounts, Activation.SIGMOID);
    }

    /**
     * Creates a neural network with the given number of nodes
     * and the type of activation function of the network.
     * @param nodeCounts The numbers of nodes per layer in the network.
     * @param activationType The type of activation function the network will use.
     */
    public NeuralNetwork(int[] nodeCounts, Activation activationType) {
        weights = new Matrix[nodeCounts.length - 1];
        biases = new Matrix[nodeCounts.length - 1];
        for (int index = 0; index < weights.length; index++) {
            if (activationType == Activation.RELU) {
                weights[index] = Matrix.random(nodeCounts[index + 1], nodeCounts[index], 0.001, 1.0);
            } else {
                weights[index] = Matrix.random(nodeCounts[index + 1], nodeCounts[index], -1.0, 1.0);
            }
            biases[index] = Matrix.random(nodeCounts[index + 1], 1, 0.0, 0.0);
        }
        this.activationType = activationType;
    }

    /**
     * Computes the 'loss' of the current weights and biases;
     * in other words, how inaccurate the results were from
     * expected.
     * @param expectedOutput The true output of the inputs.
     * @param output The model's predicted output based on the inputs.
     * @return The loss of the weights and biases.
     */
    private static double loss(Matrix expectedOutput, Matrix output) throws InvalidMatrixOperation {
        return Matrix.sumEntries(Matrix.pow(Matrix.subtract(expectedOutput, output), 2.0)) / output.getRowCount();
    }

    /**
     * Computes output sets for all input datasets and
     * returns the average loss of the neural network.
     * @return The average loss of the neural network.
     */
    public double getAverageLoss(Dataset[] datasets) throws InvalidMatrixOperation {
        if(datasets == null || datasets.length == 0) {
            return 0.0;
        }
        double loss = 0.0;
        for (int index = 0; index < datasets.length; index++) {
            loss += loss(datasets[index].getExpectedOutput(), compute(datasets[index].getInput()));
        }
        return loss / datasets.length;
    }

    /**
     * Produces an output set for the provided input set
     * based on the weights and biases of the neural network
     * @param input The input set to compute the output for.
     * @return The output of the neural network with the given input,
     * weights, and biases.
     */
    public Matrix compute(Matrix input) throws InvalidMatrixOperation {
        Matrix output = new Matrix(input);
        for (int index = 0; index < weights.length; index++) {
            output = activate(Matrix.add(Matrix.multiply(weights[index], output), biases[index]));    
        }
        return output;
    }

    /**
     * Computes the node values at each layer
     * of the forward propagation process before
     * activation (the z vectors).
     * @return An array of matrices containing the layers'
     * values.
     */
    private Matrix[] getZvalues(Matrix input) throws InvalidMatrixOperation {
        /**
         * Array of combined mx1 matrices where the first column is the
         * unactivated value at the layer.
         */
        Matrix[] output = new Matrix[weights.length + 1];
        output[0] = new Matrix(input);
        
        Matrix a = input;
        /** Calculates the partial derivative of the activation at that layer. */
        for (int index = 0; index < weights.length; index++) {
            Matrix z = Matrix.add(Matrix.multiply(weights[index], a), biases[index]);
            output[index + 1] = new Matrix(z);
            a = activate(z);
        }
        return output;
    }

    /**
     * Puts the double through a sigmoid function
     * and returns the output.
     * @param in the input to sigmoid-ify.
     * @return the sigmoid-ified double.
     */
    private static double sigmoid(double in) {
        return 1.0 / (1.0 + Math.exp(-in));
    }

    /**
     * Puts the double through a ReLU function
     * and returns the output.
     * @param in the double to linearify.
     * @return the output of the ReLU function.
     */
    private static double reLU(double in) {
        return Math.max(0.0, in);
    }

    /**
     * Puts a matrix through an activation function (sigmoid).
     * @param input the matrix to 'activate'.
     * @return the activated matrix.
     */
    private Matrix activate(Matrix input) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();

        double maxValue = 0.0001;
        Matrix activatedMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                switch(activationType) {
                    case SIGMOID:
                        activatedMatrix.set(row, col, sigmoid(input.get(row, col)));
                        break;
                    case RELU:
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
        if (activationType == Activation.RELU) {
            return Matrix.divide(activatedMatrix, maxValue);
        }
        return activatedMatrix;
    }

    /**
     * Finds the derivative of an activation function (sigmoid) for a given matrix.
     * @param input the matrix at to find the derivative of the 'activate' function.
     * @return the derivative matrix of the activation function.
     */
    private Matrix activateDerivative(Matrix input) {
        int rowCount = input.getRowCount();
        int columnCount = input.getColumnCount();

        Matrix derivativeMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                switch(activationType) {
                    case SIGMOID:
                        derivativeMatrix.set(row, col, sigmoid(input.get(row, col)) * (1 - sigmoid(input.get(row, col))));
                        break;
                    case RELU:
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
     * Given a set of Datasets, it takes the average
     * gradient learned from the model and adjusts the
     * weights and biases accordingly. It defaults to
     * one cycle of training and a batch size of one
     * dataset before updating the weights and biases.
     * @param datasets the datasets used to teach the model.
     */
    public void learn(Dataset[] datasets) {
        learn(datasets, 1, 1);
    }

    /**
     * Loops through an array of Datasets, taking 'batchSize'
     * Datasets at a time and averaging the gradients for the
     * weights and biases over these 'batchSize' datasets, adjusting the
     * weights and biases accordingly. It repeats this process
     * 'cycles' times.
     * @param datasets the Datasets used to teach the model.
     * @param cycles the amount of times to run the training datasets.
     * @param batchSize the size of each batch to train the model.
     *                  In other words, the number of datasets to
     *                  go through each time before averaging their
     *                  gradients and updating the model's weights
     *                  and biases.
     */
    public void learn(Dataset[] datasets, int cycles, int batchSize) {
        for (int cycle = 0; cycle < cycles; cycle += 1) {
            for (int batchStartIndex = 0; batchStartIndex < datasets.length; batchStartIndex += batchSize) {
                Dataset[] batch = Arrays.copyOfRange(datasets, batchStartIndex, Math.min(datasets.length, batchStartIndex + batchSize));

                Matrix[] totalWeightGradients = Matrix.multiply(weights, 0.0);
                Matrix[] totalBiasGradients = Matrix.multiply(biases, 0.0);
                
                /** Averages the gradients of the weights and biases for all datasets. */
                for (Dataset dataset : batch) {
                    Tuple<Matrix[], Matrix[]> gradients = datasetGradients(dataset);
                    totalWeightGradients = Matrix.add(totalWeightGradients, gradients.f);
                    totalBiasGradients = Matrix.add(totalBiasGradients, gradients.s);
                }
                for (int index = 0; index < totalWeightGradients.length; index += 1) {
                    totalWeightGradients[index] = Matrix.divide(totalWeightGradients[index], batch.length);
                    totalBiasGradients[index] = Matrix.divide(totalBiasGradients[index], batch.length);
                }
        
                /** Modifies the weights and biases by the calculated gradients. */
                double learningRate = 1.0;
                for (int index = 0; index < weights.length; index += 1) { 
                    weights[index] = Matrix.subtract(weights[index], Matrix.multiply(totalWeightGradients[index], learningRate));
                }
                for (int index = 0; index < biases.length; index += 1) { 
                    biases[index] = Matrix.subtract(biases[index], Matrix.multiply(totalBiasGradients[index], learningRate));
                }
            }
        }
    }

    /**
     * Takes in a dataset and returns the gradient of the
     * cost function with respect to the weights and biases
     * based on the error of the model with the given dataset.
     * @param dataset The Dataset to compute and learn from.
     * @return the weight and bias gradients learned from the dataset.
     */
    public Tuple<Matrix[], Matrix[]> datasetGradients(Dataset dataset) {
        Matrix input = dataset.getInput();
        Matrix expectedOutput = dataset.getExpectedOutput();
        
        /** Retrieves the the set of unactivated nodes. */
        Matrix[] nodeValues = getZvalues(input); 
        
        /** Backpropagates given the node values and expected output. */
        return backpropagate(nodeValues, expectedOutput);
    }

    /**
     * Given the z values and the expected output, it
     * backpropagates until it finds the gradients of
     * the cost function with respect to the weights
     * and biases.
     * @param zValues the z values of the nodes with
     *                the given input and values.
     * @param expectedOutput the expected output of
     *                       the dataset that created
     *                       the node values.
     * @return the weight and bias gradients learned
     *         from the dataset.
     */
    private Tuple<Matrix[], Matrix[]> backpropagate(Matrix[] zValues, Matrix expectedOutput) {
        /** The output of the input with the current weights and biases. */
        Matrix output = activate(zValues[zValues.length - 1]);
        /** Gradient of loss with respect to the last layer. */
        Matrix dCda_l = Matrix.subtract(output, expectedOutput);
        
        /**
         * The new gradients to modify the weights and biases with
         * based on the error of this dataset.
         */
        Matrix[] weightGradients = new Matrix[weights.length];
        Matrix[] biasGradients = new Matrix[biases.length];
        
        /** The cost at layer l. */
        Matrix sigma_l = new Matrix();

        for (int layer = weights.length - 1; layer >= 0; layer -= 1) {
            /** Unactivated node values (z) at layer l. */
            Matrix z_l = zValues[layer + 1];
            /** Unactivated node values (z) at layer l-1. */
            Matrix z_lminusOne = zValues[layer];
            /** Activated node values (a) at layer l-1. */
            Matrix a_lminusOne = activate(z_lminusOne);
            /** The derivative of the activation function at layer l. */
            Matrix sigma_lprime = activateDerivative(z_l);
            
            /** Recalculates the cost at the current layer. */
            if (layer == weights.length - 1) {
                sigma_l = Matrix.hadamardMultiply(dCda_l, sigma_lprime);
            } else {
                sigma_l = Matrix.hadamardMultiply(Matrix.multiply(Matrix.transpose(weights[layer + 1]), sigma_l), sigma_lprime);
            }

            /**
             * Adjusts the weight and bias gradients based on the error
             * at the current layer.
             */
            weightGradients[layer] = Matrix.multiply(sigma_l, Matrix.transpose(a_lminusOne));
            biasGradients[layer] = new Matrix(sigma_l);
        }

        /** Returns the gradients of the weights and biases. */
        return new Tuple<>(weightGradients, biasGradients);
    }

    /**
     * Returns the state of the Network as a readable String.
     */
    @Override
    public String toString() {
        String out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Neural Network  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";

        if(weights != null && weights.length > 0) {
            out += "\n\n\tLAYER 1 (" + weights[0].getColumnCount() + " nodes)\n";
        }

        for (int index = 0; index < weights.length; index += 1) {
            out += "\nWeights:" + weights[index] + "\nBiases:" + biases[index];
            out += "\n\n\tLAYER " + (index + 2) + " (" + weights[index].getRowCount() + " nodes)\n";
        }
        out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        return out;
    }

    /**
     * Function that facilitates the printing of objects.
     * @param in The object to print.
     */
    public static void print(Object in) {
        System.out.println(in);
    }
}