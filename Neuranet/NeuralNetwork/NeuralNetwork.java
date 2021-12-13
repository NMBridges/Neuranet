package Neuranet.NeuralNetwork;

import java.util.Arrays;

import Neuranet.Activation;
import Neuranet.Dataset;
import Neuranet.Matrix2D;
import Neuranet.Network;
import Neuranet.Tuple;
import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Class that represents a neural network.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class NeuralNetwork implements Network {
    /** The weights of the neural network. */
    private Matrix2D[] weights;
    /** The biases of the neural network. */
    private Matrix2D[] biases;
    /** The type of activation function that the neural network uses. */
    private Activation activationType;
    
    /**
     * Default no-args constructor that creates a network object.
     */
    public NeuralNetwork() {
        this.activationType = Activation.SIGMOID;
        weights = new Matrix2D[0];
        biases = new Matrix2D[0];
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
        this.activationType = activationType;
        weights = new Matrix2D[nodeCounts.length - 1];
        biases = new Matrix2D[nodeCounts.length - 1];
        for (int index = 0; index < weights.length; index++) {
            if (activationType == Activation.SIGMOID) {
                weights[index] = Matrix2D.random(nodeCounts[index + 1], nodeCounts[index], -1.0, 1.0);
            } else {
                weights[index] = Matrix2D.random(nodeCounts[index + 1], nodeCounts[index], 0.001, 1.0);
            }
            biases[index] = Matrix2D.random(nodeCounts[index + 1], 1, 0.0, 0.0);
        }
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
        for (int index = 0; index < datasets.length; index += 1) {
            loss += Network.loss(datasets[index].getExpectedOutput(), compute(datasets[index].getInput()));
        }
        return loss / datasets.length;
    }

    /**
     * Produces an output set for the provided input set
     * based on the weights and biases of the neural network.
     * @param input The input set to compute the output for.
     * @return The output of the neural network with the given input,
     * weights, and biases.
     */
    public Matrix2D compute(Matrix2D input) throws InvalidMatrixOperation {
        Matrix2D output = new Matrix2D(input);
        for (int index = 0; index < weights.length; index++) {
            output = Network.activate(Matrix2D.add(Matrix2D.multiply(weights[index], output), biases[index]), activationType);    
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
    private Matrix2D[] getZvalues(Matrix2D input) throws InvalidMatrixOperation {
        /**
         * Array of combined mx1 matrices where the first column is the
         * unactivated value at the layer.
         */
        Matrix2D[] output = new Matrix2D[weights.length + 1];
        output[0] = new Matrix2D(input);
        
        Matrix2D a = input;
        /** Calculates the partial derivative of the activation at that layer. */
        for (int index = 0; index < weights.length; index++) {
            Matrix2D z = Matrix2D.add(Matrix2D.multiply(weights[index], a), biases[index]);
            output[index + 1] = new Matrix2D(z);
            a = Network.activate(z, activationType);
        }
        return output;
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
        learn(datasets, 1, 1, 1.0);
    }

    /**
     * Loops through an array of Datasets, taking 'batchSize'
     * Datasets at a time and averaging the gradients for the
     * weights and biases over these 'batchSize' datasets, adjusting the
     * weights and biases accordingly. It repeats this process
     * 'epochs' times.
     * @param datasets the Datasets used to teach the model.
     * @param epoch the amount of times to run the training datasets.
     * @param batchSize the size of each batch to train the model.
     *                  In other words, the number of datasets to
     *                  go through each time before averaging their
     *                  gradients and updating the model's weights
     *                  and biases.
     */
    public void learn(Dataset[] datasets, int epochs, int batchSize, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch += 1) {
            for (int batchStartIndex = 0; batchStartIndex < datasets.length; batchStartIndex += batchSize) {
                Dataset[] batch = Arrays.copyOfRange(datasets, batchStartIndex, Math.min(datasets.length, batchStartIndex + batchSize));

                Matrix2D[] totalWeightGradients = Matrix2D.multiply(weights, 0.0);
                Matrix2D[] totalBiasGradients = Matrix2D.multiply(biases, 0.0);
                
                /** Averages the gradients of the weights and biases for all datasets. */
                for (Dataset dataset : batch) {
                    Tuple<Matrix2D[], Matrix2D[]> gradients = datasetGradients(dataset);
                    totalWeightGradients = Matrix2D.add(totalWeightGradients, gradients.x);
                    totalBiasGradients = Matrix2D.add(totalBiasGradients, gradients.y);
                }
                for (int index = 0; index < totalWeightGradients.length; index += 1) {
                    totalWeightGradients[index] = Matrix2D.divide(totalWeightGradients[index], batch.length);
                    totalBiasGradients[index] = Matrix2D.divide(totalBiasGradients[index], batch.length);
                }
        
                /** Modifies the weights and biases by the calculated gradients. */
                for (int index = 0; index < weights.length; index += 1) { 
                    weights[index] = Matrix2D.subtract(weights[index], Matrix2D.multiply(totalWeightGradients[index], learningRate));
                }
                for (int index = 0; index < biases.length; index += 1) { 
                    biases[index] = Matrix2D.subtract(biases[index], Matrix2D.multiply(totalBiasGradients[index], learningRate));
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
    public Tuple<Matrix2D[], Matrix2D[]> datasetGradients(Dataset dataset) {
        Matrix2D input = dataset.getInput();
        Matrix2D expectedOutput = dataset.getExpectedOutput();
        
        /** Retrieves the the set of unactivated nodes. */
        Matrix2D[] nodeValues = getZvalues(input); 
        
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
    private Tuple<Matrix2D[], Matrix2D[]> backpropagate(Matrix2D[] zValues, Matrix2D expectedOutput) {
        /** The output of the input with the current weights and biases. */
        Matrix2D output = Network.activate(zValues[zValues.length - 1], activationType);
        /** Gradient of loss with respect to the last layer. */
        Matrix2D dCda_l = Matrix2D.subtract(output, expectedOutput);
        
        /**
         * The new gradients to modify the weights and biases with
         * based on the error of this dataset.
         */
        Matrix2D[] weightGradients = new Matrix2D[weights.length];
        Matrix2D[] biasGradients = new Matrix2D[biases.length];
        
        /** The cost at layer l. */
        Matrix2D delta_l = new Matrix2D();

        for (int layer = weights.length - 1; layer >= 0; layer -= 1) {
            /** Unactivated node values (z) at layer l. */
            Matrix2D z_l = zValues[layer + 1];
            /** Unactivated node values (z) at layer l-1. */
            Matrix2D z_lminusOne = zValues[layer];
            /** Activated node values (a) at layer l-1. */
            Matrix2D a_lminusOne = Network.activate(z_lminusOne, activationType);
            /** The derivative of the activation function at layer l. */
            Matrix2D sigma_lprime = Network.activateDerivative(z_l, activationType);
            
            /** Recalculates the cost at the current layer. */
            if (layer == weights.length - 1) {
                delta_l = Matrix2D.hadamardMultiply(dCda_l, sigma_lprime);
            } else {
                delta_l = Matrix2D.hadamardMultiply(Matrix2D.multiply(Matrix2D.transpose(weights[layer + 1]), delta_l), sigma_lprime);
            }

            /**
             * Adjusts the weight and bias gradients based on the error
             * at the current layer.
             */
            weightGradients[layer] = Matrix2D.multiply(delta_l, Matrix2D.transpose(a_lminusOne));
            biasGradients[layer] = new Matrix2D(delta_l);
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
}