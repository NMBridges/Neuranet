package Neuranet;

/**
 * Class that represents a neural network.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class NeuralNetwork {
    /** The number of nodes in each of the layers. */
    private Connection[] connections;
    /**
     * The collection of inputs, expected outputs, and actual
     * outputs of the neural network.
     */
    private Dataset[] datasets;
    
    /**
     * Default no-args constructor that creates a network object.
     */
    public NeuralNetwork() {
        connections = new Connection[0];
        datasets = new Dataset[0];
    }

    /**
     * Creates a neural network with the given number of nodes.
     * @param nodeCounts The numbers of nodes per layer in the network.
     */
    public NeuralNetwork(int[] nodeCounts) {
        connections = new Connection[nodeCounts.length - 1];
        for (int index = 0; index < connections.length; index++) {
            connections[index] = new Connection(nodeCounts[index], nodeCounts[index + 1]);
        }
        datasets = new Dataset[0];
    }

    /**
     * Creates a neural network with the given number of nodes and
     * the specified datasets.
     * @param nodeCounts The numbers of nodes per layer in the network.
     * @param datasets The sets of inputs, expected outputs, and network outputs.
     */
    public NeuralNetwork(int[] nodeCounts, Dataset[] datasets) {
        connections = new Connection[nodeCounts.length - 1];
        for (int index = 0; index < connections.length; index++) {
            connections[index] = new Connection(nodeCounts[index], nodeCounts[index + 1]);
        }

        this.datasets = new Dataset[datasets.length];
        for (int index = 0; index < this.datasets.length; index++) {
            this.datasets[index] = new Dataset(datasets[index]);
        }
    }

    /**
     * Adds a dataset to the neural network's collection
     * of datasets if it is not already a member.
     * @param dataset The dataset to add to the collection.
     */
    public void addDataset(Dataset dataset) {
        Dataset[] newDatasets = new Dataset[datasets.length + 1];
        for (int index = 0; index < datasets.length; index++) {
            if (datasets[index].equals(dataset)) {
                return;
            }
            newDatasets[index] = new Dataset(datasets[index]);
        }
        newDatasets[newDatasets.length - 1] = dataset;

        datasets = new Dataset[newDatasets.length];
        for (int index = 0; index < newDatasets.length; index++) {
            datasets[index] = new Dataset(newDatasets[index]);
        }
    }

    /**
     * Removes all instances of a dataset from the
     * neural network's collection of datasets.
     * @param dataset The dataset to remove from the collection.
     */
    public void removeDataset(Dataset dataset) {
        int datasetMatchCount = 0;
        for (int index = 0; index < datasets.length; index++) {
            if (datasets[index].equals(dataset)) {
                datasetMatchCount += 1;
            }
        }
        print(datasetMatchCount);
        if (datasetMatchCount == 0) {
            return;
        }

        Dataset[] newDatasets = new Dataset[datasets.length - datasetMatchCount];
        int newIndex = 0;
        for (int index = 0; index < datasets.length; index++) {
            if (!datasets[index].equals(dataset)) {
                newDatasets[newIndex] = new Dataset(datasets[index]);
                newIndex += 1;
            }
        }

        datasets = new Dataset[datasets.length - datasetMatchCount];
        for (int index = 0; index < datasets.length; index++) {
            datasets[index] = new Dataset(newDatasets[index]);
        }
    }

    /**
     * Computes the 'cost' of the current weights and biases;
     * in other words, how inaccurate the results were from
     * expected.
     * @param expectedOutput The true output of the inputs.
     * @param output The model's predicted output based on the inputs.
     * @return The cost of the weights and biases.
     */
    public static Matrix cost(Matrix expectedOutput, Matrix output) {
        return Matrix.pow(Matrix.subtract(expectedOutput, output), 2);
    }

    /**
     * Computes output sets for all input datasets and
     * returns the average cost of the neural network.
     * @return The average cost of the neural network.
     */
    public Matrix compute() {
        if(datasets == null || datasets.length == 0) {
            return new Matrix();
        }
        Matrix totalCost = new Matrix(connections[connections.length - 1].getOutNodeCount(), 1);
        int totalCount = 0;
        for (int index = 0; index < datasets.length; index++) {
            datasets[index].setNetworkOutput(compute(datasets[index].getInput()));
            if(datasets[index].getCost() != null) {
                totalCost = Matrix.add(totalCost, datasets[index].getCost());
                totalCount += 1;
            }
        }
        return Matrix.divide(totalCost, totalCount);
    }

    /**
     * Produces an output set for the provided input set
     * based on the weights and biases of the neural network
     * @param input The input set to compute the output for.
     * @return The output of the neural network with the given input,
     * weights, and biases.
     */
    public Matrix compute(Matrix input) {
        Matrix output = new Matrix(input);
        for (int index = 0; index < connections.length; index++) {
            output = connections[index].compute(output);
        }
        return output;
    }

    /**
     * Returns the state of the Network as a readable String.
     */
    public String toString() {
        String out = stringifyNeuralNetwork() + "\n" + stringifyDatasets();
        return out;
    }

    /**
     * Converts the state of the Network to a readable String.
     * @return the state of the Network to a readable String.
     */
    public String stringifyNeuralNetwork() {
        String out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Neural Network  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
        
        if(connections != null && connections.length > 0) {
            out += "\n\n\tLAYER 1 (" + connections[0].getInNodeCount() + " nodes)\n";
        }

        int index = 1;
        for (Connection conn : connections) {
            index += 1;
            out += conn + "\n\n\tLAYER " + index + " (" + conn.getOutNodeCount() + " nodes)\n";
        }
        out += "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        return out;
    }

    /**
     * Converts the state of the neural network's datasets
     * to a readable String.
     * @return the state of the neural network's datasets
     * to a readable String.
     */
    public String stringifyDatasets() {
        String out = "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Datasets  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";

        int index = 0;
        for (Dataset dataset : datasets) {
            out +=  dataset + (index != datasets.length - 1 ? ",\n" : "");
            index += 1;
        }
        out += "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
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
