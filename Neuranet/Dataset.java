package Neuranet;

/**
 * Class that represents the input and expected output
 * of data for the neural network to parse.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Dataset {
    /** The input of the neural network entry. */
    private Matrix input;
    /** The expected output of the neural network entry. */
    private Matrix expectedOutput;
    /** The actual output of the neural network entry. */
    private Matrix networkOutput;
    /** The cost of the neural network entry. */
    private Matrix cost;

    /**
     * No-arg constructor that creates an empty dataset.
     */
    public Dataset() {
        input = new Matrix();
        expectedOutput = new Matrix();
        networkOutput = null;
        cost = null;
    }

    /**
     * Creates a dataset based on the provided input and
     * expected output 
     * @param input The input of the dataset.
     * @param expectedOutput The expected output of the dataset.
     */
    public Dataset(Matrix input, Matrix expectedOutput) {
        this.input = new Matrix(input);
        this.expectedOutput = new Matrix(expectedOutput);
        networkOutput = null;
        cost = null;
    }

    /**
     * Creates a dataset based on the provided input,
     * expected output, and actual neural network output.
     * @param input The input of the dataset.
     * @param expectedOutput The expected output of the dataset.
     * @param networkOutput The actual output of the dataset.
     */
    public Dataset(Matrix input, Matrix expectedOutput, Matrix networkOutput) {
        this.input = (input != null ? new Matrix(input) : new Matrix());
        this.expectedOutput = (expectedOutput != null ? new Matrix(expectedOutput) : new Matrix());
        this.networkOutput = (networkOutput != null ? new Matrix(networkOutput) : null);
        setCost();
    }

    /**
     * Creates a deep copy of another dataset.
     * @param dataset The dataset to create a copy of.
     */
    public Dataset(Dataset dataset) {
        this(dataset.input, dataset.expectedOutput, dataset.networkOutput);
    }

    /**
     * Returns the input of the dataset.
     * @return the input of the dataset.
     */
    public Matrix getInput() {
        return input;
    }

    /**
     * Sets the input of the dataset to the specified value.
     */
    public void setInput(Matrix input) {
        this.input = input;
    }

    /**
     * Returns the expected output of the dataset.
     * @return the expected output of the dataset.
     */
    public Matrix getExpectedOutput() {
        return expectedOutput;
    }

    /**
     * Sets the expected output of the dataset to the specified value.
     */
    public void setExpectedOutput(Matrix expectedOutput) {
        this.expectedOutput = expectedOutput;
        setCost();
    }

    /**
     * Returns the network output of the dataset.
     * @return the network output of the dataset.
     */
    public Matrix getNetworkOutput() {
        return networkOutput;
    }

    /**
     * Sets the network output of the dataset to the specified value.
     */
    public void setNetworkOutput(Matrix networkOutput) {
        this.networkOutput = networkOutput;
        setCost();
    }

    /**
     * Resets the cost of the neural network for the current dataset
     * based on the expected output and the actual neural network's
     * output.
     */
    public void setCost() {
        this.cost = (expectedOutput != null && networkOutput != null ? Matrix.pow(Matrix.subtract(expectedOutput, networkOutput), 2) : null);
    }

    /**
     * Returns the cost of the neural network for the current dataset.
     * @return the cost of the neural network for the current dataset.
     */
    public Matrix getCost() {
        return cost;
    }

    /**
     * Checks equivalence of this dataset with an inputted object.
     * @param other The object to compare with.
     */
    @Override
    public boolean equals(Object other) {
        if (other == null || !(other instanceof Dataset)) {
            return false;
        }
        final Dataset otherDataset = (Dataset) other;
        boolean inputComparison = (input == null ? otherDataset.input == null : input.equals(otherDataset.input));
        boolean expectedOutputComparison = (expectedOutput == null ? otherDataset.expectedOutput == null : expectedOutput.equals(otherDataset.expectedOutput));
        boolean networkOutputComparison = (networkOutput == null ? otherDataset.networkOutput == null : networkOutput.equals(otherDataset.networkOutput));
        return (inputComparison && expectedOutputComparison && networkOutputComparison);
    }

    /**
     * Converts the state of the dataset to a readable String.
     */
    @Override
    public String toString() {
        return "Dataset\n\tInput:" + input + "\n\tExpected Output:" + expectedOutput
                + "\n\tNetwork output:" + (networkOutput != null ? networkOutput : "\nnull");
    }
}
