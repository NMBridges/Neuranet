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

    /**
     * No-arg constructor that creates an empty dataset.
     */
    public Dataset() {
        input = new Matrix();
        expectedOutput = new Matrix();
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
    }

    /**
     * Creates a deep copy of another dataset.
     * @param dataset The dataset to create a copy of.
     */
    public Dataset(Dataset dataset) {
        this(dataset.input, dataset.expectedOutput);
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
        return (inputComparison && expectedOutputComparison);
    }

    /**
     * Converts the state of the dataset to a readable String.
     */
    @Override
    public String toString() {
        return "Dataset\n\tInput:" + input + "\n\tExpected Output:" + expectedOutput + "\n";
    }
}
