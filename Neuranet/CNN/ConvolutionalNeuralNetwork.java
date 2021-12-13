package Neuranet.CNN;

import Neuranet.Network;
import Neuranet.Matrix3D;
import Neuranet.Activation;
import Neuranet.Triple;

/**
 * Class that represents a convolutional neural network.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class ConvolutionalNeuralNetwork implements Network {
    /** The convolutions of the CNN. */
    Convolution[] convolutions;

    /**
     * Creates an empty CNN.
     */
    public ConvolutionalNeuralNetwork() {
        this.convolutions = new Convolution[0];
    }

    /**
     * Creates a CNN with the specified convolutions.
     * @param convolutions the convolutions of the CNN.
     */
    public ConvolutionalNeuralNetwork(Convolution[] convolutions) {
        this.convolutions = new Convolution[convolutions.length];
        for (int index = 0; index < convolutions.length; index += 1) {
            this.convolutions[index] = new Convolution(convolutions[index]);
        }
    }

    /**
     * Produces an output set for the provided input set
     * based on the convolutions of the CNN.
     * @param input The input set to compute the output for.
     * @return The output of the CNN with the given input and convolutions.
     */
    public Matrix3D compute(Matrix3D input) {
        Matrix3D output = new Matrix3D(input);
        for (int index = 0; index < convolutions.length; index++) {
            output = ConvolutionalNeuralNetwork.computeConvolution(convolutions[index], output);
        }
        return output;
    }

    /**
     * Produces an output set for the provided input set
     * based on the convolution.
     * @param input The input set to compute the output for.
     * @return The output of the convolution with the given input.
     */
    private static Matrix3D computeConvolution(Convolution convolution, Matrix3D input) {
        Matrix3D[] weights = convolution.getWeights();
        double[] biases = convolution.getBiases();
        int filterSize = weights[0].getRowCount();
        int filterStride = convolution.getFilterStride();
        int padding = convolution.getPadding();
        Activation activationType = convolution.getActivationType();

        int filteredRows = (input.getRowCount() - filterSize + 2 * padding) / filterStride + 1;
        int filteredCols = (input.getColumnCount() - filterSize + 2 * padding) / filterStride + 1;
        int filteredLays = weights.length;

        System.out.print("Filtering...");
        
        Matrix3D filtered = new Matrix3D(filteredRows, filteredCols, filteredLays);
        for (int layer = 0; layer < filteredLays; layer++) {
            for (int row = 0; row < filteredRows; row += 1) {
                for (int col = 0; col < filteredCols; col += 1) {
                    /** Takes a subsection of the original input. */
                    int inputRow = row * filterStride - padding;
                    int inputCol = col * filterStride - padding;
                    Matrix3D inputSection = Matrix3D.subMatrix(input, inputRow, inputCol, 0, inputRow + filterSize, inputCol + filterSize, input.getLayerCount());

                    /** Finds the dot product between the input section and the weights. */
                    Matrix3D hadamard = Matrix3D.hadamardMultiply(inputSection, weights[layer]);
                    double dot = Matrix3D.sumEntries(hadamard);

                    /** Apply bias.*/
                    double z = dot + biases[layer];
                    
                    filtered.set(row, col, layer, z);

                    System.out.print("\rFiltering: " + ((int) Math.round((col + row * filteredCols + layer * filteredCols * filteredRows + 1) * 10000.0 / (filteredRows * filteredCols * filteredLays)) / 100.0) + "%");
                }
            }
            /** Activates the layer. */
            filtered.setLayer(layer, Network.activate(filtered.getLayers()[layer], activationType));
        }

        System.out.print("\nPooling...");

        int poolSize = convolution.getPoolSize();
        int poolStride = convolution.getPoolStride();

        int pooledRows = (filteredRows - poolSize) / poolStride + 1;
        int pooledCols = (filteredCols - poolSize) / poolStride + 1;

        Matrix3D pooled = new Matrix3D(pooledRows, pooledCols, filteredLays);
        
        for (int layer = 0; layer < filteredLays; layer++) {
            for (int row = 0; row < pooledRows; row += 1) {
                for (int col = 0; col < pooledCols; col += 1) {
                    /** Takes a subsection of the original input. */
                    int inputRow = row * poolStride;
                    int inputCol = col * poolStride;
                    Matrix3D filteredSection = Matrix3D.subMatrix(filtered.getLayers()[layer], inputRow, inputCol, 0, inputRow + poolSize, inputCol + poolSize, 1);

                    switch (convolution.getPoolingType()) {
                        case MAX:
                            Triple<Integer, Integer, Integer> indexOfMax = Matrix3D.getIndexOfMax(filteredSection);
                            pooled.set(row, col, layer, filteredSection.get(indexOfMax.x, indexOfMax.y, indexOfMax.z));
                            break;
                        case AVERAGE:
                            pooled.set(row, col, layer, Matrix3D.sumEntries(filteredSection) / (poolSize * poolSize));
                            break;
                        default:
                            pooled.set(row, col, layer, Matrix3D.sumEntries(filteredSection) / (poolSize * poolSize));
                    }

                    System.out.print("\rPooling: " + ((int) Math.round((col + row * pooledCols + layer * pooledCols * pooledRows + 1) * 10000.0 / (pooledRows * pooledCols * filteredLays)) / 100.0) + "%");
                }
            }
        }

        System.out.println("");
        
        return pooled;
    }
}
