package Neuranet.CNN;

import Neuranet.Activation;
import Neuranet.Matrix3D;

/**
 * Class representing a convolution in
 * a Convolutional Neural Network.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Convolution {
    /** The weights of the filters of the convolution. */
    private Matrix3D[] weights;
    /** The biases of the filters. */
    private double[] biases;
    /** Each filter's stride length across the image. */
    private int filterStride;
    /** The padding used around the image. */
    private int padding;
    /** The activation function used by the convolution. */
    private Activation activationType;
    /** The size of the pool used during downsizing. */
    private int poolSize;
    /** The stride length of the pool across the image. */
    private int poolStride;
    /** The pooling method used when downsizing. */
    private Pooling poolType;

    /**
     * Creates a convolution object, which contains the necessary
     * state and methods to perform a convolution on a layer of a
     * Convolutional Neural Network.
     * @param numFilters the number of filters that this convolution will
     *                   use (the number of target output channels).
     * @param filterDimensions length 3 array containing the number of
     *                            rows, columns, and layers (respectively)
     *                            of each filter. All entries must be > 0.
     * @param filterStride the stride of filters in the convolution.
     * @param padding the amount of padding used on the layer when applying the convolution.
     * @param activationType the activation function type used by the convolution.
     * @param poolSize the width/height of pixels to pool when downscaling.
     * @param poolStride the stride of the pool when downscaling.
     * @param poolType the type of pooling method used when downscaling.
     */
    public Convolution(int numFilters, int[] filterDimensions, int filterStride, int padding, Activation activationType, int poolSize, int poolStride, Pooling poolType) {
        this.weights = new Matrix3D[numFilters];
        this.biases = new double[numFilters];
        for (int index = 0; index < numFilters; index += 1) {
            this.biases[index] = 0.0;
            switch (activationType) {
                case SIGMOID:
                    this.weights[index] = Matrix3D.random(filterDimensions[0], filterDimensions[1], filterDimensions[2], -1.0, 1.0);
                    break;
                case RELU:
                    this.weights[index] = Matrix3D.random(filterDimensions[0], filterDimensions[1], filterDimensions[2], 0.001, 1.0);
                    break;
                case RELU_NORMALIZED:
                    this.weights[index] = Matrix3D.random(filterDimensions[0], filterDimensions[1], filterDimensions[2], 0.001, 1.0);
                    break;
                default:
                    this.weights[index] = Matrix3D.random(filterDimensions[0], filterDimensions[1], filterDimensions[2], -1.0, 1.0);
            }
        }
        this.filterStride = filterStride;
        this.padding = padding;
        this.activationType = activationType;
        this.poolSize = poolSize;
        this.poolStride = poolStride;
        this.poolType = poolType;
    }

    /**
     * Creates a deep copy of a convolution.
     * @param convolution the convolution to copy.
     */
    public Convolution(Convolution convolution) {
        this.weights = new Matrix3D[convolution.weights.length];
        this.biases = new double[convolution.biases.length];
        for (int index = 0; index < this.weights.length; index += 1) {
            this.weights[index] = new Matrix3D(convolution.weights[index]);
            this.biases[index] = convolution.biases[index];
        }
        this.filterStride = convolution.filterStride;
        this.padding = convolution.padding;
        this.activationType = convolution.activationType;
        this.poolSize = convolution.poolSize;
        this.poolStride = convolution.poolStride;
        this.poolType = convolution.poolType;
    }

    /**
     * Sets a value for a weight.
     */
    public void setWeight(int index, Matrix3D weight) {
        weights[index] = new Matrix3D(weight);
    }

    /**
     * Returns the weights of the convolution.
     * @return the weights of the convolution.
     */
    public Matrix3D[] getWeights() {
        return weights;
    }

    /**
     * Returns the biases of the convolution.
     * @return the biases of the convolution.
     */
    public double[] getBiases() {
        return biases;
    }

    /**
     * Returns the number of filters of the convolution.
     * @return the number of filters of the convolution.
     */
    public int getFilterCount() {
        return weights.length;
    }

    /**
     * Returns the filter stride of the convolution.
     * @return the filter stride of the convolution.
     */
    public int getFilterStride() {
        return filterStride;
    }

    /**
     * Returns the padding of the convolution.
     * @return the padding of the convolution.
     */
    public int getPadding() {
        return padding;
    }

    /**
     * Returns the activation function type of the convolution.
     * @return the activation function type of the convolution.
     */
    public Activation getActivationType() {
        return activationType;
    }

    /**
     * Returns the size of the pools of the convolution.
     * @return the size of the pools of the convolution.
     */
    public int getPoolSize() {
        return poolSize;
    }

    /**
     * Returns the stride of the pools of the convolution.
     * @return the stride of the pools of the convolution.
     */
    public int getPoolStride() {
        return poolStride;
    }

    /**
     * Returns the pooling method of the convolution used when downsizing.
     * @return the pooling method of the convolution used when downsizing.
     */
    public Pooling getPoolingType() {
        return poolType;
    }

    /**
     * Converts the state of the convolution to a readable String.
     */
    @Override
    public String toString() {
        String out = "Need to implement toString method for Convolution";

        return out;
    }
}
