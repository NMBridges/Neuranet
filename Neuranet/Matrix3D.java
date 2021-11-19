package Neuranet;

import Neuranet.RuntimeExceptions.InvalidMatrixArrayValue;
import Neuranet.RuntimeExceptions.InvalidMatrixIndex;
import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Class that represents a 3D matrix of any dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Matrix3D extends Matrix {
    /** The values of the matrix. */
    private double[][][] values;

    /**
     * Default no-args constructor that creates
     * an empty matrix.
     */
    public Matrix3D() {
        values = new double[0][0][0];
    }

    /**
     * Creates a matrix with the specified values.
     * @param values The content of the matrix as a 3D array (rows, columns, layers)
     * The number of elements per row should be the same across all columns and layers,
     * as should the number of columns across all layers.
     */
    public Matrix3D(double[][][] values) {
        if (values == null) {
            this.values = new double[0][0][0];
            return;
        }
        
        int rowCount = values.length;
        int columnCount = (values.length > 0 ? values[0].length : 0);
        int layerCount = (values.length > 0 && values[0].length > 0 ? values[0][0].length : 0);
        
        this.values = new double[rowCount][columnCount][layerCount];
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    this.values[row][col][lay] = values[row][col][lay];
                }
            }
        }
    }

    /**
     * Creates a matrix with the specified dimensions. Default values: 0.0.
     * @param rows The number of rows of the matrix.
     * @param columns The number of columns of the matrix.
     * @param layers The number of layers of the matrix.
     */
    public Matrix3D(int rows, int columns, int layers) {
        this(new double[rows][columns][layers]);
    }

    /**
     * Copy constructor that copies the values of another matrix.
     * @param matrix The matrix to copy.
     */
    public Matrix3D(Matrix3D matrix) {
        int rowCount = matrix.getRowCount();
        int columnCount = matrix.getColumnCount();
        int layerCount = matrix.getLayerCount();

        values = new double[rowCount][columnCount][layerCount];
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    values[row][col][lay] = matrix.get(row, col, lay);
                }
            }
        }
    }

    /**
     * Adds two matrices together. Must be of the same dimensions.
     * @param a The first matrix to add.
     * @param b The second matrix to add.
     * @return The sum as a matrix.
     */
    public static Matrix3D add(Matrix3D a, Matrix3D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a == null || b == null || !a.getDimensions().equals(b.getDimensions())) {
            throw new InvalidMatrixOperation(a, b, "addition");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D sumMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    sumMatrix.set(row, col, lay, a.get(row, col, lay) + b.get(row, col, lay));
                }
            }
        }
        return sumMatrix;
    }

    /**
     * Subtracts the second matrix from the first. Must be of the same dimensions.
     * @param a The first matrix.
     * @param b The second matrix to subtract from the first.
     * @return The difference as a matrix.
     */
    public static Matrix3D subtract(Matrix3D a, Matrix3D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a == null || b == null || !a.getDimensions().equals(b.getDimensions())) {
            throw new InvalidMatrixOperation(a, b, "subtraction");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();
        
        Matrix3D differenceMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    differenceMatrix.set(row, col, lay, a.get(row, col, lay) - b.get(row, col, lay));
                }
            }
        }
        return differenceMatrix;
    }

    /**
     * Multiplies a matrix by a scalar value.
     * @param a The original matrix.
     * @param factor The factor that the matrix should be scaled by.
     * @return The product as a matrix.
     */
    public static Matrix3D multiply(Matrix3D a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D productMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    productMatrix.set(row, col, lay, a.get(row, col, lay) * factor);
                }
            }
        }
        return productMatrix;
    }

    /**
     * Divides a matrix by a scalar value.
     * @param a The original matrix.
     * @param factor The factor that the matrix should be divided by.
     * @return The product as a matrix.
     */
    public static Matrix3D divide(Matrix3D a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D quotientMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    quotientMatrix.set(row, col, lay, a.get(row, col, lay) / factor);
                }
            }
        }
        return quotientMatrix;
    }

    /**
     * Multiplies a matrix by another matrix. Matrix dimensions must be compatible
     * for matrix multiplication. In other words, the layerCount of a and b must
     * be the same, and the rowCount of b must equal the columnCount of a.
     * @param a The first matrix.
     * @param b The second matrix that multiplies the first.
     * @return The product as a matrix.
     */
    public static Matrix3D multiply(Matrix3D a, Matrix3D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getColumnCount() != b.getRowCount() || a.getLayerCount() != b.getLayerCount()) {
            throw new InvalidMatrixOperation(a, b, "multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = b.getColumnCount();
        int layerCount = a.getLayerCount();
        
        /** The number of terms for row/column multiplication. */
        int linComb = a.getColumnCount();

        Matrix3D productMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int lay = 0; lay < layerCount; lay += 1) {
            for (int row = 0; row < rowCount; row += 1) {
                for (int col = 0; col < columnCount; col += 1) {
                    /** Find linear combination between a's row and b's column. */
                    double sum = 0;
                    for (int i = 0; i < linComb; i += 1) {
                        sum += a.get(row, i, lay) * b.get(i, col, lay);
                    }
                    productMatrix.set(row, col, lay, sum);
                }
            }
        }
        return productMatrix;
    }

    /**
     * Multiplies two matrices together in an element-wise
     * fashion. Must be of the same dimensions.
     * @param a The first matrix to multiply.
     * @param b The second matrix to multiply.
     * @return The product as a matrix.
     */
    public static Matrix3D hadamardMultiply(Matrix3D a, Matrix3D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a == null || b == null || !a.getDimensions().equals(b.getDimensions())) {
            throw new InvalidMatrixOperation(a, b, "Hadamard multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D productMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    productMatrix.set(row, col, lay, a.get(row, col, lay) * b.get(row, col, lay));
                }
            }
        }
        return productMatrix;
    }

    /**
     * Raises the entries of the a matrix to a given power.
     * @param a The matrix.
     * @param power The power to raise the entries of the matrix to.
     * @return The resulting matrix of entries raised to an inputted power.
     */
    public static Matrix3D pow(Matrix3D a, double power) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D powerMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    powerMatrix.set(row, col, lay, Math.pow(a.get(row, col, lay), power));
                }
            }
        }
        return powerMatrix;
    }

    /**
     * Finds the absolute value of each entry in the matrix.
     * @param a The matrix.
     * @return The resulting matrix of positive entries.
     */
    public static Matrix3D abs(Matrix3D a) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        int layerCount = a.getLayerCount();

        Matrix3D absMatrix = new Matrix3D(new double[rowCount][columnCount][layerCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    absMatrix.set(row, col, lay, Math.abs(a.get(row, col, lay)));
                }
            }
        }
        return absMatrix;
    }

    /**
     * Creates a matrix of the inputted dimensions with random values
     * within the provided bounds.
     * @param rows The number of rows of the matrix.
     * @param columns The number of columns of the matrix.
     * @param layers The number of layers of the matrix.
     * @param min The minimum random value (inclusive).
     * @param max The maximum random value (exclusive).
     * @return The randomly generated matrix.
     */
    public static Matrix3D random(int rows, int columns, int layers, double min, double max) {
        Matrix3D randomMatrix = new Matrix3D(rows, columns, layers);
        for (int row = 0; row < rows; row += 1) {
            for (int col = 0; col < columns; col += 1) {
                for (int lay = 0; lay < layers; lay += 1) {
                    randomMatrix.set(row, col, lay, Math.random() * (max - min) + min);
                }
            }
        }
        return randomMatrix;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its rows.
     * @param a the matrix to split.
     * @return the array of single-rowed matrices.
     */
    public static Matrix3D[] getRows(Matrix3D a) {
        Matrix3D[] out = new Matrix3D[a.getRowCount()];
        for (int index = 0; index < a.getRowCount(); index += 1) {
            out[index] = new Matrix3D(1, a.getColumnCount(), a.getLayerCount());
            
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                for (int lay = 0; lay < a.getLayerCount(); lay += 1) {
                    out[index].set(0, col, lay, a.get(index, col, lay));
                }
            }
        }
        return out;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its columns.
     * @param a the matrix to split.
     * @return the array of single-columned matrices.
     */
    public static Matrix3D[] getColumns(Matrix3D a) {
        Matrix3D[] out = new Matrix3D[a.getColumnCount()];
        for (int index = 0; index < a.getColumnCount(); index += 1) {
            out[index] = new Matrix3D(a.getRowCount(), 1, a.getLayerCount());
            
            for (int row = 0; row < a.getRowCount(); row += 1) {
                for (int lay = 0; lay < a.getLayerCount(); lay += 1) {
                    out[index].set(row, 0, lay, a.get(row, index, lay));
                }
            }
        }
        return out;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its layers.
     * @param a the matrix to split.
     * @return the array of single-layered matrices.
     */
    public static Matrix3D[] getLayers(Matrix3D a) {
        Matrix3D[] out = new Matrix3D[a.getLayerCount()];
        for (int index = 0; index < a.getLayerCount(); index += 1) {
            out[index] = new Matrix3D(a.getRowCount(), a.getColumnCount(), 1);

            for (int row = 0; row < a.getRowCount(); row += 1) {
                for (int col = 0; col < a.getColumnCount(); col += 1) {
                    out[index].set(row, col, 0, a.get(row, col, index));
                }
            }
        }
        return out;
    }

    /**
     * Returns a sub-matrix of the inputted matrix
     * bounded by the given indices. If the bounds
     * are outside the matrix's bounds, it will make
     * the values at those indices 0.0.
     * @param a the original matrix.
     * @param rowStart the index of the top row (inclusive).
     * @param colStart the index of the left column (inclusive).
     * @param layStart the index of the front layer (inclusive);
     * @param rowEnd the index of the bottom row + 1 (inclusive).
     * @param colEnd the index of the right column + 1 (inclusive).
     * @param layEnd the index of the back layer + 1 (inclusive).
     * @return the sub matrix.
     */
    public static Matrix3D subMatrix(Matrix3D a, int rowStart, int colStart, int layStart, int rowEnd, int colEnd, int layEnd) {
        Matrix3D subMatrix = new Matrix3D(rowEnd - rowStart, colEnd - colStart, layEnd - layStart);
        for (int row = rowStart; row < rowEnd; row += 1) {
            for (int column = colStart; column < colEnd; column += 1) {
                for (int layer = layStart; layer < layEnd; layer += 1) {
                    double value = (0 <= row && row < a.getRowCount() && 0 <= column && column < a.getColumnCount() && 0 <= layer && layer < a.getLayerCount() ? a.get(row, column, layer) : 0.0);
                    subMatrix.set(row - rowStart, column - colStart, layer - layStart, value);
                }
            }
        }
        return subMatrix;
    }
    
    /**
     * Adds two arrays of matrices together, item by item.
     * Matrices and arrays must be of the same dimensions.
     * @param a The first matrix array to add.
     * @param b The second matrix array to add.
     * @return The summed matrix array.
     */
    public static Matrix3D[] add(Matrix3D[] a, Matrix3D[] b) throws InvalidMatrixOperation {
        if (a == null || b == null) {
            throw new InvalidMatrixOperation(new Matrix3D(0, 0, 0), new Matrix3D(0, 0, 0), "array addition");
        }
        Matrix3D[] summedMatrices = new Matrix3D[Math.max(a.length, b.length)];
        for (int index = 0; index < summedMatrices.length; index += 1) {
            Matrix3D a_index;
            Matrix3D b_index;
            if (a.length > 0 && a[0] != null) {
                a_index = Matrix3D.multiply(a[0], 1);
                b_index = Matrix3D.multiply(a[0], 0);
            } else if (b.length > 0 && b[0] != null) {
                a_index = Matrix3D.multiply(b[0], 0);
                b_index = Matrix3D.multiply(b[0], 1);
            } else {
                throw new InvalidMatrixOperation(new Matrix3D(0, 0, 0), new Matrix3D(0, 0, 0), "array addition");
            }
            if (index < a.length && a[index] != null) {
                a_index = a[index];
            }
            if (index < b.length && b[index] != null) {
                b_index = b[index];
            }
            summedMatrices[index] = Matrix3D.add(a_index, b_index);
        }
        return summedMatrices;
    }
    
    /**
     * Adds all like elements of an array of matrices together, item by item.
     * Matrices must be of the same dimensions.
     * @param a The matrix array to sum.
     * @return The summed array.
     */
    public static Matrix3D add(Matrix3D[] a) throws InvalidMatrixOperation {
        Matrix3D summedMatrix = a[0];
        for (int index = 1; index < a.length; index += 1) {
            summedMatrix = Matrix3D.add(summedMatrix, a[index]);
        }
        return summedMatrix;
    }
    
    /**
     * Multiplies all elements in an array of matrices
     * by a factor.
     * @param a The matrix array to multiply.
     * @return The scaled array.
     */
    public static Matrix3D[] multiply(Matrix3D[] a, double factor) throws InvalidMatrixOperation {
        Matrix3D[] productMatrices = new Matrix3D[a.length];
        for (int index = 1; index < a.length; index += 1) {
            productMatrices[index] = Matrix3D.multiply(a[index], factor);
        }
        return productMatrices;
    }

    /**
     * Sums the entries in the matrix into a scalar.
     * @param a the matrix to sum.
     * @return the sum of the entries.
     */
    public static double sumEntries(Matrix3D a) {
        double sum = 0.0;
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int column = 0; column < a.getColumnCount(); column += 1) {
                for (int layer = 0; layer < a.getLayerCount(); layer += 1) {
                    sum += a.get(row, column, layer);
                }
            }
        }
        return sum;
    }

    /**
     * Returns the index of the first occurrence of
     * the entry with the max value in the matrix.
     * @param a the matrix to find the max value of.
     * @return the index of the first occurence of the
     * entry with the max value in the matrix.
     */
    public static Triple<Integer, Integer, Integer> getIndexOfMax(Matrix3D a) {
        double maxValue = Double.MIN_VALUE;
        Triple<Integer, Integer, Integer> maxIndex = new Triple<>(0, 0, 0);
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                for (int lay = 0; lay < a.getLayerCount(); lay += 1) {
                    if (a.get(row, col, lay) > maxValue) {
                        maxValue = a.get(row, col, lay);
                        maxIndex = new Triple<>(row, col, lay);
                    }
                }
            }
        }
        return maxIndex;
    }

    /**
     * Given a 3D matrix of mxnx1 dimensions, it
     * returns an mxn 2D matrix.
     * @param a the 3D matrix to turn to 2D.
     * @return the 2D version of the matrix.
     */
    public static Matrix2D to2D(Matrix3D a) throws InvalidMatrixOperation {
        if (a.getLayerCount() != 1) {
            throw new InvalidMatrixOperation(a, new Matrix2D(a.getRowCount(), a.getColumnCount()), "conversion to 2D");
        }

        Matrix2D twoDmatrix = new Matrix2D(a.getRowCount(), a.getColumnCount());
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                twoDmatrix.set(row, col, a.get(row, col, 0));
            }
        }

        return twoDmatrix;
    }

    /**
     * Flattens the values of a 3D matrix into a
     * singled columned 2D matrix.
     * @param a the 3D matrix to flatten.
     * @return the flattened matrix.
     */
    public static Matrix2D flatten(Matrix3D a) {
        int rowCount = a.getRowCount();
        int colCount = a.getColumnCount();
        int layCount = a.getLayerCount();
        
        Matrix2D flattenedMatrix = new Matrix2D(rowCount * colCount * layCount, 1);

        for (int lay = 0; lay < layCount; lay += 1) {
            for (int row = 0; row < rowCount; row += 1) {
                for (int col = 0; col < colCount; col += 1) {
                    flattenedMatrix.set(col + row * colCount + lay * colCount * rowCount, 0, a.get(row, col, lay));
                }
            }
        }

        return flattenedMatrix;
    }

    /**
     * Gets a value at a specified index. Indices should be valid.
     * @param row The row the value is in.
     * @param column The column the value is in.
     * @param layer The layer the value is in.
     * @return The value of the specified index.
     */
    public double get(int row, int column, int layer) {
        return values[row][column][layer];
    }

    /**
     * Sets a value at a specified index. Indices should be valid.
     * @param row The row of the value to modify.
     * @param column The column of the value to modify.
     * @param layer The layer of the value to modify.
     * @param value The value to set the specified index to.
     * @throws InvalidMatrixIndex if the index is invalid
     */
    public void set(int row, int column, int layer, double value) throws InvalidMatrixIndex {
        if (row > getRowCount() - 1 || column > getColumnCount() - 1 || layer > getLayerCount() - 1) {
            throw new InvalidMatrixIndex(this, row, column);
        }
        values[row][column][layer] = value;
    }
    
    /**
     * Sets the specified column. Index should be valid.
     * @param column The index of the column to set.
     * @param values Matrix of values to set the rows/layers in the column to.
     * @throws InvalidMatrixArrayValue if the dimensions of rows/layers of
     *                  'values' do not match those of the current matrix.
     * @throws InvalidMatrixIndex if the column index is invalid.
     */
    public void setColumn(int column, Matrix3D values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int rowCount = getRowCount();
        int layerCount = getLayerCount();

        if (rowCount != values.getRowCount()) {
            throw new InvalidMatrixArrayValue(rowCount, values.getRowCount(), "column");
        } else if (layerCount != values.getLayerCount()) {
            throw new InvalidMatrixArrayValue(layerCount, values.getLayerCount(), "column");
        } else if (column >= getColumnCount()) {
            throw new InvalidMatrixIndex(this, 0, column, 0);
        }

        for (int row = 0; row < rowCount; row += 1) {
            for (int lay = 0; lay < layerCount; lay += 1) {
                this.values[row][column][lay] = values.get(row, 0, lay);
            }
        }
    }
    
    /**
     * Sets the specified row. Index should be valid.
     * @param row The index of the row to set.
     * @param values Matrix of values to set the columns/layers in the row to.
     * @throws InvalidMatrixArrayValue if the dimensions of columns/layers of
     *                  'values' do not match those of the current matrix.
     * @throws InvalidMatrixIndex if the row index is invalid.
     */
    public void setRow(int row, Matrix3D values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int columnCount = getColumnCount();
        int layerCount = getLayerCount();

        if (columnCount != values.getColumnCount()) {
            throw new InvalidMatrixArrayValue(columnCount, values.getColumnCount(), "row");
        } else if (layerCount != values.getLayerCount()) {
            throw new InvalidMatrixArrayValue(layerCount, values.getLayerCount(), "row");
        } else if (row >= getRowCount()) {
            throw new InvalidMatrixIndex(this, row, 0, 0);
        }

        for (int col = 0; col < columnCount; col += 1) {
            for (int lay = 0; lay < layerCount; lay += 1) {
                this.values[row][col][lay] = values.get(0, col, lay);
            }
        }
    }
    
    /**
     * Sets the specified layer. Index should be valid.
     * @param layer The index of the layer to set.
     * @param values Matrix of values to set the columns/rows in the layer to.
     * @throws InvalidMatrixArrayValue if the dimensions of columns/rows of
     *                  'values' do not match those of the current matrix.
     * @throws InvalidMatrixIndex if the layer index is invalid.
     */
    public void setLayer(int layer, Matrix3D values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int rowCount = getRowCount();
        int columnCount = getColumnCount();

        if (rowCount != values.getRowCount()) {
            throw new InvalidMatrixArrayValue(rowCount, values.getRowCount(), "layer");
        } else if (columnCount != values.getColumnCount()) {
            throw new InvalidMatrixArrayValue(columnCount, values.getColumnCount(), "layer");
        } else if (layer >= getLayerCount()) {
            throw new InvalidMatrixIndex(this, 0, 0, layer);
        }

        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                this.values[row][col][layer] = values.get(row, col, 0);
            }
        }
    }

    /**
     * Returns the number of rows in the matrix.
     * @return the number of rows in the matrix.
     */
    public int getRowCount() {
        return values.length;
    }

    /**
     * Returns the number of columns in the matrix.
     * @return the number of columns in the matrix.
     */
    public int getColumnCount() {
        return (values.length > 0 ? values[0].length : 0);
    }

    /**
     * Returns the number of layers in the matrix.
     * @return the number of layers in the matrix.
     */
    public int getLayerCount() {
        return (values.length > 0 && values[0].length > 0 ? values[0][0].length : 0);
    }

    /**
     * Gets the dimensions of the matrix as a String.
     * @return the dimensions of the matrix.
     */
    public String getDimensions() {
        return "" + getRowCount() + "x" + getColumnCount() + "x" + getLayerCount();
    }

    /**
     * Determines equivalence of the matrix with another object.
     * @param other The object to compare to.
     */
    @Override
    public boolean equals(Object other) {
        if (other == null || !(other instanceof Matrix3D)) {
            return false;
        }

        final Matrix3D otherMatrix = (Matrix3D) other;
        
        if (getColumnCount() != otherMatrix.getColumnCount() || getRowCount() != otherMatrix.getRowCount()
                || getLayerCount() != otherMatrix.getLayerCount()) {
            return false;
        }
        final int rowCount = getRowCount();
        final int columnCount = getColumnCount();
        final int layerCount = getLayerCount();

        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                for (int lay = 0; lay < layerCount; lay += 1) {
                    if (get(row, col, lay) != otherMatrix.get(row, col, lay)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Returns a String representation of the matrix.
     * @return a String representation of the matrix.
     */
    @Override
    public String toString() {
        String out = "\n[";
        Matrix3D[] layers = Matrix3D.getLayers(this);
        for (int index = 0; index < layers.length; index++) {
            out += Matrix3D.to2D(layers[index]);
            if (index < layers.length - 1) {
                out += ", \n";
            }
        }
        return out + "\n]";
    }
}