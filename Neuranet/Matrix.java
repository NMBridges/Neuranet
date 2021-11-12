package Neuranet;

import Neuranet.RuntimeExceptions.InvalidMatrixArrayValue;
import Neuranet.RuntimeExceptions.InvalidMatrixIndex;
import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Class that represents a matrix of any dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Matrix {
    /** The values of the matrix. */
    private double[][] values;

    /**
     * Default no-args constructor that creates
     * an empty matrix.
     */
    public Matrix() {
        values = new double[0][0];
    }

    /**
     * Creates a matrix with the specified values.
     * @param values The content of the matrix as a 2D array.
     * The number of elements per row should be the same.
     */
    public Matrix(double[][] values) {
        if (values == null) {
            this.values = new double[0][0];
            return;
        }
        
        int rowCount = values.length;
        int columnCount = (values.length > 0 ? values[0].length : 0);
        
        this.values = new double[rowCount][columnCount];
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                this.values[row][col] = values[row][col];
            }
        }
    }

    /**
     * Creates a matrix with the specified dimensions. Default values: 0.0.
     * @param rows The number of rows of the matrix.
     * @param columns The number of columns of the matrix.
     */
    public Matrix(int rows, int columns) {
        this(new double[rows][columns]);
    }

    /**
     * Copy constructor that copies the values of another matrix.
     * @param matrix The matrix to copy.
     */
    public Matrix(Matrix matrix) {
        int rowCount = matrix.getRowCount();
        int columnCount = matrix.getColumnCount();

        values = new double[rowCount][columnCount];
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                values[row][col] = matrix.get(row, col);
            }
        }
    }

    /**
     * Adds two matrices together. Must be of the same dimensions.
     * @param a The first matrix to add.
     * @param b The second matrix to add.
     * @return The sum as a matrix.
     */
    public static Matrix add(Matrix a, Matrix b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "addition");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix sumMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                sumMatrix.set(row, col, a.get(row, col) + b.get(row, col));
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
    public static Matrix subtract(Matrix a, Matrix b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "subtraction");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        
        Matrix differenceMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                differenceMatrix.set(row, col, a.get(row, col) - b.get(row, col));
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
    public static Matrix multiply(Matrix a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix productMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                productMatrix.set(row, col, a.get(row, col) * factor);
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
    public static Matrix divide(Matrix a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix quotientMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                quotientMatrix.set(row, col, a.get(row, col) / factor);
            }
        }
        return quotientMatrix;
    }

    /**
     * Multiplies a matrix by another matrix. Matrix dimensions must be compatible
     * for matrix multiplication.
     * @param a The first matrix.
     * @param b The second matrix that multiplies the first.
     * @return The product as a matrix.
     */
    public static Matrix multiply(Matrix a, Matrix b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getColumnCount() != b.getRowCount()) {
            throw new InvalidMatrixOperation(a, b, "multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = b.getColumnCount();
        
        /** The number of terms for row/column multiplication. */
        int linComb = a.getColumnCount();

        Matrix productMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                /** Find linear combination between a's row and b's column. */
                double sum = 0;
                for (int i = 0; i < linComb; i += 1) {
                    sum += a.get(row, i) * b.get(i, col);
                }
                productMatrix.set(row, col, sum);
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
    public static Matrix hadamardMultiply(Matrix a, Matrix b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "Hadamard multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix productMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                productMatrix.set(row, col, a.get(row, col) * b.get(row, col));
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
    public static Matrix pow(Matrix a, double power) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix powerMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                powerMatrix.set(row, col, Math.pow(a.get(row, col), power));
            }
        }
        return powerMatrix;
    }

    /**
     * Finds the absolute value of each entry in the
     * matrix.
     * @param a The matrix.
     * @return The resulting matrix of positive entries.
     */
    public static Matrix abs(Matrix a) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix absMatrix = new Matrix(new double[rowCount][columnCount]);
        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                absMatrix.set(row, col, Math.abs(a.get(row, col)));
            }
        }
        return absMatrix;
    }

    /**
     * Creates a matrix of the inputted dimensions with random values
     * within the provided bounds.
     * @param rows The number of rows of the matrix.
     * @param columns The number of columns of the matrix.
     * @param min The minimum random value (inclusive).
     * @param max The maximum random value (exclusive).
     * @return The randomly generated matrix.
     */
    public static Matrix random(int rows, int columns, double min, double max) {
        Matrix randomMatrix = new Matrix(rows, columns);
        for (int row = 0; row < rows; row += 1) {
            for (int col = 0; col < columns; col += 1) {
                randomMatrix.set(row, col, Math.random() * (max - min) + min);
            }
        }
        return randomMatrix;
    }

    /**
     * Transposes the inputted matrix.
     * @param a the inputted matrix.
     * @return the transposed matrix.
     */
    public static Matrix transpose(Matrix a)
        throws InvalidMatrixArrayValue, InvalidMatrixIndex, InvalidMatrixOperation {
        Matrix out = new Matrix(a.getColumnCount(), a.getRowCount());
        for (int row = 0; row < out.getRowCount(); row += 1) {
            out.setRow(row, a.getColumn(row));
        }
        return out;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its rows.
     * @param a the matrix to split.
     * @return the array of single-rowed matrices.
     */
    public static Matrix[] splitByRows(Matrix a) {
        Matrix[] out = new Matrix[a.getRowCount()];
        for (int index = 0; index < a.getRowCount(); index += 1) {
            out[index] = new Matrix(1, a.getColumnCount());
            out[index].setRow(0, a.getRow(index));
        }
        return out;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its columns.
     * @param a the matrix to split.
     * @return the array of single-columned matrices.
     */
    public static Matrix[] splitByColumns(Matrix a) {
        Matrix[] out = new Matrix[a.getColumnCount()];
        for (int index = 0; index < a.getColumnCount(); index += 1) {
            out[index] = new Matrix(a.getRowCount(), 1);
            out[index].setColumn(0, a.getColumn(index));
        }
        return out;
    }
    
    /**
     * Adds two arrays of matrices together, item by item.
     * Matrices and arrays must be of the same dimensions.
     * @param a The first matrix array to add.
     * @param b The second matrix array to add.
     * @return The summed matrix array.
     */
    public static Matrix[] add(Matrix[] a, Matrix[] b) throws InvalidMatrixOperation {
        if (a == null || b == null) {
            throw new InvalidMatrixOperation(new Matrix(0,0), new Matrix(0,0), "array addition");
        }
        Matrix[] summedMatrices = new Matrix[Math.max(a.length, b.length)];
        for (int index = 0; index < summedMatrices.length; index += 1) {
            Matrix a_index;
            Matrix b_index;
            if (a.length > 0 && a[0] != null) {
                a_index = Matrix.multiply(a[0], 0);
                b_index = Matrix.multiply(a[0], 0);
            } else if (b.length > 0 && b[0] != null) {
                a_index = Matrix.multiply(b[0], 0);
                b_index = Matrix.multiply(b[0], 0);
            } else {
                throw new InvalidMatrixOperation(new Matrix(0,0), new Matrix(0,0), "array addition");
            }
            if (index < a.length && a[index] != null) {
                a_index = a[index];
            }
            if (index < b.length && b[index] != null) {
                b_index = b[index];
            }
            summedMatrices[index] = Matrix.add(a_index, b_index);
        }
        return summedMatrices;
    }
    
    /**
     * Adds all like elements of an array of matrices together, item by item.
     * Matrices must be of the same dimensions.
     * @param a The matrix array to sum.
     * @return The summed array.
     */
    public static Matrix add(Matrix[] a) throws InvalidMatrixOperation {
        Matrix summedMatrix = a[0];
        for (int index = 1; index < a.length; index += 1) {
            summedMatrix = Matrix.add(summedMatrix, a[index]);
        }
        return summedMatrix;
    }
    
    /**
     * Multiplies all elements in an array of matrices
     * by a factor.
     * @param a The matrix array to multiply.
     * @return The scaled array.
     */
    public static Matrix[] multiply(Matrix[] a, double factor) throws InvalidMatrixOperation {
        Matrix[] productMatrices = new Matrix[a.length];
        for (int index = 1; index < a.length; index += 1) {
            productMatrices[index] = Matrix.multiply(a[index], factor);
        }
        return productMatrices;
    }

    /**
     * Sums the entries in the matrix into a scalar.
     * @param a the matrix to sum.
     * @return the sum of the entries.
     */
    public static double sumEntries(Matrix a) {
        double sum = 0.0;
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int column = 0; column < a.getColumnCount(); column += 1) {
                sum += a.get(row, column);
            }
        }
        return sum;
    }

    /**
     * Returns the determinant of a matrix.
     * @param a the matrix to find the determinant.
     * @return the determinant of the matrix.
     * @throws InvalidMatrixOperation if the matrix is not square.
     */
    public static double determinant(Matrix a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the determinant");
        } else if (a.getColumnCount() == 1) {
            return a.get(0, 0);
        } else {
            double sum = 0.0;
            for (int index = 0; index < a.getColumnCount(); index += 1) {
                Matrix minor = Matrix.minor(a, 0, index);
                sum += a.get(0, index) * Matrix.determinant(minor) * ((index % 2) * (-2) + 1);
            }
            return sum;
        }
    }

    /**
     * Returns the cofactors of a matrix.
     * @param a the matrix to find the cofactors of.
     * @return the cofactor matrix.
     * @throws InvalidMatrixOperation if the matrix is not square.
     */
    public static Matrix cofactors(Matrix a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the cofactors");
        } else {
            Matrix cofactors = new Matrix(a.getRowCount(), a.getColumnCount());
            for (int row = 0; row < a.getColumnCount(); row += 1) {
                for (int column = 0; column < a.getColumnCount(); column += 1) {
                    Matrix minor = Matrix.minor(a, row, column);
                    double det = Matrix.determinant(minor);
                    cofactors.set(row, column, det * (((row + column) % 2) * (-2) + 1));
                }
            }
            return cofactors;
        }
    }

    /**
     * Returns the adjoint of a matrix.
     * @param a the matrix to find the adjoint of.
     * @return the adjoint matrix.
     * @throws InvalidMatrixOperation if the matrix is not square.
     */
    public static Matrix adjoint(Matrix a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the adjoint matrix");
        } else {
            return Matrix.transpose(Matrix.cofactors(a));
        }
    }

    /**
     * Returns the inverse of a matrix, if there is one.
     * @param a the matrix to find the inverse of.
     * @return the inverse matrix.
     * @throws InvalidMatrixOperation if the matrix is not square or
     *                                if its determinant is zero.
     */
    public static Matrix inverse(Matrix a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the adjoint matrix");
        } else {
            double det = Matrix.determinant(a);
            if (Math.abs(det) > 0.000001) {
                return Matrix.divide(Matrix.adjoint(a), det);
            } else {
                throw new InvalidMatrixOperation(a, a, "inversion; determinant is zero.");
            }
        }
    }
    
    /**
     * Returns the minor of a matrix at a given index.
     * @param a the matrix to find the minor.
     * @param row the row index.
     * @param column the column index.
     * @return
     */
    public static Matrix minor(Matrix a, int row, int column) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the minor");
        } else if (a.getColumnCount() == 1) {
            return a;
        } else {
            Matrix minor = new Matrix(a.getRowCount() - 1, a.getColumnCount() - 1);
            for (int r = 0; r < a.getRowCount(); r += 1) {
                if (r != row) {
                    for (int c = 0; c < a.getColumnCount(); c += 1) {
                        if (c != column) {
                            minor.set((r > row ? r - 1 : r), (c > column ? c - 1 : c), a.get(r, c));
                        }
                    }
                }
            }
            return minor;
        }
    }

    /**
     * Returns the index of the first occurrence of
     * the entry with the max value in the matrix.
     * @param a the matrix to find the max value of.
     * @return the index of the first occurence of the
     * entry with the max value in the matrix.
     */
    public static Tuple<Integer, Integer> getIndexOfMax(Matrix a) {
        double maxValue = Double.MIN_VALUE;
        Tuple<Integer, Integer> maxIndex = new Tuple<>(0, 0);
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                if (a.get(row, col) > maxValue) {
                    maxValue = a.get(row, col);
                    maxIndex = new Tuple<>(row, col);
                }
            }
        }
        return maxIndex;
    }

    /**
     * Gets a value at a specified index. Indices should be valid.
     * @param row The row the value is in.
     * @param column The column the value is in.
     * @return The value of the specified index.
     */
    public double get(int row, int column) {
        return values[row][column];
    }

    /**
     * Sets a value at a specified index. Indices should be valid.
     * @param row The row of the value to modify.
     * @param column The column of the value to modify.
     * @return The value to set the specified index to.
     * @throws InvalidMatrixIndex if the index is invalid
     */
    public void set(int row, int column, double value) throws InvalidMatrixIndex {
        if (row > getRowCount() - 1 || column > getColumnCount() - 1) {
            throw new InvalidMatrixIndex(this, row, column);
        }
        values[row][column] = value;
    }

    /**
     * Gets the specified column. Index should be valid.
     * @param column The index of the column to grab.
     * @return The specified column as a double[].
     */
    public double[] getColumn(int column) {
        int rowCount = values.length;

        double[] out = new double[rowCount];
        for (int row = 0; row < rowCount; row += 1) {
            out[row] = values[row][column];
        }

        return out;
    }
    
    /**
     * Sets the specified column. Index should be valid.
     * @param column The index of the column to set.
     * @param values The values to set the column to.
     * @throws InvalidMatrixArrayValue if the length of values
     *      is not the number of rows in the matrix.
     * @throws InvalidMatrixIndex if the column index is invalid.
     */
    public void setColumn(int column, double[] values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int rowCount = this.values.length;

        if (values.length != rowCount) {
            throw new InvalidMatrixArrayValue(rowCount, values.length, "column");
        } else if (column >= getColumnCount()) {
            throw new InvalidMatrixIndex(this, 0, column);
        }

        for (int row = 0; row < rowCount; row += 1) {
            this.values[row][column] = values[row];
        }
    }

    /**
     * Gets the specified row. Index should be valid.
     * @param column The index of the row to grab.
     * @return The specified row as a double[].
     */
    public double[] getRow(int row) {
        return values[row];
    }
    
    /**
     * Sets the specified row.
     * @param row The index of the row to set.
     * @param values The values to set the row to.
     * @throws InvalidMatrixArrayValue if the length of values
     *      is not the number of columns in the matrix.
     * @throws InvalidMatrixIndex if the row index is invalid.
     */
    public void setRow(int row, double[] values) throws InvalidMatrixOperation, InvalidMatrixIndex {
        int colCount = this.values[0].length;

        if (values.length != colCount) {
            throw new InvalidMatrixArrayValue(colCount, values.length, "column");
        } else if (row >= getRowCount()) {
            throw new InvalidMatrixIndex(this, 0, row);
        }

        for (int col = 0; col < colCount; col += 1) {
            this.values[row][col] = values[col];
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
     * Determines equivalence of the matrix with another object.
     * @param other The object to compare to.
     */
    @Override
    public boolean equals(Object other) {
        if (other == null || !(other instanceof Matrix)) {
            return false;
        }

        final Matrix otherMatrix = (Matrix) other;
        
        if (getColumnCount() != otherMatrix.getColumnCount() || getRowCount() != otherMatrix.getRowCount()) {
            return false;
        }
        final int rowCount = getRowCount();
        final int columnCount = getColumnCount();

        for (int row = 0; row < rowCount; row += 1) {
            for (int col = 0; col < columnCount; col += 1) {
                if (get(row, col) != otherMatrix.get(row, col)) {
                    return false;
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
        String out = "\n";
        for (int index = 0; index < values.length; index++) {
            double[] row = values[index];
            out += "[  ";
            for (double val : row) {
                out += val + "  ";
            }
            out += (index == values.length - 1 ? "]" : "]\n");
        }
        return out;
    }
}