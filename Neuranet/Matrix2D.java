package Neuranet;

import Neuranet.RuntimeExceptions.InvalidMatrixArrayValue;
import Neuranet.RuntimeExceptions.InvalidMatrixIndex;
import Neuranet.RuntimeExceptions.InvalidMatrixOperation;

/**
 * Class that represents a 2D matrix of any dimensions.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class Matrix2D extends Matrix {
    /** The values of the matrix. */
    private double[][] values;

    /**
     * Default no-args constructor that creates
     * an empty matrix.
     */
    public Matrix2D() {
        values = new double[0][0];
    }

    /**
     * Creates a matrix with the specified values.
     * @param values The content of the matrix as a 2D array.
     * The number of elements per row should be the same.
     */
    public Matrix2D(double[][] values) {
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
    public Matrix2D(int rows, int columns) {
        this(new double[rows][columns]);
    }

    /**
     * Copy constructor that copies the values of another matrix.
     * @param matrix The matrix to copy.
     */
    public Matrix2D(Matrix2D matrix) {
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
    public static Matrix2D add(Matrix2D a, Matrix2D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "addition");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D sumMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D subtract(Matrix2D a, Matrix2D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "subtraction");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();
        
        Matrix2D differenceMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D multiply(Matrix2D a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D productMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D divide(Matrix2D a, double factor) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D quotientMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D multiply(Matrix2D a, Matrix2D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getColumnCount() != b.getRowCount()) {
            throw new InvalidMatrixOperation(a, b, "multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = b.getColumnCount();
        
        /** The number of terms for row/column multiplication. */
        int linComb = a.getColumnCount();

        Matrix2D productMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D hadamardMultiply(Matrix2D a, Matrix2D b) throws InvalidMatrixOperation {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            throw new InvalidMatrixOperation(a, b, "Hadamard multiplication");
        }

        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D productMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D pow(Matrix2D a, double power) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D powerMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D abs(Matrix2D a) {
        int rowCount = a.getRowCount();
        int columnCount = a.getColumnCount();

        Matrix2D absMatrix = new Matrix2D(new double[rowCount][columnCount]);
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
    public static Matrix2D random(int rows, int columns, double min, double max) {
        Matrix2D randomMatrix = new Matrix2D(rows, columns);
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
    public static Matrix2D transpose(Matrix2D a)
        throws InvalidMatrixArrayValue, InvalidMatrixIndex, InvalidMatrixOperation {
        Matrix2D out = new Matrix2D(a.getColumnCount(), a.getRowCount());
        for (int row = 0; row < out.getRowCount(); row += 1) {
            for (int col = 0; col < out.getColumnCount(); col += 1) {
                out.set(row, col, a.get(col, row));
            }
        }
        return out;
    }

    /**
     * Splits a matrix into an array of matrices
     * of its rows.
     * @param a the matrix to split.
     * @return the array of single-rowed matrices.
     */
    public static Matrix2D[] getRows(Matrix2D a) {
        Matrix2D[] out = new Matrix2D[a.getRowCount()];
        for (int index = 0; index < a.getRowCount(); index += 1) {
            out[index] = new Matrix2D(1, a.getColumnCount());
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                out[index].set(index, col, a.get(index, col));
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
    public static Matrix2D[] getColumns(Matrix2D a) {
        Matrix2D[] out = new Matrix2D[a.getColumnCount()];
        for (int index = 0; index < a.getColumnCount(); index += 1) {
            out[index] = new Matrix2D(a.getRowCount(), 1);
            for (int row = 0; row < a.getRowCount(); row += 1) {
                out[index].set(row, index, a.get(row, index));
            }
        }
        return out;
    }

    /**
     * Returns a sub-matrix of the inputted matrix
     * bounded by the given indices.
     * @param a the original matrix.
     * @param rowStart the index of the top row (inclusive).
     * @param colStart the index of the left column (inclusive).
     * @param rowEnd the index of the bottom row + 1 (inclusive).
     * @param colEnd the index of the right column + 1 (inclusive).
     * @return
     */
    public static Matrix2D subMatrix(Matrix2D a, int rowStart, int colStart, int rowEnd, int colEnd) {
        Matrix2D subMatrix = new Matrix2D(rowEnd - rowStart, colEnd - colStart);
        for (int row = rowStart; row < rowEnd; row += 1) {
            for (int column = colStart; column < colEnd; column += 1) {
                subMatrix.set(row - rowStart, column - colStart, a.get(row, column));
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
    public static Matrix2D[] add(Matrix2D[] a, Matrix2D[] b) throws InvalidMatrixOperation {
        if (a == null || b == null) {
            throw new InvalidMatrixOperation(new Matrix2D(0,0), new Matrix2D(0,0), "array addition");
        }
        Matrix2D[] summedMatrices = new Matrix2D[Math.max(a.length, b.length)];
        for (int index = 0; index < summedMatrices.length; index += 1) {
            Matrix2D a_index;
            Matrix2D b_index;
            if (a.length > 0 && a[0] != null) {
                a_index = Matrix2D.multiply(a[0], 1);
                b_index = Matrix2D.multiply(a[0], 0);
            } else if (b.length > 0 && b[0] != null) {
                a_index = Matrix2D.multiply(b[0], 0);
                b_index = Matrix2D.multiply(b[0], 1);
            } else {
                throw new InvalidMatrixOperation(new Matrix2D(0,0), new Matrix2D(0,0), "array addition");
            }
            if (index < a.length && a[index] != null) {
                a_index = a[index];
            }
            if (index < b.length && b[index] != null) {
                b_index = b[index];
            }
            summedMatrices[index] = Matrix2D.add(a_index, b_index);
        }
        return summedMatrices;
    }
    
    /**
     * Adds all like elements of an array of matrices together, item by item.
     * Matrices must be of the same dimensions.
     * @param a The matrix array to sum.
     * @return The summed array.
     */
    public static Matrix2D add(Matrix2D[] a) throws InvalidMatrixOperation {
        Matrix2D summedMatrix = a[0];
        for (int index = 1; index < a.length; index += 1) {
            summedMatrix = Matrix2D.add(summedMatrix, a[index]);
        }
        return summedMatrix;
    }
    
    /**
     * Multiplies all elements in an array of matrices
     * by a factor.
     * @param a The matrix array to multiply.
     * @return The scaled array.
     */
    public static Matrix2D[] multiply(Matrix2D[] a, double factor) throws InvalidMatrixOperation {
        Matrix2D[] productMatrices = new Matrix2D[a.length];
        for (int index = 1; index < a.length; index += 1) {
            productMatrices[index] = Matrix2D.multiply(a[index], factor);
        }
        return productMatrices;
    }

    /**
     * Sums the entries in the matrix into a scalar.
     * @param a the matrix to sum.
     * @return the sum of the entries.
     */
    public static double sumEntries(Matrix2D a) {
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
    public static double determinant(Matrix2D a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the determinant");
        } else if (a.getColumnCount() == 1) {
            return a.get(0, 0);
        } else {
            double sum = 0.0;
            for (int index = 0; index < a.getColumnCount(); index += 1) {
                Matrix2D minor = Matrix2D.minor(a, 0, index);
                sum += a.get(0, index) * Matrix2D.determinant(minor) * ((index % 2) * (-2) + 1);
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
    public static Matrix2D cofactors(Matrix2D a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the cofactors");
        } else {
            Matrix2D cofactors = new Matrix2D(a.getRowCount(), a.getColumnCount());
            for (int row = 0; row < a.getColumnCount(); row += 1) {
                for (int column = 0; column < a.getColumnCount(); column += 1) {
                    Matrix2D minor = Matrix2D.minor(a, row, column);
                    double det = Matrix2D.determinant(minor);
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
    public static Matrix2D adjoint(Matrix2D a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the adjoint matrix");
        } else {
            return Matrix2D.transpose(Matrix2D.cofactors(a));
        }
    }

    /**
     * Returns the inverse of a matrix, if there is one.
     * @param a the matrix to find the inverse of.
     * @return the inverse matrix.
     * @throws InvalidMatrixOperation if the matrix is not square or
     *                                if its determinant is zero.
     */
    public static Matrix2D inverse(Matrix2D a) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the adjoint matrix");
        } else {
            double det = Matrix2D.determinant(a);
            if (Math.abs(det) > 0.000001) {
                return Matrix2D.divide(Matrix2D.adjoint(a), det);
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
    public static Matrix2D minor(Matrix2D a, int row, int column) throws InvalidMatrixOperation {
        if (a.getColumnCount() != a.getRowCount()) {
            throw new InvalidMatrixOperation(a, a, "calculating the minor");
        } else if (a.getColumnCount() == 1) {
            return a;
        } else {
            Matrix2D minor = new Matrix2D(a.getRowCount() - 1, a.getColumnCount() - 1);
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
    public static Tuple<Integer, Integer> getIndexOfMax(Matrix2D a) {
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
     * Given a 2D matrix of mxn dimensions, it
     * returns an mxnx1 3D matrix.
     * @param a the 2D matrix to turn to 3D.
     * @return the 3D version of the matrix.
     */
    public static Matrix3D to3D(Matrix2D a) throws NullPointerException {
        if (a == null) {
            throw new NullPointerException();
        }

        Matrix3D threeDmatrix = new Matrix3D(a.getRowCount(), a.getColumnCount(), 1);
        for (int row = 0; row < a.getRowCount(); row += 1) {
            for (int col = 0; col < a.getColumnCount(); col += 1) {
                threeDmatrix.set(row, col, 0, a.get(row, col));
            }
        }

        return threeDmatrix;
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
     * @param value The value to set the specified index to.
     * @throws InvalidMatrixIndex if the index is invalid
     */
    public void set(int row, int column, double value) throws InvalidMatrixIndex {
        if (row > getRowCount() - 1 || column > getColumnCount() - 1) {
            throw new InvalidMatrixIndex(this, row, column);
        }
        values[row][column] = value;
    }
    
    /**
     * Sets the specified row. Index should be valid.
     * @param row The index of the row to set.
     * @param values A matrix of values, the first row of which
     *               will be the new values of the row. Preferrably
     *               single-rowed.
     * @throws InvalidMatrixArrayValue if the length of values
     *      is not the number of columns in the matrix.
     * @throws InvalidMatrixIndex if the row index is invalid.
     */
    public void setRow(int row, Matrix2D values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int colCount = this.values[0].length;

        if (values.getColumnCount() != colCount) {
            System.out.println(this);
            System.out.println(values);
            throw new InvalidMatrixArrayValue(colCount, values.getColumnCount(), "column");
        } else if (row >= getRowCount()) {
            throw new InvalidMatrixIndex(this, 0, row);
        }

        for (int col = 0; col < colCount; col += 1) {
            this.values[row][col] = values.get(0, col);
        }
    }
    
    /**
     * Sets the specified column. Index should be valid.
     * @param column The index of the column to set.
     * @param values A matrix of values, the first column of which
     *               will be the new values of the column. Preferrably
     *               single-columned.
     * @throws InvalidMatrixArrayValue if the length of values
     *      is not the number of rows in the matrix.
     * @throws InvalidMatrixIndex if the column index is invalid.
     */
    public void setColumn(int column, Matrix2D values) throws InvalidMatrixArrayValue, InvalidMatrixIndex {
        int rowCount = this.values.length;

        if (values.getRowCount() != rowCount) {
            throw new InvalidMatrixArrayValue(rowCount, values.getRowCount(), "column");
        } else if (column >= getColumnCount()) {
            throw new InvalidMatrixIndex(this, 0, column);
        }

        for (int row = 0; row < rowCount; row += 1) {
            this.values[row][column] = values.get(row, 0);
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
     * Gets the dimensions of the matrix as a String.
     * @return the dimensions of the matrix.
     */
    public String getDimensions() {
        return "" + getRowCount() + "x" + getColumnCount();
    }

    /**
     * Determines equivalence of the matrix with another object.
     * @param other The object to compare to.
     */
    @Override
    public boolean equals(Object other) {
        if (other == null || !(other instanceof Matrix2D)) {
            return false;
        }

        final Matrix2D otherMatrix = (Matrix2D) other;
        
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