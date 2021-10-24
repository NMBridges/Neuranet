package Neuranet;

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
    public static Matrix add(Matrix a, Matrix b) {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            return null;
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
    public static Matrix subtract(Matrix a, Matrix b) {
        /** Ensures compatibility. */
        if (a.getRowCount() != b.getRowCount() || a.getColumnCount() != b.getColumnCount()) {
            return null;
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
    public static Matrix multiply(Matrix a, Matrix b) {
        /** Ensures compatibility. */
        if (a.getColumnCount() != b.getRowCount()) {
            return null;
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
     */
    public void set(int row, int column, double value) {
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
     * Gets the specified row. Index should be valid.
     * @param column The index of the row to grab.
     * @return The specified row as a double[].
     */
    public double[] getRow(int row) {
        return values[row];
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