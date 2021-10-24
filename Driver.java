import Neuranet.Matrix;
import Neuranet.NeuralNetwork;
import Neuranet.Dataset;

public class Driver {

    /**
     * Function that facilitates the printing of objects.
     * @param in The object to print.
     */
    public static void print(Object in) {
        System.out.println(in);
    }

    public static void main(String[] args) {
        double[][] values1 = new double[][] {
            {  4.0  },
            {  2.0  },
            { -1.0  }
        };
        double[][] values2 = new double[][] {
            {  0.0  },
            {  0.0  },
            {  0.0  }
        };
        Matrix mat1 = new Matrix(values1);
        Matrix mat2 = new Matrix(values2);

        NeuralNetwork net = new NeuralNetwork(new int[]{3, 5, 3}, new Dataset[]{ new Dataset(mat1, mat2) });
        
        Matrix avgCost = net.compute();
        print(net);
        print(avgCost);

    }
}
