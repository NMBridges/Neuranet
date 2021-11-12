import Neuranet.Matrix;
import Neuranet.NeuralNetwork;

import java.io.FileNotFoundException;

import Neuranet.Activation;
import Neuranet.Dataset;
import Neuranet.DatasetParser;

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

        NeuralNetwork net = new NeuralNetwork(new int[]{3, 3, 3, 3}, Activation.SIGMOID);
        Dataset[] datasets = new Dataset[0];
        try {
            datasets = DatasetParser.parse("datasets.txt");
        } catch (FileNotFoundException fnfe) {
            System.out.println(fnfe.getMessage());
        }
        print(net);

        print("Average loss: " + net.getAverageLoss( datasets ));
        net.learn(datasets);
        print("Average loss after learning: " + net.getAverageLoss(datasets));

        Dataset exDataset = datasets[0];
        System.out.println("After Training:\n\nInput:" + exDataset.getInput());
        System.out.println("Output:" + net.compute(exDataset.getInput()));
        System.out.println("Expected Output:" + exDataset.getExpectedOutput());
    }
}
