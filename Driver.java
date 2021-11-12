import Neuranet.Matrix;
import Neuranet.NeuralNetwork;

import java.io.FileNotFoundException;
import java.util.Arrays;

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

        NeuralNetwork net = new NeuralNetwork(new int[]{3, 15, 3}, Activation.SIGMOID);
        Dataset[] datasets = new Dataset[0];
        try {
            datasets = DatasetParser.parse("datasets3.txt");
        } catch (FileNotFoundException fnfe) {
            System.out.println(fnfe.getMessage());
        }
        print(net);

        print("Average loss: " + net.getAverageLoss( datasets ));
        for (int iteration = 0; iteration < 20; iteration += 1) {
            for (int i = 0; i < datasets.length; i += 6) {
                net.learn(Arrays.copyOfRange(datasets, i, i + 6));
            }
        }
        print("Average loss after learning: " + net.getAverageLoss(datasets));

        try {
            datasets = DatasetParser.parse("datasets3test.txt");
        } catch (FileNotFoundException fnfe) {
            System.out.println(fnfe.getMessage());
        }

        int correct = 0;
        for (Dataset dataset : datasets) {
            Dataset exDataset = dataset;
            int guess = Matrix.getIndexOfMax(net.compute(exDataset.getInput())).f;
            int answer = Matrix.getIndexOfMax(exDataset.getExpectedOutput()).f;
            if (guess == answer) {
                correct += 1;
            } else {
                System.out.println("\nInput:" + exDataset.getInput());
                System.out.println("Output:" + net.compute(exDataset.getInput()));
                System.out.println("Expected Output:" + exDataset.getExpectedOutput());
            }
        }

        System.out.println("Accuracy: " + (100 * correct / datasets.length) + "%");

        System.out.println(net);
    }
}
