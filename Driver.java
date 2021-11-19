import Neuranet.Matrix2D;
import Neuranet.Matrix3D;
import Neuranet.NeuralNetwork.NeuralNetwork;
import Neuranet.CNN.ConvolutionalNeuralNetwork;
import Neuranet.CNN.Pooling;
import Neuranet.CNN.Convolution;
import Neuranet.Activation;
import Neuranet.Dataset;
import Neuranet.DatasetParser;
import Neuranet.ImageData;

import java.io.File;
import java.io.FileNotFoundException;


public class Driver {

    /**
     * Function that facilitates the printing of objects.
     * @param in The object to print.
     */
    public static void print(Object in) {
        System.out.println(in);
    }

    public static void main(String[] args) {
        /**
        NeuralNetwork net = new NeuralNetwork(new int[]{3, 3}, Activation.SIGMOID);
        print(net);
        
        Dataset[] datasets = new Dataset[0];
        try {
            datasets = DatasetParser.parse("datasets3.txt");
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }

        print("Average loss: " + net.getAverageLoss( datasets ));
        net.learn(datasets, 15, 1, 3.0);
        print("Average loss after learning: " + net.getAverageLoss(datasets));

        try {
            datasets = DatasetParser.parse("datasets3test.txt");
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }

        int correct = 0;
        for (Dataset dataset : datasets) {
            Dataset exDataset = dataset;
            int guess = Matrix2D.getIndexOfMax(net.compute(exDataset.getInput())).x;
            int answer = Matrix2D.getIndexOfMax(exDataset.getExpectedOutput()).x;
            if (guess == answer) {
                correct += 1;
            } else {
                print("\nInput:" + exDataset.getInput());
                print("Output:" + net.compute(exDataset.getInput()));
                print("Expected Output:" + exDataset.getExpectedOutput());
            }
        }

        print("Accuracy: " + (100 * correct / datasets.length) + "%");

        print(net);

        try {
            double[] imgData = ImageData.parseImageGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02.jpg", 400, 200);
            ImageData.writeGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02-Scaled.png", imgData, 400, 200);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }*/

        Matrix3D input = new Matrix3D();
        try {
            input = ImageData.parseImageGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02.jpg", 100, 100);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }

        Convolution[] cons = new Convolution[] {
            new Convolution(3, new int[]{3, 3, 1}, 1, 1, Activation.SIGMOID, 4, 4, Pooling.AVERAGE)
        };

        Matrix3D weight = new Matrix3D(new double[][][] {
            {
                { -1.0 },
                { -1.0 },
                { -1.0 }
            },
            {
                { -1.0 },
                {  5.0 },
                { -1.0 }
            },
            {
                { -1.0 },
                { -1.0 },
                { -1.0 }
            },
        });
        //cons[0].setWeight(0, weight);
        ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(cons);

        Matrix3D output = cnn.compute(input);
        print(Matrix3D.flatten(output));
        
        try {
            ImageData.writeRGB("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02-EEEE.jpg", output);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }
    }
}
