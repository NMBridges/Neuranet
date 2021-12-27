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
        
        long startTime = System.nanoTime();

        NeuralNetwork net = new NeuralNetwork(new int[]{3, 15, 15, 3}, Activation.SIGMOID);
        print(net);
        
        Dataset[] datasets = new Dataset[0];
        try {
            datasets = DatasetParser.parse("datasets1.txt");
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }

        print("Average loss: " + net.getAverageLoss( datasets ));
        net.learn(datasets, 15000, 1, 1.0);
        print("Average loss after learning: " + net.getAverageLoss(datasets));

        try {
            datasets = DatasetParser.parse("datasets1test.txt");
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

        System.out.println("Time: " +  (0.000000001 * (System.nanoTime() - startTime)) + " seconds.");

        /*try {
            Matrix3D imgData = ImageData.parseImageGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02.jpg", 400, 200);
            ImageData.writeGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02-Scaled.png", imgData);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }*/

        /*Matrix3D input = new Matrix3D();
        try {
            input = ImageData.parseImageGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\HandOutOfWater.png", 100, 100);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }

        Convolution[] cons = new Convolution[] {
            new Convolution(1, new int[]{3, 3, 1}, 1, 1, Activation.SIGMOID, 1, 1, Pooling.AVERAGE)
        };

        Matrix3D weight0 = new Matrix3D(new double[][][] {
            {
                { -1.0 }, { -2.0 }, {  -1.0 }
            },
            {
                {  0.0 }, {  0.0 }, {  0.0 }
            },
            {
                {  1.0 }, {  2.0 }, {  1.0 }
            },
        });

        Matrix3D weight1 = new Matrix3D(new double[][][] {
            {
                { -1.0 }, {  0.0 }, {  1.0 }
            },
            {
                { -2.0 }, {  0.0 }, {  2.0 }
            },
            {
                { -1.0 }, {  0.0 }, {  1.0 }
            },
        });
        cons[0].setWeight(0, weight0);
        //cons[0].setWeight(1, weight1);
        ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(cons);

        Matrix3D output = cnn.compute(input);
        //print(Matrix3D.flatten(output));
        
        try {
            ImageData.writeGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\HandOutOfWater-Filtered.png", output.getLayers()[0]);
            //ImageData.writeGrayscale("C:\\Users\\scrat\\Pictures\\Saved Pictures\\LowPolyEye-02-Verti.jpg", output.getLayers()[1]);
        } catch (FileNotFoundException fnfe) {
            print(fnfe.getMessage());
        }*/

        /*try {
            System.out.println("INITIAL RANDOM: " + Matrix2D.random(5, 3, 0.4, 0.6));

            Matrix2D total = new Matrix2D(5, 3);
            int iterations = 5000000;
            for (int c = 0; c < iterations; c += 1)
            {
                total = Matrix2D.add(total, Matrix2D.random(5, 3, 0.4, 0.6));
            }

            System.out.println("Average: " + Matrix2D.divide(total, iterations));

            Matrix2D percError = Matrix2D.divide(Matrix2D.abs(Matrix2D.subtract(Matrix2D.random(5, 3, 0.5, 0.5), Matrix2D.divide(total, iterations))), 0.1);
            System.out.println("% Error: " + percError);

            System.out.println("Time: " +  (0.000000001 * (System.nanoTime() - startTime)) + " seconds.");
        }
        catch (Exception e)
        {
            System.out.println("Exception");
        }*/
    }
}
