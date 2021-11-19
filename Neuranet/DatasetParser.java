package Neuranet;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import Neuranet.RuntimeExceptions.InvalidDatasetFormat;

import java.util.ArrayList;

/**
 * Class that converts Dataset values from
 * txt files into arrays of Datasets. Files
 * must be in format:
 * i_1 i_2 i_3 ... i_n | o_1 o_2 o_3 ... o_m
 * such that every line constitutes one Dataset.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class DatasetParser {
    /**
     * Returns an array of Datasets parsed from the file.
     * @param fileName the file path to parse from.
     * @return an array of Datasets parsed from the file.
     * @throws FileNotFoundException returned if the file is not valid.
     * @throws InvalidDatasetFormat returned if the dataset format in the file is invalid.
     */
    public static Dataset[] parse(String fileName) throws FileNotFoundException, InvalidDatasetFormat {
        ArrayList<Dataset> datasets = new ArrayList<>();
        
        File file = new File(fileName);
        Scanner reader = new Scanner(file);
        int lineIndex = 1;
        while (reader.hasNextLine()) {
            String line = reader.nextLine();
            if (line.split("\\|").length != 2) {
                System.out.println(line.split("\\|")[2]);
                reader.close();
                throw new InvalidDatasetFormat(lineIndex, fileName, line);
            }

            String[] inputs = line.split("\\|")[0].trim().split(" ");
            String[] outputs = line.split("\\|")[1].trim().split(" ");

            Matrix2D input = new Matrix2D(inputs.length, 1);
            Matrix2D output = new Matrix2D(outputs.length, 1);

            for (int index = 0; index < inputs.length; index += 1) {
                input.set(index, 0, Double.parseDouble(inputs[index]));
            }

            for (int index = 0; index < outputs.length; index += 1) {
                output.set(index, 0, Double.parseDouble(outputs[index]));
            }

            datasets.add(new Dataset(input, output));

            lineIndex += 1;
        }
        reader.close();
        Dataset[] out = new Dataset[datasets.size()];
        return datasets.toArray(out);
    }

    /**
     * Writes a list of datasets to the given file path.
     * @param fileName the file path to write the dataset to.
     * @param inputs the inputs to write to the file.
     * @param outputs the expected outputs to write to the file.
     * @throws FileNotFoundException returned if the file is not valid.
     * @throws InvalidDatasetFormat returned if the dataset format in the file is invalid.
     */
    public static void write(String fileName, Matrix2D[] inputs, Matrix2D[] outputs) throws FileNotFoundException, InvalidDatasetFormat {

    }
}
