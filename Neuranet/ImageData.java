package Neuranet;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import Neuranet.Matrix3D;

/**
 * Class that interprets an image by returning
 * the grayscale value of each of its pixels
 * in an array.
 * @author Nolan Bridges
 * @version 1.0.0
 */
public class ImageData {
    /**
     * Parses an image and returns the grayscale
     * values of each pixel in the image (downscaled
     * to the resolution specified in the input).
     * @param fileName the file path of the image to process.
     * @param width the width of the rescaled image in pixels.
     * @param height the height of the rescaled image in pixels.
     * @return the Matrix of grayscale values.
     * @throws FileNotFoundException if the file path is invalid.
     */
    public static Matrix3D parseImageGrayscale(String fileName, int width, int height) throws FileNotFoundException {
        if (fileName == null || fileName.trim() == "") {
            throw new FileNotFoundException("Cannot find file " + fileName);
        }
        File file = new File(fileName);
        BufferedImage image = null;
        try {
            image = ImageIO.read(file);
        } catch (IllegalArgumentException iae) {
            System.out.println(iae.getMessage());
            return null;
        } catch (IOException ioe) {
            System.out.println(ioe.getMessage());
            return null;
        }

        Matrix3D out = new Matrix3D(width, height, 1);

        /** Downscales or upscales the image to the specified width / height. */
        for (int row = 0; row < height; row += 1) {
            for (int col = 0; col < width; col += 1) {
                int rowStart = (int) Math.ceil(1.0 * row / (height + 1) * image.getHeight());
                if (height > image.getHeight()) {
                    rowStart = (int) Math.floor(1.0 * row / (height + 1) * image.getHeight());
                }
                
                int rowEnd = (int) Math.ceil(1.0 * (row + 1) / (height + 1) * image.getHeight());

                int colStart = (int) Math.ceil(1.0 * col / (width + 1) * image.getWidth());
                if (width > image.getWidth()) {
                    colStart = (int) Math.floor(1.0 * col / (width + 1) * image.getWidth());
                }
                
                int colEnd = (int) Math.ceil(1.0 * (col + 1) / (width + 1) * image.getWidth());
                

                double totalGrayscaleValue = 0.0;
                for (int originalRow = rowStart; originalRow < rowEnd; originalRow += 1) {
                    for (int originalCol = colStart; originalCol < colEnd; originalCol += 1) {
                        int argb = image.getRGB(originalCol, originalRow);
                        int a = (argb >> 24) &0xf4;
                        int r = (argb >> 16) &0xf4;
                        int g = (argb >> 8) &0xf4;
                        int b = (argb) &0xf4;

                        totalGrayscaleValue += ((r / 255.0 + g / 255.0 + b / 255.0) * (a / 255.0)) / 3.0;
                    }
                }
                
                out.set(row, col, 0, totalGrayscaleValue / ((rowEnd - rowStart) * (colEnd - colStart)));
            }
        }

        return out;
    }

    /**
     * Writes an image with the specified grayscale
     * values to the disk at the specified file path.
     * @param fileName the file path to write the image to.
     * @param values the grayscale values for the pixels.
     * @param width the width of the image.
     * @param height the height of the image.
     * @throws FileNotFoundException when the image file path is invalid.
     */
    public static void writeGrayscale(String fileName, Matrix3D values) throws FileNotFoundException {
        if (fileName == null || fileName.trim() == "") {
            throw new FileNotFoundException("Invalid file name " + fileName);
        }

        int width = values.getColumnCount();
        int height = values.getRowCount();
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        for (int row = 0; row < height; row += 1) {
            for (int col = 0; col < width; col += 1) {
                int value = (int) (values.get(row, col, 0) * 255);
                image.setRGB(col, row, (255 << 24 | value << 16 | value << 8 | value));
            }
        }

        try {
            ImageIO.write(image, "png", new File(fileName));
        } catch (IOException ioe) {
            System.out.println(ioe.getMessage());
            return;
        }
    }

    /**
     * Writes an image with the specified RGB
     * values to the disk at the specified file path.
     * @param fileName the file path to write the image to.
     * @param values the RGB values for the pixels.
     * @param width the width of the image.
     * @param height the height of the image.
     * @throws FileNotFoundException when the image file path is invalid.
     */
    public static void writeRGB(String fileName, Matrix3D values) throws FileNotFoundException {
        if (fileName == null || fileName.trim() == "") {
            throw new FileNotFoundException("Invalid file name " + fileName);
        }

        int width = values.getColumnCount();
        int height = values.getRowCount();
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        for (int row = 0; row < height; row += 1) {
            for (int col = 0; col < width; col += 1) {
                int value = (int) (values.get(row, col, 0) * 255);
                image.setRGB(col, row, (255 << 24 | (int) (values.get(row, col, 0) * 255) << 16
                                | (int) (values.get(row, col, 1) * 255) << 8
                                | (int) (values.get(row, col, 2) * 255)));
            }
        }

        try {
            ImageIO.write(image, "png", new File(fileName));
        } catch (IOException ioe) {
            System.out.println(ioe.getMessage());
            return;
        }
    }
}
