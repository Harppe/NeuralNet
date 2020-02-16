import java.util.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.*;
/**
 * Grayscale image class for conversion of image to 2d array of floats
 * all training images are 28x28 from nmist
 */
public class Image
{
    private final int DIMENSION_SIZE = 28;
    private final float COLOR_DEPTH = 255f;
    float[][] map;

    public static void test() throws Exception {
        Image i = new Image("mnist_png/testing/0/3.png");
        System.out.println(i);
    }

    public static float[][] imageToMap(String filePath) {
        try {
            Image imag = new Image(filePath);
            return imag.getMap();
        }
        catch (Exception e) {
            System.out.println(e);
        }
        return null;
    }

    public Image()
    {
    }

    public float[][] getMap() {
        return map;   
    }

    public Image(String filePath) throws Exception {
        map = new float[DIMENSION_SIZE][DIMENSION_SIZE];
        BufferedImage image = ImageIO.read(new File(filePath));
        for (int i = 0; i<DIMENSION_SIZE; i++) {
            for (int j = 0; j<DIMENSION_SIZE; j++) {
                int rgb = image.getRGB(j,i);
                short r = (short)((rgb>>16) & 0xFF);
                short g = (short)((rgb>>8) & 0xFF);
                short b = (short)((rgb>>0) & 0xFF);
                float grayScaleVal = (r+g+b)/3f;
                map[i][j] = grayScaleVal/COLOR_DEPTH;
            }
        }
    }

    /**
     * Literally an ascii representation of the image array either in 0 or 1
     */
    public String toString() {
        String str = "";
        for (int i = 0; i<DIMENSION_SIZE; i++) {
            for (int j = 0; j<DIMENSION_SIZE; j++) {
                if (map[i][j]==0)
                    str+=map[i][j]+ " ";
                else
                    str+=1.0 + " ";
            }
            str+="\n";
        }
        return str;
    }
}
