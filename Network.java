import java.util.*;
import java.io.*;
/**
 * Write a description of class Network here.
 * 
 * Input layer is 784 nodes, two hidden layers of 16 nodes, final layer of 10 nodes.
 * 
 * Error = output - targetOutput
 * 
 * deltaBiasHidden = Sigma(weight*Error) * Error*output*(1-output)
 * 
 * deltaBiasOutput = Error*output*(1-output)
 * 
 * deltaWeightOutput = -LR*output*Error*output*(1-output)
 * 
 * deltaWeightHidden = 
 * 
 * forwardFeeding-> Sigma(input*weight)+bias = E -> sigmoid(E) -> pass to next layer
 * [output*1-output is derivative of sigmoid function]
 * 
 * change = LR * Error * bias/weight
 * 
 */
public class Network
{
    private List<List<String>> trainingImagePaths; // list of list of string image file paths in order of num
    private final String trainingImageDirectory = "mnist_png/training/";
    private float[] inputNodes;
    private float[][] edgesA;
    private float[] blackLayerA;
    private float[] blackLayerAStorage;
    private float[][] edgesB;
    private float[] blackLayerB;
    private float[] blackLayerBStorage;
    private float[][] edgesC;
    private float[] outputNodes;
    private float[] outputNodesStorage;
    private float[] outputNodeSolutions;
    private float[] outputNodeErrors;
    private float[] blackLayerBErrors;
    private float[] blackLayerAErrors;
    private float learningRate = .33f;
    private int currentDigit;
    private Iterator<String> imageIter;
    private String currentImage;
    /**
     * Constructor for objects of class Network
     */
    public Network()
    {
        inputNodes = new float[784]; // 28 * 28 pixels
        edgesA = new float[784][16];
        blackLayerA = new float[16];
        edgesB = new float[16][16];
        blackLayerB = new float[16];
        edgesC = new float[16][10];
        outputNodes = new float[10];
        outputNodeSolutions = new float[10];
        outputNodeErrors = new float[10];
        blackLayerBErrors = new float[16];
        blackLayerAErrors = new float[16];

        blackLayerAStorage = new float[16];
        blackLayerBStorage = new float[16];
        outputNodesStorage = new float[10];

        randomizeBiasesWeights();
        getImagePaths();
        currentDigit = 0;
        imageIter = trainingImagePaths.get(currentDigit).listIterator();
        currentImage = nextImage();
    }
    // private List<List<String>> trainingImagePaths; // list of list of string image file paths in order of num
    public String nextImage() {
        if (imageIter.hasNext()) {
            return imageIter.next();
        }
        else {
            currentDigit++;
            if (currentDigit<=9) {
                imageIter = trainingImagePaths.get(currentDigit).listIterator();
                return imageIter.next();
            }
            else {
                return null;
            }
        }
    }

    public void resetIter() {
        imageIter = trainingImagePaths.get(0).listIterator();
        currentDigit = 0;
        currentImage = imageIter.next();
    }

    /**
     * cycle through nmist databast n number of times
     */
    public void cycle(int iterations) {
        for (int i = 0; i<iterations; i++) {
            resetIter();
            trainAllImages();
            resetIter();
        }
    }

    public void trainAllImages() {
        while (currentImage!=null) {
            iterate();
        }
    }

    public float sigmoid(float value) {
        return (float)(1/(1+Math.exp(-value)));
    }

    public float randNumb() {
        float num = (float)Math.random();
        if (Math.random()>0.5) {
            num = -num;
        }
        return num;
    }

    public void randomizeBiasesWeights() {
        for (int i = 0; i<blackLayerA.length; i++) {
            blackLayerA[i] = randNumb();
        }
        for (int i = 0; i<blackLayerB.length; i++) {
            blackLayerB[i] = randNumb();
        }
        for (int i = 0; i<outputNodes.length; i++) {
            outputNodes[i] = randNumb();
        }
        for (int i = 0; i<edgesA.length; i++) {
            for (int j = 0; j<edgesA[i].length; j++) {
                edgesA[i][j] = randNumb();
            }
        }
        for (int i = 0; i<edgesB.length; i++) { // all have same size
            for (int j = 0; j<edgesB[i].length; j++) {
                edgesB[i][j] = randNumb();
            }
        }
        for (int i = 0; i<edgesC.length; i++) {
            for (int j = 0; j<edgesC[i].length; j++) {
                edgesC[i][j] = randNumb();
            }
        }
    }

    /**
     * step through one image
     */
    public void iterate() {
        //trainingImagePaths
        if (currentImage!=null) {
            loadImageToInputNodes(currentImage);
            feedforward();
            backpropogate();
        }
        currentImage = nextImage();
    }

    public void printOutStorage() {
        String str = "[";
        for (float f : outputNodesStorage) {
            str+=f+", ";
        }
        str=str.substring(0,str.length()-2)+"]";
        System.out.println(str);
    }

    public int guess(String path) {
        loadImageToInputNodes(path);
        feedforward();
        //backpropogate(); no need to backprop
        int smallestDigit = 0;
        for (int i = 0; i<outputNodesStorage.length;i++) {
            float difference = difference(1f,outputNodesStorage[i]);
            if (difference<difference(1f,outputNodesStorage[smallestDigit]))
                smallestDigit = i;
        }
        return smallestDigit;
    }

    /**
     * val*weight summation plus bias then through sigmoid
     */

    public void feedforward() {
        float total = 0f;
        // input nodes - > black layer A
        for (int i =0; i<blackLayerA.length; i++) {
            float nodeVal = blackLayerA[i]; // just bias at this line
            for (int j = 0; j<inputNodes.length; j++) {
                nodeVal+=inputNodes[j]*edgesA[j][i];
            }
            nodeVal = sigmoid(nodeVal);
            blackLayerAStorage[i] = nodeVal;
        }
        // black layer A -> black layer B
        for (int i =0; i<blackLayerB.length; i++) {
            float nodeVal = blackLayerB[i]; // just bias at this line
            for (int j = 0; j<blackLayerA.length; j++) {
                nodeVal+=blackLayerAStorage[j]*edgesB[j][i];
            }
            nodeVal = sigmoid(nodeVal);
            blackLayerBStorage[i] = nodeVal;
        }
        // black layer B -> output nodes
        for (int i =0; i<outputNodes.length; i++) {
            float nodeVal = outputNodes[i]; // just bias at this line
            for (int j = 0; j<blackLayerB.length; j++) {
                nodeVal+=blackLayerBStorage[j]*edgesC[j][i];
            }
            nodeVal = sigmoid(nodeVal);
            outputNodesStorage[i] = nodeVal;
        }

    }

    private float difference(float a, float b) {
        return Math.abs(b-a);   
    }
    // private float[] inputNodes;
    // private float[][] edgesA;
    // private float[] blackLayerA;
    // private float[][] edgesB;
    // private float[] blackLayerB;
    // private float[][] edgesC;
    // private float[] outputNodes;
    /**
     * correcting weights and biases
     * 
     * Ei = Oi-Ti : output minus target equals error
     */
    public void backpropogate() {
        outputNodeSolutions = new float[10];
        outputNodeSolutions[currentDigit] = 1f;
        for (int i = 0; i<outputNodes.length; i++) {
            float errorBias = outputNodesStorage[i]-outputNodeSolutions[i];
            errorBias = -errorBias;
            float deltaBias = errorBias*deriveSigmoid(outputNodesStorage[i]);
            outputNodeErrors[i]=deltaBias;
            outputNodes[i]+=deltaBias;
        }
        for (int i = 0; i<blackLayerB.length; i++) {
            float errorBias = 0;
            for (int j = 0; j<outputNodes.length; j++) {
                errorBias+=edgesC[i][j]*outputNodeErrors[j];
            }
            errorBias = errorBias;
            float deltaBias = errorBias*deriveSigmoid(blackLayerBStorage[i]);
            blackLayerBErrors[i]=deltaBias;
            blackLayerB[i]+=deltaBias;
        }
        for (int i = 0; i<blackLayerA.length; i++) {
            float errorBias = 0;
            for (int j = 0; j<blackLayerB.length; j++) {
                errorBias+=edgesB[i][j]*blackLayerBErrors[j];
            }
            errorBias = errorBias;
            float deltaBias = errorBias*deriveSigmoid(blackLayerAStorage[i]);
            blackLayerAErrors[i] = deltaBias;
            blackLayerA[i]+=deltaBias;
        }
        // edges now

        for (int i = 0; i<blackLayerB.length; i++) {
            for (int j = 0; j<outputNodes.length; j++) {
                edgesC[i][j]-=outputNodeErrors[j]*blackLayerBStorage[i]*learningRate;
            }
        }
        for (int i = 0; i<blackLayerA.length; i++) {
            for (int j = 0; j<blackLayerB.length; j++) {
                edgesB[i][j]-=blackLayerBErrors[j]*blackLayerAStorage[i]*learningRate;
            }
        }
        for (int i = 0; i<inputNodes.length; i++) {
            for (int j = 0; j<blackLayerA.length; j++) {
                edgesA[i][j]-=blackLayerAErrors[j]*inputNodes[i]*learningRate;
            }
        }
    }

    private float deriveSigmoid(float val) {
        return val*(1-val);   
    }

    public void loadImageToInputNodes(String filePath) {
        float[][] pixelMap = Image.imageToMap(filePath);
        int currentIndex = 0;
        for (float[] layer:pixelMap) {
            for (float pixel:layer) {
                inputNodes[currentIndex] = pixel;
                currentIndex++;
            }
        }
    }

    public static void test() throws Exception {
        Network n = new Network();
        n.getImagePaths();
        String testPath = n.trainingImagePaths.get(2).get(0);
        System.out.println(testPath);
        System.out.println(new Image(testPath));
    }

    public void getImagePaths() {
        trainingImagePaths = new ArrayList<List<String>>();
        for (int i = 0; i<=9; i++) {
            File folder = new File("mnist_png/training/"+i);
            File[] files = folder.listFiles();
            List<String> fileNames = new ArrayList<String>();
            for (File f:files) {
                String name = "mnist_png/training/"+i+"/"+f.getName();
                fileNames.add(name);
            }
            trainingImagePaths.add(i,fileNames);
        }
    }
}
