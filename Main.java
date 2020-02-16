
/**
 * 
 */
public class Main
{
    private static String testZero = "mnist_png/testing/0/10.png";
    private static String testOne = "mnist_png/testing/1/29.png";
    private static String testNine = "mnist_png/testing/9/99.png";
    public static void main() throws Exception {
        System.out.println("\f");
        Network net = new Network();
        System.out.println();
        System.out.println("Before training: ");
        System.out.println("Zero Image: "+net.guess(testZero));
        net.printOutStorage();
        System.out.println("One Image: "+net.guess(testOne));
        net.printOutStorage();
        System.out.println("Nine Image: "+net.guess(testNine));
        net.printOutStorage();

        net.cycle(1);
        System.out.println();
        System.out.println("After training: ");
        System.out.println("Zero Image: "+net.guess(testZero));
        net.printOutStorage();
        System.out.println("One Image: "+net.guess(testOne));
        net.printOutStorage();
        System.out.println("Nine Image: "+net.guess(testNine));
        net.printOutStorage();
    }
}
