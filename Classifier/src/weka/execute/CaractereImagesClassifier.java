package weka.execute;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.DecisionStump;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;


public class CaractereImagesClassifier {
	
	
	 private static final String DIGITOS = "digitos";
	    private static final String LETRAS = "letras";
	    private static final String DIGITOS_LETRAS = "digitos_letras";
	    private static final String SEM_CARACTERES = "sem_caracteres";
	    private static final int CAPACITY = 257;
	    private static final int INDEX = 256;

	    public static void main(String[] a) throws Exception {
	       
	        String parameter = "";//args[0];
	        
	        boolean verbose = parameter.equals("-v");
			String tec = "J48";//args[0];
			String pathTrain = "G:\\projeto de metodologia científica\\training";//args[1];
			String pathTest = "G:\\projeto de metodologia científica\\test";//args[2];
	        HashMap<String, Classifier> classifers = new HashMap<String, Classifier>(){
				private static final long serialVersionUID = 1L;
			{
	            put("NaiveBayes", new NaiveBayes());
	            put("Bagging", new Bagging());
	            put("DecisionStump", new DecisionStump());
	            put("CostSensitiveClassifier", new CostSensitiveClassifier());
	        }};
	        ArrayList<Attribute> wekaAttributes = new ArrayList<Attribute>(CAPACITY);
	        for (int i = 0; i < INDEX; i++) {
	            Attribute attr = new Attribute("numeric" + i);
	            wekaAttributes.add(attr);
	        }
	        ArrayList<String> classes = new ArrayList<String>(4);
	        classes.add(DIGITOS);
	        classes.add(LETRAS);
	        classes.add(DIGITOS_LETRAS);
	        classes.add(SEM_CARACTERES);
	        Attribute attr = new Attribute("classes", classes);

	        wekaAttributes.add(attr);
	        Instances trainingSet = new Instances("Rel", wekaAttributes, 1);
	        trainingSet.setClassIndex(INDEX);
	        // Pegar no parâmetro
	        // naiveBayes
	        // bagging
	        // zeroR
	        // decisionStump
	        //classificationviaregression
	        //costsensitiveclassifier
	        Classifier cModel = classifers.get(tec);
	        String folderDigits = pathTrain + "/digitos";
	        String folderLetters = pathTrain + "/letras";
	        String folderBoth = pathTrain + "/digitos_letras";
	        String nothing = pathTrain + "/sem_caracteres";
	        buildTrainingSet(wekaAttributes, trainingSet, folderDigits, DIGITOS);
	        buildTrainingSet(wekaAttributes, trainingSet, folderLetters, LETRAS);
	        buildTrainingSet(wekaAttributes, trainingSet, folderBoth, DIGITOS_LETRAS);
	        buildTrainingSet(wekaAttributes, trainingSet, nothing, SEM_CARACTERES);
	        cModel.buildClassifier(trainingSet);

	        Evaluation eTest = new Evaluation(trainingSet);
	        Instances testingSet = new Instances("Reltst", wekaAttributes, 1);
	        testingSet.setClassIndex(INDEX);
	        String folderTestLetters = pathTest + "/letras";
	        String folderTestDigits = pathTest + "/digitos";
	        String folderTestBoth = pathTest + "/digitos_letras";
	        String nothingTest = pathTest + "/sem_caracteres";
	        buildTrainingSet(wekaAttributes, testingSet, folderTestLetters, LETRAS);
	        buildTrainingSet(wekaAttributes, testingSet, folderTestDigits, DIGITOS);
	        buildTrainingSet(wekaAttributes, testingSet, folderTestBoth, DIGITOS_LETRAS);
	        buildTrainingSet(wekaAttributes, testingSet, nothingTest, SEM_CARACTERES);
	        eTest.evaluateModel(cModel, testingSet);

	        if(verbose) {
	            System.out.println(eTest.toSummaryString(true));
	            System.out.println(eTest.toClassDetailsString());
	        }
	        System.out.println("precision: " + eTest.weightedPrecision());
	        System.out.println("recall: " + eTest.weightedRecall());
	        System.out.println("f-measure: " + eTest.weightedFMeasure());
	    }

	    private static void buildTrainingSet(ArrayList<Attribute> wekaAttributes, Instances trainingSet, String folderName, String classe) throws Exception {
	        File folder = new File(folderName);
	        File[] listOfFiles = folder.listFiles();
	        for (File f : listOfFiles) {
	            double[] histogram = buildHistogram(f);
	            createTrainingSet(trainingSet, wekaAttributes, histogram, classe);
	        }
	    }

	    private static void createTrainingSet(Instances isTrainingSet, ArrayList<Attribute> wekaAttributes, double[] histogram, String classe) {

	        Instance imageInstance = new DenseInstance(CAPACITY);
	        for (int i = 0; i < histogram.length; i++) {
	            imageInstance.setValue((Attribute) wekaAttributes.get(i), histogram[i]);
	        }
	        if (!classe.isEmpty()) {
	            imageInstance.setValue((Attribute) wekaAttributes.get(INDEX), classe);
	        }
	        isTrainingSet.add(imageInstance);
	    }

	//////////////// helper code /////////////////////////

	    private static final double LUMINANCE_RED = 0.299D;
	    private static final double LUMINANCE_GREEN = 0.587D;
	    private static final double LUMINANCE_BLUE = 0.114;
	    private static final int HIST_WIDTH = 256;
	    private static final int HIST_HEIGHT = 100;

	    /**
	     * Parses pixels out of an image file, converts the RGB values to
	     * its equivalent grayscale value (0-255), then constructs a
	     * histogram of the percentage of counts of grayscale values.
	     *
	     * @param infile - the image file.
	     * @return - a histogram of grayscale percentage counts.
	     */
	    protected static double[] buildHistogram(File infile) throws Exception {
	        BufferedImage input = ImageIO.read(infile);
	        int width = input.getWidth();
	        int height = input.getHeight();
	        List<Integer> graylevels = new ArrayList<Integer>();
	        double maxWidth = 0.0D;
	        double maxHeight = 0.0D;
	        for (int row = 0; row < width; row++) {
	            for (int col = 0; col < height; col++) {
	                Color c = new Color(input.getRGB(row, col));
	                int graylevel = (int) (LUMINANCE_RED * c.getRed() +
	                        LUMINANCE_GREEN * c.getGreen() +
	                        LUMINANCE_BLUE * c.getBlue());
	                graylevels.add(graylevel);
	                maxHeight++;
	                if (graylevel > maxWidth) {
	                    maxWidth = graylevel;
	                }
	            }
	        }
	        double[] histogram = new double[HIST_WIDTH];
	        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
	            int idx = graylevel;
	            histogram[idx] += Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / maxHeight;
	        }
	        return histogram;
	    }
	
}

