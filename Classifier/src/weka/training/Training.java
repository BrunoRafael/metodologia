package weka.training;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Training {
	private static Map<String, Classifier> classifiers;
	public static final int CLASS_INDEX = 256;
	public static final int TOTAL_CLASS = 4;
	public static final int INSTANCES_SIZE = 257;
	
	public static final String[] TECHNIQUES = {"digitos", "letras", "digitos_letras", "sem_caracteres" };
	
	public static void createClassifiers() {
		classifiers = new HashMap<String, Classifier>();
		classifiers.put("NaiveBayes", new NaiveBayes());
		classifiers.put("J48", new J48());
		classifiers.put("MultiClassClassifier", new MultiClassClassifier());
		classifiers.put("MultilayerPerceptron", new MultilayerPerceptron());
	}
	
	public Training(){
		createClassifiers();
	}
	
	public ArrayList<Attribute> generateTrainingRelation(){
		
		ArrayList<Attribute> allAttributes = new ArrayList<>();
		
		// add colorScale of all images
		for(int i = 0; i <= 255; i++){
			allAttributes.add(new Attribute("numeric " + i));
		}
		
		//add the data classes 
		ArrayList<String> classes = new ArrayList<>(TOTAL_CLASS);
		for(String t : TECHNIQUES){
			classes.add(t);
		}
		Attribute classesAttribute = new Attribute("classes", classes);
		
		allAttributes.add(classesAttribute);
		
		return allAttributes;
		
	}
	
	public void generatedTrainingSet(String path, ArrayList<Attribute> allAttributes, Instances relationInstance, String category) throws IOException {
		
		File files = new File(path);
		for(File f : files.listFiles()){
			List<Double> percentColors = buildArrayPercentageColor(f);
			trainingAlgorithm(relationInstance, allAttributes, percentColors, category);
		}
	}

	private static void trainingAlgorithm(Instances relationInstance, ArrayList<Attribute> allAttributes, List<Double> percentColors, String category) {
		Instance instance = new DenseInstance(relationInstance.numAttributes());
		for(int i = 0 ; i < percentColors.size(); i++){
			instance.setValue(allAttributes.get(i), percentColors.get(i));
		}
		
		instance.setValue(allAttributes.get(CLASS_INDEX), category);
		relationInstance.add(instance);
		
	}
	
	
	private static final double LUMINANCE_RED = 0.299D;
    private static final double LUMINANCE_GREEN = 0.587D;
    private static final double LUMINANCE_BLUE = 0.114;
    
	private static  List<Double> buildArrayPercentageColor(File f) throws IOException {
		BufferedImage img = ImageIO.read(f);
        int width = img.getWidth();
        int height = img.getHeight();
        List<Integer> colors = new ArrayList<Integer>();
        double maxWidth = 0.0D;
        double maxHeight = 0.0D;
        for (int col = 0; col < height; col++) {
            for (int row = 0; row < width; row++) {
                Color c = new Color(img.getRGB(row, col));
               /* colors.add(c.getRed());
                colors.add(c.getGreen());
                colors.add(c.getBlue());*/
                int grayScale = (int) (LUMINANCE_RED * c.getRed() +
                        LUMINANCE_GREEN * c.getGreen() +
                        LUMINANCE_BLUE * c.getBlue());
                colors.add(grayScale);
                maxHeight++;
                if (grayScale > maxWidth) {
                    maxWidth = grayScale;
                }
            }
        }
        List<Double> scalesPercents = new LinkedList<>();
        for (Integer color : (new HashSet<Integer>(colors))) {
        	scalesPercents.add(Collections.frequency(colors, color) * 100 / maxHeight);
           // scalesPercents.add((double) (Collections.frequency(colors, color) * 100 / (height * width)));
        }
        return scalesPercents;
	}

	public static void printClass(List<Instance> lettersInstances) throws Exception {
		
		/*ListIterator<Instance> it = lettersInstances.listIterator();
		while(it.hasNext()){
			Instance inst = it.next();
		}*/
		
	}

	public Classifier get(String technique) {
		return classifiers.get(technique);
	}

}
