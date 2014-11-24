package weka.execute;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.training.Training;

public class Execute {
	public static void main(String[] args) {
		
	    boolean verbose = false; //args[0].equals("-v");
	    
	    String technique = "NaiveBayes"; //args[0];
	    String trainingPath = "G:\\projeto de metodologia científica\\training";//args[1];
	    String testPath = "G:\\projeto de metodologia científica\\test"; //args[2];
	    
	    if(verbose){
	    	technique = args[1];
	    }
	    
		Training training = new Training();
		
		ArrayList<Attribute> allAttributes = training.generateTrainingRelation();
		//set relation in the memory
		Instances relationInstance =  new Instances("Relation", allAttributes, 0);
		//set the class attribute
		relationInstance.setClassIndex(Training.CLASS_INDEX);
		
		for(String category : Training.TECHNIQUES){
			String path = trainingPath + File.separator + category;
			try {
				training.generatedTrainingSet(path, allAttributes, relationInstance, category);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		Classifier classificationTechnique = training.get(technique);
		try {
			classificationTechnique.buildClassifier(relationInstance);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		try {
			Evaluation evaluation = new Evaluation(relationInstance);
			Instances relationTest = new Instances("RelationTest", allAttributes, 1);
	        relationTest.setClassIndex(Training.CLASS_INDEX);
	        
	        for(String category : Training.TECHNIQUES){
				String path = testPath + File.separator + category;
				try {
					training.generatedTrainingSet(path, allAttributes, relationTest, category);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			evaluation.evaluateModel(classificationTechnique, relationTest);
			if(verbose) {
	            System.out.println(evaluation.toSummaryString(true));
	            System.out.println(evaluation.toClassDetailsString());
	        }
	        System.out.println("precision: " + evaluation.weightedPrecision());
	        System.out.println("recall: " + evaluation.weightedRecall());
	        System.out.println("f-measure: " + evaluation.weightedFMeasure());
			
		} catch (Exception e1) {
			e1.printStackTrace();
			System.out.println("ERRO");
		}
	}
}
