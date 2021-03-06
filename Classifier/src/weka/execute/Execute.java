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
		
	    boolean verbose = Boolean.valueOf(args[0]);
	    String technique = args[1];
	    String trainingPath = args[2];
	    String testPath = args[3];
	    
		Training training = new Training();
		
		ArrayList<Attribute> allAttributes = training.generateTrainingRelation();
		//set relation in the memory
		Instances relationInstance =  new Instances("Relation", allAttributes, 0);
		//set the class attribute
		relationInstance.setClassIndex(Training.CLASS_INDEX);
		
		for(String category : Training.CLASSES){
			String path = trainingPath + File.separator + category;
			try {
				training.generatedTrainingSet(path, allAttributes, relationInstance, category);
			} catch (IOException e) {
				//e.printStackTrace();
				System.exit(1);
			}
		}
		
		Classifier classificationTechnique = training.get(technique);
		try {
			classificationTechnique.buildClassifier(relationInstance);
		} catch (Exception e) {
			//e.printStackTrace();
			System.exit(1);
		}
		
		try {
			Evaluation evaluation = new Evaluation(relationInstance);
			Instances instancesTest = new Instances("RelationTest", allAttributes, 1);
	        instancesTest.setClassIndex(Training.CLASS_INDEX);
	        
	        for(String category : Training.CLASSES){
				String path = testPath + File.separator + category;
				try {
					training.generatedTrainingSet(path, allAttributes, instancesTest, category);
				} catch (IOException e) {
					//e.printStackTrace();
					System.exit(1);
				}
			}
			evaluation.evaluateModel(classificationTechnique, instancesTest);
			if(verbose) {
				training.printClass(allAttributes, classificationTechnique, instancesTest.listIterator());
	           /* System.out.println(evaluation.toSummaryString(true));
	            System.out.println(evaluation.toClassDetailsString());*/
	        }
			
	        System.out.println("precision: " + evaluation.weightedPrecision());
	        System.out.println("recall: " + evaluation.weightedRecall());
	        System.out.println("f-measure: " + evaluation.weightedFMeasure());
 			
		} catch (Exception e1) {
			//e1.printStackTrace();
			System.exit(1);
		}
	}
}
